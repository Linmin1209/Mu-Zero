# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native GR00T N1.7 action head with shared flat decoder + per-component LoRA experts."""

from __future__ import annotations

import logging
from typing import Mapping

import torch
from torch import nn
import torch.nn.functional as F
from transformers.feature_extraction_utils import BatchFeature

from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
from gr00t.model.modules.component_action.component_layout import (
    ComponentDecoderSegment,
    build_flat_action_decoder_segments,
)
from gr00t.model.modules.component_action.component_schema import (
    DEFAULT_COMPONENT_LOSS_WEIGHTS,
    ComponentSchemaConfig,
    component_index,
)
from gr00t.model.modules.embodiment_conditioned_mlp import (
    CategorySpecificLinear,
)


logger = logging.getLogger(__name__)


class CategorySpecificLoRA(nn.Module):
    """Low-rank per-embodiment adapter; zero-init B gives Day-0 identity on the residual path."""

    def __init__(
        self,
        num_categories: int,
        input_dim: int,
        output_dim: int,
        *,
        rank: int = 8,
        alpha: float = 8.0,
    ):
        super().__init__()
        self.scaling = alpha / max(int(rank), 1)
        self.lora_a = CategorySpecificLinear(num_categories, input_dim, rank)
        self.lora_b = CategorySpecificLinear(num_categories, rank, output_dim)
        nn.init.zeros_(self.lora_b.W)
        nn.init.zeros_(self.lora_b.b)

    def forward(self, x: torch.Tensor, cat_ids: torch.Tensor) -> torch.Tensor:
        return self.lora_b(self.lora_a(x, cat_ids), cat_ids) * self.scaling


def build_component_factored_action_head(base_cls: type):
    """Build ComponentFactoredActionHead without importing gr00t_n1d7 at module load time."""

    class ComponentFactoredActionHead(base_cls):
        """AlternateVLDiT unchanged; shared flat decoder + sparse per-component LoRA."""

        def __init__(self, config: Gr00tN1d7Config):
            super().__init__(config)

            override_dims = dict(getattr(config, "component_projector_dims", None) or {})
            loss_weights = dict(DEFAULT_COMPONENT_LOSS_WEIGHTS)
            loss_weights.update(getattr(config, "component_loss_weights", None) or {})
            self.schema = ComponentSchemaConfig(
                component_dims=override_dims if override_dims else None,
                loss_weights=loss_weights,
            )

            action_key_order = list(getattr(config, "component_action_key_order", None) or [])
            action_key_dims = dict(getattr(config, "component_action_key_dims", None) or {})
            embodiment_tag = getattr(config, "component_layout_embodiment_tag", None) or ""
            if not action_key_order or not action_key_dims or not embodiment_tag:
                raise ValueError(
                    "component_factored head requires component_action_key_order, "
                    "component_action_key_dims, and component_layout_embodiment_tag in config."
                )

            self.decoder_segments: list[ComponentDecoderSegment] = (
                build_flat_action_decoder_segments(
                    action_modality_keys=action_key_order,
                    action_key_dims=action_key_dims,
                    embodiment_tag=embodiment_tag,
                    schema=self.schema,
                )
            )
            self.component_lora_rank = int(getattr(config, "component_lora_rank", 8))
            self.component_lora_alpha = float(getattr(config, "component_lora_alpha", 8.0))
            self.component_lora_train_shared_decoder = bool(
                getattr(config, "component_lora_train_shared_decoder", False)
            )

            self.component_lora_adapters = nn.ModuleDict()
            self.extra_lora_adapters = nn.ModuleDict()
            for seg in self.decoder_segments:
                out_dim = seg.end - seg.start
                adapter = CategorySpecificLoRA(
                    config.max_num_embodiments,
                    self.hidden_size,
                    out_dim,
                    rank=self.component_lora_rank,
                    alpha=self.component_lora_alpha,
                )
                if seg.is_component:
                    self.component_lora_adapters[seg.name] = adapter
                else:
                    self.extra_lora_adapters[seg.name] = adapter

            logger.info(
                "ComponentFactoredActionHead: shared action_decoder + %d LoRA segments "
                "(rank=%d alpha=%.1f train_shared_decoder=%s) segments=%s",
                len(self.decoder_segments),
                self.component_lora_rank,
                self.component_lora_alpha,
                self.component_lora_train_shared_decoder,
                [(s.name, s.start, s.end) for s in self.decoder_segments],
            )

            self.set_trainable_parameters(
                config.tune_projector, config.tune_diffusion_model, config.tune_vlln
            )

        def _lora_for_segment(self, seg: ComponentDecoderSegment) -> CategorySpecificLoRA:
            if seg.is_component:
                return self.component_lora_adapters[seg.name]
            return self.extra_lora_adapters[seg.name]

        def _segment_active_mask(
            self,
            *,
            batch_size: int,
            device: torch.device,
            dtype: torch.dtype,
            action_input: BatchFeature | None = None,
            action_mask: torch.Tensor | None = None,
        ) -> torch.Tensor:
            """Per-sample segment activity (B, num_segments); defaults to all active."""
            num_segments = len(self.decoder_segments)
            active = torch.ones(batch_size, num_segments, device=device, dtype=dtype)

            component_mask = None
            if action_input is not None:
                if hasattr(action_input, "active_component_mask"):
                    component_mask = action_input.active_component_mask
                elif isinstance(action_input, dict) and "active_component_mask" in action_input:
                    component_mask = action_input["active_component_mask"]

            for seg_idx, seg in enumerate(self.decoder_segments):
                if component_mask is not None and seg.is_component:
                    try:
                        comp_idx = component_index(seg.name)
                    except ValueError:
                        continue
                    active[:, seg_idx] = (component_mask[:, comp_idx] > 0.5).to(dtype=dtype)
                elif action_mask is not None:
                    seg_mask = action_mask[:, :, seg.start : seg.end]
                    active[:, seg_idx] = (seg_mask.sum(dim=(1, 2)) > 0).to(dtype=dtype)

            return active

        def set_trainable_parameters(
            self, tune_projector: bool, tune_diffusion_model: bool, tune_vlln: bool
        ):
            self.tune_projector = tune_projector
            self.tune_diffusion_model = tune_diffusion_model
            self.tune_vlln = tune_vlln
            for param in self.parameters():
                param.requires_grad = True
            if not tune_projector:
                if hasattr(self, "state_encoder"):
                    self.state_encoder.requires_grad_(False)
                if hasattr(self, "action_encoder"):
                    self.action_encoder.requires_grad_(False)
                self.action_decoder.requires_grad_(False)
                self.component_lora_adapters.requires_grad_(False)
                self.extra_lora_adapters.requires_grad_(False)
                if self.config.add_pos_embed and hasattr(self, "position_embedding"):
                    self.position_embedding.requires_grad_(False)
            elif not self.component_lora_train_shared_decoder:
                self.action_decoder.requires_grad_(False)
            if hasattr(self, "model") and not tune_diffusion_model:
                self.model.requires_grad_(False)
            if hasattr(self, "vlln") and not tune_vlln:
                self.vlln.requires_grad_(False)
                if hasattr(self, "vl_self_attention"):
                    self.vl_self_attention.requires_grad_(False)

        def set_frozen_modules_to_eval_mode(self):
            if self.training:
                if not self.tune_projector:
                    if hasattr(self, "state_encoder"):
                        self.state_encoder.eval()
                    if hasattr(self, "action_encoder"):
                        self.action_encoder.eval()
                    self.action_decoder.eval()
                    self.component_lora_adapters.eval()
                    self.extra_lora_adapters.eval()
                    if self.config.add_pos_embed and hasattr(self, "position_embedding"):
                        self.position_embedding.eval()
                elif not self.component_lora_train_shared_decoder:
                    self.action_decoder.eval()
                if hasattr(self, "model") and not self.tune_diffusion_model:
                    self.model.eval()
                if hasattr(self, "vlln") and not self.tune_vlln:
                    self.vlln.eval()
                    if hasattr(self, "vl_self_attention"):
                        self.vl_self_attention.eval()

        def decode_action_hidden(
            self,
            hidden: torch.Tensor,
            embodiment_id: torch.Tensor,
            *,
            gate_delta: torch.Tensor | None = None,
            action_input: BatchFeature | None = None,
            action_mask: torch.Tensor | None = None,
        ) -> torch.Tensor:
            batch_size, _, _ = hidden.shape
            pred = self.action_decoder(hidden, embodiment_id)
            active = self._segment_active_mask(
                batch_size=batch_size,
                device=hidden.device,
                dtype=pred.dtype,
                action_input=action_input,
                action_mask=action_mask,
            )

            for seg_idx, seg in enumerate(self.decoder_segments):
                h_seg = hidden
                if gate_delta is not None and seg.name in getattr(
                    self, "visor_gate_components", ()
                ):
                    h_seg = hidden + gate_delta
                delta = self._lora_for_segment(seg)(h_seg, embodiment_id)
                seg_active = active[:, seg_idx].view(batch_size, 1, 1)
                pred[:, :, seg.start : seg.end] = pred[:, :, seg.start : seg.end] + (
                    seg_active * delta
                )
            return pred

        def load_flat_decoder_into_component_decoders(
            self, flat_decoder_state: Mapping[str, torch.Tensor]
        ) -> None:
            """Legacy hook: shared action_decoder loads from checkpoint; LoRA stays zero-init."""
            layer1_w = flat_decoder_state.get("layer1.W")
            layer2_w = flat_decoder_state.get("layer2.W")
            if layer1_w is None or layer2_w is None:
                logger.warning(
                    "Flat action_decoder weights missing in checkpoint slice; "
                    "shared decoder may be randomly initialized."
                )
            else:
                logger.info(
                    "Shared action_decoder present in checkpoint; "
                    "%d component LoRA adapters remain zero-init (Day-0 equivalent).",
                    len(self.decoder_segments),
                )

        def forward(self, backbone_output, action_input):
            self.set_frozen_modules_to_eval_mode()
            backbone_output = self.process_backbone_output(backbone_output)
            vl_embeds = backbone_output.backbone_features
            device = vl_embeds.device
            embodiment_id = action_input.embodiment_id

            assert action_input.state.shape[1] == self.config.state_history_length
            action_input.state = action_input.state.view(action_input.state.shape[0], 1, -1)
            state_features = self.state_encoder(action_input.state, embodiment_id)

            if self.training and self.state_dropout_prob > 0:
                do_dropout = (
                    torch.rand(state_features.shape[0], device=state_features.device)
                    < self.state_dropout_prob
                )
                do_dropout = do_dropout[:, None, None].to(dtype=state_features.dtype)
                state_features = state_features * (1 - do_dropout)

            actions = action_input.action
            noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
            t = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)
            t = t[:, None, None]
            noisy_trajectory = (1 - t) * noise + t * actions
            velocity = actions - noise
            t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
            action_features = self.action_encoder(
                noisy_trajectory, t_discretized, embodiment_id
            )

            if self.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs

            sa_embs = torch.cat((state_features, action_features), dim=1)
            vl_attn_mask = backbone_output.backbone_attention_mask

            if self.config.use_alternate_vl_dit:
                model_output, _ = self.model(
                    hidden_states=sa_embs,
                    encoder_hidden_states=vl_embeds,
                    encoder_attention_mask=vl_attn_mask,
                    timestep=t_discretized,
                    return_all_hidden_states=True,
                    image_mask=backbone_output.image_mask,
                    backbone_attention_mask=backbone_output.backbone_attention_mask,
                )
            else:
                model_output, _ = self.model(
                    hidden_states=sa_embs,
                    encoder_hidden_states=vl_embeds,
                    encoder_attention_mask=vl_attn_mask,
                    timestep=t_discretized,
                    return_all_hidden_states=True,
                )

            hidden = model_output[:, -actions.shape[1] :]
            action_mask = action_input.action_mask
            pred_actions = self.decode_action_hidden(
                hidden,
                embodiment_id,
                action_input=action_input,
                action_mask=action_mask,
            )

            per_elem_loss = F.mse_loss(pred_actions, velocity, reduction="none")
            scaled_mask = action_mask.clone()
            for seg in self.decoder_segments:
                if seg.is_component:
                    scaled_mask[:, :, seg.start : seg.end] *= float(
                        self.schema.loss_weights.get(seg.name, 1.0)
                    )
            action_loss = per_elem_loss * action_mask
            loss = (per_elem_loss * scaled_mask).sum() / (scaled_mask.sum() + 1e-6)

            return {
                "loss": loss,
                "action_loss": action_loss,
                "action_mask": action_mask,
                "backbone_features": vl_embeds,
                "state_features": state_features,
            }

        @torch.no_grad()
        def get_action_with_features(
            self,
            backbone_features,
            state_features,
            embodiment_id,
            backbone_output,
            action_input,
            options=None,
        ):
            vl_embeds = backbone_features
            batch_size = vl_embeds.shape[0]
            device = vl_embeds.device
            actions = torch.randn(
                size=(batch_size, self.config.action_horizon, self.action_dim),
                dtype=vl_embeds.dtype,
                device=device,
            )
            dt = 1.0 / self.num_inference_timesteps
            vel_strength = torch.ones_like(actions)

            if "action" in action_input:
                assert options is not None
                action_horizon_before_padding = options["action_horizon"]
                actions[:, : options["rtc_overlap_steps"], :] = action_input["action"][
                    :,
                    action_horizon_before_padding
                    - options["rtc_overlap_steps"] : action_horizon_before_padding,
                    :,
                ]
                vel_strength[:, : options["rtc_frozen_steps"], :] = 0.0
                intermediate_steps = options["rtc_overlap_steps"] - options["rtc_frozen_steps"]
                t = torch.linspace(0.0, 1.0, intermediate_steps + 2, device=device)
                ramp = 1 - torch.exp(-options["rtc_ramp_rate"] * t)
                ramp = ramp / ramp[-1].clamp_min(1e-8)
                ramp = ramp[1:-1]
                vel_strength[
                    :,
                    options["rtc_frozen_steps"] : options["rtc_overlap_steps"],
                ] = ramp[None, :, None].to(device)

            for step in range(self.num_inference_timesteps):
                t_cont = step / float(self.num_inference_timesteps)
                t_discretized = int(t_cont * self.num_timestep_buckets)
                timesteps_tensor = torch.full(
                    size=(batch_size,), fill_value=t_discretized, device=device
                )
                action_features = self.action_encoder(actions, timesteps_tensor, embodiment_id)
                if self.config.add_pos_embed:
                    pos_ids = torch.arange(
                        action_features.shape[1], dtype=torch.long, device=device
                    )
                    pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                    action_features = action_features + pos_embs

                sa_embs = torch.cat((state_features, action_features), dim=1)
                if self.config.use_alternate_vl_dit:
                    model_output = self.model(
                        hidden_states=sa_embs,
                        encoder_hidden_states=vl_embeds,
                        timestep=timesteps_tensor,
                        image_mask=backbone_output.image_mask,
                        backbone_attention_mask=backbone_output.backbone_attention_mask,
                    )
                else:
                    model_output = self.model(
                        hidden_states=sa_embs,
                        encoder_hidden_states=vl_embeds,
                        timestep=timesteps_tensor,
                    )

                hidden = model_output[:, -self.action_horizon :]
                pred_velocity = self.decode_action_hidden(
                    hidden,
                    embodiment_id,
                    action_input=action_input,
                )
                actions = actions + dt * pred_velocity * vel_strength

            return BatchFeature(
                data={
                    "action_pred": actions,
                    "backbone_features": vl_embeds,
                    "state_features": state_features,
                }
            )

    return ComponentFactoredActionHead
