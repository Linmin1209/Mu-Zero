# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native GR00T N1.7 action head with per-component CategorySpecificMLP decoders."""

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
)
from gr00t.model.modules.embodiment_conditioned_mlp import CategorySpecificMLP


logger = logging.getLogger(__name__)


class _StateDictLinearView:
    """Minimal adapter so CategorySpecificLinear.copy_ works from flat tensors."""

    def __init__(self, w: torch.Tensor, b: torch.Tensor):
        self.W = w
        self.b = b


def build_component_factored_action_head(base_cls: type):
    """Build ComponentFactoredActionHead without importing gr00t_n1d7 at module load time."""

    class ComponentFactoredActionHead(base_cls):
        """AlternateVLDiT unchanged; flat action encoder; per-component decode MLPs."""

        def __init__(self, config: Gr00tN1d7Config):
            super().__init__(config)
            del self.action_decoder

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
            logger.info(
                "ComponentFactoredActionHead: %d decoder segments, segments=%s",
                len(self.decoder_segments),
                [(s.name, s.start, s.end) for s in self.decoder_segments],
            )

            self.component_decoders = nn.ModuleDict()
            self.extra_decoders = nn.ModuleDict()
            for seg in self.decoder_segments:
                out_dim = seg.end - seg.start
                decoder = CategorySpecificMLP(
                    num_categories=config.max_num_embodiments,
                    input_dim=self.hidden_size,
                    hidden_dim=self.hidden_size,
                    output_dim=out_dim,
                )
                if seg.is_component:
                    self.component_decoders[seg.name] = decoder
                else:
                    self.extra_decoders[seg.name] = decoder

            self.set_trainable_parameters(
                config.tune_projector, config.tune_diffusion_model, config.tune_vlln
            )

        def _decoder_for_segment(self, seg: ComponentDecoderSegment) -> CategorySpecificMLP:
            if seg.is_component:
                return self.component_decoders[seg.name]
            return self.extra_decoders[seg.name]

        def set_trainable_parameters(
            self, tune_projector: bool, tune_diffusion_model: bool, tune_vlln: bool
        ):
            self.tune_projector = tune_projector
            self.tune_diffusion_model = tune_diffusion_model
            self.tune_vlln = tune_vlln
            for param in self.parameters():
                param.requires_grad = True
            if not tune_projector:
                self.state_encoder.requires_grad_(False)
                self.action_encoder.requires_grad_(False)
                self.component_decoders.requires_grad_(False)
                self.extra_decoders.requires_grad_(False)
                if self.config.add_pos_embed:
                    self.position_embedding.requires_grad_(False)
            if not tune_diffusion_model:
                self.model.requires_grad_(False)
            if not tune_vlln:
                self.vlln.requires_grad_(False)
                self.vl_self_attention.requires_grad_(False)

        def set_frozen_modules_to_eval_mode(self):
            if self.training:
                if not self.tune_projector:
                    self.state_encoder.eval()
                    self.action_encoder.eval()
                    self.component_decoders.eval()
                    self.extra_decoders.eval()
                    if self.config.add_pos_embed:
                        self.position_embedding.eval()
                if not self.tune_diffusion_model:
                    self.model.eval()
                if not self.tune_vlln:
                    self.vlln.eval()
                    self.vl_self_attention.eval()

        def decode_action_hidden(
            self, hidden: torch.Tensor, embodiment_id: torch.Tensor
        ) -> torch.Tensor:
            batch_size, horizon, _ = hidden.shape
            pred = hidden.new_zeros(batch_size, horizon, self.action_dim)
            for seg in self.decoder_segments:
                pred[:, :, seg.start : seg.end] = self._decoder_for_segment(seg)(
                    hidden, embodiment_id
                )
            return pred

        @staticmethod
        def _copy_category_linear(src: nn.Module, dst: nn.Module) -> None:
            with torch.no_grad():
                dst.W.copy_(src.W)
                dst.b.copy_(src.b)

        def load_flat_decoder_into_component_decoders(
            self, flat_decoder_state: Mapping[str, torch.Tensor]
        ) -> None:
            layer1_w = flat_decoder_state.get("layer1.W")
            layer1_b = flat_decoder_state.get("layer1.b")
            layer2_w = flat_decoder_state.get("layer2.W")
            layer2_b = flat_decoder_state.get("layer2.b")
            if layer1_w is None or layer2_w is None:
                logger.warning(
                    "Flat action_decoder weights missing; component decoders stay random."
                )
                return

            for seg in self.decoder_segments:
                decoder = self._decoder_for_segment(seg)
                self._copy_category_linear(
                    _StateDictLinearView(layer1_w, layer1_b),
                    decoder.layer1,
                )
                with torch.no_grad():
                    decoder.layer2.W.copy_(layer2_w[:, :, seg.start : seg.end])
                    decoder.layer2.b.copy_(layer2_b[:, seg.start : seg.end])
            logger.info(
                "Initialized %d component/extra decoders from pretrained flat action_decoder.",
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
            pred_actions = self.decode_action_hidden(hidden, embodiment_id)

            action_mask = action_input.action_mask
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
                    :,
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
                pred_velocity = self.decode_action_hidden(hidden, embodiment_id)
                actions = actions + dt * pred_velocity * vel_strength

            return BatchFeature(
                data={
                    "action_pred": actions,
                    "backbone_features": vl_embeds,
                    "state_features": state_features,
                }
            )

    return ComponentFactoredActionHead
