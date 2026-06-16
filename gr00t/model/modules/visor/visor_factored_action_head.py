# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Component-factored action head with VISOR suffix IHT + tactile auxiliary loss."""

from __future__ import annotations

import logging
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F
from transformers.feature_extraction_utils import BatchFeature

from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
from gr00t.model.modules.component_action.component_factored_action_head import (
    build_component_factored_action_head,
)
from gr00t.model.modules.visor.visor import VisorModule, build_asymmetric_sa_mask


logger = logging.getLogger(__name__)


def build_visor_factored_action_head(base_cls: type):
    ComponentFactoredActionHead = build_component_factored_action_head(base_cls)

    class VisorFactoredActionHead(ComponentFactoredActionHead):
        def __init__(self, config: Gr00tN1d7Config):
            super().__init__(config)
            iht_tokens = int(getattr(config, "visor_iht_tokens", 2))
            proprio_dim = config.max_state_dim * config.state_history_length
            self.visor = VisorModule(
                action_dim=self.action_dim,
                hidden_dim=int(getattr(config, "visor_hidden_dim", 256)),
                input_embedding_dim=self.input_embedding_dim,
                action_horizon=self.action_horizon,
                vision_dim=config.backbone_embedding_dim,
                proprio_dim=proprio_dim,
                decode_hidden_dim=self.hidden_size,
                iht_tokens=iht_tokens,
                loss_weight_tactile=float(getattr(config, "visor_loss_weight_tactile", 0.5)),
                contact_loss_weight=float(
                    getattr(config, "visor_contact_loss_weight", 1.0)
                ),
            )
            logger.info(
                "VisorFactoredActionHead: iht_tokens=%d loss_weight=%.3f",
                iht_tokens,
                self.visor.loss_weight_tactile,
            )
            self.set_trainable_parameters(
                config.tune_projector, config.tune_diffusion_model, config.tune_vlln
            )

        def set_trainable_parameters(
            self, tune_projector: bool, tune_diffusion_model: bool, tune_vlln: bool
        ):
            super().set_trainable_parameters(tune_projector, tune_diffusion_model, tune_vlln)
            if hasattr(self, "visor") and getattr(self.config, "tune_visor", True):
                self.visor.requires_grad_(True)

        def set_frozen_modules_to_eval_mode(self):
            super().set_frozen_modules_to_eval_mode()
            if self.training and not getattr(self.config, "tune_visor", True):
                self.visor.eval()

        def _expand_sa_mask(
            self, sa_mask: torch.Tensor, batch_size: int
        ) -> torch.Tensor:
            if sa_mask.shape[0] == 1:
                sa_mask = sa_mask.expand(batch_size, -1, -1)
            return sa_mask.unsqueeze(1)

        def _run_dit(
            self,
            sa_embs: torch.Tensor,
            vl_embeds: torch.Tensor,
            t_discretized: torch.Tensor,
            backbone_output: BatchFeature,
            sa_self_attention_mask: torch.Tensor | None,
            *,
            return_all_hidden_states: bool,
        ):
            vl_attn_mask = backbone_output.backbone_attention_mask
            if self.config.use_alternate_vl_dit:
                return self.model(
                    hidden_states=sa_embs,
                    encoder_hidden_states=vl_embeds,
                    encoder_attention_mask=vl_attn_mask,
                    timestep=t_discretized,
                    return_all_hidden_states=return_all_hidden_states,
                    image_mask=backbone_output.image_mask,
                    backbone_attention_mask=backbone_output.backbone_attention_mask,
                    sa_self_attention_mask=sa_self_attention_mask,
                )
            return self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds,
                encoder_attention_mask=vl_attn_mask,
                timestep=t_discretized,
                return_all_hidden_states=return_all_hidden_states,
                sa_self_attention_mask=sa_self_attention_mask,
            )

        def forward(self, backbone_output, action_input):
            self.set_frozen_modules_to_eval_mode()
            backbone_output = self.process_backbone_output(backbone_output)
            vl_embeds = backbone_output.backbone_features
            device = vl_embeds.device
            embodiment_id = action_input.embodiment_id

            assert action_input.state.shape[1] == self.config.state_history_length
            proprio = action_input.state.reshape(
                action_input.state.shape[0], -1
            )
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

            vision_context = self.visor.pool_vision_context(
                vl_embeds, backbone_output.image_mask
            )
            tactile_pred = self.visor.wwm(
                noisy_trajectory,
                t,
                vision_context,
                proprio,
                use_clean_action=False,
            )
            iht_tokens = self.visor.build_iht_tokens(tactile_pred)
            sa_embs = torch.cat((state_features, action_features, iht_tokens), dim=1)
            sa_mask = build_asymmetric_sa_mask(
                self.visor.native_seq_len,
                self.visor.iht_tokens,
                device=device,
                dtype=sa_embs.dtype,
            )
            sa_mask = self._expand_sa_mask(sa_mask, sa_embs.shape[0])

            model_output, _ = self._run_dit(
                sa_embs,
                vl_embeds,
                t_discretized,
                backbone_output,
                sa_mask,
                return_all_hidden_states=True,
            )

            hidden_action = model_output[:, 1 : 1 + self.action_horizon]
            event = tactile_pred.mean(dim=1)
            hidden_action = hidden_action + self.visor.gate * self.visor.gate_proj(
                event
            ).unsqueeze(1)
            pred_actions = self.decode_action_hidden(hidden_action, embodiment_id)

            action_mask = action_input.action_mask
            per_elem_loss = F.mse_loss(pred_actions, velocity, reduction="none")
            scaled_mask = action_mask.clone()
            for seg in self.decoder_segments:
                if seg.is_component:
                    scaled_mask[:, :, seg.start : seg.end] *= float(
                        self.schema.loss_weights.get(seg.name, 1.0)
                    )
            action_loss = per_elem_loss * action_mask
            mask_sum = scaled_mask.sum()
            if mask_sum <= 0:
                flow_loss = per_elem_loss.new_zeros(())
            else:
                flow_loss = (per_elem_loss * scaled_mask).sum() / mask_sum
            loss = flow_loss
            tactile_loss = torch.zeros((), device=device, dtype=loss.dtype)

            if hasattr(action_input, "tactile_gt") and action_input.tactile_gt is not None:
                tactile_gt = action_input.tactile_gt.to(device=device, dtype=t.dtype)
                t_clean = torch.ones_like(t)
                tactile_pred_supervised = self.visor.wwm(
                    actions,
                    t_clean,
                    vision_context,
                    proprio,
                    use_clean_action=True,
                )
                tactile_loss, tactile_stats = self.visor.compute_tactile_loss(
                    tactile_pred_supervised,
                    tactile_gt,
                )
                loss = loss + tactile_loss
            else:
                tactile_stats = {}

            return {
                "loss": loss,
                "flow_loss": flow_loss.detach().reshape(()),
                "tactile_loss": tactile_loss.detach().reshape(()),
                "action_loss": action_loss,
                "action_mask": action_mask,
                "backbone_features": vl_embeds,
                "state_features": state_features,
                "tactile_pred": tactile_pred.detach(),
                **{f"visor_{k}": v for k, v in tactile_stats.items()},
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
                t_ramp = torch.linspace(0.0, 1.0, intermediate_steps + 2, device=device)
                ramp = 1 - torch.exp(-options["rtc_ramp_rate"] * t_ramp)
                ramp = ramp / ramp[-1].clamp_min(1e-8)
                ramp = ramp[1:-1]
                vel_strength[
                    :,
                    options["rtc_frozen_steps"] : options["rtc_overlap_steps"],
                    :,
                ] = ramp[None, :, None].to(device)

            vision_context = self.visor.pool_vision_context(
                vl_embeds, backbone_output.image_mask
            )
            proprio = action_input.state.reshape(action_input.state.shape[0], -1)
            sa_mask = self._expand_sa_mask(
                build_asymmetric_sa_mask(
                    self.visor.native_seq_len,
                    self.visor.iht_tokens,
                    device=device,
                    dtype=vl_embeds.dtype,
                ),
                batch_size,
            )

            for step in range(self.num_inference_timesteps):
                t_cont = step / float(self.num_inference_timesteps)
                t_discretized = int(t_cont * self.num_timestep_buckets)
                timesteps_tensor = torch.full(
                    size=(batch_size,), fill_value=t_discretized, device=device
                )
                t_broadcast = torch.full(
                    (batch_size, 1, 1), fill_value=t_cont, device=device, dtype=actions.dtype
                )
                action_features = self.action_encoder(actions, timesteps_tensor, embodiment_id)
                if self.config.add_pos_embed:
                    pos_ids = torch.arange(
                        action_features.shape[1], dtype=torch.long, device=device
                    )
                    pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                    action_features = action_features + pos_embs

                tactile_pred = self.visor.wwm(
                    actions,
                    t_broadcast,
                    vision_context,
                    proprio,
                    use_clean_action=False,
                )
                iht_tokens = self.visor.build_iht_tokens(tactile_pred)
                sa_embs = torch.cat((state_features, action_features, iht_tokens), dim=1)

                model_output = self._run_dit(
                    sa_embs,
                    vl_embeds,
                    timesteps_tensor,
                    backbone_output,
                    sa_mask,
                    return_all_hidden_states=False,
                )
                hidden_action = model_output[:, 1 : 1 + self.action_horizon]
                event = tactile_pred.mean(dim=1)
                hidden_action = hidden_action + self.visor.gate * self.visor.gate_proj(
                    event
                ).unsqueeze(1)
                pred_velocity = self.decode_action_hidden(hidden_action, embodiment_id)
                actions = actions + dt * pred_velocity * vel_strength

            return BatchFeature(
                data={
                    "action_pred": actions,
                    "backbone_features": vl_embeds,
                    "state_features": state_features,
                }
            )

    return VisorFactoredActionHead
