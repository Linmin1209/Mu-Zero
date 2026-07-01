# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native flat action head with T-Rex-style sensor VISOR (tri-path IHT + flow-late)."""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F
from transformers.feature_extraction_utils import BatchFeature

from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
from gr00t.model.modules.visor.visor import (
    VisorModule,
    build_asymmetric_sa_mask,
    resolve_sensor_tactile,
)


logger = logging.getLogger(__name__)


def build_visor_flat_action_head(base_cls: type):
    class VisorFlatActionHead(base_cls):
        """Gr00tN1d7ActionHead + VISOR IHT tokens; keeps shared flat action_decoder."""

        def __init__(self, config: Gr00tN1d7Config):
            super().__init__(config)
            self.visor_tactile_warmup_steps = int(
                getattr(config, "visor_tactile_warmup_steps", 1000)
            )
            self.visor_detach_tactile_for_gate = bool(
                getattr(config, "visor_detach_tactile_for_gate", True)
            )
            self.register_buffer(
                "_visor_train_step", torch.zeros((), dtype=torch.long), persistent=False
            )

            self.visor = VisorModule(
                input_embedding_dim=self.input_embedding_dim,
                action_horizon=self.action_horizon,
                vision_dim=config.backbone_embedding_dim,
                decode_hidden_dim=self.hidden_size,
                flow_tau_split=float(getattr(config, "visor_flow_tau_split", 0.4)),
                history_vq_tokens=int(getattr(config, "visor_history_vq_tokens", 2)),
                vq_codebook_size=int(getattr(config, "visor_vq_codebook_size", 64)),
                vq_hidden_dim=int(getattr(config, "visor_vq_hidden_dim", 64)),
                loss_weight_tactile=float(getattr(config, "visor_loss_weight_tactile", 0.1)),
                contact_loss_weight=float(
                    getattr(config, "visor_contact_loss_weight", 1.0)
                ),
                vq_commit_weight=float(getattr(config, "visor_vq_commit_weight", 0.1)),
                use_contact_rate_prior=bool(
                    getattr(config, "visor_use_contact_rate_prior", True)
                ),
                use_semantic_gate=bool(getattr(config, "visor_use_semantic_gate", True)),
                language_dim=config.backbone_embedding_dim,
            )
            logger.info(
                "VisorFlatActionHead (T-Rex sensor): tau_split=%.2f iht_tokens=%d "
                "tactile_weight=%.3f warmup=%d",
                self.visor.flow_tau_split,
                self.visor.iht_tokens,
                self.visor.loss_weight_tactile,
                self.visor_tactile_warmup_steps,
            )
            self.set_trainable_parameters(
                config.tune_projector, config.tune_diffusion_model, config.tune_vlln
            )

        def _coupling_scale(self) -> float:
            if not self.training or self.visor_tactile_warmup_steps <= 0:
                return 1.0
            step = float(self._visor_train_step.item())
            return min(1.0, step / float(self.visor_tactile_warmup_steps))

        def _get_tactile_sensor(self, action_input) -> torch.Tensor | None:
            return getattr(action_input, "tactile_sensor", None)

        def _get_tactile_gt(self, action_input) -> torch.Tensor | None:
            return getattr(action_input, "tactile_gt", None)

        def _resolve_tactile_seq(
            self,
            action_input,
            *,
            device: torch.device,
            dtype: torch.dtype,
        ) -> torch.Tensor:
            return resolve_sensor_tactile(
                tactile_sensor=self._get_tactile_sensor(action_input),
                tactile_gt=self._get_tactile_gt(action_input),
                action_horizon=self.action_horizon,
                training=self.training,
                device=device,
                dtype=dtype,
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

        def _expand_sa_mask(self, sa_mask: torch.Tensor, batch_size: int) -> torch.Tensor:
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

        def _visor_sa_mask(self, device: torch.device, dtype: torch.dtype, batch_size: int):
            sa_mask = build_asymmetric_sa_mask(
                self.visor.native_seq_len,
                self.visor.iht_tokens,
                device=device,
                dtype=dtype,
            )
            return self._expand_sa_mask(sa_mask, batch_size)

        def decode_action_hidden(
            self,
            hidden: torch.Tensor,
            embodiment_id: torch.Tensor,
            *,
            gate_delta: torch.Tensor | None = None,
        ) -> torch.Tensor:
            if gate_delta is not None:
                hidden = hidden + gate_delta
            return self.action_decoder(hidden, embodiment_id)

        def forward(self, backbone_output, action_input):
            self.set_frozen_modules_to_eval_mode()
            if self.training:
                self._visor_train_step += 1
            coupling_scale = self._coupling_scale()

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

            vision_context = self.visor.pool_vision_context(
                vl_embeds, backbone_output.image_mask
            )
            language_context = self.visor.pool_language_context(
                vl_embeds,
                backbone_output.image_mask,
                backbone_output.backbone_attention_mask,
            )
            tactile_gt = self._get_tactile_gt(action_input)
            tactile_sensor = self._get_tactile_sensor(action_input)
            coupling_lambda = self.visor.compute_coupling_lambda(
                language_context,
                tactile_gt=tactile_gt if tactile_gt is not None else tactile_sensor,
            )

            tactile_seq = self._resolve_tactile_seq(
                action_input, device=device, dtype=noisy_trajectory.dtype
            )
            iht_tokens, vq_commit = self.visor.build_iht_tokens(
                tactile_seq, vision_context, flow_time=t
            )
            sa_embs = torch.cat((state_features, action_features, iht_tokens), dim=1)

            model_output, _ = self._run_dit(
                sa_embs,
                vl_embeds,
                t_discretized,
                backbone_output,
                self._visor_sa_mask(device, sa_embs.dtype, sa_embs.shape[0]),
                return_all_hidden_states=True,
            )

            hidden_action = model_output[:, 1 : 1 + self.action_horizon]
            gate_delta = self.visor.build_gate_delta(
                tactile_seq,
                flow_time=t,
                coupling_lambda=coupling_lambda,
                coupling_scale=coupling_scale,
                detach_tactile=self.visor_detach_tactile_for_gate,
            )
            pred_actions = self.decode_action_hidden(
                hidden_action, embodiment_id, gate_delta=gate_delta
            )

            action_mask = action_input.action_mask
            action_loss = F.mse_loss(pred_actions, velocity, reduction="none") * action_mask
            flow_loss = action_loss.sum() / (action_mask.sum() + 1e-6)
            loss = flow_loss
            tactile_loss = torch.zeros((), device=device, dtype=loss.dtype)

            refine_active = self.visor.refine_active(t).float().mean()
            tactile_stats = {"vq_commit_loss": vq_commit.detach().reshape(())}
            if vq_commit is not None and self.visor.vq_commit_weight > 0:
                tactile_loss = self.visor.vq_commit_weight * vq_commit * coupling_scale
                loss = loss + tactile_loss

            return {
                "loss": loss,
                "flow_loss": flow_loss.detach().reshape(()),
                "tactile_loss": tactile_loss.detach().reshape(()),
                "action_loss": action_loss,
                "action_mask": action_mask,
                "backbone_features": vl_embeds,
                "state_features": state_features,
                "tactile_pred": tactile_seq.detach(),
                "visor_tactile_source": "sensor",
                "visor_refine_active_rate": refine_active.detach(),
                "visor_coupling_lambda": coupling_lambda.mean().detach(),
                "visor_coupling_scale": torch.tensor(coupling_scale, device=device),
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
            num_steps = self.num_inference_timesteps
            tau_split = self.visor.flow_tau_split
            slow_steps = max(1, int(round((1.0 - tau_split) * num_steps)))

            if "action" in action_input:
                assert options is not None
                action_horizon_before_padding = options["action_horizon"]
                actions[:, : options["rtc_overlap_steps"], :] = action_input["action"][
                    :,
                    action_horizon_before_padding
                    - options["rtc_overlap_steps"] : action_horizon_before_padding,
                    :,
                ]

            vision_context = self.visor.pool_vision_context(
                vl_embeds, backbone_output.image_mask
            )
            language_context = self.visor.pool_language_context(
                vl_embeds,
                backbone_output.image_mask,
                backbone_output.backbone_attention_mask,
            )
            sa_mask = self._visor_sa_mask(device, vl_embeds.dtype, batch_size)
            tactile_seq_base = self._resolve_tactile_seq(
                action_input, device=device, dtype=actions.dtype
            )

            def _euler_step(step_idx: int, *, use_visor: bool) -> None:
                nonlocal actions
                t_cont = step_idx / float(num_steps)
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

                gate_delta = None
                if use_visor:
                    tactile_seq = tactile_seq_base
                    iht_tokens, _ = self.visor.build_iht_tokens(
                        tactile_seq, vision_context, flow_time=t_broadcast
                    )
                    tactile_sensor = self._get_tactile_sensor(action_input)
                    tactile_gt = self._get_tactile_gt(action_input)
                    coupling_lambda = self.visor.compute_coupling_lambda(
                        language_context,
                        tactile_gt=tactile_gt if tactile_gt is not None else tactile_sensor,
                        tactile_pred=tactile_seq,
                    )
                    sa_embs = torch.cat((state_features, action_features, iht_tokens), dim=1)
                    gate_delta = self.visor.build_gate_delta(
                        tactile_seq,
                        flow_time=t_broadcast,
                        coupling_lambda=coupling_lambda,
                        coupling_scale=1.0,
                        detach_tactile=self.visor_detach_tactile_for_gate,
                    )
                else:
                    sa_embs = torch.cat((state_features, action_features), dim=1)

                model_output = self._run_dit(
                    sa_embs,
                    vl_embeds,
                    timesteps_tensor,
                    backbone_output,
                    sa_mask if use_visor else None,
                    return_all_hidden_states=False,
                )
                hidden_action = model_output[:, 1 : 1 + self.action_horizon]
                pred_velocity = self.decode_action_hidden(
                    hidden_action, embodiment_id, gate_delta=gate_delta
                )
                dt = 1.0 / num_steps
                actions = actions + dt * pred_velocity

            for step in range(slow_steps):
                _euler_step(step, use_visor=False)
            for step in range(slow_steps, num_steps):
                _euler_step(step, use_visor=True)

            return BatchFeature(
                data={
                    "action_pred": actions,
                    "backbone_features": vl_embeds,
                    "state_features": state_features,
                }
            )

    return VisorFlatActionHead
