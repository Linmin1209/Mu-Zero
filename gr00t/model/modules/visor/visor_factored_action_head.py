# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Component-factored action head with VISOR suffix IHT + tactile auxiliary loss."""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F
from transformers.feature_extraction_utils import BatchFeature

from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
from gr00t.model.modules.component_action.component_factored_action_head import (
    build_component_factored_action_head,
)
from gr00t.model.modules.visor.visor import (
    VisorModule,
    align_tactile_horizon,
    build_asymmetric_sa_mask,
    normalize_visor_tactile_mode,
)


logger = logging.getLogger(__name__)


def build_visor_factored_action_head(base_cls: type):
    ComponentFactoredActionHead = build_component_factored_action_head(base_cls)

    class VisorFactoredActionHead(ComponentFactoredActionHead):
        def __init__(self, config: Gr00tN1d7Config):
            super().__init__(config)
            proprio_dim = config.max_state_dim * config.state_history_length
            gate_components = getattr(config, "visor_gate_components", ("right_hand",))
            if isinstance(gate_components, list):
                gate_components = tuple(gate_components)
            self.visor_gate_components = frozenset(gate_components)
            self.visor_tactile_warmup_steps = int(
                getattr(config, "visor_tactile_warmup_steps", 1000)
            )
            self.visor_detach_tactile_for_gate = bool(
                getattr(config, "visor_detach_tactile_for_gate", True)
            )
            self.visor_tactile_mode = normalize_visor_tactile_mode(
                getattr(config, "visor_tactile_mode", "imagine")
            )
            self.visor_train_wwm = bool(getattr(config, "visor_train_wwm", True))
            self.register_buffer("_visor_train_step", torch.zeros((), dtype=torch.long), persistent=False)

            self.visor = VisorModule(
                action_dim=self.action_dim,
                hidden_dim=int(getattr(config, "visor_hidden_dim", 256)),
                input_embedding_dim=self.input_embedding_dim,
                action_horizon=self.action_horizon,
                vision_dim=config.backbone_embedding_dim,
                proprio_dim=proprio_dim,
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
                "VisorFactoredActionHead: tau_split=%.2f iht_tokens=%d gate_components=%s "
                "tactile_mode=%s train_wwm=%s tactile_weight=%.3f warmup=%d",
                self.visor.flow_tau_split,
                self.visor.iht_tokens,
                sorted(self.visor_gate_components),
                self.visor_tactile_mode,
                self.visor_train_wwm,
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

        def _resolve_visor_tactile(
            self,
            action_input,
            *,
            trajectory: torch.Tensor,
            flow_time: torch.Tensor,
            vision_context: torch.Tensor,
            proprio: torch.Tensor,
            use_clean_action: bool,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
            """Build tactile sequence + IHT tokens. Returns seq, iht, vq_commit, source."""
            tactile_sensor = self._get_tactile_sensor(action_input)
            use_sensor = self.visor_tactile_mode == "sensor" or (
                self.visor_tactile_mode == "hybrid" and tactile_sensor is not None
            )
            if use_sensor:
                if tactile_sensor is None:
                    raise ValueError(
                        f"visor_tactile_mode={self.visor_tactile_mode!r} requires "
                        "tactile_sensor in the action batch."
                    )
                tactile_seq = tactile_sensor.to(
                    device=trajectory.device, dtype=trajectory.dtype
                )
                if tactile_seq.shape[1] >= self.action_horizon:
                    tactile_seq = align_tactile_horizon(tactile_seq, self.action_horizon)
                source = "sensor"
            else:
                tactile_seq = self.visor.wwm(
                    trajectory,
                    flow_time,
                    vision_context,
                    proprio,
                    use_clean_action=use_clean_action,
                )
                source = "imagine"
            iht_tokens, vq_commit = self.visor.build_iht_tokens(
                tactile_seq, vision_context, flow_time=flow_time
            )
            return tactile_seq, iht_tokens, vq_commit, source

        def decode_action_hidden(
            self,
            hidden: torch.Tensor,
            embodiment_id: torch.Tensor,
            *,
            gate_delta: torch.Tensor | None = None,
        ) -> torch.Tensor:
            batch_size, horizon, _ = hidden.shape
            pred = hidden.new_zeros(batch_size, horizon, self.action_dim)
            for seg in self.decoder_segments:
                h_seg = hidden
                if gate_delta is not None and seg.name in self.visor_gate_components:
                    h_seg = hidden + gate_delta
                pred[:, :, seg.start : seg.end] = self._decoder_for_segment(seg)(
                    h_seg, embodiment_id
                )
            return pred

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

        def _visor_sa_mask(self, device: torch.device, dtype: torch.dtype, batch_size: int):
            sa_mask = build_asymmetric_sa_mask(
                self.visor.native_seq_len,
                self.visor.iht_tokens,
                device=device,
                dtype=dtype,
            )
            return self._expand_sa_mask(sa_mask, batch_size)

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
            proprio = action_input.state.reshape(action_input.state.shape[0], -1)
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

            tactile_seq, iht_tokens, vq_commit, tactile_source = self._resolve_visor_tactile(
                action_input,
                trajectory=noisy_trajectory,
                flow_time=t,
                vision_context=vision_context,
                proprio=proprio,
                use_clean_action=False,
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

            refine_active = self.visor.refine_active(t).float().mean()
            tactile_gt_future = self._get_tactile_gt(action_input)
            if (
                tactile_gt_future is not None
                and self.visor_train_wwm
                and tactile_gt_future.shape[1] >= self.action_horizon
            ):
                tactile_gt_future = tactile_gt_future.to(device=device, dtype=t.dtype)
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
                    tactile_gt_future,
                    vq_commit_loss=vq_commit,
                    coupling_lambda=coupling_lambda,
                )
                tactile_loss = tactile_loss * coupling_scale
                loss = loss + tactile_loss
            else:
                tactile_stats = {"vq_commit_loss": vq_commit.detach().reshape(())}

            return {
                "loss": loss,
                "flow_loss": flow_loss.detach().reshape(()),
                "tactile_loss": tactile_loss.detach().reshape(()),
                "action_loss": action_loss,
                "action_mask": action_mask,
                "backbone_features": vl_embeds,
                "state_features": state_features,
                "tactile_pred": tactile_seq.detach(),
                "visor_tactile_source": tactile_source,
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
            proprio = action_input.state.reshape(action_input.state.shape[0], -1)
            sa_mask = self._visor_sa_mask(device, vl_embeds.dtype, batch_size)

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
                    tactile_seq, iht_tokens, _, _ = self._resolve_visor_tactile(
                        action_input,
                        trajectory=actions,
                        flow_time=t_broadcast,
                        vision_context=vision_context,
                        proprio=proprio,
                        use_clean_action=False,
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

    return VisorFactoredActionHead
