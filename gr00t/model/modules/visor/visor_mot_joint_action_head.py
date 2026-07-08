# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""VISOR action head with MoT inpaint tokens injected into shared DiT (VT-WAM style)."""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F

from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
from gr00t.model.modules.mot.asymmetric_mot_mask import build_mot_inpaint_sa_mask
from gr00t.model.modules.mot.flux_inpaint_expert import FluxInpaintExpert
from gr00t.model.modules.visor.visor import (
    apply_decoupled_action_mask,
    compute_visor_tactile_training_loss,
    compute_visor_visual_training_loss,
    sanitize_finite_tensor,
)
from gr00t.model.modules.visor.visor_flat_action_head import build_visor_flat_action_head

logger = logging.getLogger(__name__)


def build_visor_mot_joint_action_head(base_cls: type):
    VisorFlatActionHead = build_visor_flat_action_head(base_cls)

    class VisorMotJointActionHead(VisorFlatActionHead):
        """Shared DiT joint training: action denoise + inpaint latent flow in one forward."""

        def __init__(self, config: Gr00tN1d7Config):
            super().__init__(config)
            self.mot_inpaint_tokens = int(getattr(config, "mot_inpaint_tokens", 4))
            self.inpaint_expert = FluxInpaintExpert(
                vae_path=str(config.joint_flux_model_path),
                embed_dim=self.input_embedding_dim,
                num_tokens=self.mot_inpaint_tokens,
                num_timestep_buckets=self.num_timestep_buckets,
                logit_mean=float(getattr(config, "joint_flux_logit_mean", 0.0)),
                logit_std=float(getattr(config, "joint_flux_logit_std", 1.0)),
            )
            logger.info(
                "VisorMotJointActionHead: inpaint_tokens=%d vae=%s asymmetric_mot_mask=on",
                self.mot_inpaint_tokens,
                config.joint_flux_model_path,
            )

        def set_trainable_parameters(
            self, tune_projector: bool, tune_diffusion_model: bool, tune_vlln: bool
        ):
            super().set_trainable_parameters(tune_projector, tune_diffusion_model, tune_vlln)
            if not tune_projector:
                self.inpaint_expert.requires_grad_(False)
            else:
                for name, param in self.inpaint_expert.named_parameters():
                    if name.startswith("_vae") or "vae" in name:
                        param.requires_grad = False

        def set_frozen_modules_to_eval_mode(self):
            super().set_frozen_modules_to_eval_mode()
            if self.training and self.inpaint_expert._vae is not None:
                self.inpaint_expert._vae.eval()

        def _flux_tensors(self, action_input) -> dict[str, torch.Tensor] | None:
            future = getattr(action_input, "flux_pixel_values", None)
            anchor = getattr(action_input, "flux_masked_images", None)
            if future is None or anchor is None:
                return None
            return {"future": future, "anchor": anchor}

        def _should_run_mot(self, action_input) -> bool:
            return self.training and self._flux_tensors(action_input) is not None

        def _mot_sa_mask(
            self,
            *,
            num_native: int,
            device: torch.device,
            dtype: torch.dtype,
            batch_size: int,
        ) -> torch.Tensor:
            sa_mask = build_mot_inpaint_sa_mask(
                num_native,
                self.visor.iht_tokens,
                self.mot_inpaint_tokens,
                self.mot_inpaint_tokens,
                device=device,
                dtype=dtype,
            )
            return self._expand_sa_mask(sa_mask, batch_size)

        def forward(self, backbone_output, action_input):
            if not self._should_run_mot(action_input):
                return super().forward(backbone_output, action_input)

            self.set_frozen_modules_to_eval_mode()
            if self.training:
                self._visor_train_step += 1
            coupling_scale = self._coupling_scale()
            aux_scale = self._aux_scale()

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
            coupling_lambda = sanitize_finite_tensor(
                self._compute_coupling_lambda(language_context, action_input),
                fill=1.0,
            ).clamp(0.0, 1.0)

            tactile_seq = self._resolve_tactile_seq(
                action_input, device=device, dtype=noisy_trajectory.dtype
            )
            visual_gt_dict = self._get_visual_gt_dict(action_input)
            visual_summary = None
            if visual_gt_dict is not None and "manip" in visual_gt_dict:
                visual_summary = sanitize_finite_tensor(
                    visual_gt_dict["manip"].to(device=device, dtype=torch.float32),
                    fill=0.0,
                    clamp=1e4,
                ).to(device=device, dtype=noisy_trajectory.dtype)

            iht_tokens, vq_commit = self.visor.build_iht_tokens(
                tactile_seq,
                vision_context,
                flow_time=t,
                visual_summary=visual_summary,
            )

            flux = self._flux_tensors(action_input)
            assert flux is not None
            inpaint_pack = self.inpaint_expert.build_training_pack(
                future_images=flux["future"].to(device=device, dtype=noisy_trajectory.dtype),
                anchor_images=flux["anchor"].to(device=device, dtype=noisy_trajectory.dtype),
                device=device,
                dtype=noisy_trajectory.dtype,
            )
            anchor_tokens = inpaint_pack["anchor_tokens"]
            future_tokens = inpaint_pack["future_tokens"]
            velocity_target = inpaint_pack["velocity_target"]

            sa_embs = torch.cat(
                (state_features, action_features, iht_tokens, anchor_tokens, future_tokens),
                dim=1,
            )
            num_native = state_features.shape[1] + action_features.shape[1]

            model_output, _ = self._run_dit(
                sa_embs,
                vl_embeds,
                t_discretized,
                backbone_output,
                self._mot_sa_mask(
                    num_native=num_native,
                    device=device,
                    dtype=sa_embs.dtype,
                    batch_size=sa_embs.shape[0],
                ),
                return_all_hidden_states=True,
            )
            model_output = sanitize_finite_tensor(
                model_output,
                fill=0.0,
                clamp=65504.0,
            )

            hidden_action = model_output[:, 1 : 1 + actions.shape[1]]
            tactile_pred = self.visor.predict_tactile_from_hidden(hidden_action)
            visual_streams = self.visor.predict_visual_streams_from_hidden(hidden_action)
            arm_gate, base_gate, hand_gate = self._build_action_gates(
                tactile_seq,
                flow_time=t,
                coupling_lambda=coupling_lambda,
                coupling_scale=coupling_scale,
                tactile_pred=tactile_pred,
                visual_streams=visual_streams,
            )
            pred_actions = self.decode_action_hidden(
                model_output,
                embodiment_id,
                arm_gate=arm_gate,
                base_gate=base_gate,
                hand_gate=hand_gate,
                action_horizon=actions.shape[1],
            )

            action_mask = action_input.action_mask
            action_mask = apply_decoupled_action_mask(
                action_mask,
                base_slice=self.visor_base_slice,
                decouple=bool(getattr(self.config, "decouple_base_arm", False)),
            )
            action_loss = (
                F.mse_loss(pred_actions.float(), velocity.float(), reduction="none")
                * action_mask.float()
            )
            flow_loss = action_loss.sum() / (action_mask.sum() + 1e-6)
            if not torch.isfinite(flow_loss):
                flow_loss = flow_loss * 0.0

            future_start = num_native + self.visor.iht_tokens + self.mot_inpaint_tokens
            future_end = future_start + self.mot_inpaint_tokens
            hidden_future = model_output[:, future_start:future_end]
            pred_inpaint_velocity = self.inpaint_expert.predict_velocity(hidden_future)
            inpaint_flow_loss = F.mse_loss(
                pred_inpaint_velocity.float(),
                velocity_target.float(),
            )
            if not torch.isfinite(inpaint_flow_loss):
                inpaint_flow_loss = inpaint_flow_loss * 0.0

            loss = flow_loss

            refine_active = self.visor.refine_active(t).float().mean()
            tactile_gt_supervision = self._get_tactile_supervision_gt(
                action_input, device=device, dtype=noisy_trajectory.dtype
            )
            use_tactile_sup = bool(getattr(self.config, "visor_use_tactile_supervision", True))
            tactile_loss, tactile_stats = compute_visor_tactile_training_loss(
                self.visor,
                hidden_action=hidden_action,
                tactile_gt=tactile_gt_supervision,
                vq_commit=vq_commit,
                coupling_lambda=coupling_lambda,
                coupling_scale=coupling_scale,
                aux_scale=aux_scale,
                tactile_mask=self._get_tactile_mask(action_input),
                enabled=use_tactile_sup and self.training,
            )
            visual_loss, visual_stats = compute_visor_visual_training_loss(
                self.visor,
                hidden_action=hidden_action,
                visual_gt=visual_gt_dict,
                coupling_scale=coupling_scale,
                aux_scale=aux_scale,
                enabled=self.training,
            )

            loss = flow_loss + tactile_loss + visual_loss

            return {
                "loss": loss,
                "flow_loss": flow_loss.detach().reshape(()),
                "inpaint_flow_loss": inpaint_flow_loss,
                "tactile_loss": tactile_loss.detach().reshape(()),
                "visual_loss": visual_loss.detach().reshape(()),
                "action_loss": action_loss,
                "action_mask": action_mask,
                "backbone_features": vl_embeds,
                "state_features": state_features,
                "tactile_pred": tactile_pred.detach(),
                "visual_pred": visual_streams["manip"].detach(),
                "visual_pred_nav": visual_streams["nav"].detach(),
                "visual_pred_hand": visual_streams["hand"].detach(),
                "visor_tactile_source": "readout",
                "visor_refine_active_rate": refine_active.detach(),
                "visor_coupling_lambda": coupling_lambda.mean().detach(),
                "visor_coupling_scale": torch.tensor(coupling_scale, device=device),
                "visor_aux_scale": torch.tensor(aux_scale, device=device),
                **{f"visor_{k}": v for k, v in tactile_stats.items()},
                **{f"visor_{k}": v for k, v in visual_stats.items()},
            }

    return VisorMotJointActionHead
