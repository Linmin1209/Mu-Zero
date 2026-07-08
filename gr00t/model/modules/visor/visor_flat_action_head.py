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
    apply_decoupled_action_mask,
    build_asymmetric_sa_mask,
    compute_visor_aux_scales,
    compute_visor_tactile_training_loss,
    compute_visor_visual_training_loss,
    dit_accepts_sa_self_attention_mask,
    expand_asymmetric_sa_mask,
    resolve_sensor_tactile,
    sanitize_finite_tensor,
)


logger = logging.getLogger(__name__)


def build_visor_flat_action_head(base_cls: type):
    class VisorFlatActionHead(base_cls):
        """Gr00tN1d7ActionHead + VISOR IHT tokens; keeps shared flat action_decoder."""

        def __init__(self, config: Gr00tN1d7Config):
            super().__init__(config)
            self.visor_tactile_warmup_steps = int(
                getattr(config, "visor_aux_warmup_steps", None)
                or getattr(config, "visor_tactile_warmup_steps", 2000)
            )
            self.visor_aux_delay_steps = int(getattr(config, "visor_aux_delay_steps", 500))
            self.visor_gate_mode = str(getattr(config, "visor_gate_mode", "tactile_hand_only"))
            self.visor_tactile_align_mode = str(
                getattr(config, "visor_tactile_align_mode", "hold_last")
            )
            self.visor_use_readout_fed_gates = bool(
                getattr(config, "visor_use_readout_fed_gates", False)
            )
            self.visor_detach_tactile_for_gate = bool(
                getattr(config, "visor_detach_tactile_for_gate", True)
            )
            self.register_buffer(
                "_visor_train_step", torch.zeros((), dtype=torch.long), persistent=False
            )
            self._joint_schedule = None
            arm_slice = getattr(config, "visor_arm_action_slice", (1, 7))
            base_slice = getattr(config, "visor_base_action_slice", (7, 11))
            hand_slice = getattr(config, "visor_hand_action_slice", (0, 1))
            self.visor_arm_slice = (int(arm_slice[0]), int(arm_slice[1]))
            self.visor_base_slice = (int(base_slice[0]), int(base_slice[1]))
            self.visor_hand_slice = (int(hand_slice[0]), int(hand_slice[1]))
            self._dit_accepts_sa_mask = dit_accepts_sa_self_attention_mask(self.model)

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
                use_split_action_gates=bool(
                    getattr(config, "visor_use_split_action_gates", True)
                ),
                arm_action_dim=int(getattr(config, "visor_arm_action_dim", 6)),
                base_action_dim=int(getattr(config, "visor_base_action_dim", 4)),
                hand_action_dim=int(getattr(config, "visor_hand_action_dim", 1)),
                tactile_num_force=int(getattr(config, "visor_tactile_num_force", 2)),
                tactile_num_contact=int(getattr(config, "visor_tactile_num_contact", 1)),
                gate_mode=self.visor_gate_mode,
                tactile_align_mode=self.visor_tactile_align_mode,
                use_visual_supervision=bool(
                    getattr(config, "visor_use_visual_supervision", False)
                ),
                visual_waypoints=int(getattr(config, "visor_visual_waypoints", 8)),
                visual_dim=int(getattr(config, "visor_visual_dim", 256)),
                loss_weight_visual=float(getattr(config, "visor_loss_weight_visual", 0.03)),
                visual_vq_tokens=int(getattr(config, "visor_visual_vq_tokens", 1)),
                use_readout_fed_gates=self.visor_use_readout_fed_gates,
                decouple_base_arm=bool(getattr(config, "decouple_base_arm", False)),
            )
            logger.info(
                "VisorFlatActionHead (v4 sensor): tau_split=%.2f iht_tokens=%d "
                "tactile_weight=%.3f gate_mode=%s align=%s readout_gates=%s warmup=%d "
                "split_gates=arm%s hand%s dit_sa_mask=%s",
                self.visor.flow_tau_split,
                self.visor.iht_tokens,
                self.visor.loss_weight_tactile,
                self.visor_gate_mode,
                self.visor_tactile_align_mode,
                self.visor_use_readout_fed_gates,
                self.visor_tactile_warmup_steps,
                self.visor_arm_slice,
                self.visor_hand_slice,
                self._dit_accepts_sa_mask,
            )
            self.set_trainable_parameters(
                config.tune_projector, config.tune_diffusion_model, config.tune_vlln
            )

        def _coupling_scale(self) -> float:
            if not self.training or self.visor_tactile_warmup_steps <= 0:
                return 1.0
            step = float(self._visor_train_step.item())
            scale, _ = compute_visor_aux_scales(
                int(step),
                warmup_steps=self.visor_tactile_warmup_steps,
                aux_delay_steps=self.visor_aux_delay_steps,
            )
            return scale

        def _aux_scale(self) -> float:
            if not self.training or self.visor_tactile_warmup_steps <= 0:
                return 1.0
            step = float(self._visor_train_step.item())
            _, aux = compute_visor_aux_scales(
                int(step),
                warmup_steps=self.visor_tactile_warmup_steps,
                aux_delay_steps=self.visor_aux_delay_steps,
            )
            return aux

        def _get_visual_gt(self, action_input) -> torch.Tensor | None:
            return getattr(action_input, "visual_gt", None)

        def _get_visual_gt_dict(self, action_input) -> dict[str, torch.Tensor] | None:
            manip = getattr(action_input, "visual_gt_manip", None)
            nav = getattr(action_input, "visual_gt_nav", None)
            hand = getattr(action_input, "visual_gt_hand", None)
            legacy = self._get_visual_gt(action_input)
            if manip is not None or nav is not None or hand is not None:
                out: dict[str, torch.Tensor] = {}
                if manip is not None:
                    out["manip"] = manip
                if nav is not None:
                    out["nav"] = nav
                if hand is not None:
                    out["hand"] = hand
                elif manip is not None:
                    out["hand"] = manip
                return out
            if legacy is not None:
                return {"manip": legacy, "nav": legacy, "hand": legacy}
            return None

        def _compute_coupling_lambda(self, language_context, action_input):
            tactile_sensor = self._get_tactile_sensor(action_input)
            return self.visor.compute_coupling_lambda(
                language_context,
                tactile_gt=None,
                tactile_pred=tactile_sensor,
            )

        def _get_tactile_sensor(self, action_input) -> torch.Tensor | None:
            return getattr(action_input, "tactile_sensor", None)

        def _get_tactile_gt(self, action_input) -> torch.Tensor | None:
            return getattr(action_input, "tactile_gt", None)

        def _get_tactile_mask(self, action_input) -> torch.Tensor | None:
            return getattr(action_input, "tactile_mask", None)

        def _get_tactile_supervision_gt(
            self,
            action_input,
            *,
            device: torch.device,
            dtype: torch.dtype,
        ) -> torch.Tensor | None:
            tactile_gt = self._get_tactile_gt(action_input)
            if tactile_gt is None:
                return None
            return tactile_gt[:, : self.action_horizon].to(device=device, dtype=dtype)

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
                for_supervision=False,
                align_mode=self.visor_tactile_align_mode,
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
            return expand_asymmetric_sa_mask(sa_mask, batch_size)

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
            dit_kwargs = dict(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds,
                encoder_attention_mask=vl_attn_mask,
                timestep=t_discretized,
                return_all_hidden_states=return_all_hidden_states,
            )
            if self.config.use_alternate_vl_dit:
                dit_kwargs["image_mask"] = backbone_output.image_mask
                dit_kwargs["backbone_attention_mask"] = backbone_output.backbone_attention_mask
            if sa_self_attention_mask is not None and self._dit_accepts_sa_mask:
                dit_kwargs["sa_self_attention_mask"] = sa_self_attention_mask
            return self.model(**dit_kwargs)

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
            model_output: torch.Tensor,
            embodiment_id: torch.Tensor,
            *,
            arm_gate: torch.Tensor | None = None,
            base_gate: torch.Tensor | None = None,
            hand_gate: torch.Tensor | None = None,
            action_horizon: int | None = None,
        ) -> torch.Tensor:
            horizon = int(action_horizon or self.action_horizon)
            native_len = 1 + horizon
            # fp32 decode avoids bf16 overflow in action_decoder + gate addition.
            hidden_fp32 = model_output[:, :native_len].float()
            pred = self.action_decoder(hidden_fp32, embodiment_id)
            pred_actions = pred[:, -horizon:]

            def _apply_gate(
                gate: torch.Tensor | None,
                slice_pair: tuple[int, int],
                *,
                stream_name: str | None = None,
            ) -> None:
                if gate is None:
                    return
                start, end = slice_pair
                width = end - start
                gate_b = gate.float()
                if gate_b.dim() == 2:
                    gate_b = gate_b.unsqueeze(1)
                if gate_b.dim() == 4:
                    gate_b = gate_b.squeeze(1)
                # Recover from accidental (B, 1, K*Dv) flattened visual stream.
                expected_flat = self.visor.visual_waypoints * self.visor.visual_dim
                if (
                    stream_name in {"manip", "nav", "hand"}
                    and gate_b.shape[-1] == expected_flat
                    and self.visor.visual_iht is not None
                ):
                    kv = self.visor.visual_waypoints
                    dv = self.visor.visual_dim
                    stream = gate_b.reshape(gate_b.shape[0], 1, kv, dv).squeeze(1)
                    if stream_name == "hand":
                        event = stream[:, -1, :]
                        proj = self.visor.visual_hand_gate_proj
                    elif stream_name == "nav":
                        event = self.visor.visual_iht.arm_gate_event(stream)
                        proj = self.visor.visual_base_gate_proj
                    else:
                        event = self.visor.visual_iht.arm_gate_event(stream)
                        proj = self.visor.visual_arm_gate_proj
                    gate_b = proj(event).unsqueeze(1)
                if gate_b.shape[1] not in (1, pred_actions.shape[1]):
                    gate_b = gate_b.mean(dim=1, keepdim=True)
                if gate_b.shape[-1] != width:
                    gate_b = gate_b[..., :width]
                pred_actions[:, :, start:end] = pred_actions[:, :, start:end] + gate_b

            _apply_gate(arm_gate, self.visor_arm_slice, stream_name="manip")
            _apply_gate(base_gate, self.visor_base_slice, stream_name="nav")
            _apply_gate(hand_gate, self.visor_hand_slice, stream_name="hand")
            # Action velocities are O(1–10); tight clamp avoids bf16 DiT blow-ups (~512² loss).
            return sanitize_finite_tensor(pred_actions, fill=0.0, clamp=32.0)

        def _needs_readout_gates(self) -> bool:
            return self.visor_use_readout_fed_gates or self.visor_gate_mode in (
                "dual_split",
                "visual_manip_nav_tactile_hand",
            )

        def _build_action_gates(
            self,
            tactile_seq: torch.Tensor,
            *,
            flow_time: torch.Tensor,
            coupling_lambda: torch.Tensor,
            coupling_scale: float,
            tactile_pred: torch.Tensor | None = None,
            visual_streams: dict[str, torch.Tensor] | None = None,
        ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
            gate_tactile = tactile_pred if self.visor_use_readout_fed_gates else tactile_seq
            manip = visual_streams.get("manip") if visual_streams else None
            nav = visual_streams.get("nav") if visual_streams else None
            hand = visual_streams.get("hand") if visual_streams else None
            return self.visor.build_split_action_gates(
                gate_tactile,
                flow_time=flow_time,
                coupling_lambda=coupling_lambda,
                coupling_scale=coupling_scale,
                detach_tactile=self.visor_detach_tactile_for_gate,
                tactile_pred=tactile_pred,
                visual_pred=manip,
                visual_pred_nav=nav,
                visual_pred_hand=hand,
                gate_mode=self.visor_gate_mode,
            )

        def forward(self, backbone_output, action_input):
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
            tactile_gt = self._get_tactile_gt(action_input)
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
            sa_embs = torch.cat((state_features, action_features, iht_tokens), dim=1)

            model_output, _ = self._run_dit(
                sa_embs,
                vl_embeds,
                t_discretized,
                backbone_output,
                self._visor_sa_mask(device, sa_embs.dtype, sa_embs.shape[0]),
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
            # fp32 MSE avoids bf16 overflow on large action/velocity magnitudes early in training.
            action_loss = (
                F.mse_loss(pred_actions.float(), velocity.float(), reduction="none")
                * action_mask.float()
            )
            flow_loss = action_loss.sum() / (action_mask.sum() + 1e-6)
            if not torch.isfinite(flow_loss):
                flow_loss = flow_loss * 0.0
            elif flow_loss > 10.0 and int(self._visor_train_step.item()) < 20:
                logger.warning(
                    "High flow_loss=%.4g at visor step %d (not clamping; check pred/velocity scale)",
                    float(flow_loss.detach()),
                    int(self._visor_train_step.item()),
                )
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

                arm_gate = None
                hand_gate = None
                if use_visor:
                    tactile_seq = tactile_seq_base
                    visual_gt_dict = self._get_visual_gt_dict(action_input)
                    visual_summary = None
                    if visual_gt_dict is not None and "manip" in visual_gt_dict:
                        visual_summary = visual_gt_dict["manip"].to(
                            device=device, dtype=actions.dtype
                        )
                    iht_tokens, _ = self.visor.build_iht_tokens(
                        tactile_seq,
                        vision_context,
                        flow_time=t_broadcast,
                        visual_summary=visual_summary,
                    )
                    coupling_lambda = self._compute_coupling_lambda(
                        language_context, action_input
                    )
                    sa_embs = torch.cat((state_features, action_features, iht_tokens), dim=1)
                else:
                    sa_embs = torch.cat((state_features, action_features), dim=1)

                need_hidden = use_visor and self._needs_readout_gates()
                model_output = self._run_dit(
                    sa_embs,
                    vl_embeds,
                    timesteps_tensor,
                    backbone_output,
                    sa_mask if use_visor else None,
                    return_all_hidden_states=need_hidden,
                )
                arm_gate = None
                base_gate = None
                hand_gate = None
                if use_visor:
                    if need_hidden and isinstance(model_output, tuple):
                        hidden_states, _ = model_output
                    else:
                        hidden_states = model_output
                    if need_hidden:
                        hidden_action = hidden_states[:, 1 : 1 + self.action_horizon]
                        tactile_pred = self.visor.predict_tactile_from_hidden(hidden_action)
                        visual_streams = self.visor.predict_visual_streams_from_hidden(
                            hidden_action
                        )
                    else:
                        tactile_pred = None
                        visual_streams = None
                    arm_gate, base_gate, hand_gate = self._build_action_gates(
                        tactile_seq,
                        flow_time=t_broadcast,
                        coupling_lambda=coupling_lambda,
                        coupling_scale=1.0,
                        tactile_pred=tactile_pred,
                        visual_streams=visual_streams,
                    )
                    if need_hidden:
                        model_output = hidden_states
                pred_velocity = self.decode_action_hidden(
                    model_output,
                    embodiment_id,
                    arm_gate=arm_gate,
                    base_gate=base_gate,
                    hand_gate=hand_gate,
                    action_horizon=actions.shape[1],
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
