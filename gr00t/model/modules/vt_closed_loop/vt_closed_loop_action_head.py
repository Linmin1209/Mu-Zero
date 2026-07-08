# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""VT closed-loop action head — replaces VISOR; keeps MOSS on vision backbone."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from transformers.feature_extraction_utils import BatchFeature

from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
from gr00t.model.modules.vt_closed_loop.action_groups import get_action_groups, groups_to_index_dict
from gr00t.model.modules.vt_closed_loop.batch_types import MultiModalRobotBatch
from gr00t.model.modules.vt_closed_loop.closed_loop_policy import (
    GR00TN17VisuoTactileClosedLoopPolicy,
)
from gr00t.model.modules.vt_closed_loop.flux_teacher import VTFluxVaeTeacher
from gr00t.model.modules.vt_closed_loop.losses import (
    build_action_group_weight_vector,
    compute_group_weighted_flow_loss,
    compute_router_losses,
    intent_token_diversity_loss,
    masked_flux_feature_loss,
    _sanitize_loss_term,
)
from gr00t.model.modules.vt_closed_loop.structured_action_dit import build_base_arm_sa_mask
from gr00t.model.modules.visor.visor import (
    apply_decoupled_action_mask,
    expand_asymmetric_sa_mask,
    dit_accepts_sa_self_attention_mask,
    sanitize_finite_tensor,
)


logger = logging.getLogger(__name__)


def _vt_cfg_from_model(config: Gr00tN1d7Config) -> SimpleNamespace:
    """Build VT policy config namespace from unified model config."""
    return SimpleNamespace(
        embodiment_tag=str(
            getattr(config, "component_layout_embodiment_tag", None)
            or getattr(config, "embodiment_tag", "robocasa_panda_omron")
        ),
        tactile_dim=int(getattr(config, "vt_tactile_dim", 3)),
        tactile_use_pressure_map=bool(getattr(config, "vt_tactile_use_pressure_map", False)),
        tactile_history_len=int(getattr(config, "vt_tactile_history_len", 16)),
        tactile_num_tokens=int(getattr(config, "vt_tactile_num_tokens", 4)),
        num_route_modes=int(getattr(config, "vt_num_route_modes", 16)),
        num_global_intent_tokens=int(getattr(config, "vt_num_global_intent_tokens", 4)),
        num_motion_intent_tokens=int(getattr(config, "vt_num_motion_intent_tokens", 8)),
        num_contact_intent_tokens=int(getattr(config, "vt_num_contact_intent_tokens", 4)),
        num_recovery_intent_tokens=int(getattr(config, "vt_num_recovery_intent_tokens", 2)),
        num_intent_tokens=int(getattr(config, "vt_num_intent_tokens", 16)),
        num_intent_phases=int(getattr(config, "vt_num_intent_phases", 8)),
        decouple_base_arm=bool(getattr(config, "decouple_base_arm", True)),
        tactile_refiner_layers=int(getattr(config, "vt_tactile_refiner_layers", 2)),
        tactile_refiner_heads=int(getattr(config, "vt_tactile_refiner_heads", 8)),
        enable_intent_adapter=bool(getattr(config, "vt_enable_intent_adapter", True)),
        enable_tactile_encoder=bool(getattr(config, "vt_enable_tactile_encoder", True)),
        enable_contact_gate=bool(getattr(config, "vt_enable_contact_gate", True)),
        enable_structured_action_dit=bool(
            getattr(config, "vt_enable_structured_action_dit", True)
        ),
        enable_future_head=bool(getattr(config, "vt_enable_future_head", False)),
        num_future_tokens=int(getattr(config, "vt_num_future_tokens", 8)),
        action_decoder_use_future_tokens=bool(
            getattr(config, "vt_action_decoder_use_future_tokens", False)
        ),
        detach_future_tokens=bool(getattr(config, "vt_detach_future_tokens", True)),
        detach_vlm_for_refiner=bool(getattr(config, "vt_detach_vlm_for_refiner", True)),
        action_clamp=float(getattr(config, "vt_action_clamp", 1.0)),
        enable_tactile_refiner=bool(getattr(config, "vt_enable_tactile_refiner", True)),
        enable_execution_monitor=bool(getattr(config, "vt_enable_execution_monitor", False)),
        enable_recovery_expert=bool(getattr(config, "vt_enable_recovery_expert", False)),
        closed_loop_stage=int(getattr(config, "vt_closed_loop_stage", 1)),
    )


def _batch_get(batch, key: str):
    if hasattr(batch, key):
        val = getattr(batch, key)
        if val is not None:
            return val
    data = getattr(batch, "data", None)
    if isinstance(data, dict) and key in data:
        return data[key]
    if isinstance(batch, dict):
        return batch.get(key)
    return None


def build_vt_closed_loop_action_head(base_cls: type):
    class VTClosedLoopActionHead(base_cls):
        """Gr00tN1d7ActionHead + VT closed-loop stack.

        Does **not** use VISOR (no IHT, split gates, visor aux losses).
        MOSS remains on the frozen/trainable vision path via ``use_motion`` on backbone.
        """

        def __init__(self, config: Gr00tN1d7Config):
            if getattr(config, "use_visor", False):
                raise ValueError(
                    "use_vt_closed_loop requires use_visor=False — "
                    "VT closed-loop replaces the VISOR tactile action decoder."
                )
            super().__init__(config)
            self._dit_accepts_sa_mask = dit_accepts_sa_self_attention_mask(self.model)
            self._action_groups = get_action_groups(
                str(
                    getattr(config, "component_layout_embodiment_tag", None)
                    or "robocasa_panda_omron"
                )
            )
            state_dim = int(getattr(config, "max_state_dim", 64)) * int(
                getattr(config, "state_history_length", 1)
            )
            self.flux_teacher = None
            vae_latent_channels = 16
            if bool(getattr(config, "vt_use_flux_teacher", False)):
                self.flux_teacher = VTFluxVaeTeacher(
                    vae_path=str(getattr(config, "vt_flux_model_path", "")),
                    hidden_dim=self.hidden_size,
                    num_tokens=int(getattr(config, "vt_num_future_tokens", 8)),
                )
                vae_latent_channels = self.flux_teacher._latent_channels
            self.vt_policy = GR00TN17VisuoTactileClosedLoopPolicy(
                _vt_cfg_from_model(config),
                action_dim=self.action_dim,
                state_dim=state_dim,
                hidden_dim=self.hidden_size,
                vlm_dim=int(config.backbone_embedding_dim),
                vae_latent_channels=vae_latent_channels,
            )
            self._flux_vae_preloaded = False
            logger.info(
                "VTClosedLoopActionHead: VISOR disabled; MOSS use_motion=%s motion_use_gating=%s "
                "future=%s monitor=%s recovery=%s flux_teacher=%s",
                getattr(config, "use_motion", False),
                getattr(config, "motion_use_gating", True),
                getattr(config, "vt_enable_future_head", False),
                getattr(config, "vt_enable_execution_monitor", False),
                getattr(config, "vt_enable_recovery_expert", False),
                self.flux_teacher is not None,
            )

        def _vt_sa_mask(self, device, dtype, batch_size: int, horizon: int):
            # Per-timestep action tokens encode all dims jointly; base/arm SA split
            # requires structured token layout (future). Loss mask handles decoupling.
            return None

        def _resolve_tactile(self, action_input, *, device, dtype, batch_size: int, horizon: int):
            t = _batch_get(action_input, "tactile_sensor")
            if t is None:
                t = _batch_get(action_input, "tactile")
            if t is not None:
                t = t.to(device=device, dtype=dtype)
                t = sanitize_finite_tensor(t)
            else:
                t = torch.zeros(
                    batch_size,
                    horizon,
                    int(getattr(self.config, "vt_tactile_dim", 3)),
                    device=device,
                    dtype=dtype,
                )
            if t.dim() == 2:
                t = t.unsqueeze(0).expand(batch_size, -1, -1)
            if t.shape[0] != batch_size:
                t = t.expand(batch_size, *t.shape[1:])
            return {"tactile": t}

        def _resolve_contact_label(self, action_input, *, device, batch_size: int):
            t = _batch_get(action_input, "tactile_sensor")
            if t is None:
                return None
            t = t.to(device=device)
            if t.dim() == 2:
                t = t.unsqueeze(0)
            if t.shape[-1] >= 3:
                contact = t[..., 2:3].amax(dim=1)
            else:
                contact = t.abs().amax(dim=1, keepdim=True)
            if contact.shape[0] != batch_size:
                contact = contact.expand(batch_size, -1)
            return contact.squeeze(-1).clamp(0.0, 1.0)

        def _tactile_future_target(self, action_input, *, device, dtype):
            tactile_gt = _batch_get(action_input, "tactile_gt")
            if tactile_gt is None:
                return None
            tactile_gt = tactile_gt.to(device=device, dtype=dtype)
            with torch.no_grad():
                enc = self.vt_policy.tactile_encoder({"tactile": tactile_gt})
                return enc["tactile_summary"]

        def _maybe_preload_flux_vae(self, device):
            if self.flux_teacher is None or self._flux_vae_preloaded:
                return
            self.flux_teacher.preload_vae(device)
            self._flux_vae_preloaded = True

        def _finite_mse(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            pred_f = sanitize_finite_tensor(pred.float())
            target_f = sanitize_finite_tensor(target.float())
            return F.mse_loss(pred_f, target_f)

        def _compute_aux_losses(self, vt_out, action_input, *, device, dtype, batch_size: int):
            zero = torch.zeros((), device=device, dtype=dtype)
            w_future = float(getattr(self.config, "vt_loss_future", 0.1))
            w_flux = float(getattr(self.config, "vt_loss_flux", 0.05))
            w_contact = float(getattr(self.config, "vt_loss_contact", 0.2))
            w_recovery = float(getattr(self.config, "vt_loss_recovery", 0.01))
            w_router = float(getattr(self.config, "vt_loss_router", 0.05))
            w_intent_div = float(getattr(self.config, "vt_loss_intent_diversity", 0.01))

            future_loss = zero
            flux_loss = zero
            contact_loss = zero
            recovery_loss = zero
            router_loss = zero
            intent_diversity_loss = zero

            future_out = vt_out.get("future")
            use_tactile = bool(getattr(self.config, "vt_enable_tactile_encoder", True))
            use_intent = bool(getattr(self.config, "vt_enable_intent_adapter", True))
            use_contact = bool(getattr(self.config, "vt_enable_contact_gate", True))

            if future_out is not None and use_tactile:
                tac_target = self._tactile_future_target(action_input, device=device, dtype=dtype)
                if tac_target is not None:
                    future_loss = self._finite_mse(
                        future_out["future_tactile_latent"],
                        tac_target,
                    )

            if (
                self.flux_teacher is not None
                and future_out is not None
                and _batch_get(action_input, "flux_pixel_values") is not None
            ):
                self._maybe_preload_flux_vae(device)
                flux_img = _batch_get(action_input, "flux_pixel_values")
                if flux_img.dim() == 5:
                    flux_img = flux_img[:, 0]
                flux_target = self.flux_teacher.future_tokens_target(
                    flux_img, device=device, dtype=torch.float32
                )
                flux_pred = sanitize_finite_tensor(
                    future_out["future_tokens"].float()
                )
                flux_target = sanitize_finite_tensor(flux_target.float())
                flux_masks = _batch_get(action_input, "flux_masks")
                token_mask = self.flux_teacher.pool_mask_to_token_weights(
                    flux_masks,
                    num_tokens=flux_pred.shape[1],
                )
                if token_mask is not None:
                    flux_loss = masked_flux_feature_loss(
                        flux_pred, flux_target, token_mask
                    )
                else:
                    flux_loss = self._finite_mse(flux_pred, flux_target)
                latent_target = self.flux_teacher.future_pooled_latent_target(
                    flux_img, device=device, dtype=torch.float32
                )
                latent_pred = sanitize_finite_tensor(
                    future_out["future_vae_latent"].float()
                )
                flux_loss = flux_loss + 0.25 * self._finite_mse(
                    latent_pred, sanitize_finite_tensor(latent_target.float())
                )

            tactile_out = vt_out.get("tactile") or {}
            contact_label = self._resolve_contact_label(
                action_input, device=device, batch_size=batch_size
            )
            if (
                use_tactile
                and use_contact
                and contact_label is not None
                and "contact_logits" in tactile_out
            ):
                contact_label = torch.nan_to_num(contact_label.float(), nan=0.0).clamp(0.0, 1.0)
                with torch.autocast(device_type=device.type, enabled=False):
                    contact_loss = F.binary_cross_entropy_with_logits(
                        sanitize_finite_tensor(
                            tactile_out["contact_logits"].squeeze(-1).float()
                        ),
                        contact_label,
                    )

            recovery_out = vt_out.get("recovery") or {}
            if (
                bool(getattr(self.config, "vt_enable_recovery_expert", False))
                and "delta_recovery" in recovery_out
            ):
                delta = sanitize_finite_tensor(recovery_out["delta_recovery"].float())
                recovery_loss = delta.pow(2).mean()

            intent_out = vt_out.get("intent") or {}
            if use_intent and "route_probs" in intent_out:
                with torch.autocast(device_type=device.type, enabled=False):
                    router_parts = compute_router_losses(
                        intent_out["route_logits"].float(),
                        intent_out["route_probs"].float(),
                        contact_gate=vt_out.get("contact_gate"),
                    )
                if router_parts:
                    router_loss = sum(v.float() for v in router_parts.values())
            if use_intent and "all_intent_tokens" in intent_out:
                intent_diversity_loss = intent_token_diversity_loss(
                    intent_out["all_intent_tokens"]
                )

            def _wterm(raw: torch.Tensor, weight: float) -> torch.Tensor:
                raw = _sanitize_loss_term(raw) if torch.is_tensor(raw) else zero
                return weight * raw

            aux = (
                _wterm(future_loss, w_future)
                + _wterm(flux_loss, w_flux)
                + _wterm(contact_loss, w_contact)
                + _wterm(recovery_loss, w_recovery)
                + _wterm(router_loss, w_router)
                + _wterm(intent_diversity_loss, w_intent_div)
            )
            if not torch.isfinite(aux):
                aux = zero
            return {
                "aux_loss": aux,
                "future_loss": future_loss.detach().reshape(()),
                "flux_loss": flux_loss.detach().reshape(()),
                "tactile_loss": contact_loss.detach().reshape(()),
                "recovery_loss": recovery_loss.detach().reshape(()),
                "router_loss": _sanitize_loss_term(router_loss).detach().reshape(()),
                "intent_diversity_loss": _sanitize_loss_term(intent_diversity_loss).detach().reshape(()),
                "weighted_future_loss": (w_future * future_loss).detach().reshape(()),
                "weighted_flux_loss": (w_flux * flux_loss).detach().reshape(()),
                "weighted_tactile_loss": (w_contact * contact_loss).detach().reshape(()),
                "weighted_recovery_loss": (w_recovery * recovery_loss).detach().reshape(()),
                "weighted_router_loss": _sanitize_loss_term(w_router * router_loss).detach().reshape(()),
                "weighted_intent_diversity_loss": _sanitize_loss_term(
                    w_intent_div * intent_diversity_loss
                ).detach().reshape(()),
            }

        def forward(self, backbone_output: BatchFeature, action_input: BatchFeature) -> dict:
            self.set_frozen_modules_to_eval_mode()
            backbone_output = self.process_backbone_output(backbone_output)
            vl_embeds = backbone_output.backbone_features
            device = vl_embeds.device
            embodiment_id = action_input.embodiment_id

            assert action_input.state.shape[1] == self.config.state_history_length
            action_input.state = action_input.state.view(action_input.state.shape[0], 1, -1)
            state_features = self.state_encoder(action_input.state, embodiment_id)

            actions = action_input.action
            noise = torch.randn(actions.shape, device=device, dtype=actions.dtype)
            t = self.sample_time(actions.shape[0], device=device, dtype=actions.dtype)
            t = t[:, None, None]
            noisy_trajectory = (1 - t) * noise + t * actions
            velocity = actions - noise
            t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
            action_features = self.action_encoder(
                noisy_trajectory, t_discretized, embodiment_id
            )
            if self.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                action_features = action_features + self.position_embedding(pos_ids).unsqueeze(0)

            sa_embs = torch.cat((state_features, action_features), dim=1)
            sa_mask = (
                self._vt_sa_mask(device, sa_embs.dtype, sa_embs.shape[0], actions.shape[1])
                if self._dit_accepts_sa_mask
                else None
            )

            dit_kwargs = dict(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds,
                encoder_attention_mask=backbone_output.backbone_attention_mask,
                timestep=t_discretized,
                return_all_hidden_states=True,
            )
            if self.config.use_alternate_vl_dit:
                dit_kwargs["image_mask"] = backbone_output.image_mask
                dit_kwargs["backbone_attention_mask"] = backbone_output.backbone_attention_mask
            if sa_mask is not None:
                dit_kwargs["sa_self_attention_mask"] = sa_mask

            model_output, _ = self.model(**dit_kwargs)
            hidden_action = model_output[:, 1 : 1 + actions.shape[1]]
            base_pred = self.action_decoder(model_output, embodiment_id)
            base_velocity = base_pred[:, -actions.shape[1] :]

            mm_batch = MultiModalRobotBatch.from_gr00t_batch(
                action_input,
                action_groups=groups_to_index_dict(self._action_groups),
            )
            mm_batch.robot_state = action_input.state
            mm_batch.diffusion_timestep = t[:, 0, 0]
            mm_batch.tactile = self._resolve_tactile(
                action_input,
                device=device,
                dtype=actions.dtype,
                batch_size=actions.shape[0],
                horizon=actions.shape[1],
            )

            vt_out = self.vt_policy.forward_stages(
                mm_batch,
                h_vlm=vl_embeds,
                hidden_action=hidden_action,
                base_velocity=base_velocity,
            )
            pred_velocity = vt_out["coarse"]["a_mid"]
            if bool(getattr(self.config, "vt_enable_tactile_refiner", True)):
                pred_velocity = vt_out["tactile_refine"]["a_refined"]

            action_mask = action_input.action_mask
            action_mask = apply_decoupled_action_mask(
                action_mask,
                base_slice=(
                    self._action_groups["base"].start,
                    self._action_groups["base"].end,
                ),
                decouple=bool(getattr(self.config, "decouple_base_arm", False)),
            )
            pred_velocity = sanitize_finite_tensor(pred_velocity.float())
            velocity_f = sanitize_finite_tensor(velocity.float())
            group_weights = build_action_group_weight_vector(
                actions.shape[-1],
                self._action_groups,
                device=device,
                dtype=pred_velocity.dtype,
            )
            action_loss = (
                F.mse_loss(pred_velocity, velocity_f, reduction="none")
                * action_mask.float()
            )
            flow_loss = compute_group_weighted_flow_loss(
                pred_velocity, velocity_f, action_mask, group_weights
            )
            w_action = float(getattr(self.config, "vt_loss_action", 1.0))
            weighted_action_loss = w_action * flow_loss

            aux = self._compute_aux_losses(
                vt_out,
                action_input,
                device=device,
                dtype=actions.dtype,
                batch_size=actions.shape[0],
            )
            total_loss = weighted_action_loss + aux["aux_loss"]
            if not torch.isfinite(total_loss):
                total_loss = weighted_action_loss

            return {
                "loss": total_loss,
                "flow_loss": flow_loss.detach().reshape(()),
                "weighted_action_loss": weighted_action_loss.detach().reshape(()),
                "future_loss": aux["future_loss"],
                "flux_loss": aux["flux_loss"],
                "tactile_loss": aux["tactile_loss"],
                "recovery_loss": aux["recovery_loss"],
                "router_loss": aux["router_loss"],
                "intent_diversity_loss": aux["intent_diversity_loss"],
                "weighted_future_loss": aux["weighted_future_loss"],
                "weighted_flux_loss": aux["weighted_flux_loss"],
                "weighted_tactile_loss": aux["weighted_tactile_loss"],
                "weighted_recovery_loss": aux["weighted_recovery_loss"],
                "weighted_router_loss": aux["weighted_router_loss"],
                "weighted_intent_diversity_loss": aux["weighted_intent_diversity_loss"],
                "action_loss": action_loss,
                "action_mask": action_mask,
                "contact_gate": vt_out["contact_gate"].detach(),
                "backbone_features": vl_embeds,
            }

    return VTClosedLoopActionHead
