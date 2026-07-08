# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Top-level VT closed-loop policy wrapping GR00T N1.7 (skeleton)."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from gr00t.model.modules.vt_closed_loop.action_groups import ERROR_TYPES, get_action_groups
from gr00t.model.modules.vt_closed_loop.batch_types import MultiModalRobotBatch
from gr00t.model.modules.vt_closed_loop.contact_gate import ContactGate
from gr00t.model.modules.vt_closed_loop.execution_monitor import ExecutionMonitor
from gr00t.model.modules.vt_closed_loop.future_head import VisuoTactileFutureHead
from gr00t.model.modules.vt_closed_loop.intent_manifold_adapter import (
    HierarchicalIntentManifoldAdapter,
)
from gr00t.model.modules.vt_closed_loop.recovery_expert import RecoveryExpert
from gr00t.model.modules.vt_closed_loop.safety_projector import SafetyProjector
from gr00t.model.modules.vt_closed_loop.structured_action_dit import StructuredActionDiT
from gr00t.model.modules.vt_closed_loop.tactile_encoder import TactileEncoder
from gr00t.model.modules.vt_closed_loop.tactile_late_denoising_refiner import (
    TactileLateDenoisingRefiner,
)


class GR00TN17VisuoTactileClosedLoopPolicy(nn.Module):
    """Orchestrates adapters around an existing GR00T model (backbone + action head).

    Integration path: ``VTClosedLoopActionHead`` calls this policy's stages after
    the native DiT produces ``base_velocity`` / ``hidden_action``.
    """

    def __init__(self, cfg: Any, *, action_dim: int, state_dim: int, hidden_dim: int, vlm_dim: int, vae_latent_channels: int = 16):
        super().__init__()
        self.cfg = cfg
        groups = get_action_groups(cfg.embodiment_tag)

        self.tactile_encoder = TactileEncoder(
            tactile_dim=cfg.tactile_dim,
            hidden_dim=hidden_dim,
            use_pressure_map=cfg.tactile_use_pressure_map,
            history_len=cfg.tactile_history_len,
            num_tokens=cfg.tactile_num_tokens,
        )
        self.intent_adapter = HierarchicalIntentManifoldAdapter(
            hidden_dim=hidden_dim,
            state_dim=state_dim,
            vlm_dim=vlm_dim,
            num_route_modes=cfg.num_route_modes,
            num_global_tokens=cfg.num_global_intent_tokens,
            num_motion_tokens=cfg.num_motion_intent_tokens,
            num_contact_tokens=cfg.num_contact_intent_tokens,
            num_recovery_tokens=cfg.num_recovery_intent_tokens,
        )
        self.contact_gate = ContactGate(
            hidden_dim=hidden_dim,
            state_dim=state_dim,
            num_route_modes=cfg.num_route_modes,
        )
        self.future_head = VisuoTactileFutureHead(
            hidden_dim=hidden_dim,
            num_future_tokens=cfg.num_future_tokens,
            state_dim=state_dim,
            vlm_dim=vlm_dim,
            vae_latent_channels=vae_latent_channels,
        )
        self.coarse_action_dit = StructuredActionDiT(
            hidden_dim=hidden_dim,
            action_dim=action_dim,
            action_groups=groups,
            decouple_base_arm=cfg.decouple_base_arm,
        )
        self.tactile_refiner = TactileLateDenoisingRefiner(
            hidden_dim=hidden_dim,
            action_dim=action_dim,
            action_groups=groups,
            num_layers=cfg.tactile_refiner_layers,
            num_heads=cfg.tactile_refiner_heads,
            vlm_dim=vlm_dim,
        )
        self.execution_monitor = ExecutionMonitor(
            hidden_dim=hidden_dim,
            vlm_dim=vlm_dim,
            state_dim=state_dim,
            num_error_types=len(ERROR_TYPES),
        )
        self.recovery_expert = RecoveryExpert(
            hidden_dim=hidden_dim,
            action_dim=action_dim,
            vlm_dim=vlm_dim,
            state_dim=state_dim,
            num_error_types=len(ERROR_TYPES),
        )
        self.safety_projector = SafetyProjector(action_dim=action_dim, clamp=cfg.action_clamp)

    @staticmethod
    def _dummy_tactile_out(
        batch_size: int,
        hidden_dim: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
        num_tokens: int,
    ) -> dict[str, torch.Tensor]:
        return {
            "tactile_tokens": torch.zeros(
                batch_size, num_tokens, hidden_dim, device=device, dtype=dtype
            ),
            "tactile_summary": torch.zeros(batch_size, hidden_dim, device=device, dtype=dtype),
            "contact_logits": torch.zeros(batch_size, 1, device=device, dtype=dtype),
        }

    @staticmethod
    def _dummy_intent_out(
        batch_size: int,
        hidden_dim: int,
        num_route_modes: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
        num_global: int = 4,
        num_motion: int = 8,
        num_contact: int = 4,
        num_recovery: int = 2,
    ) -> dict[str, torch.Tensor]:
        zeros = lambda n: torch.zeros(batch_size, n, hidden_dim, device=device, dtype=dtype)
        route_logits = torch.zeros(batch_size, num_route_modes, device=device, dtype=dtype)
        route_probs = torch.full(
            (batch_size, num_route_modes),
            1.0 / max(num_route_modes, 1),
            device=device,
            dtype=dtype,
        )
        all_tokens = torch.zeros(
            batch_size,
            num_global + num_motion + num_contact + num_recovery,
            hidden_dim,
            device=device,
            dtype=dtype,
        )
        return {
            "global_intent_tokens": zeros(num_global),
            "motion_intent_tokens": zeros(num_motion),
            "contact_intent_tokens": zeros(num_contact),
            "recovery_intent_tokens": zeros(num_recovery),
            "all_intent_tokens": all_tokens,
            "intent_tokens": all_tokens,
            "route_logits": route_logits,
            "route_probs": route_probs,
        }

    def forward_stages(
        self,
        batch: MultiModalRobotBatch,
        *,
        h_vlm: torch.Tensor,
        hidden_action: torch.Tensor,
        base_velocity: torch.Tensor,
        mode: str = "train",
    ) -> dict[str, Any]:
        batch_size = h_vlm.shape[0]
        device = h_vlm.device
        dtype = h_vlm.dtype
        use_tactile = bool(getattr(self.cfg, "enable_tactile_encoder", True))
        use_intent = bool(getattr(self.cfg, "enable_intent_adapter", True))
        use_gate = bool(getattr(self.cfg, "enable_contact_gate", True))

        if use_tactile:
            tactile_out = self.tactile_encoder(batch.tactile)
        else:
            tactile_out = self._dummy_tactile_out(
                batch_size,
                self.intent_adapter.hidden_dim,
                device=device,
                dtype=dtype,
                num_tokens=int(getattr(self.cfg, "tactile_num_tokens", 4)),
            )

        if use_intent:
            intent_pre = self.intent_adapter(
                h_vlm=h_vlm,
                robot_state=batch.robot_state,
                tactile_tokens=tactile_out["tactile_tokens"] if use_tactile else None,
            )
            sim_contact = batch.tactile.get("contact_flag") if use_tactile else None
            g_contact = (
                self.contact_gate(
                    tactile_summary=tactile_out["tactile_summary"],
                    robot_state=batch.robot_state,
                    route_probs=intent_pre["route_probs"],
                    sim_contact_flag=sim_contact,
                )
                if use_gate
                else torch.zeros(batch_size, 1, device=device, dtype=dtype)
            )
            intent_out = self.intent_adapter(
                h_vlm=h_vlm,
                robot_state=batch.robot_state,
                tactile_tokens=tactile_out["tactile_tokens"] if use_tactile else None,
                contact_gate=g_contact.detach(),
            )
        else:
            intent_out = self._dummy_intent_out(
                batch_size,
                self.intent_adapter.hidden_dim,
                int(getattr(self.cfg, "num_route_modes", 16)),
                device=device,
                dtype=dtype,
                num_global=int(getattr(self.cfg, "num_global_intent_tokens", 4)),
                num_motion=int(getattr(self.cfg, "num_motion_intent_tokens", 8)),
                num_contact=int(getattr(self.cfg, "num_contact_intent_tokens", 4)),
                num_recovery=int(getattr(self.cfg, "num_recovery_intent_tokens", 2)),
            )
            g_contact = torch.zeros(batch_size, 1, device=device, dtype=dtype)

        future_out = None
        future_tokens = None
        if self.training and self.cfg.enable_future_head:
            future_out = self.future_head(
                h_vlm=h_vlm,
                intent_tokens=intent_out["all_intent_tokens"] if use_intent else None,
                robot_state=batch.robot_state,
            )
            if self.cfg.action_decoder_use_future_tokens:
                future_tokens = future_out["future_tokens"]
                if self.cfg.detach_future_tokens:
                    future_tokens = future_tokens.detach()

        coarse_out = self.coarse_action_dit(
            hidden_action=hidden_action,
            base_velocity=base_velocity,
        )
        a_mid = coarse_out["a_mid"]
        refine_out = {"a_refined": a_mid, "delta_tactile": torch.zeros_like(a_mid)}
        if self.cfg.enable_tactile_refiner and use_tactile:
            h_cache = h_vlm.detach() if self.cfg.detach_vlm_for_refiner else h_vlm
            refine_out = self.tactile_refiner(
                a_mid=a_mid,
                timestep=batch.diffusion_timestep,
                tactile_tokens=tactile_out["tactile_tokens"],
                h_vlm_cache=h_cache,
                intent_tokens=intent_out["all_intent_tokens"],
                robot_state=batch.robot_state,
                contact_gate=g_contact,
            )

        monitor_out = None
        if self.cfg.enable_execution_monitor:
            monitor_out = self.execution_monitor(
                h_vlm=h_vlm,
                tactile_tokens=tactile_out["tactile_tokens"],
                robot_state=batch.robot_state,
                predicted_future_tokens=future_tokens,
                executed_action=batch.executed_action,
            )

        a_before_safety = refine_out["a_refined"]
        recovery_out = {"a_recovered": a_before_safety, "delta_recovery": torch.zeros_like(a_before_safety)}
        if self.cfg.enable_recovery_expert and monitor_out is not None:
            recovery_out = self.recovery_expert(
                error_logits=monitor_out["error_logits"],
                recovery_gate=monitor_out["recovery_gate"],
                h_vlm=h_vlm,
                intent_tokens=intent_out["all_intent_tokens"],
                tactile_tokens=tactile_out["tactile_tokens"],
                robot_state=batch.robot_state,
                a_refined=a_before_safety,
            )
            a_before_safety = recovery_out["a_recovered"]

        action = self.safety_projector(
            a_before_safety,
            robot_state=batch.robot_state,
        )

        return {
            "action": action,
            "h_vlm": h_vlm,
            "tactile": tactile_out,
            "intent": intent_out,
            "future": future_out,
            "coarse": coarse_out,
            "tactile_refine": refine_out,
            "monitor": monitor_out,
            "recovery": recovery_out,
            "contact_gate": g_contact,
            "future_tokens": future_tokens,
        }
