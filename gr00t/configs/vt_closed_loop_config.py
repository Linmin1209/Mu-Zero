# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration for VT closed-loop policy (extends finetune with stage ablations)."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

from gr00t.configs.finetune_config import FinetuneConfig
from gr00t.model.modules.vt_closed_loop.action_groups import AVTAGGroupWeights


@dataclass
class VTClosedLoopConfig(FinetuneConfig):
    """Flags for visual-tactile closed-loop adapters around GR00T N1.7.

    **Mutually exclusive with VISOR** (``use_visor=False`` when ``use_vt_closed_loop=True``).
    **MOSS is retained** via ``use_motion=True`` on the vision backbone (default on).
    """

    use_vt_closed_loop: bool = False
    """Master switch; selects VTClosedLoopActionHead (not Visor* heads)."""

    vt_closed_loop_stage: int = 1
    """Training stage 1–6 per VT_CLOSED_LOOP_DESIGN.md."""

    # --- MOSS (vision motion module; unchanged from moss finetune) ---
    use_motion: bool = True
    tune_motion: bool = True
    motion_use_gating: bool = True
    motion_gate_init_bias: float = 1.5

    # --- VISOR explicitly off for this stack ---
    use_visor: bool = False
    use_joint_dual_branch: bool = False

    # --- Intent (hierarchical v2) ---
    enable_intent_adapter: bool = True
    num_route_modes: int = 16
    num_global_intent_tokens: int = 4
    num_motion_intent_tokens: int = 8
    num_contact_intent_tokens: int = 4
    num_recovery_intent_tokens: int = 2
    num_intent_tokens: int = 16
    num_intent_phases: int = 8

    # --- Precision ---
    train_fp32: bool = False
    """Full fp32 training (load_bf16=False, sdpa attention, Trainer bf16=False)."""

    # --- Tactile (VT encoder; not VISOR IHT) ---
    enable_tactile_encoder: bool = True
    tactile_dim: int = 3
    tactile_history_len: int = 16
    tactile_num_tokens: int = 4
    tactile_use_pressure_map: bool = False

    # --- Contact gate ---
    enable_contact_gate: bool = True
    contact_gate_use_sim_flag: bool = True

    # --- Structured coarse decoder ---
    enable_structured_action_dit: bool = True
    decouple_base_arm: bool = True

    # --- Late tactile refiner ---
    enable_tactile_refiner: bool = True
    tactile_refiner_layers: int = 2
    tactile_refiner_heads: int = 8

    # --- Future / FLUX (train only; optional, separate from VISOR-MoT joint) ---
    enable_future_head: bool = False
    num_future_tokens: int = 8
    action_decoder_use_future_tokens: bool = False
    detach_future_tokens: bool = True
    use_flux_teacher: bool = False
    flux_training_only: bool = True
    flux_model_path: str = (
        "/HOME/sysu_xdliang/sysu_xdliang_1/HDD_POOL/linmin/models/FLUX.1-Fill-dev"
    )
    flux_future_delta: int = 5
    flux_resolution: int = 256
    flux_mask_mode: str = "keep_reference"

    # --- AVTAG ---
    enable_avtag: bool = False
    avtag_margin: float = 0.05
    avtag_weight: float = 0.02
    avtag_group_weights: AVTAGGroupWeights = field(default_factory=AVTAGGroupWeights)

    # --- Monitor / recovery ---
    enable_execution_monitor: bool = False
    enable_recovery_expert: bool = False
    detach_vlm_for_refiner: bool = True

    # --- Inference ---
    inference_execute_steps: int = 4
    inference_use_flux: bool = False
    inference_use_recovery: bool = True
    action_clamp: float = 1.0

    # --- Loss weights (design v2 §15 weighted mix) ---
    loss_action: float = 1.0
    loss_future: float = 0.1
    loss_flux: float = 0.05
    loss_contact: float = 0.2
    loss_avtag: float = 0.02
    loss_monitor: float = 0.2
    loss_recovery: float = 0.5
    loss_router: float = 0.05
    loss_intent_diversity: float = 0.01

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.train_fp32:
            self.load_bf16 = False
        if self.use_vt_closed_loop:
            if self.use_visor:
                warnings.warn(
                    "use_vt_closed_loop=True: forcing use_visor=False (VISOR stack replaced by VT).",
                    stacklevel=2,
                )
                self.use_visor = False
            if self.use_joint_dual_branch:
                warnings.warn(
                    "use_vt_closed_loop=True: forcing use_joint_dual_branch=False "
                    "(VisorMotJoint head requires VISOR).",
                    stacklevel=2,
                )
                self.use_joint_dual_branch = False

    def ablation_level(self) -> int:
        """Map enabled modules to A0–A11 style ablation index."""
        if not self.use_vt_closed_loop:
            return 0
        level = 0
        if self.enable_intent_adapter:
            level = max(level, 1)
        if self.enable_structured_action_dit:
            level = max(level, 2)
        if self.enable_tactile_refiner:
            level = max(level, 4)
        if self.enable_contact_gate:
            level = max(level, 5)
        if self.enable_avtag:
            level = max(level, 6)
        if self.enable_future_head:
            level = max(level, 7)
        if self.use_flux_teacher:
            level = max(level, 8)
        if self.enable_execution_monitor:
            level = max(level, 9)
        if self.enable_recovery_expert:
            level = max(level, 10)
        if all(
            [
                self.enable_intent_adapter,
                self.enable_structured_action_dit,
                self.enable_tactile_refiner,
                self.enable_contact_gate,
                self.enable_future_head,
                self.enable_execution_monitor,
                self.enable_recovery_expert,
            ]
        ):
            level = 11
        return level

    def sync_vt_model_fields(self) -> dict[str, object]:
        """Fields to copy onto ``Gr00tN1d7Config`` when building the model."""
        return {
            "use_vt_closed_loop": self.use_vt_closed_loop,
            "use_visor": False,
            "use_joint_dual_branch": False,
            "use_motion": self.use_motion,
            "tune_motion": self.tune_motion,
            "motion_use_gating": self.motion_use_gating,
            "motion_gate_init_bias": self.motion_gate_init_bias,
            "decouple_base_arm": self.decouple_base_arm,
            "vt_tactile_dim": self.tactile_dim,
            "vt_tactile_history_len": self.tactile_history_len,
            "vt_tactile_num_tokens": self.tactile_num_tokens,
            "vt_tactile_use_pressure_map": self.tactile_use_pressure_map,
            "vt_num_intent_tokens": self.num_intent_tokens,
            "vt_num_intent_phases": self.num_intent_phases,
            "vt_num_route_modes": self.num_route_modes,
            "vt_num_global_intent_tokens": self.num_global_intent_tokens,
            "vt_num_motion_intent_tokens": self.num_motion_intent_tokens,
            "vt_num_contact_intent_tokens": self.num_contact_intent_tokens,
            "vt_num_recovery_intent_tokens": self.num_recovery_intent_tokens,
            "vt_tactile_refiner_layers": self.tactile_refiner_layers,
            "vt_tactile_refiner_heads": self.tactile_refiner_heads,
            "vt_enable_intent_adapter": self.enable_intent_adapter,
            "vt_enable_tactile_encoder": self.enable_tactile_encoder,
            "vt_enable_contact_gate": self.enable_contact_gate,
            "vt_enable_structured_action_dit": self.enable_structured_action_dit,
            "vt_enable_future_head": self.enable_future_head,
            "vt_num_future_tokens": self.num_future_tokens,
            "vt_action_decoder_use_future_tokens": self.action_decoder_use_future_tokens,
            "vt_detach_future_tokens": self.detach_future_tokens,
            "vt_detach_vlm_for_refiner": self.detach_vlm_for_refiner,
            "vt_action_clamp": self.action_clamp,
            "vt_closed_loop_stage": self.vt_closed_loop_stage,
            "vt_enable_tactile_refiner": self.enable_tactile_refiner,
            "vt_enable_execution_monitor": self.enable_execution_monitor,
            "vt_enable_recovery_expert": self.enable_recovery_expert,
            "vt_use_flux_teacher": self.use_flux_teacher,
            "vt_build_flux_batch": self.use_flux_teacher,
            "vt_flux_model_path": self.flux_model_path,
            "vt_flux_future_delta": self.flux_future_delta,
            "vt_flux_resolution": self.flux_resolution,
            "vt_flux_mask_mode": self.flux_mask_mode,
            "vt_loss_action": self.loss_action,
            "vt_loss_future": self.loss_future,
            "vt_loss_flux": self.loss_flux,
            "vt_loss_contact": self.loss_contact,
            "vt_loss_recovery": self.loss_recovery,
            "vt_loss_router": self.loss_router,
            "vt_loss_intent_diversity": self.loss_intent_diversity,
        }
