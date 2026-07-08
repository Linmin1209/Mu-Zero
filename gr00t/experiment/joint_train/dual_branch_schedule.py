# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Alpha/beta funnel schedule for dual-branch flow matching (L_img + L_act)."""

from __future__ import annotations

from dataclasses import dataclass

from gr00t.configs.joint_finetune_config import JointFinetuneConfig


@dataclass(frozen=True)
class DualBranchWeights:
    """Per-step loss multipliers for joint training."""

    alpha: float
    """Weight on L_img (FLUX Fill LoRA flow matching)."""
    beta: float
    """Weight on L_act (GR00T action flow matching)."""
    visor_visual_scale: float
    """Extra scale on VISOR visual auxiliary (multiplies visor_loss_weight_visual)."""
    visor_tactile_scale: float
    """Extra scale on VISOR tactile auxiliary."""
    visor_aux_scale: float
    """Gate/aux ramp 0..1 after aux_delay (compatible with legacy VISOR warmup)."""
    visor_coupling_scale: float
    """Gate coupling ramp 0..1 over warmup_steps."""
    phase: int
    """1 = visual alignment, 2 = action / correction."""
    in_phase1: bool


@dataclass(frozen=True)
class DualBranchSchedule:
    max_steps: int
    phase1_end: int
    alpha_phase1: float
    beta_phase1: float
    alpha_phase2: float
    beta_phase2: float
    visor_aux_delay_steps: int
    visor_warmup_steps: int
    visor_visual_weight_phase1: float
    visor_visual_weight_phase2: float
    visor_tactile_weight_phase1: float
    visor_tactile_weight_phase2: float

    def weights(self, global_step: int) -> DualBranchWeights:
        step = max(int(global_step), 0)
        in_phase1 = step < self.phase1_end
        if in_phase1:
            alpha, beta = self.alpha_phase1, self.beta_phase1
            vis_vis = self.visor_visual_weight_phase1
            vis_tac = self.visor_tactile_weight_phase1
            phase = 1
        else:
            alpha, beta = self.alpha_phase2, self.beta_phase2
            vis_vis = self.visor_visual_weight_phase2
            vis_tac = self.visor_tactile_weight_phase2
            phase = 2

        warmup = max(self.visor_warmup_steps, 1)
        coupling_scale = min(1.0, step / float(warmup))
        aux_scale = min(
            1.0, max(0, step - self.visor_aux_delay_steps) / float(warmup)
        )

        return DualBranchWeights(
            alpha=alpha,
            beta=beta,
            visor_visual_scale=vis_vis,
            visor_tactile_scale=vis_tac,
            visor_aux_scale=aux_scale,
            visor_coupling_scale=coupling_scale,
            phase=phase,
            in_phase1=in_phase1,
        )


def build_dual_branch_schedule(cfg: JointFinetuneConfig) -> DualBranchSchedule:
    max_steps = max(int(cfg.max_steps), 1)
    phase1_ratio = float(cfg.joint_phase1_ratio)
    phase1_end = int(max_steps * phase1_ratio)
    return DualBranchSchedule(
        max_steps=max_steps,
        phase1_end=phase1_end,
        alpha_phase1=float(cfg.joint_alpha_phase1),
        beta_phase1=float(cfg.joint_beta_phase1),
        alpha_phase2=float(cfg.joint_alpha_phase2),
        beta_phase2=float(cfg.joint_beta_phase2),
        visor_aux_delay_steps=cfg.resolved_visor_aux_delay_steps(),
        visor_warmup_steps=int(cfg.visor_aux_warmup_steps or cfg.visor_tactile_warmup_steps),
        visor_visual_weight_phase1=float(cfg.joint_visor_visual_weight_phase1),
        visor_visual_weight_phase2=float(cfg.joint_visor_visual_weight_phase2),
        visor_tactile_weight_phase1=float(cfg.joint_visor_tactile_weight_phase1),
        visor_tactile_weight_phase2=float(cfg.joint_visor_tactile_weight_phase2),
    )


def build_dual_branch_schedule_from_model_config(model_cfg) -> DualBranchSchedule:
    """Build schedule from Gr00tN1d7Config fields set by launch_joint_finetune."""
    max_steps = max(int(getattr(model_cfg, "max_steps", 30000)), 1)
    phase1_ratio = float(getattr(model_cfg, "joint_phase1_ratio", 0.2))
    phase1_end = int(max_steps * phase1_ratio)
    aux_delay = getattr(model_cfg, "joint_visor_aux_delay_steps", None)
    if aux_delay is None:
        aux_delay = phase1_end
    return DualBranchSchedule(
        max_steps=max_steps,
        phase1_end=phase1_end,
        alpha_phase1=float(getattr(model_cfg, "joint_alpha_phase1", 1.0)),
        beta_phase1=float(getattr(model_cfg, "joint_beta_phase1", 0.1)),
        alpha_phase2=float(getattr(model_cfg, "joint_alpha_phase2", 0.2)),
        beta_phase2=float(getattr(model_cfg, "joint_beta_phase2", 2.0)),
        visor_aux_delay_steps=int(aux_delay),
        visor_warmup_steps=int(
            getattr(model_cfg, "visor_aux_warmup_steps", 2000)
            or getattr(model_cfg, "visor_tactile_warmup_steps", 2000)
        ),
        visor_visual_weight_phase1=float(
            getattr(model_cfg, "joint_visor_visual_weight_phase1", 1.0)
        ),
        visor_visual_weight_phase2=float(
            getattr(model_cfg, "joint_visor_visual_weight_phase2", 0.1)
        ),
        visor_tactile_weight_phase1=float(
            getattr(model_cfg, "joint_visor_tactile_weight_phase1", 0.01)
        ),
        visor_tactile_weight_phase2=float(
            getattr(model_cfg, "joint_visor_tactile_weight_phase2", 0.02)
        ),
    )
