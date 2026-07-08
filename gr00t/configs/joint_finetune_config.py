# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration for MoT joint flow-matching training (shared DiT inpaint + action)."""

from __future__ import annotations

from dataclasses import dataclass

from gr00t.configs.finetune_config import FinetuneConfig


@dataclass
class JointFinetuneConfig(FinetuneConfig):
    """Extends single-branch finetune with MoT joint image + action training."""

    use_joint_dual_branch: bool = False
    """If True, use JointGr00tTrainer with VisorMotJointActionHead."""

    joint_train_mode: str = "simultaneous"
    """simultaneous | gr00t_only | visor_only (no legacy alternate FLUX branch)."""

    decouple_base_arm: bool = True
    """Suppress base_motion training loss and nav/base VISOR coupling."""

    mot_inpaint_tokens: int = 4

    # --- FLUX VAE inpaint (frozen; tokens injected into shared DiT) ---
    joint_flux_model_path: str = (
        "/HOME/sysu_xdliang/sysu_xdliang_1/HDD_POOL/linmin/models/FLUX.1-Fill-dev"
    )
    joint_flux_future_delta: int = 5
    joint_flux_resolution: int = 256
    joint_flux_mask_mode: str = "keep_reference"
    joint_flux_logit_mean: float = 0.0
    joint_flux_logit_std: float = 1.0

    # --- Alpha/beta funnel (L_inpaint + L_act) ---
    joint_phase1_ratio: float = 0.2
    joint_alpha_phase1: float = 1.0
    joint_beta_phase1: float = 0.1
    joint_alpha_phase2: float = 0.2
    joint_beta_phase2: float = 2.0

    joint_visor_aux_delay_steps: int | None = None
    joint_visor_visual_weight_phase1: float = 1.0
    joint_visor_visual_weight_phase2: float = 0.15
    joint_visor_tactile_weight_phase1: float = 0.01
    joint_visor_tactile_weight_phase2: float = 0.02

    visor_loss_weight_visual: float = 0.15
    visor_loss_weight_tactile: float = 0.01
    visor_aux_delay_steps: int = 6000

    def resolved_visor_aux_delay_steps(self) -> int:
        if self.joint_visor_aux_delay_steps is not None:
            return int(self.joint_visor_aux_delay_steps)
        return int(self.max_steps * self.joint_phase1_ratio)


def joint_finetune_config_from_run(config) -> JointFinetuneConfig:
    """Build JointFinetuneConfig from a full training Config (model + training sections)."""
    mc = config.model
    tc = config.training
    jcfg = JointFinetuneConfig(base_model_path="", embodiment_tag="new_embodiment")
    jcfg.max_steps = tc.max_steps
    jcfg.use_joint_dual_branch = True
    for name in JointFinetuneConfig.__dataclass_fields__:
        if name in ("base_model_path", "embodiment_tag"):
            continue
        if hasattr(mc, name):
            setattr(jcfg, name, getattr(mc, name))
    return jcfg
