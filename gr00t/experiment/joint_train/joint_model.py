# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MoT joint training wrapper: shared DiT inpaint + action flow matching."""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
from transformers.feature_extraction_utils import BatchFeature

from gr00t.configs.joint_finetune_config import JointFinetuneConfig
from gr00t.experiment.joint_train.dual_branch_schedule import (
    DualBranchSchedule,
    build_dual_branch_schedule,
)

logger = logging.getLogger(__name__)


class JointMotModel(nn.Module):
    """HF Trainer-facing module: L_total = L_act + L_visor_aux + alpha(s) * L_inpaint."""

    def __init__(self, gr00t_model: nn.Module, cfg: JointFinetuneConfig):
        super().__init__()
        self.gr00t = gr00t_model
        self.cfg = cfg
        self.schedule: DualBranchSchedule = build_dual_branch_schedule(cfg)
        self._global_step = 0

    def set_global_step(self, step: int) -> None:
        self._global_step = int(step)

    def _has_flux_batch(self, inputs: dict) -> bool:
        return "flux_pixel_values" in inputs and inputs["flux_pixel_values"] is not None

    def forward(self, inputs: dict) -> BatchFeature:
        weights = self.schedule.weights(self._global_step)
        mode = self.cfg.joint_train_mode.lower()
        device = self.gr00t.device
        run_inpaint = mode not in ("gr00t_only", "visor_only") and self._has_flux_batch(inputs)

        if self._global_step < 10:
            logger.info(
                "mot joint step=%d mode=%s has_flux_batch=%s",
                self._global_step,
                mode,
                self._has_flux_batch(inputs),
            )

        gr00t_out = self.gr00t(inputs)
        if isinstance(gr00t_out, BatchFeature):
            data = dict(gr00t_out.data)
        elif isinstance(gr00t_out, dict):
            data = dict(gr00t_out)
        else:
            data = dict(gr00t_out)

        loss = data["loss"]
        inpaint_loss = data.get("inpaint_flow_loss")
        if run_inpaint and inpaint_loss is not None:
            scaled_img = weights.alpha * inpaint_loss
            loss = loss + scaled_img
            joint_loss_img = scaled_img
        else:
            joint_loss_img = torch.zeros((), device=loss.device)
            if mode == "simultaneous" and not run_inpaint and self._global_step == 0:
                logger.warning(
                    "MoT joint step %d: flux_pixel_values missing from batch",
                    self._global_step,
                )

        data["loss"] = loss
        data["joint_loss_img"] = joint_loss_img.detach().reshape(())
        data["joint_alpha"] = torch.tensor(weights.alpha, device=loss.device)
        data["joint_beta"] = torch.tensor(weights.beta, device=loss.device)
        data["joint_phase"] = torch.tensor(float(weights.phase), device=loss.device)
        return data

    @property
    def config(self):
        return self.gr00t.config

    def gradient_checkpointing_enable(self, **kwargs):
        if hasattr(self.gr00t, "gradient_checkpointing_enable"):
            self.gr00t.gradient_checkpointing_enable(**kwargs)


# Backward-compatible alias for experiment.py / trainer imports.
JointDualBranchModel = JointMotModel
