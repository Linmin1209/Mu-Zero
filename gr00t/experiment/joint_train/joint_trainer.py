# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gr00tTrainer with MoT joint global_step sync and joint metrics."""

from __future__ import annotations

import logging

from gr00t.experiment.joint_train.joint_model import JointMotModel
from gr00t.experiment.trainer import Gr00tTrainer, _scalar_from_outputs

logger = logging.getLogger(__name__)


class JointGr00tTrainer(Gr00tTrainer):
    def _log_motion_optimizer_coverage(self, optimizer) -> None:
        opt_param_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
        gate_in_opt = gate_total = block_in_opt = block_total = 0
        gr00t = self.model
        if hasattr(gr00t, "module"):
            gr00t = gr00t.module
        if hasattr(gr00t, "gr00t"):
            gr00t = gr00t.gr00t
        for name, param in gr00t.named_parameters():
            if "motion_gate" in name:
                gate_total += param.numel()
                if id(param) in opt_param_ids:
                    gate_in_opt += param.numel()
            elif "motion_block" in name:
                block_total += param.numel()
                if id(param) in opt_param_ids:
                    block_in_opt += param.numel()
        if gate_total == 0 and block_total == 0:
            return
        logger.info(
            "Optimizer motion coverage: motion_block %s/%s trainable params, "
            "motion_gate %s/%s trainable params",
            f"{block_in_opt:,}",
            f"{block_total:,}",
            f"{gate_in_opt:,}",
            f"{gate_total:,}",
        )
        if gate_total > 0 and gate_in_opt == 0:
            logger.warning(
                "motion_gate parameters are NOT in the optimizer — MOSS gating will not learn."
            )

    def create_optimizer(self):
        optimizer = super().create_optimizer()
        if self.args.local_rank in (-1, 0):
            self._log_motion_optimizer_coverage(optimizer)
        return optimizer

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        unwrapped = model.module if hasattr(model, "module") else model
        if isinstance(unwrapped, JointMotModel):
            unwrapped.set_global_step(self.state.global_step)

        loss, outputs = super().compute_loss(
            model,
            inputs,
            return_outputs=True,
            num_items_in_batch=num_items_in_batch,
        )

        step = self.state.global_step
        should_log_joint = (
            step % self.args.logging_steps == 0 and model.training and self.args.local_rank in (-1, 0)
        )

        if should_log_joint:
            joint_metrics: dict[str, float] = {}
            for key in (
                "visual_loss",
                "joint_loss_img",
                "inpaint_flow_loss",
                "joint_alpha",
                "joint_beta",
                "joint_phase",
            ):
                value = _scalar_from_outputs(outputs, key)
                if value is not None:
                    joint_metrics[key] = value
            if joint_metrics:
                self.log(joint_metrics)
                logger.info(
                    "Step %d — flow_loss=%s inpaint_flow_loss=%s joint_loss_img=%s "
                    "alpha=%s beta=%s phase=%s",
                    step,
                    _scalar_from_outputs(outputs, "flow_loss"),
                    joint_metrics.get("inpaint_flow_loss", "n/a"),
                    joint_metrics.get("joint_loss_img", "n/a"),
                    joint_metrics.get("joint_alpha", "n/a"),
                    joint_metrics.get("joint_beta", "n/a"),
                    joint_metrics.get("joint_phase", "n/a"),
                )

        if return_outputs:
            return loss, outputs
        return loss
