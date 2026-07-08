# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Batch schema for VT closed-loop training (extends existing GR00T BatchFeature)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class MultiModalRobotBatch:
    """Logical batch view; maps from processor / collator outputs."""

    images: torch.Tensor | None = None
    language_input: dict[str, Any] | None = None
    robot_state: torch.Tensor | None = None
    tactile: dict[str, torch.Tensor] = field(default_factory=dict)
    actions: torch.Tensor | None = None
    action_mask: torch.Tensor | None = None
    action_groups: dict[str, tuple[int, int]] = field(default_factory=dict)
    noisy_action: torch.Tensor | None = None
    diffusion_timestep: torch.Tensor | None = None
    future_images: torch.Tensor | None = None
    future_masks: dict[str, torch.Tensor] = field(default_factory=dict)
    labels: dict[str, torch.Tensor] = field(default_factory=dict)
    executed_action: torch.Tensor | None = None

    @classmethod
    def from_gr00t_batch(
        cls,
        action_input: Any,
        backbone_input: Any | None = None,
        *,
        action_groups: dict[str, tuple[int, int]] | None = None,
    ) -> MultiModalRobotBatch:
        """Best-effort adapter from existing ``BatchFeature`` / dict inputs."""
        tactile: dict[str, torch.Tensor] = {}
        for key in (
            "tactile",
            "tactile_history",
            "tactile_force",
            "contact_flag",
            "slip_signal",
        ):
            if hasattr(action_input, key):
                val = getattr(action_input, key)
                if val is not None:
                    tactile[key.split("_", 1)[-1] if key.startswith("tactile_") else key] = val
            elif isinstance(action_input, dict) and key in action_input:
                tactile[key] = action_input[key]

        actions = getattr(action_input, "action", None)
        if actions is None and isinstance(action_input, dict):
            actions = action_input.get("action")

        return cls(
            robot_state=getattr(action_input, "state", None),
            tactile=tactile,
            actions=actions,
            action_mask=getattr(action_input, "action_mask", None),
            action_groups=action_groups or {},
            labels={},
        )
