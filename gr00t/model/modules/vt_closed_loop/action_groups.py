# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Action group index maps for structured decoding (RoboCasa / GR1)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


@dataclass(frozen=True)
class ActionGroupSpec:
    """Half-open index range ``[start, end)`` into flat action vector."""

    name: str
    start: int
    end: int
    decode_type: str = "delta"
    loss_weight: float = 1.0
    refiner_scale: float = 1.0

    @property
    def dim(self) -> int:
        return self.end - self.start

    @property
    def indices(self) -> tuple[int, ...]:
        return tuple(range(self.start, self.end))


ROBOCASA_PANDA_OMRON_GROUPS: dict[str, ActionGroupSpec] = {
    "gripper": ActionGroupSpec("gripper", 0, 1, "open_close_or_force", 2.0, 1.5),
    "arm": ActionGroupSpec("arm", 1, 7, "relative_eef_delta", 1.0, 1.0),
    "base": ActionGroupSpec("base", 7, 11, "velocity_or_delta_pose", 1.0, 0.1),
}

# GR1 tabletop (32-dim placeholder — adjust when enabling full-body)
GR1_TABLETOP_GROUPS: dict[str, ActionGroupSpec] = {
    "gripper": ActionGroupSpec("gripper", 0, 1, "open_close_or_force", 2.0, 1.5),
    "arm": ActionGroupSpec("arm", 1, 7, "relative_eef_delta", 1.0, 1.0),
    "base": ActionGroupSpec("base", 7, 10, "velocity_or_delta_pose", 1.0, 0.1),
    "posture": ActionGroupSpec("posture", 10, 14, "posture_balance_residual", 0.5, 0.5),
}


def get_action_groups(embodiment: str) -> dict[str, ActionGroupSpec]:
    key = embodiment.lower().replace("-", "_")
    if "robocasa" in key and "gr1" not in key:
        return dict(ROBOCASA_PANDA_OMRON_GROUPS)
    if "gr1" in key or "tabletop" in key:
        return dict(GR1_TABLETOP_GROUPS)
    return dict(ROBOCASA_PANDA_OMRON_GROUPS)


def groups_to_index_dict(groups: Mapping[str, ActionGroupSpec]) -> dict[str, tuple[int, int]]:
    return {name: (g.start, g.end) for name, g in groups.items()}


def build_refiner_scale_vector(action_dim: int, groups: Mapping[str, ActionGroupSpec]) -> list[float]:
    scales = [0.0] * action_dim
    for g in groups.values():
        for i in range(g.start, min(g.end, action_dim)):
            scales[i] = g.refiner_scale
    return scales


@dataclass
class AVTAGGroupWeights:
    arm: float = 1.0
    gripper: float = 1.5
    recovery: float = 1.5
    posture: float = 0.5
    base_stop: float = 0.5
    base_navigation: float = 0.1

    def as_dict(self) -> dict[str, float]:
        return {
            "arm": self.arm,
            "gripper": self.gripper,
            "recovery": self.recovery,
            "posture": self.posture,
            "base_stop": self.base_stop,
            "base_navigation": self.base_navigation,
        }


ERROR_TYPES: tuple[str, ...] = (
    "none",
    "no_contact_when_expected",
    "unexpected_contact",
    "slip_detected",
    "grasp_unstable",
    "object_lost",
    "force_too_large",
    "base_misaligned",
    "collision_risk",
    "posture_unstable",
    "progress_stalled",
)

INTENT_PHASES: tuple[str, ...] = (
    "navigation",
    "pre_grasp",
    "contact",
    "transport",
    "place",
    "release",
    "recovery",
    "idle",
)
