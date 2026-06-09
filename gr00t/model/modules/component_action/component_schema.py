# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical hardware component schema for adaptive embodiment action heads."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np


CANONICAL_COMPONENTS: tuple[str, ...] = (
    "left_arm",
    "right_arm",
    "left_hand",
    "right_hand",
    "base",
    "waist",
)

DEFAULT_COMPONENT_DIMS: dict[str, int] = {
    "left_arm": 7,
    "right_arm": 7,
    "left_hand": 1,
    "right_hand": 1,
    "base": 3,
    "waist": 2,
}

DEFAULT_COMPONENT_LOSS_WEIGHTS: dict[str, float] = {
    "left_arm": 1.0,
    "right_arm": 1.0,
    "left_hand": 1.0,
    "right_hand": 1.0,
    "base": 1.0,
    "waist": 1.0,
}

# Dataset modality key groups merged into each canonical component.
DEFAULT_EMBODIMENT_COMPONENT_GROUPS: dict[str, dict[str, list[str]]] = {
    "robocasa_panda_omron": {
        "right_arm": ["end_effector_position", "end_effector_rotation"],
        "right_hand": ["gripper_close"],
        "base": ["base_motion"],
    },
    "robocasa_gr1_tabletop": {
        "left_arm": ["left_arm"],
        "right_arm": ["right_arm"],
        "left_hand": ["left_hand"],
        "right_hand": ["right_hand"],
        "base": ["base_motion"],
        "waist": ["waist"],
    },
    "unitree_g1_full_body_with_waist_height_nav_cmd": {
        "left_arm": ["left_arm"],
        "right_arm": ["right_arm"],
        "left_hand": ["left_hand"],
        "right_hand": ["right_hand"],
        "waist": ["waist"],
        "base": ["navigate_command"],
    },
}

# Keys skipped when building component dict (non-physical / meta controls).
DEFAULT_SKIP_DATASET_KEYS: frozenset[str] = frozenset({"control_mode"})


@dataclass
class ComponentSchemaConfig:
    """Runtime component schema for a training or inference job."""

    component_dims: dict[str, int] = field(default_factory=lambda: dict(DEFAULT_COMPONENT_DIMS))
    loss_weights: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_COMPONENT_LOSS_WEIGHTS)
    )
    embodiment_groups: dict[str, dict[str, list[str]]] = field(
        default_factory=lambda: dict(DEFAULT_EMBODIMENT_COMPONENT_GROUPS)
    )
    skip_dataset_keys: frozenset[str] = DEFAULT_SKIP_DATASET_KEYS

    def projector_dims(self) -> dict[str, int]:
        """Return input dims for components that have a projector (subset of canonical)."""
        dims: dict[str, int] = {}
        for comp in CANONICAL_COMPONENTS:
            if comp in self.component_dims:
                dims[comp] = self.component_dims[comp]
        return dims

    def groups_for_embodiment(self, embodiment_tag: str) -> dict[str, list[str]]:
        return self.embodiment_groups.get(embodiment_tag, {})


def component_index(name: str) -> int:
    return CANONICAL_COMPONENTS.index(name)


def build_active_mask(active_components: Sequence[str]) -> np.ndarray:
    mask = np.zeros(len(CANONICAL_COMPONENTS), dtype=np.float32)
    for name in active_components:
        mask[component_index(name)] = 1.0
    return mask


def merge_dataset_actions_to_components(
    normalized_actions: Mapping[str, np.ndarray],
    embodiment_tag: str,
    schema: ComponentSchemaConfig,
) -> tuple[dict[str, np.ndarray], list[str]]:
    """Map dataset modality keys to canonical component tensors without zero padding."""
    groups = schema.groups_for_embodiment(embodiment_tag)
    if not groups:
        raise ValueError(
            f"No component mapping registered for embodiment {embodiment_tag!r}. "
            "Add an entry to DEFAULT_EMBODIMENT_COMPONENT_GROUPS or ComponentSchemaConfig."
        )

    components: dict[str, np.ndarray] = {}
    active: list[str] = []
    for comp in CANONICAL_COMPONENTS:
        keys = groups.get(comp)
        if not keys:
            continue
        parts = [normalized_actions[k] for k in keys if k in normalized_actions]
        if not parts:
            continue
        merged = np.concatenate(parts, axis=-1).astype(np.float32, copy=False)
        expected = schema.component_dims.get(comp)
        if expected is not None and merged.shape[-1] != expected:
            raise ValueError(
                f"Component {comp!r} for {embodiment_tag!r} has dim {merged.shape[-1]}, "
                f"expected {expected}. Update component_dims in config."
            )
        components[comp] = merged
        active.append(comp)

    if not components:
        raise ValueError(
            f"No active components produced for embodiment {embodiment_tag!r} "
            f"from keys {list(normalized_actions.keys())}."
        )
    return components, active


def split_components_to_dataset_actions(
    component_actions: Mapping[str, np.ndarray],
    embodiment_tag: str,
    schema: ComponentSchemaConfig,
) -> dict[str, np.ndarray]:
    """Split canonical components back to dataset modality keys for policy output."""
    groups = schema.groups_for_embodiment(embodiment_tag)
    out: dict[str, np.ndarray] = {}
    for comp, tensor in component_actions.items():
        keys = groups.get(comp, [comp])
        if len(keys) == 1:
            out[keys[0]] = tensor
            continue
        start = 0
        for key in keys:
            if key not in schema.skip_dataset_keys and key in groups.get(comp, []):
                # Infer split sizes from stored component dim fractions is ambiguous;
                # use equal split only when groups were concatenated in merge order.
                pass
        # Default: only support explicit single-key or pre-merged single output key list
        if len(keys) == 1:
            out[keys[0]] = tensor
        else:
            # Recover by re-reading merge order and known dataset dims from tensor splits
            # stored in group order — caller should use decode_component_actions in processor.
            raise NotImplementedError(
                f"Multi-key split for component {comp!r} requires processor-side dims."
            )
    return out
