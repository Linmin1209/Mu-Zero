# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Flat action layout helpers for component-factored native decoders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from gr00t.model.modules.component_action.component_schema import ComponentSchemaConfig


@dataclass(frozen=True)
class ComponentDecoderSegment:
    name: str
    start: int
    end: int
    is_component: bool


def build_flat_action_decoder_segments(
    action_modality_keys: Sequence[str],
    action_key_dims: Mapping[str, int],
    embodiment_tag: str,
    schema: ComponentSchemaConfig,
) -> list[ComponentDecoderSegment]:
    """Map concatenated flat action keys to per-component / extra decoder segments."""
    groups = schema.groups_for_embodiment(embodiment_tag)
    key_to_component: dict[str, str] = {}
    for comp, keys in groups.items():
        for key in keys:
            key_to_component[key] = comp

    component_ranges: dict[str, tuple[int, int]] = {}
    extra_ranges: dict[str, tuple[int, int]] = {}
    offset = 0
    for key in action_modality_keys:
        if key not in action_key_dims:
            raise ValueError(
                f"Missing action_key_dims entry for {key!r}. "
                "Provide dims in modality config (e.g. ROBOCASA365_ACTION_KEY_DIMS)."
            )
        dim = int(action_key_dims[key])
        start, end = offset, offset + dim
        offset = end

        if key in schema.skip_dataset_keys:
            extra_ranges[key] = (start, end)
            continue

        comp = key_to_component.get(key)
        if comp is None:
            extra_ranges[key] = (start, end)
            continue

        if comp not in component_ranges:
            component_ranges[comp] = (start, end)
        else:
            prev_start, _ = component_ranges[comp]
            component_ranges[comp] = (prev_start, end)

    segments: list[ComponentDecoderSegment] = []
    for comp, (start, end) in component_ranges.items():
        segments.append(ComponentDecoderSegment(comp, start, end, is_component=True))
    for key, (start, end) in extra_ranges.items():
        segments.append(ComponentDecoderSegment(key, start, end, is_component=False))
    segments.sort(key=lambda seg: seg.start)
    return segments
