# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inference helpers for component-level action heads (RTC, smoothing)."""

from __future__ import annotations

from typing import Mapping

import numpy as np
import torch


def build_rtc_velocity_strength(
    *,
    horizon: int,
    action_dim: int,
    overlap_steps: int,
    frozen_steps: int,
    ramp_rate: float,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int = 1,
) -> torch.Tensor:
    """Per-step velocity gate matching native Gr00tN1d7ActionHead RTC."""
    strength = torch.ones(batch_size, horizon, action_dim, device=device, dtype=dtype)
    if overlap_steps <= 0:
        return strength
    frozen_steps = min(frozen_steps, overlap_steps)
    strength[:, :frozen_steps, :] = 0.0
    intermediate = overlap_steps - frozen_steps
    if intermediate > 0:
        t = torch.linspace(0.0, 1.0, intermediate + 2, device=device, dtype=dtype)
        ramp = 1.0 - torch.exp(-ramp_rate * t)
        ramp = ramp / ramp[-1].clamp_min(1e-8)
        ramp = ramp[1:-1]
        strength[:, frozen_steps:overlap_steps, :] = ramp[None, :, None]
    return strength


def inpaint_component_rtc_prefix(
    component_noise: dict[str, torch.Tensor],
    prev_components: Mapping[str, torch.Tensor],
    overlap_steps: int,
) -> None:
    """Inpaint the prefix of a new chunk from the tail of the previous chunk."""
    if overlap_steps <= 0:
        return
    for comp, noise in component_noise.items():
        if comp not in prev_components:
            continue
        prev = prev_components[comp]
        tail = prev[:, -overlap_steps:, :]
        overlap = min(overlap_steps, noise.shape[1], tail.shape[1])
        noise[:, :overlap, :] = tail[:, -overlap:, :].to(device=noise.device, dtype=noise.dtype)


def smooth_normalized_components(
    components: dict[str, np.ndarray],
    *,
    window: int = 5,
) -> dict[str, np.ndarray]:
    """Light temporal smoothing along the action horizon (eval-only helper)."""
    if window <= 1:
        return components
    out: dict[str, np.ndarray] = {}
    kernel = np.ones(window, dtype=np.float32) / float(window)
    for comp, arr in components.items():
        if arr.ndim != 3 or arr.shape[1] < 2:
            out[comp] = arr
            continue
        smoothed = np.empty_like(arr, dtype=np.float32)
        for b in range(arr.shape[0]):
            for d in range(arr.shape[2]):
                smoothed[b, :, d] = np.convolve(arr[b, :, d], kernel, mode="same")
        out[comp] = smoothed
    return out


def smooth_decoded_actions(
    actions: dict[str, np.ndarray],
    *,
    window: int = 3,
) -> dict[str, np.ndarray]:
    """Smooth executed action keys after denormalization."""
    if window <= 1:
        return actions
    kernel = np.ones(window, dtype=np.float32) / float(window)
    skip = {"control_mode"}
    out: dict[str, np.ndarray] = {}
    for key, arr in actions.items():
        if key in skip or arr.ndim != 3 or arr.shape[1] < 2:
            out[key] = arr
            continue
        smoothed = np.empty_like(arr, dtype=np.float32)
        for b in range(arr.shape[0]):
            for d in range(arr.shape[2]):
                smoothed[b, :, d] = np.convolve(arr[b, :, d], kernel, mode="same")
        out[key] = smoothed
    return out
