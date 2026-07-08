# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Online waypoint optical-flow features for VISOR visual supervision (v4.2b)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from gr00t.data.flux_inpainting import FluxFillFuturePredictor

try:
    import cv2
except ImportError as exc:  # pragma: no cover - optional at import, required at runtime
    cv2 = None
    _CV2_IMPORT_ERROR = exc
else:
    _CV2_IMPORT_ERROR = None


def _require_cv2() -> None:
    if cv2 is None:
        raise ImportError(
            "opencv-python is required for VISOR flow GT. Install with: uv pip install opencv-python"
        ) from _CV2_IMPORT_ERROR


def _to_uint8_rgb(frame: np.ndarray) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    return arr


def mean_optical_flow(reference: np.ndarray, future: np.ndarray) -> np.ndarray:
    """Mean (dx, dy) pixel displacement from reference to future frame, normalized by image size."""
    _require_cv2()
    ref = _to_uint8_rgb(reference)
    fut = _to_uint8_rgb(future)
    height, width = ref.shape[:2]
    ref_gray = cv2.cvtColor(ref, cv2.COLOR_RGB2GRAY)
    fut_gray = cv2.cvtColor(fut, cv2.COLOR_RGB2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        ref_gray,
        fut_gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )
    dx = float(flow[..., 0].mean()) / max(width, 1)
    dy = float(flow[..., 1].mean()) / max(height, 1)
    return np.asarray([dx, dy], dtype=np.float32)


def compute_waypoint_flow_features(frames: np.ndarray) -> np.ndarray:
    """Compute flow from waypoint-0 frame to each future waypoint.

    Args:
        frames: (K, H, W, C) uint8/float frames at deltas [0, 5, ..., 35].

    Returns:
        (K, 2) float32 normalized mean optical flow per waypoint.
    """
    frames = np.asarray(frames)
    if frames.ndim != 4:
        raise ValueError(f"Expected frames (K,H,W,C), got shape {frames.shape}")
    num_waypoints = frames.shape[0]
    out = np.zeros((num_waypoints, 2), dtype=np.float32)
    if num_waypoints == 0:
        return out
    reference = frames[0]
    for idx in range(num_waypoints):
        if idx == 0:
            continue
        out[idx] = mean_optical_flow(reference, frames[idx])
    return out


def compute_nav_flow_features(left_frames: np.ndarray, right_frames: np.ndarray) -> np.ndarray:
    """Average manip-style waypoint flow from left and right agentview streams."""
    left_flow = compute_waypoint_flow_features(left_frames)
    right_flow = compute_waypoint_flow_features(right_frames)
    return 0.5 * (left_flow + right_flow)


def compute_waypoint_flow_from_flux_inpaint(
    reference: np.ndarray,
    *,
    prompt: str,
    predictor: FluxFillFuturePredictor | None = None,
    num_waypoints: int | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """FLUX.1-Fill-dev full-frame inpaint → optical flow vs reference (I_t → Î_{t+k})."""
    from gr00t.data.flux_inpainting import FluxFillFuturePredictor, get_flux_fill_predictor

    pred = predictor or get_flux_fill_predictor()
    k = num_waypoints or reference.shape[0] if reference.ndim == 4 else 8
    if reference.ndim == 3:
        inpainted = pred.predict_waypoint_frames(
            reference, prompt=prompt, num_waypoints=k, seed=seed
        )
    else:
        inpainted = np.stack(
            [
                reference[0]
                if idx == 0
                else pred.predict_future_frame(
                    reference[0], prompt=prompt, seed=None if seed is None else seed + idx
                )
                for idx in range(k)
            ],
            axis=0,
        )
    return compute_waypoint_flow_features(inpainted)
