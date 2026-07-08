# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build FLUX Fill batch tensors from the same VLA step as GR00T (shared anchor)."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from PIL import Image

from gr00t.data.flux_inpainting import DEFAULT_FLUX_PROMPT
from gr00t.data.flux_lora import mask_image_from_mode, prepare_mask_and_masked_image, snap_dim


def _frame_to_pil(frame: np.ndarray) -> Image.Image:
    arr = np.asarray(frame)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    return Image.fromarray(arr, mode="RGB")


def _waypoint_index(future_delta: int) -> int:
    """Map frame offset k to index in [0,5,10,...,35] waypoint list."""
    delta = max(0, int(future_delta))
    return min(delta // 5, 7)


def compact_flux_meta_from_vla_metadata(
    metadata: dict[str, Any],
    *,
    video_key: str = "robot0_eye_in_hand",
    future_delta: int = 5,
    resolution: int = 256,
    mask_mode: str = "keep_reference",
) -> dict[str, Any]:
    """Store only the two frames + prompt needed for deferred FLUX collate."""
    manip = metadata.get("video_future_manip") or metadata.get("video_future")
    if manip is None:
        raise KeyError("metadata missing video_future_manip for joint FLUX batch")
    if isinstance(manip, dict):
        frames = manip.get(video_key)
    else:
        frames = manip
    if frames is None:
        raise KeyError(f"video_future_manip missing key {video_key}")

    frames_np = np.asarray(frames)
    if frames_np.ndim != 4 or frames_np.shape[0] < 1:
        raise ValueError(f"Expected (K,H,W,C) future frames, got {frames_np.shape}")

    future_idx = _waypoint_index(future_delta)
    lang = metadata.get("language") or metadata.get("annotation.human.task_description")
    if isinstance(lang, (list, tuple)) and lang:
        language = str(lang[0])
    elif isinstance(lang, str) and lang.strip():
        language = lang
    else:
        language = DEFAULT_FLUX_PROMPT

    return {
        "eye_frames": np.stack([frames_np[0], frames_np[future_idx]], axis=0),
        "language": language,
        "future_delta": int(future_delta),
        "resolution": int(resolution),
        "mask_mode": str(mask_mode),
        "video_key": video_key,
    }


def _build_flux_tensors_from_pil_pair(
    current_pil: Image.Image,
    future_pil: Image.Image,
    *,
    resolution: int,
    mask_mode: str,
    prompt: str,
) -> dict[str, Any]:
    width = snap_dim(current_pil.width)
    height = snap_dim(current_pil.height)
    if current_pil.size != (width, height):
        current_pil = current_pil.resize((width, height), Image.Resampling.BILINEAR)
        future_pil = future_pil.resize((width, height), Image.Resampling.BILINEAR)

    from torchvision import transforms

    to_tensor = transforms.Compose(
        [
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    current_resized = to_tensor(current_pil)
    future_tensor = to_tensor(future_pil)

    current_for_mask = transforms.ToPILImage()(current_resized * 0.5 + 0.5)
    mask_pil = mask_image_from_mode(current_for_mask.size, mask_mode)
    mask_tensor, masked_image = prepare_mask_and_masked_image(current_for_mask, mask_pil)

    return {
        "pixel_values": future_tensor,
        "masks": mask_tensor.squeeze(0),
        "masked_images": masked_image.squeeze(0),
        "prompts": prompt,
    }


def build_flux_batch_from_compact_meta(meta: dict[str, Any]) -> dict[str, Any]:
    """Rebuild FLUX tensors from compact cache metadata (collate-time)."""
    eye_frames = np.asarray(meta["eye_frames"])
    if eye_frames.shape[0] < 2:
        raise ValueError(f"compact flux meta needs 2 frames, got {eye_frames.shape}")
    current_pil = _frame_to_pil(eye_frames[0])
    future_pil = _frame_to_pil(eye_frames[1])
    prompt = str(meta.get("language") or DEFAULT_FLUX_PROMPT)
    return _build_flux_tensors_from_pil_pair(
        current_pil,
        future_pil,
        resolution=int(meta.get("resolution", 256)),
        mask_mode=str(meta.get("mask_mode", "keep_reference")),
        prompt=prompt,
    )

def build_flux_batch_from_vla_metadata(
    metadata: dict[str, Any],
    *,
    video_key: str = "robot0_eye_in_hand",
    future_delta: int = 5,
    resolution: int = 256,
    mask_mode: str = "keep_reference",
    default_prompt: str = DEFAULT_FLUX_PROMPT,
) -> dict[str, Any]:
    """Extract (I_t, I_{t+k}, mask, prompt) from processor/dataset metadata.

    Expected metadata keys (from ``robocasa365_config_4frame`` + episode loader):
      - ``video_future_manip``[video_key]: stack (K, H, W, C) at deltas [0,5,...,35]
      - or ``language`` / task description for prompt

    Returns dict compatible with ``collate_flux_fill_batch`` after stacking batches.
    """
    manip = metadata.get("video_future_manip") or metadata.get("video_future")
    if manip is None:
        raise KeyError("metadata missing video_future_manip for joint FLUX batch")

    if isinstance(manip, dict):
        frames = manip.get(video_key)
    else:
        frames = manip
    if frames is None:
        raise KeyError(f"video_future_manip missing key {video_key}")

    frames_np = np.asarray(frames)
    if frames_np.ndim != 4 or frames_np.shape[0] < 1:
        raise ValueError(f"Expected (K,H,W,C) future frames, got {frames_np.shape}")

    future_idx = _waypoint_index(future_delta)
    current_pil = _frame_to_pil(frames_np[0])
    future_pil = _frame_to_pil(frames_np[future_idx])

    lang = metadata.get("language") or metadata.get("annotation.human.task_description")
    if isinstance(lang, (list, tuple)) and lang:
        prompt = str(lang[0])
    elif isinstance(lang, str) and lang.strip():
        prompt = lang
    else:
        prompt = default_prompt

    return _build_flux_tensors_from_pil_pair(
        current_pil,
        future_pil,
        resolution=resolution,
        mask_mode=mask_mode,
        prompt=prompt,
    )
