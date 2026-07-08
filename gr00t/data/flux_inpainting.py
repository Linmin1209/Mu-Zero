# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FLUX.1-Fill-dev wrapper for VISOR future-frame inpainting (no manual mask → full-frame)."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

DEFAULT_FLUX_FILL_PATH = Path(
    os.environ.get(
        "VISOR_FLUX_FILL_MODEL_PATH",
        "/HOME/sysu_xdliang/sysu_xdliang_1/HDD_POOL/linmin/models/FLUX.1-Fill-dev",
    )
)

DEFAULT_FLUX_PROMPT = (
    "robot manipulation scene, same camera view, natural next moment, photorealistic"
)


def _to_pil_rgb(frame: np.ndarray) -> Image.Image:
    arr = np.asarray(frame)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    return Image.fromarray(arr, mode="RGB")


def full_inpaint_mask(size: tuple[int, int]) -> Image.Image:
    """No user mask: inpaint the entire frame (next-state prediction)."""
    width, height = size
    mask = Image.new("L", (width, height), 255)
    return mask


class FluxFillFuturePredictor:
    """Lazy-loaded FLUX.1-Fill-dev for waypoint future image prediction."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        lora_path: str | Path | None = None,
        lora_scale: float = 1.0,
        mask_mode: str = "full_inpaint",
        device: str | None = None,
        torch_dtype: Any | None = None,
        num_inference_steps: int = 28,
        guidance_scale: float = 30.0,
        max_sequence_length: int = 512,
    ):
        self.model_path = Path(model_path or DEFAULT_FLUX_FILL_PATH)
        self.lora_path = Path(lora_path) if lora_path else None
        self.lora_scale = float(lora_scale)
        self.mask_mode = mask_mode
        self.device = device or ("cuda" if _cuda_available() else "cpu")
        self.torch_dtype = torch_dtype
        self.num_inference_steps = int(num_inference_steps)
        self.guidance_scale = float(guidance_scale)
        self.max_sequence_length = int(max_sequence_length)
        self._pipe = None
        self._lora_loaded = False

    def _load_pipeline(self):
        if self._pipe is not None:
            return self._pipe
        import torch
        from diffusers import FluxFillPipeline

        dtype = self.torch_dtype
        if dtype is None:
            dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32

        local_only = os.environ.get("HF_HUB_OFFLINE", "0") == "1"
        self._pipe = FluxFillPipeline.from_pretrained(
            str(self.model_path),
            torch_dtype=dtype,
            local_files_only=local_only,
        )
        self._pipe.to(self.device)
        if self.lora_path is not None and not self._lora_loaded:
            from gr00t.data.flux_lora import DEFAULT_LORA_WEIGHT_NAME

            weight_name = DEFAULT_LORA_WEIGHT_NAME
            lora_file = self.lora_path / weight_name
            if self.lora_path.is_file():
                weight_name = self.lora_path.name
                lora_source = str(self.lora_path.parent)
            elif lora_file.exists():
                lora_source = str(self.lora_path)
            else:
                lora_source = str(self.lora_path)
                weight_name = None
            self._pipe.load_lora_weights(lora_source, weight_name=weight_name, adapter_name="default")
            if self.lora_scale != 1.0:
                self._pipe.set_adapters(["default"], adapter_weights=[self.lora_scale])
            self._lora_loaded = True
        return self._pipe

    @staticmethod
    def _snap_dim(value: int, multiple: int = 16) -> int:
        return max(multiple, (value // multiple) * multiple)

    def predict_future_frame(
        self,
        reference: np.ndarray,
        *,
        prompt: str = DEFAULT_FLUX_PROMPT,
        mask: Image.Image | None = None,
        seed: int | None = None,
    ) -> np.ndarray:
        """Predict next visual state from ``reference`` (H,W,3) uint8 RGB."""
        import torch

        pipe = self._load_pipeline()
        image = _to_pil_rgb(reference)
        width, height = image.size
        width = self._snap_dim(width)
        height = self._snap_dim(height)
        if image.size != (width, height):
            image = image.resize((width, height), Image.Resampling.BILINEAR)
        if mask is not None:
            mask_image = mask
        elif self.mask_mode == "keep_reference":
            from gr00t.data.flux_lora import mask_image_from_mode

            mask_image = mask_image_from_mode((width, height), "keep_reference")
        else:
            mask_image = full_inpaint_mask((width, height))

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(int(seed))

        result = pipe(
            prompt=prompt,
            image=image,
            mask_image=mask_image,
            height=height,
            width=width,
            guidance_scale=self.guidance_scale,
            num_inference_steps=self.num_inference_steps,
            max_sequence_length=self.max_sequence_length,
            generator=generator,
        ).images[0]
        return np.asarray(result.convert("RGB"), dtype=np.uint8)

    def predict_waypoint_frames(
        self,
        reference: np.ndarray,
        *,
        prompt: str = DEFAULT_FLUX_PROMPT,
        num_waypoints: int = 8,
        seed: int | None = None,
    ) -> np.ndarray:
        """Predict K future frames (same reference, stochastic seeds per waypoint)."""
        frames = []
        for idx in range(num_waypoints):
            if idx == 0:
                frames.append(_to_pil_rgb(reference))
                frames[-1] = np.asarray(frames[-1], dtype=np.uint8)
                continue
            wp_seed = None if seed is None else int(seed) + idx
            frames.append(
                self.predict_future_frame(reference, prompt=prompt, seed=wp_seed)
            )
        return np.stack(frames, axis=0)


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


@lru_cache(maxsize=4)
def get_flux_fill_predictor(
    model_path: str | None = None,
    lora_path: str | None = None,
    lora_scale: float = 1.0,
    mask_mode: str = "full_inpaint",
    device: str | None = None,
) -> FluxFillFuturePredictor:
    return FluxFillFuturePredictor(
        model_path=model_path,
        lora_path=lora_path,
        lora_scale=lora_scale,
        mask_mode=mask_mode,
        device=device,
    )
