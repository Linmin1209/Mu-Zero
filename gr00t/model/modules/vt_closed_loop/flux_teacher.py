# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Frozen FLUX VAE teacher targets for VT FutureHead (training-only)."""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL

logger = logging.getLogger(__name__)


class VTFluxVaeTeacher(nn.Module):
    """Encode future RGB frames to pooled VAE latents projected to DiT hidden size."""

    def __init__(
        self,
        *,
        vae_path: str,
        hidden_dim: int,
        num_tokens: int = 8,
    ):
        super().__init__()
        self.vae_path = vae_path
        self.hidden_dim = hidden_dim
        self.num_tokens = int(num_tokens)
        self._vae: AutoencoderKL | None = None
        vae_cfg = AutoencoderKL.load_config(self.vae_path, subfolder="vae")
        self._latent_channels = int(vae_cfg["latent_channels"])
        self.latent_proj = nn.Linear(self._latent_channels, hidden_dim)
        self._init_frozen_teacher_projection()

    def _init_frozen_teacher_projection(self) -> None:
        """Fixed random projection — must not be zero-init or FLUX targets collapse to 0."""
        nn.init.xavier_uniform_(self.latent_proj.weight)
        nn.init.zeros_(self.latent_proj.bias)
        for param in self.latent_proj.parameters():
            param.requires_grad_(False)

    def _maybe_reinit_zero_projection(self) -> None:
        """Old checkpoints saved zero-init proj; re-init so FLUX targets carry VAE signal."""
        with torch.no_grad():
            if self.latent_proj.weight.abs().sum().item() == 0.0:
                logger.warning(
                    "VTFluxVaeTeacher: zero latent_proj weights detected; "
                    "reinitializing frozen teacher projection"
                )
                self._init_frozen_teacher_projection()

    def _ensure_vae(self, device: torch.device) -> AutoencoderKL:
        if self._vae is None:
            logger.info("VTFluxVaeTeacher: loading frozen VAE from %s", self.vae_path)
            self._vae = AutoencoderKL.from_pretrained(
                self.vae_path, subfolder="vae", torch_dtype=torch.float32
            )
            self._vae.requires_grad_(False)
            self._vae.eval()
        if next(self._vae.parameters()).device != device:
            self._vae.to(device=device, dtype=torch.float32)
        return self._vae

    def preload_vae(self, device: torch.device | str) -> None:
        """Load VAE before first training step (avoids lazy-load + autocast issues)."""
        self._maybe_reinit_zero_projection()
        self._ensure_vae(torch.device(device))

    @torch.no_grad()
    def encode_images(self, images: torch.Tensor, *, device: torch.device) -> torch.Tensor:
        vae = self._ensure_vae(device)
        pixels = images.to(device=device, dtype=torch.float32)
        with torch.autocast(device_type=device.type, enabled=False):
            dist = vae.encode(pixels).latent_dist
            latent = dist.mode() if hasattr(dist, "mode") else dist.sample()
        shift = float(vae.config.shift_factor)
        scale = float(vae.config.scaling_factor)
        return (latent - shift) * scale

    def _pool_latent_tokens(self, latent: torch.Tensor) -> torch.Tensor:
        pooled = F.adaptive_avg_pool2d(latent, (1, self.num_tokens))
        return pooled.flatten(2).transpose(1, 2).contiguous()

    @staticmethod
    def pool_mask_to_token_weights(
        flux_masks: torch.Tensor | None,
        *,
        num_tokens: int,
    ) -> torch.Tensor | None:
        """Image mask ``(B, C, H, W)`` or ``(B, H, W)`` -> token weights ``(B, T, 1)``."""
        if flux_masks is None:
            return None
        mask = flux_masks.float()
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        if mask.shape[1] > 1:
            mask = mask.mean(dim=1, keepdim=True)
        pooled = F.adaptive_avg_pool2d(mask, (1, num_tokens))
        return pooled.flatten(2).transpose(1, 2).contiguous().clamp(min=0.0, max=1.0)

    @torch.no_grad()
    def future_pooled_latent_target(
        self,
        future_images: torch.Tensor,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """``(B, num_tokens, latent_channels)`` pooled VAE latents (teacher, no grad)."""
        latent = self.encode_images(future_images, device=device)
        return self._pool_latent_tokens(latent).to(dtype=dtype)

    @torch.no_grad()
    def future_tokens_target(
        self,
        future_images: torch.Tensor,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """``(B, num_tokens, hidden_dim)`` teacher targets from RGB ``(B, 3, H, W)``."""
        latent = self.encode_images(future_images, device=device)
        pooled = self._pool_latent_tokens(latent)
        with torch.autocast(device_type=device.type, enabled=False):
            tokens = self.latent_proj(pooled.float())
        if not torch.isfinite(tokens).all():
            logger.warning("VTFluxVaeTeacher: non-finite VAE target; zeroing")
            tokens = torch.where(torch.isfinite(tokens), tokens, torch.zeros_like(tokens))
        return tokens.to(dtype=dtype)
