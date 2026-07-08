# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lightweight FLUX VAE latent inpaint expert for shared DiT token injection."""

from __future__ import annotations

import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL
from diffusers.training_utils import compute_density_for_timestep_sampling

logger = logging.getLogger(__name__)


class FluxInpaintExpert(nn.Module):
    """Encode anchor/future latents into DiT tokens and predict latent flow velocity.

    Uses frozen FLUX VAE; trainable projection + velocity head live in the action head.
    """

    def __init__(
        self,
        *,
        vae_path: str,
        embed_dim: int,
        num_tokens: int = 4,
        num_timestep_buckets: int = 1000,
        logit_mean: float = 0.0,
        logit_std: float = 1.0,
    ):
        super().__init__()
        self.vae_path = vae_path
        self.embed_dim = embed_dim
        self.num_tokens = int(num_tokens)
        self.num_timestep_buckets = int(num_timestep_buckets)
        self.logit_mean = float(logit_mean)
        self.logit_std = float(logit_std)

        grid = int(math.sqrt(self.num_tokens))
        if grid * grid != self.num_tokens:
            raise ValueError(f"mot_inpaint_tokens must be a perfect square, got {num_tokens}")

        self._grid = grid
        self._vae: AutoencoderKL | None = None
        vae_cfg = AutoencoderKL.load_config(self.vae_path, subfolder="vae")
        self._latent_channels = int(vae_cfg["latent_channels"])

        self.latent_proj = nn.Linear(self._latent_channels, embed_dim)
        self.time_embed = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.velocity_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, self._latent_channels),
        )
        self._init_trainable()

    def _init_trainable(self) -> None:
        nn.init.zeros_(self.latent_proj.weight)
        if self.latent_proj.bias is not None:
            nn.init.zeros_(self.latent_proj.bias)
        for module in (self.time_embed, self.velocity_head):
            for layer in module.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.zeros_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    @property
    def latent_channels(self) -> int:
        return self._latent_channels

    def _ensure_vae(self, device: torch.device) -> AutoencoderKL:
        if self._vae is None:
            logger.info("FluxInpaintExpert: loading frozen VAE from %s", self.vae_path)
            self._vae = AutoencoderKL.from_pretrained(self.vae_path, subfolder="vae")
            self._vae.requires_grad_(False)
        if next(self._vae.parameters()).device != device:
            self._vae.to(device=device)
        return self._vae

    @torch.no_grad()
    def encode_images(self, images: torch.Tensor, *, device: torch.device) -> torch.Tensor:
        """VAE-encode RGB tensors ``(B, 3, H, W)`` to scaled latents ``(B, C, h, w)``."""
        vae = self._ensure_vae(device)
        vae_dtype = next(vae.parameters()).dtype
        pixels = images.to(device=device, dtype=vae_dtype)
        latent = vae.encode(pixels).latent_dist.sample()
        shift = float(vae.config.shift_factor)
        scale = float(vae.config.scaling_factor)
        return (latent - shift) * scale

    def _pool_latent_tokens(self, latent: torch.Tensor) -> torch.Tensor:
        """``(B, C, H, W)`` -> ``(B, num_tokens, C)``."""
        pooled = F.adaptive_avg_pool2d(latent, (self._grid, self._grid))
        return pooled.flatten(2).transpose(1, 2).contiguous()

    def _timestep_embedding(self, t: torch.Tensor, *, device: torch.device, dtype: torch.dtype):
        """Sinusoidal embedding for continuous flow time ``t`` in ``(B,)``."""
        half = self.embed_dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=device, dtype=torch.float32)
            / max(half - 1, 1)
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.embed_dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb.to(dtype=dtype)

    def latent_to_tokens(
        self,
        latent: torch.Tensor,
        *,
        flow_time: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Project pooled latents to DiT tokens; optional per-token flow time bias."""
        tokens = self.latent_proj(self._pool_latent_tokens(latent))
        if flow_time is not None:
            t_embed = self._timestep_embedding(
                flow_time.reshape(-1),
                device=tokens.device,
                dtype=tokens.dtype,
            )
            tokens = tokens + self.time_embed(t_embed).unsqueeze(1)
        return tokens

    def sample_flow_time(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        u = compute_density_for_timestep_sampling(
            weighting_scheme="logit_normal",
            batch_size=batch_size,
            logit_mean=self.logit_mean,
            logit_std=self.logit_std,
        )
        return u.to(device=device, dtype=dtype)

    def build_training_pack(
        self,
        *,
        future_images: torch.Tensor,
        anchor_images: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, torch.Tensor]:
        """Return anchor tokens, noisy future tokens, and velocity target in token space."""
        self._ensure_vae(device)
        future_latent = self.encode_images(future_images, device=device).to(dtype=dtype)
        anchor_latent = self.encode_images(anchor_images, device=device).to(dtype=dtype)

        batch_size = future_latent.shape[0]
        t_img = self.sample_flow_time(batch_size, device=device, dtype=dtype)

        future_tokens_clean = self._pool_latent_tokens(future_latent)
        noise = torch.randn_like(future_tokens_clean)
        t_view = t_img[:, None, None]
        noisy_future_latent_tokens = (1.0 - t_view) * noise + t_view * future_tokens_clean
        velocity_target = future_tokens_clean - noise

        anchor_tokens = self.latent_to_tokens(anchor_latent)
        future_tokens = self.latent_proj(noisy_future_latent_tokens)
        future_tokens = future_tokens + self.time_embed(
            self._timestep_embedding(t_img, device=device, dtype=dtype)
        ).unsqueeze(1)

        return {
            "anchor_tokens": anchor_tokens,
            "future_tokens": future_tokens,
            "velocity_target": velocity_target,
            "t_img": t_img,
        }

    def predict_velocity(self, hidden_future: torch.Tensor) -> torch.Tensor:
        """Map future inpaint hidden states to latent-token velocity ``(B, N, C)``."""
        return self.velocity_head(hidden_future)
