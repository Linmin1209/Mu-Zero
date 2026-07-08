# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Future visual / tactile prediction head (training auxiliary)."""

from __future__ import annotations

import torch
from torch import nn


class VisuoTactileFutureHead(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_future_tokens: int = 8,
        state_dim: int = 64,
        vlm_dim: int | None = None,
        vae_latent_channels: int = 16,
    ):
        super().__init__()
        self.num_future_tokens = num_future_tokens
        self.vae_latent_channels = int(vae_latent_channels)
        vlm_dim = hidden_dim if vlm_dim is None else vlm_dim
        self.vlm_proj = (
            nn.Identity() if vlm_dim == hidden_dim else nn.Linear(vlm_dim, hidden_dim)
        )
        self.state_proj = (
            nn.Identity() if state_dim == hidden_dim else nn.Linear(state_dim, hidden_dim)
        )
        in_dim = hidden_dim * 3
        self.ctx_proj = nn.Linear(in_dim, hidden_dim)
        self.future_tokens = nn.Linear(hidden_dim, num_future_tokens * hidden_dim)
        self.future_vae_latent = nn.Linear(
            hidden_dim, num_future_tokens * self.vae_latent_channels
        )
        self.slip_head = nn.Linear(hidden_dim, 1)
        self.stability_head = nn.Linear(hidden_dim, 1)
        self.contact_affordance = nn.Linear(hidden_dim, 1)
        self.future_tactile = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        h_vlm: torch.Tensor,
        intent_tokens: torch.Tensor | None,
        robot_state: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        vlm_vec = self.vlm_proj(h_vlm).mean(dim=1)
        if intent_tokens is None:
            intent_vec = torch.zeros_like(vlm_vec)
        else:
            intent_vec = intent_tokens.mean(dim=1)
        if robot_state.dim() == 3:
            state_vec = robot_state.mean(dim=1)
        else:
            state_vec = robot_state
        state_vec = self.state_proj(state_vec)
        ctx = torch.cat([vlm_vec, intent_vec, state_vec], dim=-1)
        h = self.ctx_proj(ctx)
        flat = self.future_tokens(h)
        future_tokens = flat.view(h.shape[0], self.num_future_tokens, -1)
        vae_flat = self.future_vae_latent(h)
        future_vae_latent = vae_flat.view(
            h.shape[0], self.num_future_tokens, self.vae_latent_channels
        )
        return {
            "future_tokens": future_tokens,
            "future_vae_latent": future_vae_latent,
            "slip_logits": self.slip_head(h),
            "grasp_stability_logits": self.stability_head(h),
            "contact_affordance_logits": self.contact_affordance(h),
            "future_tactile_latent": self.future_tactile(h),
            "future_visual_latent": future_tokens.mean(dim=1),
        }
