# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Synchronous tactile late-denoising: a_refined = a_mid + g_contact * delta_tactile."""

from __future__ import annotations

from typing import Mapping

import torch
from torch import nn

from gr00t.model.modules.vt_closed_loop.action_groups import ActionGroupSpec, build_refiner_scale_vector


class TactileLateDenoisingRefiner(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        action_dim: int,
        action_groups: Mapping[str, ActionGroupSpec],
        num_layers: int = 2,
        num_heads: int = 8,
        vlm_dim: int | None = None,
    ):
        super().__init__()
        self.action_dim = action_dim
        vlm_dim = hidden_dim if vlm_dim is None else vlm_dim
        self.vlm_proj = (
            nn.Identity() if vlm_dim == hidden_dim else nn.Linear(vlm_dim, hidden_dim)
        )
        self.register_buffer(
            "group_scale",
            torch.tensor(build_refiner_scale_vector(action_dim, action_groups)),
            persistent=False,
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.delta_head = nn.Linear(hidden_dim, action_dim)
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)
        self.action_proj = nn.Linear(action_dim, hidden_dim)

    def forward(
        self,
        a_mid: torch.Tensor,
        timestep: torch.Tensor,
        tactile_tokens: torch.Tensor,
        h_vlm_cache: torch.Tensor,
        intent_tokens: torch.Tensor,
        robot_state: torch.Tensor,
        contact_gate: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch_size, horizon, _ = a_mid.shape
        a_proj = self.action_proj(a_mid)
        vlm_ctx = self.vlm_proj(h_vlm_cache[:, : min(8, h_vlm_cache.shape[1])])
        memory = torch.cat([intent_tokens, tactile_tokens, vlm_ctx], dim=1)
        h = self.transformer(torch.cat([a_proj, memory[:, :1].expand(-1, horizon, -1)], dim=1))
        h_action = h[:, :horizon]
        delta = self.delta_head(h_action)
        delta = delta * self.group_scale.view(1, 1, -1).to(delta.dtype)
        g = contact_gate.view(batch_size, 1, 1).to(delta.dtype)
        delta_tactile = g * delta
        a_refined = a_mid + delta_tactile
        return {
            "delta_tactile": delta_tactile,
            "a_refined": a_refined,
        }
