# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recovery residual expert — does not replace main policy."""

from __future__ import annotations

import torch
from torch import nn


class RecoveryExpert(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        action_dim: int,
        num_error_types: int,
        vlm_dim: int | None = None,
        state_dim: int | None = None,
    ):
        super().__init__()
        vlm_dim = hidden_dim if vlm_dim is None else vlm_dim
        state_dim = hidden_dim if state_dim is None else state_dim
        self.vlm_proj = (
            nn.Identity() if vlm_dim == hidden_dim else nn.Linear(vlm_dim, hidden_dim)
        )
        self.state_proj = (
            nn.Identity() if state_dim == hidden_dim else nn.Linear(state_dim, hidden_dim)
        )
        self.error_embed = nn.Embedding(num_error_types, hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.delta_head = nn.Linear(hidden_dim, action_dim)
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)

    def forward(
        self,
        error_logits: torch.Tensor,
        recovery_gate: torch.Tensor,
        h_vlm: torch.Tensor,
        intent_tokens: torch.Tensor,
        tactile_tokens: torch.Tensor,
        robot_state: torch.Tensor,
        a_refined: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        error_idx = error_logits.argmax(dim=-1)
        err_emb = self.error_embed(error_idx)
        if robot_state.dim() == 3:
            state_vec = robot_state.mean(dim=1)
        else:
            state_vec = robot_state
        state_vec = self.state_proj(state_vec)
        ctx = torch.cat(
            [
                self.vlm_proj(h_vlm).mean(dim=1),
                intent_tokens.mean(dim=1),
                tactile_tokens.mean(dim=1),
                err_emb,
            ],
            dim=-1,
        )
        h = self.net(ctx).unsqueeze(1).expand(-1, a_refined.shape[1], -1)
        delta_recovery = self.delta_head(h)
        g = recovery_gate.view(-1, 1, 1).to(delta_recovery.dtype)
        a_recovered = a_refined + g * delta_recovery
        return {"delta_recovery": delta_recovery, "a_recovered": a_recovered}
