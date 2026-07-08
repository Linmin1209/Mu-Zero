# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Execution monitor: progress, error type, recovery gate."""

from __future__ import annotations

import torch
from torch import nn

from gr00t.model.modules.vt_closed_loop.action_groups import ERROR_TYPES


class ExecutionMonitor(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_error_types: int | None = None,
        vlm_dim: int | None = None,
        state_dim: int | None = None,
    ):
        super().__init__()
        self.num_error_types = num_error_types or len(ERROR_TYPES)
        vlm_dim = hidden_dim if vlm_dim is None else vlm_dim
        state_dim = hidden_dim if state_dim is None else state_dim
        self.vlm_proj = (
            nn.Identity() if vlm_dim == hidden_dim else nn.Linear(vlm_dim, hidden_dim)
        )
        self.state_proj = (
            nn.Identity() if state_dim == hidden_dim else nn.Linear(state_dim, hidden_dim)
        )
        self.net = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.progress_head = nn.Linear(hidden_dim, 1)
        self.error_head = nn.Linear(hidden_dim, self.num_error_types)
        self.recovery_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        h_vlm: torch.Tensor,
        tactile_tokens: torch.Tensor,
        robot_state: torch.Tensor,
        predicted_future_tokens: torch.Tensor | None = None,
        executed_action: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        del predicted_future_tokens, executed_action
        vlm_vec = self.vlm_proj(h_vlm).mean(dim=1)
        tac_vec = tactile_tokens.mean(dim=1)
        if robot_state.dim() == 3:
            state_vec = robot_state.mean(dim=1)
        else:
            state_vec = robot_state
        state_vec = self.state_proj(state_vec)
        ctx = torch.cat([vlm_vec, tac_vec, state_vec], dim=-1)
        h = self.net(ctx)
        return {
            "progress_score": torch.sigmoid(self.progress_head(h)),
            "error_logits": self.error_head(h),
            "recovery_gate": torch.sigmoid(self.recovery_head(h)),
        }
