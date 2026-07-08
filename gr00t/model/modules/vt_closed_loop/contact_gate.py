# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contact gate g_contact in [0, 1] for tactile late-denoising strength."""

from __future__ import annotations

import torch
from torch import nn


class ContactGate(nn.Module):
    """Uses route_probs (learned latent modes), not discrete phase_probs."""

    def __init__(
        self,
        hidden_dim: int,
        state_dim: int,
        num_route_modes: int = 16,
    ):
        super().__init__()
        in_dim = hidden_dim + state_dim + num_route_modes
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        tactile_summary: torch.Tensor,
        robot_state: torch.Tensor,
        route_probs: torch.Tensor,
        sim_contact_flag: torch.Tensor | None = None,
        *,
        phase_probs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del phase_probs  # legacy kwarg; ignored
        if robot_state.dim() == 3:
            state_vec = robot_state.mean(dim=1)
        else:
            state_vec = robot_state
        x = torch.cat([tactile_summary, state_vec, route_probs], dim=-1)
        g = torch.sigmoid(self.net(x))

        if sim_contact_flag is not None:
            flag = sim_contact_flag.float().reshape(-1, 1)
            g = 0.5 * g + 0.5 * flag
        return g.clamp(0.0, 1.0)
