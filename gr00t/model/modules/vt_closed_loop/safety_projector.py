# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Clamp / project actions to safe ranges before env execution."""

from __future__ import annotations

import torch
from torch import nn


class SafetyProjector(nn.Module):
    def __init__(self, action_dim: int, clamp: float = 1.0):
        super().__init__()
        self.clamp = clamp

    def forward(
        self,
        action: torch.Tensor,
        robot_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del robot_state  # hook for joint-limit projection later
        return action.clamp(-self.clamp, self.clamp)
