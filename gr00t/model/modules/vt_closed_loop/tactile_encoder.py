# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tactile history encoder (lightweight; can delegate to VISOR IHT in integrated head)."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class TactileEncoder(nn.Module):
    """1D temporal conv + small transformer over force/torque/contact signals."""

    def __init__(
        self,
        tactile_dim: int = 3,
        hidden_dim: int = 256,
        use_pressure_map: bool = False,
        history_len: int = 16,
        num_tokens: int = 4,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_tokens = num_tokens
        self.use_pressure_map = use_pressure_map

        self.input_proj = nn.Linear(tactile_dim, hidden_dim)
        conv_k = 5
        pad = conv_k // 2
        self.temporal = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, conv_k, padding=pad),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, hidden_dim, conv_k, padding=pad, stride=2),
            nn.SiLU(),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=max(1, hidden_dim // 64),
            dim_feedforward=hidden_dim * 2,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.to_tokens = nn.Linear(hidden_dim, hidden_dim * num_tokens)
        self.contact_head = nn.Linear(hidden_dim, 1)
        self.slip_head = nn.Linear(hidden_dim, 1)

    def _resolve_sequence(self, tactile: dict[str, torch.Tensor]) -> torch.Tensor:
        if "force_torque" in tactile:
            return tactile["force_torque"]
        if "tactile" in tactile:
            return tactile["tactile"]
        for v in tactile.values():
            if v.dim() >= 2:
                return v
        raise ValueError("TactileEncoder: no usable tactile sequence in batch.tactile")

    def forward(self, tactile: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        seq = self._resolve_sequence(tactile).float()
        if seq.dim() == 2:
            seq = seq.unsqueeze(1)
        h = self.input_proj(seq)
        h = self.temporal(h.transpose(1, 2)).transpose(1, 2)
        h = self.transformer(h)
        summary = h.mean(dim=1)
        flat = self.to_tokens(summary)
        tactile_tokens = flat.view(summary.shape[0], self.num_tokens, self.hidden_dim)
        return {
            "tactile_tokens": tactile_tokens,
            "tactile_summary": summary,
            "contact_logits": self.contact_head(summary),
            "slip_logits": self.slip_head(summary),
            "force_summary": summary,
        }
