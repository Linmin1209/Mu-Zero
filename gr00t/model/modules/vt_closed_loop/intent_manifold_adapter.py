# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hierarchical intent manifold + route_probs (design v2)."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class HierarchicalIntentManifoldAdapter(nn.Module):
    """Multi-scale continuous intent tokens + learned route_probs (not phase classifier)."""

    def __init__(
        self,
        hidden_dim: int,
        state_dim: int,
        vlm_dim: int | None = None,
        num_route_modes: int = 16,
        num_global_tokens: int = 4,
        num_motion_tokens: int = 8,
        num_contact_tokens: int = 4,
        num_recovery_tokens: int = 2,
        num_heads: int = 8,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        vlm_dim = hidden_dim if vlm_dim is None else vlm_dim
        self.vlm_proj = (
            nn.Identity() if vlm_dim == hidden_dim else nn.Linear(vlm_dim, hidden_dim)
        )
        self.num_route_modes = num_route_modes

        self.global_queries = nn.Parameter(
            torch.randn(1, num_global_tokens, hidden_dim) * 0.02
        )
        self.motion_queries = nn.Parameter(
            torch.randn(1, num_motion_tokens, hidden_dim) * 0.02
        )
        self.contact_queries = nn.Parameter(
            torch.randn(1, num_contact_tokens, hidden_dim) * 0.02
        )
        self.recovery_queries = nn.Parameter(
            torch.randn(1, num_recovery_tokens, hidden_dim) * 0.02
        )

        self.state_proj = nn.Sequential(
            nn.LayerNorm(state_dim),
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        attn_kw = dict(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.global_attn = nn.MultiheadAttention(**attn_kw)
        self.motion_attn = nn.MultiheadAttention(**attn_kw)
        self.contact_attn = nn.MultiheadAttention(**attn_kw)
        self.recovery_attn = nn.MultiheadAttention(**attn_kw)

        self.router_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_route_modes),
        )
        self.intent_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        h_vlm: torch.Tensor,
        robot_state: torch.Tensor,
        tactile_tokens: torch.Tensor | None = None,
        contact_gate: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch_size = h_vlm.shape[0]
        if robot_state.dim() == 3:
            state_summary = robot_state[:, -1]
        elif robot_state.dim() == 2:
            state_summary = robot_state
        else:
            state_summary = robot_state.reshape(batch_size, -1)
        state_token = self.state_proj(state_summary).unsqueeze(1)
        base_context = torch.cat([self.vlm_proj(h_vlm), state_token], dim=1)

        global_q = self.global_queries.expand(batch_size, -1, -1)
        motion_q = self.motion_queries.expand(batch_size, -1, -1)
        global_tokens, _ = self.global_attn(global_q, base_context, base_context)
        motion_tokens, _ = self.motion_attn(motion_q, base_context, base_context)

        contact_context = base_context
        if tactile_tokens is not None:
            contact_context = torch.cat([base_context, tactile_tokens], dim=1)

        contact_q = self.contact_queries.expand(batch_size, -1, -1)
        contact_tokens, _ = self.contact_attn(contact_q, contact_context, contact_context)
        if contact_gate is not None:
            g = contact_gate
            if g.dim() == 2:
                g = g.unsqueeze(-1)
            contact_tokens = g * contact_tokens

        recovery_q = self.recovery_queries.expand(batch_size, -1, -1)
        recovery_tokens, _ = self.recovery_attn(recovery_q, contact_context, contact_context)

        all_tokens = self.intent_norm(
            torch.cat([global_tokens, motion_tokens, contact_tokens, recovery_tokens], dim=1)
        )
        pooled = all_tokens.mean(dim=1)
        route_logits = self.router_head(pooled)
        route_probs = F.softmax(route_logits, dim=-1)

        return {
            "global_intent_tokens": global_tokens,
            "motion_intent_tokens": motion_tokens,
            "contact_intent_tokens": contact_tokens,
            "recovery_intent_tokens": recovery_tokens,
            "all_intent_tokens": all_tokens,
            "intent_tokens": all_tokens,
            "route_logits": route_logits,
            "route_probs": route_probs,
        }


# Backward-compatible alias (legacy flat adapter name).
IntentManifoldAdapter = HierarchicalIntentManifoldAdapter
