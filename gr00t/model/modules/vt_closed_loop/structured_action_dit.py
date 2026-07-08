# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Coarse structured action DiT — wraps existing AlternateVLDiT with group-specific decoders."""

from __future__ import annotations

from typing import Mapping

import torch
from torch import nn

from gr00t.model.modules.vt_closed_loop.action_groups import ActionGroupSpec


def build_base_arm_sa_mask(
    num_state: int,
    horizon: int,
    group_specs: Mapping[str, ActionGroupSpec],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Block cross-attention between manip (gripper+arm) and base action tokens."""
    g = group_specs
    gripper = g.get("gripper", ActionGroupSpec("gripper", 0, 0))
    arm = g.get("arm", ActionGroupSpec("arm", 0, 0))
    base = g.get("base", ActionGroupSpec("base", 0, 0))
    manip_dim = gripper.dim + arm.dim
    base_dim = base.dim
    total_action = manip_dim + base_dim
    if total_action == 0:
        total_action = horizon  # fallback: treat each step as one token block

    num_native = num_state + total_action
    mask = torch.zeros(1, num_native, num_native, device=device, dtype=dtype)
    blocked = torch.finfo(dtype).min
    s0 = num_state
    manip_end = s0 + manip_dim
    base_end = manip_end + base_dim
    if manip_dim > 0 and base_dim > 0:
        mask[:, s0:manip_end, manip_end:base_end] = blocked
        mask[:, manip_end:base_end, s0:manip_end] = blocked
    return mask


class StructuredActionDiT(nn.Module):
    """Group-specific velocity heads on shared DiT trunk (injected at integration time)."""

    def __init__(
        self,
        hidden_dim: int,
        action_dim: int,
        action_groups: Mapping[str, ActionGroupSpec],
        *,
        decouple_base_arm: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.action_groups = dict(action_groups)
        self.decouple_base_arm = decouple_base_arm
        self.group_heads = nn.ModuleDict(
            {
                name: nn.Linear(hidden_dim, spec.dim)
                for name, spec in self.action_groups.items()
                if spec.dim > 0
            }
        )
        self.coupling_head = nn.Linear(hidden_dim, action_dim)
        nn.init.zeros_(self.coupling_head.weight)
        nn.init.zeros_(self.coupling_head.bias)

    def decode_grouped_velocity(
        self,
        hidden_action: torch.Tensor,
        base_velocity: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Fuse per-group linear heads into full ``(B, H, action_dim)`` velocity."""
        out = base_velocity.clone()
        group_outputs: dict[str, torch.Tensor] = {}
        for name, spec in self.action_groups.items():
            if name not in self.group_heads:
                continue
            delta = self.group_heads[name](hidden_action)
            group_outputs[name] = delta
            out[..., spec.start : spec.end] = out[..., spec.start : spec.end] + delta
        coupling = self.coupling_head(hidden_action)
        out = out + coupling
        if self.decouple_base_arm and "base" in self.action_groups:
            spec = self.action_groups["base"]
            out[..., spec.start : spec.end] = base_velocity[..., spec.start : spec.end]
        return {"a_mid": out, "group_outputs": group_outputs, "coupling": coupling}

    def forward(
        self,
        hidden_action: torch.Tensor,
        base_velocity: torch.Tensor,
        *,
        attn_maps: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        result = self.decode_grouped_velocity(hidden_action, base_velocity)
        if attn_maps is not None:
            result["attn_maps"] = attn_maps
        return result
