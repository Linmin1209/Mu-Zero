# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Training losses for VT closed-loop policy."""

from __future__ import annotations

from typing import Mapping

import torch
import torch.nn.functional as F
from torch import nn

from gr00t.model.modules.vt_closed_loop.action_groups import AVTAGGroupWeights, ActionGroupSpec
from gr00t.model.modules.visor.visor import sanitize_finite_tensor


def build_action_group_weight_vector(
    action_dim: int,
    groups: Mapping[str, ActionGroupSpec],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Per-dimension loss weights from action group specs (design v2 §4)."""
    weights = torch.ones(action_dim, device=device, dtype=dtype)
    for spec in groups.values():
        for idx in range(spec.start, min(spec.end, action_dim)):
            weights[idx] = spec.loss_weight
    return weights


def compute_group_weighted_flow_loss(
    pred_velocity: torch.Tensor,
    target_velocity: torch.Tensor,
    action_mask: torch.Tensor,
    group_weights: torch.Tensor,
) -> torch.Tensor:
    """Masked MSE with per-action-dim group weights."""
    per_elem = F.mse_loss(pred_velocity, target_velocity, reduction="none") * action_mask.float()
    gw = group_weights.view(1, 1, -1).to(device=per_elem.device, dtype=per_elem.dtype)
    weighted = per_elem * gw
    denom = (action_mask.float() * gw).sum().clamp_min(1e-6)
    return weighted.sum() / denom


def intent_token_diversity_loss(tokens: torch.Tensor) -> torch.Tensor:
    """Penalize redundant intent tokens (design v2 §15.1)."""
    tokens = sanitize_finite_tensor(tokens.float())
    if tokens.numel() == 0:
        return torch.zeros((), device=tokens.device, dtype=tokens.dtype)
    tokens = F.normalize(tokens, dim=-1, eps=1e-6)
    sim = torch.matmul(tokens, tokens.transpose(-1, -2))
    eye = torch.eye(sim.shape[-1], device=tokens.device, dtype=tokens.dtype).unsqueeze(0)
    off_diag = sim * (1.0 - eye)
    loss = off_diag.pow(2).mean()
    return loss if torch.isfinite(loss) else torch.zeros((), device=tokens.device, dtype=tokens.dtype)


def _sanitize_loss_term(loss: torch.Tensor) -> torch.Tensor:
    loss = loss.float()
    if not torch.isfinite(loss):
        return torch.zeros((), device=loss.device, dtype=loss.dtype)
    return loss


def compute_router_losses(
    route_logits: torch.Tensor,
    route_probs: torch.Tensor,
    *,
    contact_gate: torch.Tensor | None = None,
    contact_route_indices: tuple[int, ...] = (2, 3),
) -> dict[str, torch.Tensor]:
    """Router regularizers (design v2 §5.4); no pseudo CE by default."""
    losses: dict[str, torch.Tensor] = {}
    mean_prob = route_probs.mean(dim=0).clamp(min=1e-8)
    mean_prob = mean_prob / mean_prob.sum()
    uniform = torch.full_like(mean_prob, 1.0 / mean_prob.numel())
    losses["router_balance"] = _sanitize_loss_term(
        (mean_prob * (mean_prob.log() - uniform.log())).sum()
    )
    if contact_gate is not None and route_probs.shape[-1] > max(contact_route_indices):
        idx = list(contact_route_indices)
        contact_prob = route_probs[:, idx].sum(dim=-1, keepdim=True).float()
        target = contact_gate.detach().float().reshape_as(contact_prob).clamp(0, 1)
        target = torch.nan_to_num(target, nan=0.0)
        losses["router_contact_consistency"] = _sanitize_loss_term(
            F.mse_loss(
                contact_prob.clamp(0, 1),
                target,
            )
        )
    return losses


def compute_avtag_loss(
    attn_maps: dict[str, torch.Tensor],
    token_groups: dict[str, slice],
    contact_gate: torch.Tensor,
    action_group_weights: Mapping[str, float] | AVTAGGroupWeights,
    margin: float = 0.05,
) -> torch.Tensor:
    """VT-WAM style: encourage tactile attention over visual during contact.

    ``L = g_contact * max(0, p_vis - p_tac + margin)`` (training-only).
    """
    if "action_to_visual" not in attn_maps or "action_to_tactile" not in attn_maps:
        return torch.zeros((), device=contact_gate.device)
    p_vis = attn_maps["action_to_visual"].mean(dim=-1)
    p_tac = attn_maps["action_to_tactile"].mean(dim=-1)
    hinge = F.relu(p_vis - p_tac + margin)
    g = contact_gate.view(-1, 1).to(hinge.dtype)
    loss = (g * hinge).mean()
    return loss


def masked_flux_feature_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None,
) -> torch.Tensor:
    if mask is None:
        return F.mse_loss(pred, target)
    m = mask.to(pred.dtype)
    while m.dim() < pred.dim():
        m = m.unsqueeze(-1)
    diff = (pred - target) ** 2 * m
    return diff.sum() / m.sum().clamp(min=1.0)


class ClosedLoopPolicyLoss(nn.Module):
    def __init__(
        self,
        *,
        weight_action: float = 1.0,
        weight_future: float = 0.1,
        weight_flux: float = 0.05,
        weight_contact: float = 0.2,
        weight_tactile: float = 0.1,
        weight_avtag: float = 0.02,
        weight_monitor: float = 0.2,
        weight_recovery: float = 0.5,
        weight_smooth: float = 0.01,
        weight_safe: float = 0.01,
    ):
        super().__init__()
        self.w_action = weight_action
        self.w_future = weight_future
        self.w_flux = weight_flux
        self.w_contact = weight_contact
        self.w_tactile = weight_tactile
        self.w_avtag = weight_avtag
        self.w_monitor = weight_monitor
        self.w_recovery = weight_recovery
        self.w_smooth = weight_smooth
        self.w_safe = weight_safe

    def forward(self, outputs: dict, batch: dict) -> dict[str, torch.Tensor]:
        losses: dict[str, torch.Tensor] = {}
        device = outputs.get("action", torch.tensor(0.0)).device
        zero = torch.zeros((), device=device)

        velocity_target = batch.get("velocity_target")
        if velocity_target is not None and "action" in outputs:
            losses["action"] = F.mse_loss(outputs["action"], velocity_target)
        else:
            losses["action"] = zero

        if outputs.get("future") and batch.get("future_target") is not None:
            losses["future"] = F.mse_loss(
                outputs["future"]["future_tokens"],
                batch["future_target"],
            )
        else:
            losses["future"] = zero

        if batch.get("flux_target") is not None and outputs.get("future"):
            losses["flux"] = masked_flux_feature_loss(
                outputs["future"].get("future_visual_latent", outputs["future"]["future_tokens"]),
                batch["flux_target"],
                batch.get("flux_mask"),
            )
        else:
            losses["flux"] = zero

        tactile_out = outputs.get("tactile") or {}
        if batch.get("contact_label") is not None and "contact_logits" in tactile_out:
            losses["contact"] = F.binary_cross_entropy_with_logits(
                tactile_out["contact_logits"].squeeze(-1),
                batch["contact_label"].float(),
            )
        else:
            losses["contact"] = zero

        if outputs.get("attn_maps") and outputs.get("contact_gate") is not None:
            losses["avtag"] = compute_avtag_loss(
                outputs["attn_maps"],
                batch.get("token_groups", {}),
                outputs["contact_gate"],
                batch.get("avtag_weights", AVTAGGroupWeights()),
            )
        else:
            losses["avtag"] = zero

        losses["tactile"] = zero
        losses["monitor"] = zero
        losses["recovery"] = zero
        losses["smooth"] = zero
        losses["safe"] = zero

        total = (
            self.w_action * losses["action"]
            + self.w_future * losses["future"]
            + self.w_flux * losses["flux"]
            + self.w_contact * losses["contact"]
            + self.w_tactile * losses["tactile"]
            + self.w_avtag * losses["avtag"]
            + self.w_monitor * losses["monitor"]
            + self.w_recovery * losses["recovery"]
            + self.w_smooth * losses["smooth"]
            + self.w_safe * losses["safe"]
        )
        losses["total"] = total
        return losses
