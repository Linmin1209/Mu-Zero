"""VISOR: WWM + suffix IHT + decoder gate + tactile auxiliary loss."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def build_asymmetric_sa_mask(
    num_native: int,
    num_iht: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return additive mask (B=1, L, L): 0 allowed, -inf blocked (native -> IHT)."""
    total = num_native + num_iht
    mask = torch.zeros(1, total, total, device=device, dtype=dtype)
    if num_iht > 0:
        mask[:, :num_native, num_native:] = float("-inf")
    return mask


def _zero_init_linear(linear: nn.Linear) -> None:
    nn.init.zeros_(linear.weight)
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)


class WristWorldModel(nn.Module):
    """Predict future tactile from action trajectory + vision + proprio."""

    def __init__(
        self,
        action_dim: int,
        hidden_dim: int,
        horizon: int,
        vision_dim: int,
        proprio_dim: int,
        conv_kernel: int = 5,
    ):
        super().__init__()
        self.horizon = horizon
        in_dim = action_dim + vision_dim + proprio_dim
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.step_embed = nn.Embedding(horizon, hidden_dim)
        nn.init.normal_(self.step_embed.weight, mean=0.0, std=0.02)
        padding = conv_kernel // 2
        self.temporal = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=conv_kernel, padding=padding),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=conv_kernel, padding=padding),
            nn.SiLU(),
        )
        self.out_proj = nn.Linear(hidden_dim, 3)
        _zero_init_linear(self.out_proj)

    def forward(
        self,
        action: torch.Tensor,
        flow_time: torch.Tensor,
        vision_context: torch.Tensor,
        proprio: torch.Tensor,
        *,
        use_clean_action: bool = False,
    ) -> torch.Tensor:
        # action: (B, H, D_a); flow_time: (B, 1, 1); vision_context: (B, D_v); proprio: (B, D_p)
        if use_clean_action:
            action_input = action
        else:
            action_input = action * (1.0 - flow_time)
        ctx = vision_context.unsqueeze(1).expand(-1, action_input.shape[1], -1)
        prop = proprio.unsqueeze(1).expand(-1, action_input.shape[1], -1)
        x = torch.cat([action_input, ctx, prop], dim=-1)
        x = self.input_proj(x)
        step_ids = torch.arange(action_input.shape[1], device=action.device, dtype=torch.long)
        x = x + self.step_embed(step_ids).unsqueeze(0)
        x = self.temporal(x.transpose(1, 2)).transpose(1, 2)
        return self.out_proj(x)


class VisorModule(nn.Module):
    def __init__(
        self,
        *,
        action_dim: int,
        hidden_dim: int,
        input_embedding_dim: int,
        action_horizon: int,
        vision_dim: int,
        proprio_dim: int,
        decode_hidden_dim: int,
        iht_tokens: int = 2,
        loss_weight_tactile: float = 0.5,
        contact_loss_weight: float = 1.0,
        contact_onset_boost: float = 3.0,
        use_contact_rate_prior: bool = False,
    ):
        super().__init__()
        self.action_horizon = action_horizon
        self.iht_tokens = iht_tokens
        self.native_seq_len = 1 + action_horizon
        self.loss_weight_tactile = loss_weight_tactile
        self.contact_loss_weight = contact_loss_weight
        self.contact_onset_boost = contact_onset_boost
        self.use_contact_rate_prior = use_contact_rate_prior

        self.wwm = WristWorldModel(
            action_dim,
            hidden_dim,
            action_horizon,
            vision_dim,
            proprio_dim,
        )
        self.iht_proj = nn.Linear(3, input_embedding_dim)
        _zero_init_linear(self.iht_proj)
        self.iht_pos_embed = nn.Embedding(iht_tokens, input_embedding_dim)
        nn.init.normal_(self.iht_pos_embed.weight, mean=0.0, std=0.02)

        self.gate = nn.Parameter(torch.zeros(1))
        self.gate_proj = nn.Linear(3, decode_hidden_dim)
        _zero_init_linear(self.gate_proj)

    def pool_vision_context(
        self, vl_embeds: torch.Tensor, image_mask: torch.Tensor
    ) -> torch.Tensor:
        mask = image_mask.to(dtype=vl_embeds.dtype).unsqueeze(-1)
        summed = (vl_embeds * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        return summed / denom

    def build_iht_tokens(self, tactile_pred: torch.Tensor) -> torch.Tensor:
        # tactile_pred: (B, H, 3) -> (B, K, D)
        pooled = tactile_pred.mean(dim=1)
        base = self.iht_proj(pooled).unsqueeze(1).expand(-1, self.iht_tokens, -1)
        pos = torch.arange(self.iht_tokens, device=tactile_pred.device, dtype=torch.long)
        return base + self.iht_pos_embed(pos).unsqueeze(0)

    def modulate_action_hidden(
        self, hidden: torch.Tensor, tactile_pred: torch.Tensor
    ) -> torch.Tensor:
        h_action = hidden[:, 1 : 1 + self.action_horizon, :].clone()
        event = tactile_pred.mean(dim=1)
        h_action = h_action + self.gate * self.gate_proj(event).unsqueeze(1)
        return h_action

    def compute_tactile_loss(
        self,
        tactile_pred: torch.Tensor,
        tactile_gt: torch.Tensor,
        *,
        tactile_valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Sparse tactile GT: approach steps are zero; force applies only after grasp."""
        force_pred = tactile_pred[..., :2].clamp_min(0)
        force_gt = tactile_gt[..., :2].clamp_min(0)
        contact_logits = tactile_pred[..., 2]
        contact_gt = tactile_gt[..., 2].float()
        if contact_gt.ndim == 3:
            contact_gt = contact_gt.squeeze(-1)

        contact_bin = (contact_gt > 0.5).float()
        contact_rate = contact_bin.mean(dim=1, keepdim=True).clamp(min=1e-3)
        pos_weight = ((1.0 - contact_rate) / contact_rate).clamp(max=20.0)

        first_contact = contact_bin * (contact_bin.cumsum(dim=1) == 1)
        prev_onset = F.pad(first_contact[:, :-1], (1, 0), value=0.0)
        next_onset = F.pad(first_contact[:, 1:], (0, 1), value=0.0)
        step_weight = torch.ones_like(contact_gt)
        boost = self.contact_onset_boost - 1.0
        step_weight = step_weight + boost * (first_contact + prev_onset + next_onset)

        l_contact = F.binary_cross_entropy_with_logits(
            contact_logits,
            contact_gt,
            reduction="none",
            pos_weight=pos_weight.expand_as(contact_gt),
        )
        l_contact = l_contact * step_weight
        if tactile_valid_mask is not None:
            l_contact = l_contact * tactile_valid_mask

        l_force = F.huber_loss(
            torch.log1p(force_pred),
            torch.log1p(force_gt),
            reduction="none",
        )
        force_mask = contact_bin.unsqueeze(-1)
        l_force = l_force * force_mask
        if tactile_valid_mask is not None:
            l_force = l_force * tactile_valid_mask.unsqueeze(-1)

        force_denom = force_mask.sum(dim=(1, 2)).clamp(min=1.0)
        l_force_per_sample = l_force.sum(dim=(1, 2)) / force_denom
        contact_denom = step_weight.sum(dim=1).clamp(min=1.0)
        l_contact_per_sample = l_contact.sum(dim=1) / contact_denom

        per_sample = l_force_per_sample + self.contact_loss_weight * l_contact_per_sample
        if self.use_contact_rate_prior:
            lambda_prior = contact_rate.squeeze(-1) / (contact_rate.squeeze(-1) + 0.05)
            per_sample = per_sample * lambda_prior
            lambda_eff = lambda_prior.mean()
        else:
            lambda_eff = torch.ones((), device=tactile_pred.device)

        loss = self.loss_weight_tactile * per_sample.mean()
        return loss, {
            "lambda_eff": lambda_eff.detach(),
            "contact_rate": contact_rate.mean().detach(),
            "force_step_rate": force_mask.mean().detach(),
        }
