"""VISOR: WWM + tri-path IHT + flow-late refine + tactile auxiliary loss."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

VALID_VISOR_TACTILE_MODES = frozenset({"imagine", "sensor", "hybrid"})


def normalize_visor_tactile_mode(mode: str) -> str:
    normalized = str(mode).strip().lower()
    if normalized not in VALID_VISOR_TACTILE_MODES:
        raise ValueError(
            f"visor_tactile_mode must be one of {sorted(VALID_VISOR_TACTILE_MODES)}, got {mode!r}"
        )
    return normalized


def align_tactile_horizon(tactile: torch.Tensor, horizon: int) -> torch.Tensor:
    """Pad or truncate tactile (B, T, C) to action horizon."""
    if tactile.shape[1] == horizon:
        return tactile
    if tactile.shape[1] > horizon:
        return tactile[:, :horizon]
    pad = torch.zeros(
        tactile.shape[0],
        horizon - tactile.shape[1],
        tactile.shape[2],
        device=tactile.device,
        dtype=tactile.dtype,
    )
    return torch.cat([tactile, pad], dim=1)


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


class TactileVectorQuantizer(nn.Module):
    """Discretize tactile force history (B, T, 3) into K VQ tokens (T-Rex-style)."""

    def __init__(
        self,
        *,
        in_channels: int = 3,
        hidden_dim: int = 64,
        num_tokens: int = 2,
        codebook_size: int = 64,
        embed_dim: int = 256,
        conv_kernel: int = 5,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.embed_dim = embed_dim
        padding = conv_kernel // 2
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=conv_kernel, padding=padding),
            nn.SiLU(),
            nn.Conv1d(
                hidden_dim,
                hidden_dim,
                kernel_size=conv_kernel,
                stride=2,
                padding=padding,
            ),
            nn.SiLU(),
            nn.Conv1d(
                hidden_dim,
                hidden_dim,
                kernel_size=conv_kernel,
                stride=2,
                padding=padding,
            ),
            nn.SiLU(),
        )
        self.to_code = nn.Linear(hidden_dim, embed_dim)
        self.codebook = nn.Embedding(codebook_size, embed_dim)
        nn.init.normal_(self.codebook.weight, mean=0.0, std=0.02)

    def _pool_temporal(self, h: torch.Tensor) -> torch.Tensor:
        # h: (B, D, T') -> (B, K, D)
        batch_size, channels, length = h.shape
        if length >= self.num_tokens:
            chunk = length // self.num_tokens
            trimmed = h[:, :, : chunk * self.num_tokens]
            pooled = trimmed.reshape(batch_size, channels, self.num_tokens, chunk).mean(dim=-1)
            return pooled.transpose(1, 2)
        pooled = F.adaptive_avg_pool1d(h, self.num_tokens)
        return pooled.transpose(1, 2)

    def forward(self, tactile_seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # tactile_seq: (B, T, 3)
        h = self.encoder(tactile_seq.transpose(1, 2))
        z = self.to_code(self._pool_temporal(h))
        flat = z.reshape(-1, self.embed_dim)
        distances = torch.cdist(flat, self.codebook.weight)
        indices = distances.argmin(dim=-1)
        quantized = self.codebook(indices).view_as(z)
        z_q = z + (quantized - z).detach()
        commit_loss = F.mse_loss(z, quantized.detach())
        return z_q, commit_loss


class TriPathTactileEncoder(nn.Module):
    """Instant + VQ history + spatial (vision proxy) tactile tokenization."""

    def __init__(
        self,
        *,
        input_embedding_dim: int,
        vision_dim: int,
        history_vq_tokens: int = 2,
        vq_codebook_size: int = 64,
        vq_hidden_dim: int = 64,
    ):
        super().__init__()
        self.history_vq_tokens = history_vq_tokens
        self.num_iht_tokens = history_vq_tokens + 2

        self.history_vq = TactileVectorQuantizer(
            in_channels=3,
            hidden_dim=vq_hidden_dim,
            num_tokens=history_vq_tokens,
            codebook_size=vq_codebook_size,
            embed_dim=input_embedding_dim,
        )
        self.instant_proj = nn.Linear(3, input_embedding_dim)
        _zero_init_linear(self.instant_proj)
        self.spatial_proj = nn.Sequential(
            nn.Linear(vision_dim, input_embedding_dim),
            nn.SiLU(),
            nn.Linear(input_embedding_dim, input_embedding_dim),
        )
        _zero_init_linear(self.spatial_proj[-1])
        self.iht_pos_embed = nn.Embedding(self.num_iht_tokens, input_embedding_dim)
        nn.init.normal_(self.iht_pos_embed.weight, mean=0.0, std=0.02)

    def build_iht_tokens(
        self,
        tactile_pred: torch.Tensor,
        vision_context: torch.Tensor,
        *,
        active: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # tactile_pred: (B, H, 3); vision_context: (B, D_v)
        history_tokens, vq_commit = self.history_vq(tactile_pred)
        # Sequence is chronological (e.g. deltas [-6,-4,-2,0] → last row is current t=0).
        current = tactile_pred[:, -1, :]
        instant = self.instant_proj(current).unsqueeze(1)
        spatial = self.spatial_proj(vision_context).unsqueeze(1)
        tokens = torch.cat((history_tokens, instant, spatial), dim=1)
        pos = torch.arange(self.num_iht_tokens, device=tactile_pred.device, dtype=torch.long)
        tokens = tokens + self.iht_pos_embed(pos).unsqueeze(0)
        if active is not None:
            tokens = tokens * active.to(dtype=tokens.dtype).view(-1, 1, 1)
        return tokens, vq_commit

    def gate_event(self, tactile_pred: torch.Tensor) -> torch.Tensor:
        return tactile_pred[:, -1, :]


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
        flow_tau_split: float = 0.4,
        history_vq_tokens: int = 2,
        vq_codebook_size: int = 64,
        vq_hidden_dim: int = 64,
        loss_weight_tactile: float = 0.5,
        contact_loss_weight: float = 1.0,
        contact_onset_boost: float = 3.0,
        vq_commit_weight: float = 0.1,
        use_contact_rate_prior: bool = True,
        use_semantic_gate: bool = True,
        language_dim: int | None = None,
        # Legacy alias; ignored when tri-path encoder is active.
        iht_tokens: int | None = None,
    ):
        super().__init__()
        self.action_horizon = action_horizon
        self.flow_tau_split = float(flow_tau_split)
        self.loss_weight_tactile = loss_weight_tactile
        self.contact_loss_weight = contact_loss_weight
        self.contact_onset_boost = contact_onset_boost
        self.vq_commit_weight = vq_commit_weight
        self.use_contact_rate_prior = use_contact_rate_prior
        self.use_semantic_gate = use_semantic_gate
        lang_dim = language_dim if language_dim is not None else vision_dim

        self.wwm = WristWorldModel(
            action_dim,
            hidden_dim,
            action_horizon,
            vision_dim,
            proprio_dim,
        )
        self.tri_path = TriPathTactileEncoder(
            input_embedding_dim=input_embedding_dim,
            vision_dim=vision_dim,
            history_vq_tokens=history_vq_tokens,
            vq_codebook_size=vq_codebook_size,
            vq_hidden_dim=vq_hidden_dim,
        )
        self.iht_tokens = self.tri_path.num_iht_tokens
        self.native_seq_len = 1 + action_horizon

        self.gate = nn.Parameter(torch.zeros(1))
        self.gate_proj = nn.Linear(3, decode_hidden_dim)
        _zero_init_linear(self.gate_proj)
        self.semantic_gate_proj = nn.Linear(lang_dim, 1)
        _zero_init_linear(self.semantic_gate_proj)

    def pool_language_context(
        self,
        vl_embeds: torch.Tensor,
        image_mask: torch.Tensor,
        backbone_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        text_mask = (~image_mask) & backbone_attention_mask
        mask = text_mask.to(dtype=vl_embeds.dtype).unsqueeze(-1)
        summed = (vl_embeds * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        return summed / denom

    def compute_coupling_lambda(
        self,
        language_context: torch.Tensor,
        *,
        tactile_gt: torch.Tensor | None = None,
        tactile_pred: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """λ_eff = λ_sem * λ_prior; down-weights nav / no-contact episodes."""
        batch_size = language_context.shape[0]
        device = language_context.device
        if self.use_semantic_gate:
            lambda_sem = torch.sigmoid(self.semantic_gate_proj(language_context))
        else:
            lambda_sem = torch.ones(batch_size, 1, device=device, dtype=language_context.dtype)

        if self.use_contact_rate_prior:
            if tactile_gt is not None:
                contact = tactile_gt[..., 2].float()
                if contact.ndim == 3:
                    contact = contact.squeeze(-1)
                contact_rate = (contact > 0.5).float().mean(dim=1, keepdim=True)
            elif tactile_pred is not None:
                contact_rate = torch.sigmoid(tactile_pred[..., 2]).mean(dim=1, keepdim=True)
            else:
                contact_rate = torch.zeros(batch_size, 1, device=device, dtype=language_context.dtype)
            lambda_prior = contact_rate / (contact_rate + 0.05)
        else:
            lambda_prior = torch.ones(batch_size, 1, device=device, dtype=language_context.dtype)

        return (lambda_sem * lambda_prior).clamp(0.0, 1.0)

    def build_gate_delta(
        self,
        tactile_pred: torch.Tensor,
        *,
        flow_time: torch.Tensor,
        coupling_lambda: torch.Tensor,
        coupling_scale: float = 1.0,
        detach_tactile: bool = True,
    ) -> torch.Tensor:
        """Per-timestep hidden delta for gated components (zero-init → Day-0 safe)."""
        active = self.refine_active(flow_time).to(dtype=tactile_pred.dtype)
        event_source = tactile_pred.detach() if detach_tactile else tactile_pred
        event = self.tri_path.gate_event(event_source)
        gate = (
            active.view(-1, 1, 1)
            * coupling_scale
            * coupling_lambda.view(-1, 1, 1).to(dtype=event.dtype)
            * self.gate
            * self.gate_proj(event).unsqueeze(1)
        )
        return gate

    def modulate_action_hidden(
        self,
        hidden: torch.Tensor,
        tactile_pred: torch.Tensor,
        *,
        flow_time: torch.Tensor,
        coupling_lambda: torch.Tensor | None = None,
        coupling_scale: float = 1.0,
    ) -> torch.Tensor:
        if coupling_lambda is None:
            batch_size = hidden.shape[0]
            coupling_lambda = torch.ones(
                batch_size, 1, device=hidden.device, dtype=hidden.dtype
            )
        gate = self.build_gate_delta(
            tactile_pred,
            flow_time=flow_time,
            coupling_lambda=coupling_lambda,
            coupling_scale=coupling_scale,
        )
        return hidden + gate

    def refine_active(self, flow_time: torch.Tensor) -> torch.Tensor:
        """GR00T flow time: 0=noise, 1=clean. VISOR refines only on the late segment."""
        t = flow_time.reshape(flow_time.shape[0], -1)[:, 0]
        return t >= self.flow_tau_split

    def pool_vision_context(
        self, vl_embeds: torch.Tensor, image_mask: torch.Tensor
    ) -> torch.Tensor:
        mask = image_mask.to(dtype=vl_embeds.dtype).unsqueeze(-1)
        summed = (vl_embeds * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        return summed / denom

    def build_iht_tokens(
        self,
        tactile_pred: torch.Tensor,
        vision_context: torch.Tensor,
        *,
        flow_time: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        active = self.refine_active(flow_time).to(dtype=tactile_pred.dtype)
        return self.tri_path.build_iht_tokens(
            tactile_pred,
            vision_context,
            active=active,
        )

    def compute_tactile_loss(
        self,
        tactile_pred: torch.Tensor,
        tactile_gt: torch.Tensor,
        *,
        tactile_valid_mask: torch.Tensor | None = None,
        vq_commit_loss: torch.Tensor | None = None,
        coupling_lambda: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
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

        force_mag = force_gt.sum(dim=-1, keepdim=True).clamp(min=1.0)
        force_weight = 1.0 + force_mag
        l_force = F.huber_loss(
            torch.log1p(force_pred),
            torch.log1p(force_gt),
            reduction="none",
        )
        force_mask = contact_bin.unsqueeze(-1)
        l_force = l_force * force_mask * force_weight
        if tactile_valid_mask is not None:
            l_force = l_force * tactile_valid_mask.unsqueeze(-1)

        force_denom = (force_mask * force_weight).sum(dim=(1, 2)).clamp(min=1.0)
        l_force_per_sample = l_force.sum(dim=(1, 2)) / force_denom
        contact_denom = step_weight.sum(dim=1).clamp(min=1.0)
        l_contact_per_sample = l_contact.sum(dim=1) / contact_denom

        per_sample = l_force_per_sample + self.contact_loss_weight * l_contact_per_sample
        if coupling_lambda is not None:
            per_sample = per_sample * coupling_lambda.reshape(-1).to(dtype=per_sample.dtype)
            lambda_eff = coupling_lambda.mean()
        elif self.use_contact_rate_prior:
            lambda_prior = contact_rate.squeeze(-1) / (contact_rate.squeeze(-1) + 0.05)
            per_sample = per_sample * lambda_prior
            lambda_eff = lambda_prior.mean()
        else:
            lambda_eff = torch.ones((), device=tactile_pred.device)

        loss = self.loss_weight_tactile * per_sample.mean()
        if vq_commit_loss is not None:
            loss = loss + self.vq_commit_weight * vq_commit_loss
        return loss, {
            "lambda_eff": lambda_eff.detach(),
            "contact_rate": contact_rate.mean().detach(),
            "force_step_rate": force_mask.mean().detach(),
            "vq_commit_loss": (
                vq_commit_loss.detach().reshape(())
                if vq_commit_loss is not None
                else torch.zeros((), device=tactile_pred.device)
            ),
        }
