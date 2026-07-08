"""VISOR: T-Rex-style sensor tactile + tri-path IHT + flow-late refine."""

from __future__ import annotations

import inspect

import torch
from torch import nn
import torch.nn.functional as F


def align_tactile_horizon(
    tactile: torch.Tensor,
    horizon: int,
    *,
    mode: str = "hold_last",
) -> torch.Tensor:
    """Pad or truncate tactile (B, T, C) to action horizon."""
    if tactile.shape[1] == horizon:
        return tactile
    if tactile.shape[1] > horizon:
        return tactile[:, :horizon]
    pad_len = horizon - tactile.shape[1]
    if mode == "zero_pad":
        pad = torch.zeros(
            tactile.shape[0],
            pad_len,
            tactile.shape[2],
            device=tactile.device,
            dtype=tactile.dtype,
        )
    elif mode == "hold_last":
        last = tactile[:, -1:, :].expand(-1, pad_len, -1)
        pad = last
    else:
        raise ValueError(f"Unknown tactile align mode: {mode}")
    return torch.cat([tactile, pad], dim=1)


def sanitize_finite_tensor(
    x: torch.Tensor,
    *,
    fill: float = 0.0,
    clamp: float | None = None,
) -> torch.Tensor:
    """Replace non-finite values; optional symmetric clamp for overflow safety."""
    if torch.isfinite(x).all():
        out = x
    else:
        out = torch.where(torch.isfinite(x), x, torch.full_like(x, fill))
    if clamp is not None:
        out = out.clamp(-clamp, clamp)
    return out


def compute_visor_aux_scales(
    train_step: int,
    *,
    warmup_steps: int = 2000,
    aux_delay_steps: int = 500,
) -> tuple[float, float]:
    """Return (coupling_scale, aux_loss_scale) for VISOR gate ramp and aux warmup."""
    warmup = max(int(warmup_steps), 1)
    step = max(int(train_step), 0)
    coupling_scale = min(1.0, step / float(warmup))
    aux_scale = min(1.0, max(0, step - int(aux_delay_steps)) / float(warmup))
    return coupling_scale, aux_scale


def compute_tactile_sample_mask(
    tactile_gt: torch.Tensor | None,
    *,
    tactile_mask: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Return per-sample tactile validity mask ``(B, 1)`` in ``{0, 1}``."""
    if tactile_gt is None:
        return None
    batch_size = tactile_gt.shape[0]
    device = tactile_gt.device
    dtype = tactile_gt.dtype
    if tactile_mask is not None:
        mask = tactile_mask.to(device=device, dtype=dtype)
        if mask.ndim == 1:
            mask = mask.unsqueeze(-1)
        elif mask.ndim == 0:
            mask = mask.reshape(1, 1)
        if mask.shape[0] == 1 and batch_size > 1:
            mask = mask.expand(batch_size, -1)
        return mask.clamp(0.0, 1.0)
    return torch.ones(batch_size, 1, device=device, dtype=dtype)


def compute_tactile_gt_stats(tactile_gt: torch.Tensor) -> dict[str, torch.Tensor]:
    """Batch-level contact / force-step rates from ground-truth tactile (B, T, C)."""
    channels = tactile_gt.shape[-1]
    if channels >= 4 and channels % 2 == 0:
        contact = tactile_gt[..., channels // 2 :].float()
    elif channels == 3:
        contact = tactile_gt[..., 2].float()
        if contact.ndim == 3:
            contact = contact.unsqueeze(-1)
    else:
        contact = tactile_gt[..., -1:].float()
    if contact.ndim == 2:
        contact = contact.unsqueeze(-1)
    contact_bin = (contact > 0.5).float()
    rate = contact_bin.mean().detach()
    return {
        "contact_rate": rate,
        "force_step_rate": rate,
    }


def compute_visor_tactile_training_loss(
    visor: "VisorModule",
    *,
    hidden_action: torch.Tensor,
    tactile_gt: torch.Tensor | None,
    vq_commit: torch.Tensor | None,
    coupling_lambda: torch.Tensor,
    coupling_scale: float,
    aux_scale: float = 1.0,
    tactile_mask: torch.Tensor | None = None,
    enabled: bool = True,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """DiT readout tactile supervision + VQ commit (scaled by warmup coupling_scale)."""
    device = hidden_action.device
    dtype = hidden_action.dtype
    zero = torch.zeros((), device=device, dtype=dtype)
    stats: dict[str, torch.Tensor] = {}

    if tactile_gt is not None:
        stats.update(compute_tactile_gt_stats(tactile_gt))

    sample_mask = compute_tactile_sample_mask(tactile_gt, tactile_mask=tactile_mask)
    if sample_mask is not None:
        stats["tactile_valid_rate"] = sample_mask.mean().detach()

    if not enabled or tactile_gt is None:
        stats.setdefault(
            "vq_commit_loss",
            vq_commit.detach().reshape(()) if vq_commit is not None else zero,
        )
        stats.setdefault("lambda_eff", torch.ones((), device=device, dtype=dtype))
        return zero, stats

    if sample_mask is not None and sample_mask.sum() <= 0:
        stats.setdefault(
            "vq_commit_loss",
            vq_commit.detach().reshape(()) if vq_commit is not None else zero,
        )
        stats.setdefault("lambda_eff", torch.ones((), device=device, dtype=dtype))
        return zero, stats

    scale = float(coupling_scale) * float(aux_scale)
    if scale <= 0:
        stats.setdefault(
            "vq_commit_loss",
            vq_commit.detach().reshape(()) if vq_commit is not None else zero,
        )
        stats.setdefault("lambda_eff", torch.ones((), device=device, dtype=dtype))
        return zero, stats

    tactile_pred = visor.predict_tactile_from_hidden(hidden_action)
    tactile_loss, loss_stats = visor.compute_tactile_loss(
        tactile_pred,
        tactile_gt,
        tactile_valid_mask=sample_mask,
        vq_commit_loss=vq_commit,
        coupling_lambda=coupling_lambda,
    )
    stats.update(loss_stats)
    return tactile_loss * scale, stats


def compute_visor_visual_training_loss(
    visor: "VisorModule",
    *,
    hidden_action: torch.Tensor,
    visual_gt: torch.Tensor | dict[str, torch.Tensor] | None,
    coupling_scale: float,
    aux_scale: float = 1.0,
    enabled: bool = True,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """DiT readout visual supervision (manip / nav / hand waypoint latents)."""
    device = hidden_action.device
    dtype = hidden_action.dtype
    zero = torch.zeros((), device=device, dtype=dtype)
    if not enabled or visual_gt is None or not visor.use_visual_supervision:
        return zero, {}
    scale = float(coupling_scale) * float(aux_scale)
    if scale <= 0:
        return zero, {}
    preds = visor.predict_visual_streams_from_hidden(hidden_action)
    if isinstance(visual_gt, dict):
        visual_loss, stats = visor.compute_multi_visual_loss(preds, visual_gt)
    else:
        visual_loss, stats = visor.compute_visual_loss(preds["manip"], visual_gt)
    return visual_loss * scale, stats


def _uses_visual_iht(gate_mode: str, use_visual_supervision: bool) -> bool:
    return use_visual_supervision or gate_mode in (
        "dual_split",
        "dual_hand_only",
        "visual_manip_nav_tactile_hand",
    )


def _uses_visual_gates(gate_mode: str) -> bool:
    return gate_mode in ("dual_split", "visual_manip_nav_tactile_hand")


def resolve_sensor_tactile(
    *,
    tactile_sensor: torch.Tensor | None,
    tactile_gt: torch.Tensor | None,
    action_horizon: int,
    training: bool,
    device: torch.device,
    dtype: torch.dtype,
    for_supervision: bool = False,
    align_mode: str = "hold_last",
) -> torch.Tensor:
    """Real tactile only (T-Rex sensor path). No imagined/WWM branch."""
    if for_supervision and training and tactile_gt is not None:
        if tactile_gt.shape[1] >= action_horizon:
            seq = tactile_gt[:, :action_horizon]
        else:
            seq = align_tactile_horizon(tactile_gt, action_horizon, mode=align_mode)
    elif tactile_sensor is not None:
        seq = tactile_sensor
        if seq.shape[1] != action_horizon:
            seq = align_tactile_horizon(seq, action_horizon, mode=align_mode)
    elif training and tactile_gt is not None:
        seq = align_tactile_horizon(tactile_gt, action_horizon, mode=align_mode)
    else:
        raise ValueError(
            "VISOR requires tactile_sensor (eval) or tactile_gt/tactile_sensor (train). "
            "Use robocasa365_config_4frame.py with haptic labels and TactileObservationWrapper at eval."
        )
    return seq.to(device=device, dtype=dtype)


def build_asymmetric_sa_mask(
    num_native: int,
    num_iht: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return additive mask (1, L, L): 0 allowed, large negative blocked (native -> IHT)."""
    total = num_native + num_iht
    mask = torch.zeros(1, total, total, device=device, dtype=dtype)
    if num_iht > 0:
        blocked = torch.finfo(dtype).min
        mask[:, :num_native, num_native:] = blocked
    return mask


def apply_decoupled_action_mask(
    action_mask: torch.Tensor,
    *,
    base_slice: tuple[int, int],
    decouple: bool,
) -> torch.Tensor:
    """Zero base_motion dims in the flow-matching mask when chassis is decoupled."""
    if not decouple:
        return action_mask
    masked = action_mask.clone()
    masked[..., int(base_slice[0]) : int(base_slice[1])] = 0.0
    return masked


def expand_asymmetric_sa_mask(sa_mask: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Broadcast mask to (B, L, L) for diffusers Attention (not B, 1, L, L)."""
    if sa_mask.shape[0] == 1:
        sa_mask = sa_mask.expand(batch_size, -1, -1)
    return sa_mask.contiguous()


def dit_accepts_sa_self_attention_mask(model: nn.Module) -> bool:
    """True when DiT/AlternateVLDiT forward accepts VISOR asymmetric SA mask."""
    try:
        params = inspect.signature(model.forward).parameters
    except (TypeError, ValueError):
        return False
    return "sa_self_attention_mask" in params


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


class TactileIHTEncoder(nn.Module):
    """Instant + VQ history tactile tokenization (sensor-only, no vision mixing)."""

    def __init__(
        self,
        *,
        input_embedding_dim: int,
        tactile_dim: int = 3,
        history_vq_tokens: int = 2,
        vq_codebook_size: int = 64,
        vq_hidden_dim: int = 64,
    ):
        super().__init__()
        self.tactile_dim = tactile_dim
        self.history_vq_tokens = history_vq_tokens
        self.num_iht_tokens = history_vq_tokens + 1

        self.history_vq = TactileVectorQuantizer(
            in_channels=tactile_dim,
            hidden_dim=vq_hidden_dim,
            num_tokens=history_vq_tokens,
            codebook_size=vq_codebook_size,
            embed_dim=input_embedding_dim,
        )
        self.instant_proj = nn.Linear(tactile_dim, input_embedding_dim)
        _zero_init_linear(self.instant_proj)
        self.iht_pos_embed = nn.Embedding(self.num_iht_tokens, input_embedding_dim)
        nn.init.normal_(self.iht_pos_embed.weight, mean=0.0, std=0.02)

    def build_iht_tokens(
        self,
        tactile_seq: torch.Tensor,
        *,
        active: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        history_tokens, vq_commit = self.history_vq(tactile_seq)
        current = tactile_seq[:, -1, :]
        instant = self.instant_proj(current).unsqueeze(1)
        tokens = torch.cat((history_tokens, instant), dim=1)
        pos = torch.arange(self.num_iht_tokens, device=tactile_seq.device, dtype=torch.long)
        tokens = tokens + self.iht_pos_embed(pos).unsqueeze(0)
        if active is not None:
            tokens = tokens * active.to(dtype=tokens.dtype).view(-1, 1, 1)
        return tokens, vq_commit

    def gate_event(self, tactile_seq: torch.Tensor) -> torch.Tensor:
        return tactile_seq[:, -1, :]


# Backward-compatible alias (v4 removes spatial vision token).
TriPathTactileEncoder = TactileIHTEncoder


class VisualIHTEncoder(nn.Module):
    """Compact visual future summary tokens for dual-modal VISOR (v4.2)."""

    def __init__(
        self,
        *,
        input_embedding_dim: int,
        visual_dim: int,
        visual_vq_tokens: int = 1,
        vq_codebook_size: int = 64,
        vq_hidden_dim: int = 64,
    ):
        super().__init__()
        self.visual_dim = visual_dim
        self.visual_vq_tokens = visual_vq_tokens
        self.num_iht_tokens = visual_vq_tokens + 1
        self.history_vq = TactileVectorQuantizer(
            in_channels=visual_dim,
            hidden_dim=vq_hidden_dim,
            num_tokens=visual_vq_tokens,
            codebook_size=vq_codebook_size,
            embed_dim=input_embedding_dim,
        )
        self.instant_proj = nn.Linear(visual_dim, input_embedding_dim)
        _zero_init_linear(self.instant_proj)
        self.iht_pos_embed = nn.Embedding(self.num_iht_tokens, input_embedding_dim)
        nn.init.normal_(self.iht_pos_embed.weight, mean=0.0, std=0.02)

    def build_iht_tokens(
        self,
        visual_summary: torch.Tensor,
        vision_context: torch.Tensor,
        *,
        active: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # visual_summary: (B, K, D_v); vision_context: (B, D_v) pooled eye-in-hand context
        history_tokens, vq_commit = self.history_vq(visual_summary)
        if visual_summary.shape[1] > 0:
            instant_source = visual_summary[:, -1, :]
        else:
            instant_source = vision_context
        instant = self.instant_proj(instant_source).unsqueeze(1)
        tokens = torch.cat((history_tokens, instant), dim=1)
        pos = torch.arange(self.num_iht_tokens, device=visual_summary.device, dtype=torch.long)
        tokens = tokens + self.iht_pos_embed(pos).unsqueeze(0)
        if active is not None:
            tokens = tokens * active.to(dtype=tokens.dtype).view(-1, 1, 1)
        return tokens, vq_commit

    def arm_gate_event(self, visual_pred: torch.Tensor) -> torch.Tensor:
        """Waypoint-to-waypoint motion summary for arm gate (B, D_v)."""
        if visual_pred.shape[1] < 2:
            return visual_pred[:, 0, :]
        delta = visual_pred[:, 1:, :] - visual_pred[:, :-1, :]
        return delta.mean(dim=1)


class VisorModule(nn.Module):
    def __init__(
        self,
        *,
        input_embedding_dim: int,
        action_horizon: int,
        vision_dim: int,
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
        use_split_action_gates: bool = False,
        arm_action_dim: int = 6,
        base_action_dim: int = 4,
        hand_action_dim: int = 1,
        tactile_num_force: int = 2,
        tactile_num_contact: int = 1,
        gate_mode: str = "tactile_hand_only",
        tactile_align_mode: str = "hold_last",
        use_visual_supervision: bool = False,
        visual_waypoints: int = 8,
        visual_dim: int = 2,
        loss_weight_visual: float = 0.03,
        visual_vq_tokens: int = 1,
        use_readout_fed_gates: bool = False,
        decouple_base_arm: bool = False,
        # Legacy alias; ignored when tri-path encoder is active.
        iht_tokens: int | None = None,
    ):
        super().__init__()
        self.action_horizon = action_horizon
        self.flow_tau_split = float(flow_tau_split)
        self.loss_weight_tactile = loss_weight_tactile
        self.loss_weight_visual = loss_weight_visual
        self.contact_loss_weight = contact_loss_weight
        self.contact_onset_boost = contact_onset_boost
        self.vq_commit_weight = vq_commit_weight
        self.use_contact_rate_prior = use_contact_rate_prior
        self.use_semantic_gate = use_semantic_gate
        self.tactile_num_force = int(tactile_num_force)
        self.tactile_num_contact = int(tactile_num_contact)
        self.tactile_dim = self.tactile_num_force + self.tactile_num_contact
        self.gate_mode = gate_mode
        self.decouple_base_arm = bool(decouple_base_arm)
        self.tactile_align_mode = tactile_align_mode
        self.use_visual_supervision = use_visual_supervision
        self.visual_waypoints = int(visual_waypoints)
        self.visual_dim = int(visual_dim)
        self.use_readout_fed_gates = use_readout_fed_gates
        lang_dim = language_dim if language_dim is not None else vision_dim

        self.tactile_iht = TactileIHTEncoder(
            input_embedding_dim=input_embedding_dim,
            tactile_dim=self.tactile_dim,
            history_vq_tokens=history_vq_tokens,
            vq_codebook_size=vq_codebook_size,
            vq_hidden_dim=vq_hidden_dim,
        )
        self.tri_path = self.tactile_iht
        self.visual_iht: VisualIHTEncoder | None = None
        if _uses_visual_iht(gate_mode, use_visual_supervision):
            self.visual_iht = VisualIHTEncoder(
                input_embedding_dim=input_embedding_dim,
                visual_dim=self.visual_dim,
                visual_vq_tokens=visual_vq_tokens,
                vq_codebook_size=vq_codebook_size,
                vq_hidden_dim=vq_hidden_dim,
            )
        self.iht_tokens = self.tactile_iht.num_iht_tokens + (
            self.visual_iht.num_iht_tokens if self.visual_iht is not None else 0
        )
        self.native_seq_len = 1 + action_horizon

        self.use_split_action_gates = use_split_action_gates
        self.arm_action_dim = arm_action_dim
        self.base_action_dim = base_action_dim
        self.hand_action_dim = hand_action_dim

        self.gate = nn.Parameter(torch.zeros(1))
        self.visual_gate = nn.Parameter(torch.zeros(1))
        self.visual_hand_gate = nn.Parameter(torch.zeros(1))
        self.gate_proj = nn.Linear(self.tactile_dim, decode_hidden_dim)
        _zero_init_linear(self.gate_proj)
        if use_split_action_gates:
            self.arm_gate_proj = nn.Linear(self.tactile_num_force, arm_action_dim)
            self.hand_gate_proj = nn.Linear(self.tactile_num_contact, hand_action_dim)
            _zero_init_linear(self.arm_gate_proj)
            _zero_init_linear(self.hand_gate_proj)
        else:
            self.arm_gate_proj = None
            self.hand_gate_proj = None
        if _uses_visual_gates(gate_mode):
            self.visual_arm_gate_proj = nn.Linear(self.visual_dim, arm_action_dim)
            _zero_init_linear(self.visual_arm_gate_proj)
            if gate_mode == "visual_manip_nav_tactile_hand":
                self.visual_base_gate_proj = nn.Linear(self.visual_dim, base_action_dim)
                self.visual_hand_gate_proj = nn.Linear(self.visual_dim, hand_action_dim)
                _zero_init_linear(self.visual_base_gate_proj)
                _zero_init_linear(self.visual_hand_gate_proj)
            else:
                self.visual_base_gate_proj = None
                self.visual_hand_gate_proj = None
        else:
            self.visual_arm_gate_proj = None
            self.visual_base_gate_proj = None
            self.visual_hand_gate_proj = None
        self.semantic_gate_proj = nn.Linear(lang_dim, 1)
        _zero_init_linear(self.semantic_gate_proj)
        self.tactile_readout = nn.Linear(decode_hidden_dim, self.tactile_dim)
        _zero_init_linear(self.tactile_readout)
        self.visual_readout_manip = nn.Linear(decode_hidden_dim, self.visual_dim)
        self.visual_readout_nav = nn.Linear(decode_hidden_dim, self.visual_dim)
        self.visual_readout_hand = nn.Linear(decode_hidden_dim, self.visual_dim)
        _zero_init_linear(self.visual_readout_manip)
        _zero_init_linear(self.visual_readout_nav)
        _zero_init_linear(self.visual_readout_hand)
        # Backward-compatible alias
        self.visual_readout = self.visual_readout_manip
        self.visual_context_proj = nn.Linear(vision_dim, self.visual_dim)
        _zero_init_linear(self.visual_context_proj)
        default_waypoints = [0, 5, 10, 15, 20, 25, 30, 35]
        indices = [
            min(d, action_horizon - 1)
            for d in default_waypoints[: self.visual_waypoints]
        ]
        while len(indices) < self.visual_waypoints:
            indices.append(action_horizon - 1)
        self.register_buffer(
            "visual_waypoint_indices",
            torch.tensor(indices, dtype=torch.long),
            persistent=False,
        )

    def predict_tactile_from_hidden(self, hidden_action: torch.Tensor) -> torch.Tensor:
        """Map DiT action hidden states (B, H, D) to tactile predictions (B, H, C)."""
        return self.tactile_readout(hidden_action)

    def _select_visual_waypoints(self, hidden_action: torch.Tensor) -> torch.Tensor:
        idx = self.visual_waypoint_indices.to(device=hidden_action.device)
        return hidden_action.index_select(1, idx)

    def predict_visual_streams_from_hidden(
        self, hidden_action: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Waypoint visual latents: manip/hand (eye_in_hand), nav (agentview)."""
        selected = self._select_visual_waypoints(hidden_action)
        return {
            "manip": self.visual_readout_manip(selected),
            "nav": self.visual_readout_nav(selected),
            "hand": self.visual_readout_hand(selected),
        }

    def predict_visual_from_hidden(self, hidden_action: torch.Tensor) -> torch.Tensor:
        """Backward-compatible manip stream readout (B, K, D_v)."""
        return self.predict_visual_streams_from_hidden(hidden_action)["manip"]

    def _force_slice(self, tactile: torch.Tensor) -> torch.Tensor:
        return tactile[..., : self.tactile_num_force]

    def _contact_slice(self, tactile: torch.Tensor) -> torch.Tensor:
        return tactile[..., self.tactile_num_force : self.tactile_dim]

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
                contact = self._contact_slice(tactile_gt).float()
                if contact.ndim == 3 and contact.shape[-1] == 1:
                    contact = contact.squeeze(-1)
                contact_rate = (contact > 0.5).float().mean(dim=tuple(range(1, contact.ndim)))
                contact_rate = contact_rate.reshape(-1, 1)
            elif tactile_pred is not None:
                contact = self._contact_slice(tactile_pred)
                contact_rate = torch.sigmoid(contact).mean(dim=tuple(range(1, contact.ndim)))
                contact_rate = contact_rate.reshape(-1, 1)
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

    @staticmethod
    def _broadcast_gate_scale(scale: torch.Tensor, gate_delta: torch.Tensor) -> torch.Tensor:
        """Combine per-batch scale (B, 1, 1) with gate delta (B, D) -> (B, 1, D)."""
        return scale * gate_delta.unsqueeze(1)

    def build_split_action_gates(
        self,
        tactile_seq: torch.Tensor,
        *,
        flow_time: torch.Tensor,
        coupling_lambda: torch.Tensor,
        coupling_scale: float = 1.0,
        detach_tactile: bool = True,
        tactile_pred: torch.Tensor | None = None,
        visual_pred: torch.Tensor | None = None,
        visual_pred_nav: torch.Tensor | None = None,
        visual_pred_hand: torch.Tensor | None = None,
        gate_mode: str | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """Per-part action-space gates: arm, base, hand (zero-init → Day-0 safe)."""
        if not self.use_split_action_gates:
            raise RuntimeError("build_split_action_gates requires use_split_action_gates=True")
        mode = gate_mode or self.gate_mode
        active = self.refine_active(flow_time).to(dtype=tactile_seq.dtype)
        scale_base = (
            active.view(-1, 1, 1)
            * coupling_scale
            * coupling_lambda.view(-1, 1, 1).to(dtype=tactile_seq.dtype)
        )

        tactile_source = tactile_seq
        if tactile_pred is not None and self.use_readout_fed_gates:
            tactile_source = tactile_pred.detach() if detach_tactile else tactile_pred
        event = self.tactile_iht.gate_event(tactile_source)
        tactile_scale = scale_base * self.gate

        arm_gate: torch.Tensor | None = None
        base_gate: torch.Tensor | None = None
        hand_gate: torch.Tensor | None = None

        if mode in ("tactile_split",):
            arm_gate = self._broadcast_gate_scale(
                tactile_scale, self.arm_gate_proj(self._force_slice(event))
            )
            hand_gate = self._broadcast_gate_scale(
                tactile_scale, self.hand_gate_proj(self._contact_slice(event))
            )
        elif mode in ("tactile_hand_only", "dual_hand_only"):
            hand_gate = self._broadcast_gate_scale(
                tactile_scale, self.hand_gate_proj(self._contact_slice(event))
            )
        elif mode == "dual_split":
            if visual_pred is not None and self.visual_arm_gate_proj is not None:
                visual_source = visual_pred.detach() if detach_tactile else visual_pred
                visual_event = self.visual_iht.arm_gate_event(visual_source)
                arm_gate = self._broadcast_gate_scale(
                    scale_base * self.visual_gate,
                    self.visual_arm_gate_proj(visual_event),
                )
            hand_gate = self._broadcast_gate_scale(
                tactile_scale, self.hand_gate_proj(self._contact_slice(event))
            )
        elif mode == "visual_manip_nav_tactile_hand":
            hand_tac = self._broadcast_gate_scale(
                tactile_scale, self.hand_gate_proj(self._contact_slice(event))
            )
            if visual_pred is not None and self.visual_arm_gate_proj is not None:
                manip = visual_pred.detach() if detach_tactile else visual_pred
                if manip.dim() == 3 and manip.shape[-1] != self.visual_dim:
                    manip = manip[..., : self.visual_dim]
                arm_event = self.visual_iht.arm_gate_event(manip)
                arm_gate = self._broadcast_gate_scale(
                    scale_base * self.visual_gate,
                    self.visual_arm_gate_proj(arm_event),
                )
            if visual_pred_nav is not None and self.visual_base_gate_proj is not None:
                nav = visual_pred_nav.detach() if detach_tactile else visual_pred_nav
                base_gate = self._broadcast_gate_scale(
                    scale_base * self.visual_gate,
                    self.visual_base_gate_proj(self.visual_iht.arm_gate_event(nav)),
                )
            if visual_pred_hand is not None and self.visual_hand_gate_proj is not None:
                hand_vis_src = visual_pred_hand.detach() if detach_tactile else visual_pred_hand
                hand_event = hand_vis_src[:, -1, :]
                hand_vis = self._broadcast_gate_scale(
                    scale_base * self.visual_hand_gate,
                    self.visual_hand_gate_proj(hand_event),
                )
                hand_gate = hand_vis + hand_tac
            else:
                hand_gate = hand_tac
        else:
            raise ValueError(f"Unknown visor gate_mode: {mode}")

        if self.decouple_base_arm:
            base_gate = None

        return arm_gate, base_gate, hand_gate

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
        tactile_seq: torch.Tensor,
        vision_context: torch.Tensor,
        *,
        flow_time: torch.Tensor,
        visual_summary: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        active = self.refine_active(flow_time).to(dtype=tactile_seq.dtype)
        tactile_tokens, vq_commit = self.tactile_iht.build_iht_tokens(
            tactile_seq,
            active=active,
        )
        if self.visual_iht is None:
            return tactile_tokens, vq_commit
        if visual_summary is None:
            ctx = self.visual_context_proj(vision_context)
            visual_summary = ctx.unsqueeze(1).expand(-1, self.visual_waypoints, -1)
        visual_ctx = (
            vision_context[..., : self.visual_dim]
            if vision_context.shape[-1] >= self.visual_dim
            else F.pad(
                vision_context,
                (0, self.visual_dim - vision_context.shape[-1]),
            )
        )
        visual_tokens, visual_commit = self.visual_iht.build_iht_tokens(
            visual_summary,
            visual_ctx,
            active=active,
        )
        commit = vq_commit + visual_commit
        return torch.cat((tactile_tokens, visual_tokens), dim=1), commit

    def compute_visual_loss(
        self,
        visual_pred: torch.Tensor,
        visual_gt: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        pred = F.normalize(visual_pred, dim=-1)
        target = F.normalize(visual_gt.to(dtype=pred.dtype), dim=-1)
        cosine = 1.0 - (pred * target).sum(dim=-1).mean()
        mse = F.mse_loss(pred, target)
        loss = self.loss_weight_visual * (0.5 * cosine + 0.5 * mse)
        return loss, {
            "visual_cosine_loss": cosine.detach(),
            "visual_mse_loss": mse.detach(),
        }

    def compute_multi_visual_loss(
        self,
        preds: dict[str, torch.Tensor],
        gts: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        total = torch.zeros((), device=preds["manip"].device, dtype=preds["manip"].dtype)
        stats: dict[str, torch.Tensor] = {}
        nav_weight = 0.0 if self.decouple_base_arm else 1.0
        stream_weights = {"manip": 1.0, "nav": nav_weight, "hand": 0.5}
        for key, weight in stream_weights.items():
            gt = gts.get(key)
            if gt is None:
                continue
            loss, sub = self.compute_visual_loss(preds[key], gt)
            total = total + float(weight) * loss
            for name, val in sub.items():
                stats[f"{key}_{name}"] = val
        return total, stats

    def compute_tactile_loss(
        self,
        tactile_pred: torch.Tensor,
        tactile_gt: torch.Tensor,
        *,
        tactile_valid_mask: torch.Tensor | None = None,
        vq_commit_loss: torch.Tensor | None = None,
        coupling_lambda: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        force_pred = self._force_slice(tactile_pred).clamp_min(0)
        force_gt = self._force_slice(tactile_gt).clamp_min(0)
        contact_logits = self._contact_slice(tactile_pred)
        contact_gt = self._contact_slice(tactile_gt).float()
        if contact_logits.ndim == 3 and contact_logits.shape[-1] == 1:
            contact_logits = contact_logits.squeeze(-1)
        if contact_gt.ndim == 3 and contact_gt.shape[-1] == 1:
            contact_gt = contact_gt.squeeze(-1)

        contact_bin = (contact_gt > 0.5).float()
        contact_rate = contact_bin.mean(dim=tuple(range(1, contact_bin.ndim))).reshape(-1, 1)
        contact_rate = contact_rate.clamp(min=1e-3)
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
        )
        if pos_weight.shape[1] == 1 and contact_logits.ndim > 2:
            l_contact = l_contact * pos_weight.view(-1, 1, *([1] * (l_contact.ndim - 2)))
        l_contact = l_contact * step_weight
        if tactile_valid_mask is not None:
            sample_mask = tactile_valid_mask.reshape(
                -1, *([1] * (l_contact.ndim - 1))
            ).to(dtype=l_contact.dtype)
            l_contact = l_contact * sample_mask

        force_mag = force_gt.sum(dim=-1, keepdim=True).clamp(min=1.0)
        force_weight = 1.0 + force_mag
        l_force = F.huber_loss(
            torch.log1p(force_pred),
            torch.log1p(force_gt),
            reduction="none",
        )
        force_mask = contact_bin
        if force_mask.ndim == 2:
            force_mask = force_mask.unsqueeze(-1)
        l_force = l_force * force_mask * force_weight
        if tactile_valid_mask is not None:
            sample_mask = tactile_valid_mask.reshape(
                -1, *([1] * (l_force.ndim - 1))
            ).to(dtype=l_force.dtype)
            l_force = l_force * sample_mask

        force_denom = (force_mask * force_weight).sum(dim=tuple(range(1, force_mask.ndim))).clamp(
            min=1.0
        )
        l_force_per_sample = l_force.sum(dim=tuple(range(1, l_force.ndim))) / force_denom
        contact_denom = step_weight.sum(dim=tuple(range(1, step_weight.ndim))).clamp(min=1.0)
        if contact_logits.ndim > 2:
            l_contact_per_sample = (
                l_contact.sum(dim=tuple(range(1, l_contact.ndim))) / contact_denom
            )
        else:
            l_contact_per_sample = (
                l_contact.sum(dim=tuple(range(1, l_contact.ndim))) / contact_denom
            )

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

        if tactile_valid_mask is not None:
            m = tactile_valid_mask.reshape(-1).to(dtype=per_sample.dtype)
            valid_count = m.sum().clamp(min=1.0)
            loss = self.loss_weight_tactile * (per_sample * m).sum() / valid_count
        else:
            loss = self.loss_weight_tactile * per_sample.mean()
        if vq_commit_loss is not None:
            if tactile_valid_mask is not None:
                m = tactile_valid_mask.reshape(-1).to(dtype=per_sample.dtype)
                valid_count = m.sum().clamp(min=1.0)
                vq_scalar = vq_commit_loss.reshape(())
                loss = loss + self.vq_commit_weight * vq_scalar * (m.sum() / valid_count)
            else:
                loss = loss + self.vq_commit_weight * vq_commit_loss
        flat_contact = contact_bin.reshape(contact_bin.shape[0], -1).mean(dim=1)
        force_step = force_mask.reshape(force_mask.shape[0], -1).mean(dim=1)
        return loss, {
            "lambda_eff": lambda_eff.detach(),
            "contact_rate": flat_contact.mean().detach(),
            "force_step_rate": force_step.mean().detach(),
            "vq_commit_loss": (
                vq_commit_loss.detach().reshape(())
                if vq_commit_loss is not None
                else torch.zeros((), device=tactile_pred.device)
            ),
        }
