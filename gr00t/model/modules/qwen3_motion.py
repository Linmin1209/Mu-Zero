# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""STSS/MOSS motion module integration for Qwen3-VL vision encoder (ported from RLDX-1)."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from gr00t.model.modules.motion import MotionModule

logger = logging.getLogger(__name__)


def pool_motion_gate_text_context(
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    input_ids: torch.Tensor,
    image_token_id: int,
) -> torch.Tensor:
    """Mean-pool text token embeddings (exclude image placeholder positions)."""
    text_mask = attention_mask.bool() & (input_ids != image_token_id)
    if not torch.any(text_mask):
        return inputs_embeds.mean(dim=1)
    mask = text_mask.unsqueeze(-1).to(dtype=inputs_embeds.dtype)
    summed = (inputs_embeds * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1.0)
    return summed / counts


@dataclass
class MotionConfig:
    use_motion: bool = False
    motion_insert_layer: int = 9
    motion_injection_point: str = "vision_encoder"
    motion_d_hid: int = 512
    motion_window: tuple[int, int, int] = (5, 9, 9)
    motion_ext_chnls: tuple[int, ...] = (256,)
    motion_int_chnls: tuple[int, ...] = (256, 256, 512)
    motion_corr_func: str = "cosine"
    motion_n_encoders: int = 1
    motion_use_layerscale: bool = False
    motion_layerscale_init: float = 1e-5
    motion_use_layernorm: bool = False
    motion_use_syncbn: bool = False
    motion_gradient_check: bool = False
    motion_int_mode: str = "lite"
    tune_motion: bool = True
    motion_use_gating: bool = True
    motion_gate_hidden: int = 256


class MotionFusionGate(nn.Module):
    """Scalar gate per batch: h' = h + g * moss_delta."""

    def __init__(
        self,
        vision_dim: int,
        text_dim: int | None = None,
        hidden_dim: int = 256,
    ):
        super().__init__()
        text_dim = vision_dim if text_dim is None else text_dim
        self.vision_dim = vision_dim
        self.text_dim = text_dim
        in_dim = text_dim + vision_dim * 2
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, 3.0)

    def forward(
        self,
        text_ctx: torch.Tensor,
        vision_ctx: torch.Tensor,
        temporal_ctx: torch.Tensor,
    ) -> torch.Tensor:
        return torch.sigmoid(self.net(torch.cat([text_ctx, vision_ctx, temporal_ctx], dim=-1)))


def _pool_hidden_for_gate(hidden_5d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    frame_tokens = hidden_5d.mean(dim=(2, 3))
    vision_ctx = frame_tokens.mean(dim=1)
    temporal_ctx = frame_tokens.std(dim=1)
    return vision_ctx, temporal_ctx


def _apply_motion_gate(
    visual: Any,
    moss_delta: torch.Tensor,
    hidden_5d: torch.Tensor,
    true_batch: int,
) -> torch.Tensor:
    gate_module = getattr(visual, "motion_gate", None)
    if gate_module is None:
        return moss_delta

    text_ctx = getattr(visual, "_gr00t_motion_text_context", None)
    vision_ctx, temporal_ctx = _pool_hidden_for_gate(hidden_5d)
    if text_ctx is None:
        text_ctx = torch.zeros_like(vision_ctx)
    if text_ctx.shape[0] != true_batch:
        raise ValueError(
            f"motion gate text context batch ({text_ctx.shape[0]}) != vision batch ({true_batch})"
        )

    gate_dtype = moss_delta.dtype
    gate = gate_module(
        text_ctx.to(dtype=gate_dtype),
        vision_ctx.to(dtype=gate_dtype),
        temporal_ctx.to(dtype=gate_dtype),
    )
    return moss_delta * gate.view(true_batch, 1, 1, 1, 1)


def apply_moss(
    visual: Any,
    hidden_states: torch.Tensor,
    grid_thw: torch.Tensor,
    num_frames: int,
    num_views: int,
) -> torch.Tensor:
    """Apply MotionModule with proper reshape for batch containing num_frames and num_views."""
    motion_block = visual.motion_block
    num_images = grid_thw.shape[0]
    hidden_dim = hidden_states.shape[-1]
    true_batch = num_images // (num_frames * num_views)
    if true_batch * num_frames * num_views != num_images:
        raise ValueError(
            f"image_grid_thw rows ({num_images}) must equal batch * num_frames * num_views "
            f"({true_batch} * {num_frames} * {num_views}) for motion module"
        )

    h = grid_thw[0, 1].item()
    w = grid_thw[0, 2].item()
    num_patches = h * w

    hidden_3d = hidden_states.reshape(num_images, num_patches, hidden_dim)
    hidden_5d = hidden_3d.reshape(true_batch, num_frames, num_views, num_patches, hidden_dim)

    merge_size = visual.spatial_merge_size
    merged_h, merged_w = h // merge_size, w // merge_size
    hidden_5d = hidden_5d.reshape(
        true_batch,
        num_frames,
        num_views,
        merged_h,
        merge_size,
        merged_w,
        merge_size,
        hidden_dim,
    )
    hidden_5d = hidden_5d.permute(0, 1, 2, 3, 5, 4, 6, 7).contiguous()
    hidden_5d = hidden_5d.reshape(true_batch, num_frames, num_views, num_patches, hidden_dim)

    hidden_bvtpd = hidden_5d.permute(0, 2, 1, 3, 4).contiguous()
    moss_input = hidden_bvtpd.reshape(
        true_batch * num_views * num_frames * num_patches, hidden_dim
    )

    moss_grid_sizes = torch.tensor(
        [[num_frames, h, w]] * (true_batch * num_views),
        dtype=torch.long,
        device=hidden_states.device,
    )

    use_motion_ckpt = _use_motion_activation_checkpoint(visual)
    ckpt_fn = _motion_checkpoint_fn(visual)
    if use_motion_ckpt and moss_input.requires_grad:

        def _motion_forward(moss_input_in: torch.Tensor) -> torch.Tensor:
            return motion_block(moss_input_in, moss_grid_sizes)

        moss_out = ckpt_fn(_motion_forward, moss_input)
    else:
        moss_out = motion_block(moss_input, moss_grid_sizes)

    moss_out = (
        moss_out.reshape(true_batch, num_views, num_frames, num_patches, hidden_dim)
        .permute(0, 2, 1, 3, 4)
        .contiguous()
    )
    moss_out = moss_out.reshape(
        true_batch,
        num_frames,
        num_views,
        merged_h,
        merge_size,
        merged_w,
        merge_size,
        hidden_dim,
    )
    moss_out = moss_out.permute(0, 1, 2, 3, 5, 4, 6, 7).contiguous()
    moss_out = moss_out.reshape(true_batch, num_frames, num_views, num_patches, hidden_dim)

    moss_delta = moss_out.reshape(true_batch, num_frames, num_views, num_patches, hidden_dim)
    moss_delta = _apply_motion_gate(visual, moss_delta, hidden_5d, true_batch)

    injection_point = getattr(visual, "motion_injection_point", "vision_encoder")
    if injection_point == "vision_encoder":
        return hidden_states + moss_delta.reshape(-1, hidden_dim)
    visual._moss_features = moss_delta
    visual._moss_meta = (true_batch, num_frames, num_views, int(h), int(w))
    return hidden_states


def _motion_checkpoint_fn(visual: Any):
    return getattr(visual, "_gradient_checkpointing_func", checkpoint)


def _use_motion_activation_checkpoint(visual: Any) -> bool:
    # visual stays in eval() when tune_visual=False; still checkpoint for tune_motion runs.
    return bool(
        getattr(visual, "_gr00t_use_motion_checkpoint", False)
        or getattr(visual, "gradient_checkpointing", False)
    )


def _tune_motion_only(visual: Any) -> bool:
    return bool(getattr(visual, "_gr00t_tune_motion_only", False))


def _run_vision_block(
    block: Any,
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    block_kwargs: dict,
) -> torch.Tensor:
    return block(
        hidden_states,
        cu_seqlens=cu_seqlens,
        position_embeddings=position_embeddings,
        **block_kwargs,
    )


def _checkpoint_vision_block(
    ckpt_fn: Any,
    block: Any,
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    block_kwargs: dict,
) -> torch.Tensor:
    def forward_hidden(hidden_states_in: torch.Tensor) -> torch.Tensor:
        return _run_vision_block(
            block, hidden_states_in, cu_seqlens, position_embeddings, block_kwargs
        )

    return ckpt_fn(forward_hidden, hidden_states)


def _visual_forward_with_motion(
    visual: Any,
    hidden_states: torch.Tensor,
    grid_thw: torch.Tensor,
    num_frames: int,
    num_views: int,
    **kwargs,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """HF Qwen3VLVisionModel forward with optional MOSS insertion."""
    hidden_states = visual.patch_embed(hidden_states)

    pos_embeds = visual.fast_pos_embed_interpolate(grid_thw)
    hidden_states = hidden_states + pos_embeds

    rotary_pos_emb = visual.rot_pos_emb(grid_thw)

    seq_len, _ = hidden_states.size()
    hidden_states = hidden_states.reshape(seq_len, -1)
    rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
    emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
    position_embeddings = (emb.cos(), emb.sin())

    cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
        dim=0,
        dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
    )
    cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

    deepstack_feature_lists = []
    motion_insert_layer = getattr(visual, "motion_insert_layer", 9)
    use_motion_ckpt = _use_motion_activation_checkpoint(visual)
    ckpt_fn = _motion_checkpoint_fn(visual)
    tune_motion_only = _tune_motion_only(visual)
    for layer_num, blk in enumerate(visual.blocks):
        run_no_grad = tune_motion_only and layer_num <= motion_insert_layer
        if run_no_grad:
            with torch.no_grad():
                hidden_states = _run_vision_block(
                    blk, hidden_states, cu_seqlens, position_embeddings, kwargs
                )
        elif use_motion_ckpt and hidden_states.requires_grad:
            hidden_states = _checkpoint_vision_block(
                ckpt_fn, blk, hidden_states, cu_seqlens, position_embeddings, kwargs
            )
        else:
            hidden_states = _run_vision_block(
                blk, hidden_states, cu_seqlens, position_embeddings, kwargs
            )
        if visual.motion_block is not None and layer_num == motion_insert_layer:
            hidden_states = apply_moss(
                visual, hidden_states, grid_thw, num_frames=num_frames, num_views=num_views
            )
        if layer_num in visual.deepstack_visual_indexes:
            deepstack_feature = visual.deepstack_merger_list[
                visual.deepstack_visual_indexes.index(layer_num)
            ](hidden_states)
            deepstack_feature_lists.append(deepstack_feature)

    hidden_states = visual.merger(hidden_states)
    return hidden_states, deepstack_feature_lists


def resolve_motion_text_hidden_size(model_or_config: Any) -> int | None:
    """Language embedding width for MOSS gate text context (often != vision hidden_size)."""
    config = getattr(model_or_config, "config", model_or_config)
    text_config = getattr(config, "text_config", None)
    if text_config is not None:
        hidden = int(getattr(text_config, "hidden_size", 0))
        if hidden > 0:
            return hidden
    hidden = int(getattr(config, "hidden_size", 0))
    return hidden if hidden > 0 else None


def _motion_gate_input_dim(vision_dim: int, text_dim: int | None) -> int:
    text_dim = vision_dim if text_dim is None else text_dim
    return text_dim + vision_dim * 2


def ensure_motion_gate(
    visual: Any,
    config: MotionConfig,
    *,
    text_hidden_size: int | None = None,
) -> None:
    if not config.use_motion or not config.motion_use_gating:
        return
    vision_dim = visual.config.hidden_size
    text_dim = text_hidden_size or vision_dim
    expected_in = _motion_gate_input_dim(vision_dim, text_dim)

    existing = getattr(visual, "motion_gate", None)
    if existing is not None:
        first_linear = existing.net[0]
        if isinstance(first_linear, nn.Linear) and first_linear.in_features == expected_in:
            return
        logger.warning(
            "Replacing MotionFusionGate (in_features %s -> %s; text=%s vision=%s)",
            getattr(first_linear, "in_features", "?"),
            expected_in,
            text_dim,
            vision_dim,
        )

    visual.motion_gate = MotionFusionGate(
        vision_dim, text_dim, config.motion_gate_hidden
    )
    visual._gr00t_motion_text_context = None
    logger.info(
        "Installed MotionFusionGate (text=%s vision=%s in=%s gate_hidden=%s)",
        text_dim,
        vision_dim,
        expected_in,
        config.motion_gate_hidden,
    )


def install_motion_module(
    visual: Any,
    config: MotionConfig,
    *,
    text_hidden_size: int | None = None,
) -> None:
    """Attach MotionModule to a HF Qwen3VLVisionModel and patch its forward pass."""
    if not config.use_motion:
        return

    visual.motion_insert_layer = config.motion_insert_layer
    visual.motion_injection_point = config.motion_injection_point
    visual.motion_use_gating = config.motion_use_gating
    visual._moss_features = None
    visual._moss_meta = None
    visual._gr00t_num_frames = 1
    visual._gr00t_num_views = 1
    visual._gr00t_motion_text_context = None
    visual._gr00t_use_motion_checkpoint = True
    visual._gr00t_tune_motion_only = False
    if not hasattr(visual, "_gradient_checkpointing_func") or visual._gradient_checkpointing_func is None:
        visual._gradient_checkpointing_func = checkpoint

    hidden_size = visual.config.hidden_size
    visual.motion_block = MotionModule(
        d_in=hidden_size,
        d_hid=config.motion_d_hid,
        d_out=hidden_size,
        window=config.motion_window,
        ext_chnls=config.motion_ext_chnls,
        int_chnls=config.motion_int_chnls,
        corr_func=config.motion_corr_func,
        n_encoders=config.motion_n_encoders,
        use_layerscale=config.motion_use_layerscale,
        layerscale_init=config.motion_layerscale_init,
        use_layernorm=config.motion_use_layernorm,
        use_syncbn=config.motion_use_syncbn,
        gradient_check=config.motion_gradient_check,
        int_mode=config.motion_int_mode,
    )
    visual.motion_block.initialize_weights()
    if config.motion_use_gating:
        ensure_motion_gate(visual, config, text_hidden_size=text_hidden_size)
        logger.info("MOSS fusion: task-modality gated residual")
    else:
        visual.motion_gate = None
    logger.info(
        "Installed MotionModule at vision layer %s (d_hid=%s, window=%s, injection=%s)",
        config.motion_insert_layer,
        config.motion_d_hid,
        config.motion_window,
        config.motion_injection_point,
    )

    def patched_forward(hidden_states, grid_thw, **kwargs):
        num_frames = int(kwargs.pop("num_frames", getattr(visual, "_gr00t_num_frames", 1)))
        num_views = int(kwargs.pop("num_views", getattr(visual, "_gr00t_num_views", 1)))
        return _visual_forward_with_motion(
            visual,
            hidden_states,
            grid_thw,
            num_frames=num_frames,
            num_views=num_views,
            **kwargs,
        )

    visual.forward = patched_forward


def set_motion_trainable(visual: Any, tune_motion: bool, tune_visual: bool) -> None:
    if not hasattr(visual, "motion_block") or visual.motion_block is None:
        return
    if tune_visual:
        visual.requires_grad_(True)
    elif tune_motion:
        for param in visual.parameters():
            param.requires_grad = False
        for param in visual.motion_block.parameters():
            param.requires_grad = True
        motion_gate = getattr(visual, "motion_gate", None)
        if motion_gate is not None:
            for param in motion_gate.parameters():
                param.requires_grad = True
    else:
        for param in visual.motion_block.parameters():
            param.requires_grad = False
        motion_gate = getattr(visual, "motion_gate", None)
        if motion_gate is not None:
            for param in motion_gate.parameters():
                param.requires_grad = False


def motion_state_dict_prefixes() -> tuple[str, ...]:
    return (
        "motion_block.",
        "motion_gate.",
        "backbone.motion_block.",
        "backbone.motion_gate.",
        "model.visual.motion_block.",
        "model.visual.motion_gate.",
    )


def is_motion_missing_key(key: str) -> bool:
    return "motion_block" in key or "motion_gate" in key


def motion_config_from_model_config(config: Any) -> MotionConfig:
    """Build MotionConfig from Gr00tN1d7Config (or any object with motion fields)."""
    return MotionConfig(
        use_motion=getattr(config, "use_motion", False),
        motion_insert_layer=getattr(config, "motion_insert_layer", 9),
        motion_injection_point=getattr(config, "motion_injection_point", "vision_encoder"),
        motion_d_hid=getattr(config, "motion_d_hid", 512),
        motion_window=getattr(config, "motion_window", (5, 9, 9)),
        motion_ext_chnls=getattr(config, "motion_ext_chnls", (256,)),
        motion_int_chnls=getattr(config, "motion_int_chnls", (256, 256, 512)),
        motion_corr_func=getattr(config, "motion_corr_func", "cosine"),
        motion_n_encoders=getattr(config, "motion_n_encoders", 1),
        motion_use_layerscale=getattr(config, "motion_use_layerscale", False),
        motion_layerscale_init=getattr(config, "motion_layerscale_init", 1e-5),
        motion_use_layernorm=getattr(config, "motion_use_layernorm", False),
        motion_use_syncbn=getattr(config, "motion_use_syncbn", False),
        motion_gradient_check=getattr(config, "motion_gradient_check", False),
        motion_int_mode=getattr(config, "motion_int_mode", "lite"),
        tune_motion=getattr(config, "tune_motion", True),
        motion_use_gating=getattr(config, "motion_use_gating", True),
        motion_gate_hidden=getattr(config, "motion_gate_hidden", 256),
    )
