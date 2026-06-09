# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pack/unpack variable-length component action token streams."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
import torch.nn as nn

from .component_schema import CANONICAL_COMPONENTS


@dataclass
class PackedActionStream:
    """Batch of packed SA tokens with metadata for dynamic slicing."""

    tokens: torch.Tensor  # (B, N_max, D)
    attention_mask: torch.Tensor  # (B, N_max), 1=valid
    horizon_ids: torch.Tensor  # (B, N_max), RoPE horizon index (-1 for non-horizon)
    layout: list[list[tuple[str, int, int]]]  # per batch: [(comp, start, end), ...]
    state_len: int
    tau_len: int = 1


def pack_action_stream(
    *,
    state_features: torch.Tensor,
    tau_token: torch.Tensor,
    component_tokens: Mapping[str, torch.Tensor],
    active_components: Sequence[Sequence[str]],
    component_projectors: nn.ModuleDict,
    component_type_embed: nn.Embedding,
) -> PackedActionStream:
    """Embed and pack per-sample component streams into a padded batch."""
    batch_size = tau_token.shape[0]
    hidden_dim = tau_token.shape[-1]
    device = tau_token.device
    dtype = tau_token.dtype

    per_sample_tokens: list[torch.Tensor] = []
    per_sample_horizon: list[torch.Tensor] = []
    layout: list[list[tuple[str, int, int]]] = []

    state_len = state_features.shape[1]
    tau_len = tau_token.shape[1]

    for b in range(batch_size):
        chunks: list[torch.Tensor] = [state_features[b : b + 1], tau_token[b : b + 1]]
        horizon_chunks: list[torch.Tensor] = [
            torch.full((1, state_len), -1, device=device, dtype=torch.long),
            torch.full((1, tau_len), -1, device=device, dtype=torch.long),
        ]
        sample_layout: list[tuple[str, int, int]] = []
        cursor = state_len + tau_len

        for comp in active_components[b]:
            if comp not in component_tokens:
                continue
            raw = component_tokens[comp][b : b + 1]
            proj = component_projectors[comp]
            tok = proj(raw) + component_type_embed.weight[CANONICAL_COMPONENTS.index(comp)]
            horizon = torch.arange(raw.shape[1], device=device, dtype=torch.long).unsqueeze(0)
            start, end = cursor, cursor + tok.shape[1]
            sample_layout.append((comp, start, end))
            chunks.append(tok)
            horizon_chunks.append(horizon)
            cursor = end

        tokens_b = torch.cat(chunks, dim=1)
        horizon_b = torch.cat(horizon_chunks, dim=1)
        per_sample_tokens.append(tokens_b.squeeze(0))
        per_sample_horizon.append(horizon_b.squeeze(0))
        layout.append(sample_layout)

    max_len = max(t.shape[0] for t in per_sample_tokens)
    tokens = torch.zeros(batch_size, max_len, hidden_dim, device=device, dtype=dtype)
    attention_mask = torch.zeros(batch_size, max_len, device=device, dtype=torch.float32)
    horizon_ids = torch.full((batch_size, max_len), -1, device=device, dtype=torch.long)

    for b, (tok, hid) in enumerate(zip(per_sample_tokens, per_sample_horizon)):
        n = tok.shape[0]
        tokens[b, :n] = tok
        attention_mask[b, :n] = 1.0
        horizon_ids[b, :n] = hid

    return PackedActionStream(
        tokens=tokens,
        attention_mask=attention_mask,
        horizon_ids=horizon_ids,
        layout=layout,
        state_len=state_len,
        tau_len=tau_len,
    )


def unpack_action_predictions(
    sa_output: torch.Tensor,
    packed: PackedActionStream,
    inverse_projectors: nn.ModuleDict,
) -> dict[str, torch.Tensor]:
    """Slice MSAT SA output and decode to per-component velocity fields."""
    predictions: dict[str, torch.Tensor] = {}
    batch_size = sa_output.shape[0]
    for b in range(batch_size):
        for comp, start, end in packed.layout[b]:
            if comp in predictions:
                # Lazy init batched tensor on first pass
                continue
    # Build batched dict
    for comp in CANONICAL_COMPONENTS:
        comp_slices = []
        valid_batch = []
        for b in range(batch_size):
            entry = next(((c, s, e) for c, s, e in packed.layout[b] if c == comp), None)
            if entry is None:
                continue
            _, start, end = entry
            hidden = sa_output[b : b + 1, start:end]
            comp_slices.append(inverse_projectors[comp](hidden))
            valid_batch.append(b)
        if not comp_slices:
            continue
        if len(comp_slices) == batch_size:
            predictions[comp] = torch.cat(comp_slices, dim=0)
        else:
            # Mixed batch: store only for samples that have this component
            predictions[comp] = comp_slices  # type: ignore[assignment]

    # Normalize mixed-batch path: require all samples share same active set for training
    for comp, val in list(predictions.items()):
        if isinstance(val, list):
            predictions[comp] = torch.cat(val, dim=0)
    return predictions


def unpack_action_predictions_batched(
    sa_output: torch.Tensor,
    packed: PackedActionStream,
    inverse_projectors: nn.ModuleDict,
) -> dict[str, torch.Tensor]:
    """Decode when every sample in the batch shares the same active component layout."""
    predictions: dict[str, torch.Tensor] = {}
    if not packed.layout:
        return predictions
    ref_layout = packed.layout[0]
    for comp, start, end in ref_layout:
        hidden = sa_output[:, start:end]
        predictions[comp] = inverse_projectors[comp](hidden)
    return predictions
