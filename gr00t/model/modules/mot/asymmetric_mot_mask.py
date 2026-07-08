# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Asymmetric MoT self-attention masks (VT-WAM / TwinBrainVLA style).

Token layout: ``[native | IHT | anchor_inpaint | future_inpaint]``.

Rules (additive mask: 0 = attend, large negative = block):
- Native (state+action) may attend anchor inpaint, not IHT or future inpaint.
- Inpaint experts do not attend native keys (visual isolated from action KV).
- IHT and inpaint experts are mutually isolated.
- Within each segment, full bidirectional attention is allowed.
"""

from __future__ import annotations

import torch


def build_mot_inpaint_sa_mask(
    num_native: int,
    num_iht: int,
    num_anchor: int,
    num_future: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    block_native_to_iht: bool = True,
    block_native_to_future: bool = True,
    block_inpaint_to_native: bool = True,
    block_iht_inpaint_cross: bool = True,
) -> torch.Tensor:
    """Return additive SA mask ``(1, L, L)`` for MoT inpaint + VISOR layout."""
    total = num_native + num_iht + num_anchor + num_future
    mask = torch.zeros(1, total, total, device=device, dtype=dtype)
    if total == 0:
        return mask

    blocked = torch.finfo(dtype).min
    i_native = num_native
    i_iht_end = i_native + num_iht
    i_anchor_end = i_iht_end + num_anchor
    i_end = i_anchor_end + num_future

    if block_native_to_iht and num_iht > 0:
        mask[:, :i_native, i_native:i_iht_end] = blocked

    if block_native_to_future and num_future > 0:
        mask[:, :i_native, i_anchor_end:i_end] = blocked

    if block_inpaint_to_native and (num_anchor + num_future) > 0:
        mask[:, i_iht_end:i_end, :i_native] = blocked

    if block_iht_inpaint_cross and num_iht > 0 and (num_anchor + num_future) > 0:
        mask[:, i_native:i_iht_end, i_iht_end:i_end] = blocked
        mask[:, i_iht_end:i_end, i_native:i_iht_end] = blocked

    return mask
