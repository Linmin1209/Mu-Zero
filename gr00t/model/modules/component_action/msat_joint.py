# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lightweight MSAT-style joint attention for component action streams."""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F
from diffusers.models.embeddings import TimestepEmbedding, Timesteps


def _split_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    b, n, d = x.shape
    dh = d // num_heads
    return x.view(b, n, num_heads, dh).permute(0, 2, 1, 3).contiguous()


def _merge_heads(x: torch.Tensor) -> torch.Tensor:
    b, h, n, dh = x.shape
    return x.permute(0, 2, 1, 3).contiguous().view(b, n, h * dh)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * norm * self.weight


class TimestepEncoder(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=1)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        dtype = next(self.parameters()).dtype
        t = self.time_proj(timesteps).to(dtype)
        return self.timestep_embedder(t)


class RoPEEmbedder1D(nn.Module):
    """RoPE over a single axis (horizon within each component block)."""

    def __init__(self, head_dim: int, theta: float = 10000.0, max_seq_len: int = 512):
        super().__init__()
        assert head_dim % 2 == 0
        self.head_dim = head_dim
        freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(max_seq_len)
        freqs = torch.outer(t, freqs).float()
        self.register_buffer("freqs_cis", torch.polar(torch.ones_like(freqs), freqs), persistent=False)

    def forward(self, pos_ids: torch.Tensor) -> torch.Tensor:
        # pos_ids: (B, N) with -1 for non-horizon tokens -> use 0 angle
        pos = pos_ids.clamp(min=0)
        return self.freqs_cis[pos]


def apply_rotary_emb(
    q: torch.Tensor, k: torch.Tensor, freqs_cis: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    q_ = torch.view_as_complex(q.float().reshape(*q.shape[:-1], -1, 2))
    k_ = torch.view_as_complex(k.float().reshape(*k.shape[:-1], -1, 2))
    freqs = freqs_cis.unsqueeze(1)
    q_out = torch.view_as_real(q_ * freqs).flatten(-2)
    k_out = torch.view_as_real(k_ * freqs).flatten(-2)
    return q_out.type_as(q), k_out.type_as(k)


class SwiGLUFFN(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w12 = nn.Linear(dim, hidden_dim * 2)
        self.w3 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(x1) * x2)


class JointCrossAttentionBlock(nn.Module):
    """SA stream cross-attends to VL; both streams get FFN updates."""

    def __init__(
        self,
        sa_dim: int,
        vl_dim: int,
        num_heads: int,
        head_dim: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        inner = num_heads * head_dim

        self.sa_norm1 = RMSNorm(sa_dim, eps=norm_eps)
        self.vl_norm1 = RMSNorm(vl_dim, eps=norm_eps)
        self.sa_q = nn.Linear(sa_dim, inner, bias=True)
        self.sa_kv = nn.Linear(sa_dim, inner * 2, bias=True)
        self.vl_q = nn.Linear(vl_dim, inner, bias=True)
        self.vl_kv = nn.Linear(vl_dim, inner * 2, bias=True)
        self.sa_out = nn.Linear(inner, sa_dim, bias=True)
        self.vl_out = nn.Linear(inner, vl_dim, bias=True)
        self.q_norm_sa = RMSNorm(head_dim, eps=norm_eps)
        self.k_norm_sa = RMSNorm(head_dim, eps=norm_eps)
        self.q_norm_vl = RMSNorm(head_dim, eps=norm_eps)
        self.k_norm_vl = RMSNorm(head_dim, eps=norm_eps)

        sa_hidden = int(sa_dim * mlp_ratio)
        vl_hidden = int(vl_dim * mlp_ratio)
        self.sa_norm2 = RMSNorm(sa_dim, eps=norm_eps)
        self.vl_norm2 = RMSNorm(vl_dim, eps=norm_eps)
        self.sa_mlp = SwiGLUFFN(sa_dim, sa_hidden)
        self.vl_mlp = SwiGLUFFN(vl_dim, vl_hidden)
        self.dropout = nn.Dropout(dropout)

    def _attend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q_norm: RMSNorm,
        k_norm: RMSNorm,
        pe: Optional[torch.Tensor],
        attn_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        q = _split_heads(q, self.num_heads)
        k = _split_heads(k, self.num_heads)
        v = _split_heads(v, self.num_heads)
        q = q_norm(q)
        k = k_norm(k)
        if pe is not None:
            q, k = apply_rotary_emb(q, k, pe)
        b, h, nq, _ = q.shape
        nk = k.shape[2]
        qf = q.reshape(b * h, nq, self.head_dim)
        kf = k.reshape(b * h, nk, self.head_dim)
        vf = v.reshape(b * h, nk, self.head_dim)
        if attn_mask is not None:
            # attn_mask: (B, N_k) with 1=keep
            mask = attn_mask[:, None, None, :].expand(b, h, nq, nk).reshape(b * h, nq, nk)
            mask = mask.to(dtype=qf.dtype)
            mask = (1.0 - mask) * -1e4
        else:
            mask = None
        out = F.scaled_dot_product_attention(qf, kf, vf, attn_mask=mask)
        return _merge_heads(out.reshape(b, h, nq, self.head_dim))

    def forward(
        self,
        sa: torch.Tensor,
        vl: torch.Tensor,
        sa_pe: Optional[torch.Tensor] = None,
        vl_pe: Optional[torch.Tensor] = None,
        sa_mask: Optional[torch.Tensor] = None,
        vl_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sa_n = self.sa_norm1(sa)
        vl_n = self.vl_norm1(vl)

        sa_q = self.sa_q(sa_n)
        sa_k, sa_v = self.sa_kv(sa_n).chunk(2, dim=-1)
        vl_q = self.vl_q(vl_n)
        vl_k, vl_v = self.vl_kv(vl_n).chunk(2, dim=-1)

        q = torch.cat([sa_q, vl_q], dim=1)
        k = torch.cat([sa_k, vl_k], dim=1)
        v = torch.cat([sa_v, vl_v], dim=1)
        if sa_pe is not None and vl_pe is not None:
            pe = torch.cat([sa_pe, vl_pe], dim=1)
        else:
            pe = sa_pe if sa_pe is not None else vl_pe
        if sa_mask is not None and vl_mask is not None:
            mask = torch.cat([sa_mask, vl_mask], dim=1)
        else:
            mask = sa_mask if sa_mask is not None else vl_mask

        joint = self._attend(q, k, v, self.q_norm_sa, self.k_norm_sa, pe, mask)
        sa_len = sa.shape[1]
        sa_attn = self.sa_out(joint[:, :sa_len])
        vl_attn = self.vl_out(joint[:, sa_len:])
        sa = sa + self.dropout(sa_attn)
        vl = vl + self.dropout(vl_attn)

        sa = sa + self.dropout(self.sa_mlp(self.sa_norm2(sa)))
        vl = vl + self.dropout(self.vl_mlp(self.vl_norm2(vl)))
        return sa, vl


class JointSelfAttentionBlock(nn.Module):
    """Single-stream joint self-attention on concatenated [VL | SA]."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        head_dim: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        inner = num_heads * head_dim
        self.norm1 = RMSNorm(hidden_dim, eps=norm_eps)
        self.qkv = nn.Linear(hidden_dim, inner * 3, bias=True)
        self.out = nn.Linear(inner, hidden_dim, bias=True)
        self.q_norm = RMSNorm(head_dim, eps=norm_eps)
        self.k_norm = RMSNorm(head_dim, eps=norm_eps)
        self.norm2 = RMSNorm(hidden_dim, eps=norm_eps)
        self.mlp = SwiGLUFFN(hidden_dim, int(hidden_dim * mlp_ratio))
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        pe: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        n = self.norm1(x)
        q, k, v = self.qkv(n).chunk(3, dim=-1)
        q = _split_heads(q, self.num_heads)
        k = _split_heads(k, self.num_heads)
        v = _split_heads(v, self.num_heads)
        q = self.q_norm(q)
        k = self.k_norm(k)
        if pe is not None:
            q, k = apply_rotary_emb(q, k, pe)
        b, h, seq, _ = q.shape
        qf = q.reshape(b * h, seq, self.head_dim)
        kf = k.reshape(b * h, seq, self.head_dim)
        vf = v.reshape(b * h, seq, self.head_dim)
        if attn_mask is not None:
            mask = attn_mask[:, None, None, :].expand(b, h, seq, seq).reshape(b * h, seq, seq)
            mask = (1.0 - mask) * -1e4
        else:
            mask = None
        attn = F.scaled_dot_product_attention(qf, kf, vf, attn_mask=mask)
        attn = _merge_heads(attn.reshape(b, h, seq, self.head_dim))
        x = x + self.dropout(self.out(attn))
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x


class ComponentActionMSAT(nn.Module):
    """MSAT-style action transformer with tau input token and RoPE on horizon."""

    def __init__(
        self,
        sa_dim: int = 1536,
        vl_dim: int = 2048,
        num_attention_heads: int = 24,
        attention_head_dim: int = 64,
        depth_multi_stream: int = 4,
        depth_single_stream: int = 8,
        dropout: float = 0.2,
        mlp_ratio: float = 4.0,
        output_dim: int = 1024,
        rope_theta: float = 10000.0,
        max_seq_len: int = 1024,
    ):
        super().__init__()
        self.sa_dim = sa_dim
        self.inner_dim = num_attention_heads * attention_head_dim
        self.output_dim = output_dim

        self.timestep_encoder = TimestepEncoder(self.inner_dim)
        self.time_token_proj = (
            nn.Linear(self.inner_dim, sa_dim)
            if self.inner_dim != sa_dim
            else nn.Identity()
        )
        self.vl_proj = nn.Linear(vl_dim, sa_dim) if vl_dim != sa_dim else nn.Identity()

        self.double_blocks = nn.ModuleList(
            [
                JointCrossAttentionBlock(
                    sa_dim=sa_dim,
                    vl_dim=vl_dim,
                    num_heads=num_attention_heads,
                    head_dim=attention_head_dim,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(depth_multi_stream)
            ]
        )
        self.single_blocks = nn.ModuleList(
            [
                JointSelfAttentionBlock(
                    hidden_dim=sa_dim,
                    num_heads=num_attention_heads,
                    head_dim=attention_head_dim,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(depth_single_stream)
            ]
        )
        self.rope_sa = RoPEEmbedder1D(
            head_dim=attention_head_dim, theta=rope_theta, max_seq_len=max_seq_len
        )
        self.norm_out = RMSNorm(sa_dim)
        self.proj_out = nn.Linear(sa_dim, output_dim) if output_dim != sa_dim else nn.Identity()

    def encode_tau_token(self, timesteps: torch.Tensor) -> torch.Tensor:
        temb = self.timestep_encoder(timesteps)
        return self.time_token_proj(temb).unsqueeze(1)

    def forward(
        self,
        sa_tokens: torch.Tensor,
        vl_tokens: torch.Tensor,
        timesteps: torch.Tensor,
        sa_horizon_ids: torch.Tensor,
        sa_attention_mask: Optional[torch.Tensor] = None,
        vl_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        sa_pe = self.rope_sa(sa_horizon_ids)
        vl_len = vl_tokens.shape[1]
        vl_horizon = torch.full(
            (sa_horizon_ids.shape[0], vl_len),
            -1,
            device=sa_horizon_ids.device,
            dtype=sa_horizon_ids.dtype,
        )
        vl_pe = self.rope_sa(vl_horizon)

        sa, vl = sa_tokens, vl_tokens
        for block in self.double_blocks:
            sa, vl = block(
                sa,
                vl,
                sa_pe=sa_pe,
                vl_pe=vl_pe,
                sa_mask=sa_attention_mask,
                vl_mask=vl_attention_mask,
            )

        vl_sa = self.vl_proj(vl)
        joint = torch.cat([vl_sa, sa], dim=1)
        if vl_attention_mask is not None and sa_attention_mask is not None:
            joint_mask = torch.cat([vl_attention_mask, sa_attention_mask], dim=1)
        else:
            joint_mask = sa_attention_mask
        joint_pe = torch.cat(
            [
                torch.full(
                    (sa_horizon_ids.shape[0], vl_len),
                    -1,
                    device=sa_horizon_ids.device,
                    dtype=sa_horizon_ids.dtype,
                ),
                sa_horizon_ids,
            ],
            dim=1,
        )
        joint_rope = self.rope_sa(joint_pe)

        for block in self.single_blocks:
            joint = block(joint, pe=joint_rope, attn_mask=joint_mask)

        sa_out = joint[:, vl_len:]
        sa_out = self.norm_out(sa_out)
        return self.proj_out(sa_out)
