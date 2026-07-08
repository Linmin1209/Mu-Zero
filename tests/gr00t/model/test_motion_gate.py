"""Unit tests for MOSS MotionFusionGate input dimensions."""

from __future__ import annotations

import torch

from gr00t.model.modules.qwen3_motion import MotionFusionGate, _motion_gate_input_dim


def test_motion_gate_accepts_mismatched_text_and_vision_dims():
    gate = MotionFusionGate(
        vision_dim=1024,
        text_dim=2048,
        hidden_dim=256,
        mode="full",
    )
    text_ctx = torch.randn(4, 2048)
    vision_ctx = torch.randn(4, 1024)
    temporal_ctx = torch.randn(4, 1024)
    out = gate(text_ctx, vision_ctx, temporal_ctx)
    assert out.shape == (4, 1)
    assert gate.net[0].in_features == _motion_gate_input_dim(1024, 2048, mode="full")


def test_motion_gate_text_only_uses_text_dim_only():
    gate = MotionFusionGate(vision_dim=1024, text_dim=2048, hidden_dim=64, mode="text_only")
    assert gate.net[0].in_features == 2048
    text_ctx = torch.randn(3, 2048)
    vision_ctx = torch.randn(3, 1024)
    temporal_ctx = torch.randn(3, 1024)
    out = gate(text_ctx, vision_ctx, temporal_ctx)
    assert out.shape == (3, 1)


def test_motion_gate_init_bias_controls_initial_scale():
    gate = MotionFusionGate(
        vision_dim=64,
        hidden_dim=32,
        init_bias=0.0,
        mode="text_only",
        g_min=0.0,
        g_max=0.8,
    )
    text_ctx = torch.zeros(2, 64)
    vision_ctx = torch.zeros(2, 64)
    temporal_ctx = torch.zeros(2, 64)
    out = gate(text_ctx, vision_ctx, temporal_ctx)
    expected = torch.tensor(0.4)
    assert torch.allclose(out, expected.expand_as(out), atol=1e-4)


def test_motion_gate_text_only_responds_to_different_text():
    torch.manual_seed(0)
    gate = MotionFusionGate(
        vision_dim=64,
        hidden_dim=32,
        init_bias=0.0,
        mode="text_only",
        g_min=0.0,
        g_max=0.8,
    )
    vision_ctx = torch.randn(4, 64)
    temporal_ctx = torch.randn(4, 64)
    text_a = torch.randn(4, 64)
    text_b = text_a + torch.randn(4, 64) * 0.5
    out_a = gate(text_a, vision_ctx, temporal_ctx)
    out_b = gate(text_b, vision_ctx, temporal_ctx)
    assert not torch.allclose(out_a, out_b, atol=1e-5)
