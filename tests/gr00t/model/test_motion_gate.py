"""Unit tests for MOSS MotionFusionGate input dimensions."""

from __future__ import annotations

import torch

from gr00t.model.modules.qwen3_motion import MotionFusionGate, _motion_gate_input_dim


def test_motion_gate_accepts_mismatched_text_and_vision_dims():
    gate = MotionFusionGate(vision_dim=1024, text_dim=2048, hidden_dim=256)
    text_ctx = torch.randn(4, 2048)
    vision_ctx = torch.randn(4, 1024)
    temporal_ctx = torch.randn(4, 1024)
    out = gate(text_ctx, vision_ctx, temporal_ctx)
    assert out.shape == (4, 1)
    assert gate.net[0].in_features == _motion_gate_input_dim(1024, 2048)
