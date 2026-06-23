"""Unit tests for VISOR flow-late refine and tri-path tactile encoding."""

from __future__ import annotations

import torch

from gr00t.model.modules.visor.visor import TriPathTactileEncoder, VisorModule, build_asymmetric_sa_mask


def test_refine_active_only_on_late_flow_segment():
    visor = VisorModule(
        action_dim=8,
        hidden_dim=32,
        input_embedding_dim=16,
        action_horizon=4,
        vision_dim=12,
        proprio_dim=6,
        decode_hidden_dim=20,
        flow_tau_split=0.4,
        history_vq_tokens=2,
    )
    t_early = torch.tensor([[0.1], [0.2]])
    t_late = torch.tensor([[0.5], [0.9]])
    assert not visor.refine_active(t_early).any()
    assert visor.refine_active(t_late).all()


def test_tri_path_iht_token_count_and_gating():
    encoder = TriPathTactileEncoder(
        input_embedding_dim=16,
        vision_dim=12,
        history_vq_tokens=2,
    )
    tactile = torch.randn(2, 8, 3)
    vision = torch.randn(2, 12)
    active = torch.tensor([1.0, 0.0])
    tokens, commit = encoder.build_iht_tokens(tactile, vision, active=active)
    assert tokens.shape == (2, 4, 16)
    assert commit.ndim == 0
    assert torch.allclose(tokens[1], torch.zeros_like(tokens[1]))


def test_asymmetric_mask_blocks_native_to_iht():
    mask = build_asymmetric_sa_mask(5, 4, device=torch.device("cpu"), dtype=torch.float32)
    assert torch.isinf(mask[0, :5, 5:]).all()
    assert torch.all(mask[0, 5:, :5] == 0)
