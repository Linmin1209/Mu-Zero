"""Unit tests for VISOR v4 flow-late refine and sensor tactile encoding."""

from __future__ import annotations

import torch

from gr00t.model.modules.visor.visor import (
    TactileIHTEncoder,
    TriPathTactileEncoder,
    VisorModule,
    align_tactile_horizon,
    build_asymmetric_sa_mask,
    compute_tactile_gt_stats,
    compute_visor_aux_scales,
    expand_asymmetric_sa_mask,
    resolve_sensor_tactile,
)


def test_refine_active_only_on_late_flow_segment():
    visor = VisorModule(
        input_embedding_dim=16,
        action_horizon=4,
        vision_dim=12,
        decode_hidden_dim=20,
        flow_tau_split=0.4,
        history_vq_tokens=2,
    )
    t_early = torch.tensor([[0.1], [0.2]])
    t_late = torch.tensor([[0.5], [0.9]])
    assert not visor.refine_active(t_early).any()
    assert visor.refine_active(t_late).all()


def test_tactile_iht_token_count_and_gating():
    encoder = TactileIHTEncoder(
        input_embedding_dim=16,
        history_vq_tokens=2,
    )
    tactile = torch.randn(2, 8, 3)
    active = torch.tensor([1.0, 0.0])
    tokens, commit = encoder.build_iht_tokens(tactile, active=active)
    assert tokens.shape == (2, 3, 16)
    assert commit.ndim == 0
    assert torch.allclose(tokens[1], torch.zeros_like(tokens[1]))


def test_tri_path_alias_matches_tactile_iht():
    alias = TriPathTactileEncoder(input_embedding_dim=16, history_vq_tokens=2)
    assert alias.num_iht_tokens == 3


def test_align_tactile_horizon_hold_last():
    tactile = torch.full((1, 4, 3), 2.0)
    out = align_tactile_horizon(tactile, 40, mode="hold_last")
    assert out.shape == (1, 40, 3)
    assert out[0, -1, 0] == 2.0
    assert out[0, 5, 0] == 2.0


def test_asymmetric_mask_blocks_native_to_iht():
    mask = build_asymmetric_sa_mask(5, 3, device=torch.device("cpu"), dtype=torch.float32)
    assert mask.shape == (1, 8, 8)
    blocked = torch.finfo(torch.float32).min
    assert torch.all(mask[0, :5, 5:] == blocked)
    assert torch.all(mask[0, 5:, :5] == 0)


def test_expand_asymmetric_sa_mask_shape():
    mask = build_asymmetric_sa_mask(5, 3, device=torch.device("cpu"), dtype=torch.float32)
    expanded = expand_asymmetric_sa_mask(mask, batch_size=3)
    assert expanded.shape == (3, 8, 8)
    assert expanded.ndim == 3


def test_resolve_sensor_tactile_supervision_only_with_flag():
    gt = torch.ones(2, 40, 3)
    sensor = torch.zeros(2, 4, 3)
    seq = resolve_sensor_tactile(
        tactile_sensor=sensor,
        tactile_gt=gt,
        action_horizon=40,
        training=True,
        device=torch.device("cpu"),
        dtype=torch.float32,
        for_supervision=True,
    )
    assert seq.shape == (2, 40, 3)
    assert seq[0, 0, 0] == 1.0

    iht_seq = resolve_sensor_tactile(
        tactile_sensor=sensor,
        tactile_gt=gt,
        action_horizon=40,
        training=True,
        device=torch.device("cpu"),
        dtype=torch.float32,
        for_supervision=False,
        align_mode="hold_last",
    )
    assert iht_seq[0, -1, 0] == 0.0
    assert iht_seq[0, 3, 0] == 0.0


def test_resolve_sensor_tactile_uses_sensor_at_eval():
    gt = torch.ones(2, 40, 3)
    sensor = torch.full((2, 4, 3), 2.0)
    seq = resolve_sensor_tactile(
        tactile_sensor=sensor,
        tactile_gt=gt,
        action_horizon=40,
        training=False,
        device=torch.device("cpu"),
        dtype=torch.float32,
        align_mode="hold_last",
    )
    assert seq.shape == (2, 40, 3)
    assert seq[0, 3, 0] == 2.0
    assert seq[0, -1, 0] == 2.0


def test_resolve_sensor_tactile_requires_input():
    try:
        resolve_sensor_tactile(
            tactile_sensor=None,
            tactile_gt=None,
            action_horizon=40,
            training=False,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "VISOR requires" in str(exc)


def test_build_split_action_gates_hand_only():
    visor = VisorModule(
        input_embedding_dim=16,
        action_horizon=4,
        vision_dim=12,
        decode_hidden_dim=20,
        flow_tau_split=0.4,
        history_vq_tokens=2,
        use_split_action_gates=True,
        arm_action_dim=6,
        hand_action_dim=1,
        gate_mode="tactile_hand_only",
    )
    tactile = torch.randn(2, 4, 3)
    coupling = torch.ones(2, 1)
    t_late = torch.tensor([[0.5], [0.9]])

    arm_gate, base_gate, hand_gate = visor.build_split_action_gates(
        tactile, flow_time=t_late, coupling_lambda=coupling
    )
    assert arm_gate is None
    assert base_gate is None
    assert hand_gate.shape == (2, 1, 1)

    visor.gate.data.fill_(1.0)
    with torch.no_grad():
        visor.hand_gate_proj.weight.fill_(0.1)
    _, _, hand_gate = visor.build_split_action_gates(
        tactile, flow_time=t_late, coupling_lambda=coupling
    )
    assert not torch.allclose(hand_gate, torch.zeros_like(hand_gate))


def test_dual_split_arm_gate_from_visual_pred():
    visor = VisorModule(
        input_embedding_dim=16,
        action_horizon=40,
        vision_dim=12,
        decode_hidden_dim=20,
        flow_tau_split=0.4,
        history_vq_tokens=2,
        use_split_action_gates=True,
        gate_mode="dual_split",
        visual_dim=8,
        visual_waypoints=4,
    )
    tactile = torch.randn(2, 40, 3)
    visual_pred = torch.randn(2, 4, 8)
    coupling = torch.ones(2, 1)
    t_late = torch.tensor([[0.9], [0.8]])
    visor.visual_gate.data.fill_(1.0)
    with torch.no_grad():
        visor.visual_arm_gate_proj.weight.fill_(0.1)
        visor.hand_gate_proj.weight.fill_(0.1)
        visor.gate.data.fill_(1.0)
    arm_gate, base_gate, hand_gate = visor.build_split_action_gates(
        tactile,
        flow_time=t_late,
        coupling_lambda=coupling,
        visual_pred=visual_pred,
        gate_mode="dual_split",
    )
    assert arm_gate is not None
    assert base_gate is None
    assert hand_gate is not None


def test_visual_manip_nav_tactile_hand_triple_gate():
    visor = VisorModule(
        input_embedding_dim=16,
        action_horizon=40,
        vision_dim=12,
        decode_hidden_dim=20,
        flow_tau_split=0.4,
        history_vq_tokens=2,
        use_split_action_gates=True,
        gate_mode="visual_manip_nav_tactile_hand",
        visual_dim=8,
        visual_waypoints=4,
        base_action_dim=4,
    )
    tactile = torch.randn(2, 40, 3)
    visual_manip = torch.randn(2, 4, 8)
    visual_nav = torch.randn(2, 4, 8)
    visual_hand = torch.randn(2, 4, 8)
    coupling = torch.ones(2, 1)
    t_late = torch.tensor([[0.9], [0.8]])
    visor.visual_gate.data.fill_(1.0)
    visor.visual_hand_gate.data.fill_(1.0)
    with torch.no_grad():
        visor.visual_arm_gate_proj.weight.fill_(0.1)
        visor.visual_base_gate_proj.weight.fill_(0.1)
        visor.visual_hand_gate_proj.weight.fill_(0.1)
        visor.hand_gate_proj.weight.fill_(0.1)
        visor.gate.data.fill_(1.0)
    arm_gate, base_gate, hand_gate = visor.build_split_action_gates(
        tactile,
        flow_time=t_late,
        coupling_lambda=coupling,
        visual_pred=visual_manip,
        visual_pred_nav=visual_nav,
        visual_pred_hand=visual_hand,
        gate_mode="visual_manip_nav_tactile_hand",
    )
    assert arm_gate is not None
    assert base_gate is not None
    assert hand_gate is not None
    assert arm_gate.shape[-1] == 6
    assert base_gate.shape[-1] == 4
    assert hand_gate.shape[-1] == 1


def test_compute_visor_aux_scales():
    coupling, aux = compute_visor_aux_scales(0, warmup_steps=2000, aux_delay_steps=500)
    assert coupling == 0.0
    assert aux == 0.0
    coupling, aux = compute_visor_aux_scales(2500, warmup_steps=2000, aux_delay_steps=500)
    assert coupling == 1.0
    assert aux == 1.0


def test_compute_tactile_gt_stats_reports_contact_fraction():
    tactile_gt = torch.ones(2, 8, 3)
    stats = compute_tactile_gt_stats(tactile_gt)
    assert abs(float(stats["contact_rate"]) - 1.0) < 1e-5
    assert abs(float(stats["force_step_rate"]) - 1.0) < 1e-5


def test_compute_visor_tactile_training_loss_supervises_readout():
    from gr00t.model.modules.visor.visor import compute_visor_tactile_training_loss

    visor = VisorModule(
        input_embedding_dim=16,
        action_horizon=4,
        vision_dim=12,
        decode_hidden_dim=20,
        loss_weight_tactile=1.0,
        vq_commit_weight=0.0,
    )
    hidden = torch.randn(2, 4, 20)
    tactile_gt = torch.zeros(2, 4, 3)
    tactile_gt[..., 2] = 1.0
    coupling = torch.ones(2, 1)
    loss, stats = compute_visor_tactile_training_loss(
        visor,
        hidden_action=hidden,
        tactile_gt=tactile_gt,
        vq_commit=torch.tensor(0.0),
        coupling_lambda=coupling,
        coupling_scale=1.0,
        aux_scale=1.0,
    )
    assert float(loss) > 0.0
    assert abs(float(stats["contact_rate"]) - 1.0) < 1e-5


def test_tactile_mask_zeros_invalid_sample_loss():
    from gr00t.model.modules.visor.visor import compute_visor_tactile_training_loss

    visor = VisorModule(
        input_embedding_dim=16,
        action_horizon=4,
        vision_dim=12,
        decode_hidden_dim=20,
        loss_weight_tactile=1.0,
        vq_commit_weight=0.0,
    )
    hidden = torch.randn(2, 4, 20)
    tactile_gt = torch.zeros(2, 4, 3)
    tactile_gt[0, :, 2] = 1.0
    tactile_gt[1, :2, 2] = 1.0
    coupling = torch.ones(2, 1)
    full_loss, _ = compute_visor_tactile_training_loss(
        visor,
        hidden_action=hidden,
        tactile_gt=tactile_gt,
        vq_commit=torch.tensor(0.0),
        coupling_lambda=coupling,
        coupling_scale=1.0,
        aux_scale=1.0,
        tactile_mask=torch.ones(2, 1),
    )
    masked_loss, stats = compute_visor_tactile_training_loss(
        visor,
        hidden_action=hidden,
        tactile_gt=tactile_gt,
        vq_commit=torch.tensor(0.0),
        coupling_lambda=coupling,
        coupling_scale=1.0,
        aux_scale=1.0,
        tactile_mask=torch.tensor([[1.0], [0.0]]),
    )
    assert float(full_loss) > 0.0
    assert float(masked_loss) > 0.0
    assert abs(float(stats["tactile_valid_rate"]) - 0.5) < 1e-5


def test_tactile_mask_all_invalid_returns_zero():
    from gr00t.model.modules.visor.visor import compute_visor_tactile_training_loss

    visor = VisorModule(
        input_embedding_dim=16,
        action_horizon=4,
        vision_dim=12,
        decode_hidden_dim=20,
        loss_weight_tactile=1.0,
        vq_commit_weight=0.0,
    )
    hidden = torch.randn(2, 4, 20)
    tactile_gt = torch.zeros(2, 4, 3)
    tactile_gt[..., 2] = 1.0
    coupling = torch.ones(2, 1)
    loss, stats = compute_visor_tactile_training_loss(
        visor,
        hidden_action=hidden,
        tactile_gt=tactile_gt,
        vq_commit=torch.tensor(0.0),
        coupling_lambda=coupling,
        coupling_scale=1.0,
        aux_scale=1.0,
        tactile_mask=torch.zeros(2, 1),
    )
    assert float(loss) == 0.0
    assert float(stats["tactile_valid_rate"]) == 0.0


def test_visual_loss_when_supervision_enabled():
    visor = VisorModule(
        input_embedding_dim=16,
        action_horizon=40,
        vision_dim=12,
        decode_hidden_dim=20,
        use_visual_supervision=True,
        visual_dim=8,
        visual_waypoints=4,
        loss_weight_visual=1.0,
    )
    hidden = torch.randn(2, 40, 20)
    pred = visor.predict_visual_from_hidden(hidden)
    gt = torch.randn(2, 4, 8)
    loss, stats = visor.compute_visual_loss(pred, gt)
    assert float(loss) > 0.0
    assert "visual_cosine_loss" in stats
