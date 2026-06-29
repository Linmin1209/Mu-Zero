"""Tests for shared flat decoder + per-component LoRA action head."""

from __future__ import annotations

import torch

from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
from gr00t.model.modules.component_action.component_factored_action_head import (
    CategorySpecificLoRA,
    build_component_factored_action_head,
)
from gr00t.model.modules.embodiment_conditioned_mlp import CategorySpecificMLP


class _StubActionHead(torch.nn.Module):
    hidden_size = 64
    action_dim = 12
    state_dropout_prob = 0.0
    num_timestep_buckets = 10
    num_inference_timesteps = 4

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.action_decoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            output_dim=self.action_dim,
        )

    def sample_time(self, batch_size, device, dtype):
        return torch.full((batch_size, 1, 1), 0.5, device=device, dtype=dtype)


def _minimal_config() -> Gr00tN1d7Config:
    return Gr00tN1d7Config(
        component_action_key_order=[
            "gripper_close",
            "end_effector_position",
            "end_effector_rotation",
            "base_motion",
            "control_mode",
        ],
        component_action_key_dims={
            "gripper_close": 1,
            "end_effector_position": 3,
            "end_effector_rotation": 3,
            "base_motion": 4,
            "control_mode": 1,
        },
        component_layout_embodiment_tag="robocasa_panda_omron",
        component_lora_rank=4,
        component_lora_alpha=4.0,
        tune_projector=True,
        tune_diffusion_model=False,
        tune_vlln=False,
    )


def test_lora_zero_init_is_identity_residual():
    lora = CategorySpecificLoRA(2, 32, 6, rank=4, alpha=4.0)
    x = torch.randn(3, 5, 32)
    cat = torch.zeros(3, dtype=torch.long)
    out = lora(x, cat)
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)


def test_decode_matches_flat_decoder_at_init():
    Head = build_component_factored_action_head(_StubActionHead)
    config = _minimal_config()
    head = Head(config)
    hidden = torch.randn(2, 8, head.hidden_size)
    cat = torch.zeros(2, dtype=torch.long)
    flat = head.action_decoder(hidden, cat)
    factored = head.decode_action_hidden(hidden, cat)
    assert torch.allclose(flat, factored, atol=1e-5)


def test_inactive_segment_lora_does_not_change_output():
    Head = build_component_factored_action_head(_StubActionHead)
    config = _minimal_config()
    head = Head(config)
    hidden = torch.randn(2, 8, head.hidden_size)
    cat = torch.zeros(2, dtype=torch.long)

    for adapter in head.component_lora_adapters.values():
        adapter.lora_b.W.data.fill_(0.01)
    for adapter in head.extra_lora_adapters.values():
        adapter.lora_b.W.data.fill_(0.01)

    action_mask = torch.zeros(2, 8, head.action_dim)

    full = head.decode_action_hidden(hidden, cat, action_mask=action_mask)
    flat = head.action_decoder(hidden, cat)
    assert torch.allclose(full, flat, atol=1e-5)
