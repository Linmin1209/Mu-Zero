# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import MISSING, asdict, dataclass, field, is_dataclass
from enum import Enum
import json
from pathlib import Path

import torch
from transformers import PretrainedConfig

from . import register_model_config


@dataclass
class Gr00tN1d7Config(PretrainedConfig):
    """Unified configuration for Gr00tN1d7 model with backbone and action head.

    Gr00tN1d7 uses the Cosmos-Reason2-2B (Qwen3-VL architecture) VLM backbone,
    replacing the Eagle backbone used in Gr00tN1d6.
    """

    # Model identification
    model_type: str = "Gr00tN1d7"
    model_dtype: str = "bfloat16"  # Use bfloat16 for Flash Attention compatibility

    # Backbone configuration
    model_name: str = "nvidia/Cosmos-Reason2-2B"
    backbone_model_type: str = "qwen"
    model_revision: str | None = None
    tune_top_llm_layers: int = 0  # Number of top LLM layers to tune
    backbone_embedding_dim: int = 2048  # project_to_dim; must match Cosmos-Reason2-2B hidden size
    tune_llm: bool = False
    tune_visual: bool = False
    # STSS/MOSS motion module (ported from RLDX-1; requires multi-frame video input)
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
    motion_gate_init_bias: float = 0.0
    motion_gate_mode: str = "text_only"
    motion_gate_g_min: float = 0.0
    motion_gate_g_max: float = 0.8
    motion_gate_lr_scale: float = 5.0
    select_layer: int = 12
    reproject_vision: bool = False
    use_flash_attention: bool = True
    load_bf16: bool = False  # Enable BF16 loading
    backbone_trainable_params_fp32: bool = True

    ### Processing parameters
    image_crop_size: tuple[int, int] | None = (230, 230)
    image_target_size: tuple[int, int] | None = (256, 256)

    shortest_image_edge: int | None = None
    crop_fraction: float | None = None

    random_rotation_angle: int | None = None
    color_jitter_params: dict[str, float] | None = None
    use_albumentations_transforms: bool = True
    letter_box_transform: bool = False
    # Extra augmentation config (mask-based and others).
    extra_augmentation_config: dict | None = None
    formalize_language: bool = True
    apply_sincos_state_encoding: bool = (
        False  # Global flag to enable per-embodiment sin/cos encoding
    )
    use_percentiles: bool = True
    use_relative_action: bool = False

    # Action head configuration parameters
    max_state_dim: int = 132  # Default from state_shape
    max_action_dim: int = 132  # Default from action_shape
    action_horizon: int = 40
    hidden_size: int = 1024
    input_embedding_dim: int = 1536

    # State history: number of consecutive state timesteps fed to the state encoder
    state_history_length: int = 1

    # Global parameters
    add_pos_embed: bool = True
    attn_dropout: float = 0.2
    use_vlln: bool = True
    max_seq_len: int = 1024
    use_alternate_vl_dit: bool = True  # True for AlternateVLDiT, False for DiT
    attend_text_every_n_blocks: int = 2

    diffusion_model_cfg: dict = field(
        default_factory=lambda: {
            "positional_embeddings": None,
            "num_layers": 16,
            "num_attention_heads": 32,
            "attention_head_dim": 48,
            "norm_type": "ada_norm",
            "dropout": 0.2,
            "final_dropout": True,
            "output_dim": 1024,
            "interleave_self_attention": True,
        }
    )

    # Flow matching parameters
    num_inference_timesteps: int = 4
    noise_beta_alpha: float = 1.5
    noise_beta_beta: float = 1.0
    noise_s: float = 0.999
    num_timestep_buckets: int = 1000

    # Training parameters
    tune_projector: bool = True
    tune_diffusion_model: bool = True
    tune_vlln: bool = True

    # State augmentation parameters
    state_dropout_prob: float = 0.8  # State dropout probability
    exclude_state: bool = False  # Zero out all state inputs (ablation)
    use_mean_std: bool = False  # Use mean/std normalization instead of min/max

    # Multi-embodiment parameters
    max_num_embodiments: int = 32

    # Adaptive component-level action head (zero-padding-free MSAT decoder)
    use_adaptive_component_head: bool = False
    component_projector_dims: dict[str, int] | None = None
    component_loss_weights: dict[str, float] | None = None
    component_msat_cfg: dict | None = None

    # Native DiT + per-component CategorySpecificMLP decoders (pretrained DiT load)
    use_component_factored_head: bool = False
    component_action_key_order: list[str] | None = None
    component_action_key_dims: dict[str, int] | None = None
    component_layout_embodiment_tag: str | None = None

    # VISOR: T-Rex-style sensor tactile (flat head default; component_factored optional)
    use_vt_closed_loop: bool = False
    """VT closed-loop action stack (replaces VISOR tactile decoder; MOSS unchanged)."""

    vt_tactile_dim: int = 3
    vt_tactile_history_len: int = 16
    vt_tactile_num_tokens: int = 4
    vt_tactile_use_pressure_map: bool = False
    vt_num_intent_tokens: int = 16
    vt_num_intent_phases: int = 8
    vt_num_route_modes: int = 16
    vt_num_global_intent_tokens: int = 4
    vt_num_motion_intent_tokens: int = 8
    vt_num_contact_intent_tokens: int = 4
    vt_num_recovery_intent_tokens: int = 2
    vt_tactile_refiner_layers: int = 2
    vt_tactile_refiner_heads: int = 8
    vt_enable_intent_adapter: bool = True
    vt_enable_tactile_encoder: bool = True
    vt_enable_contact_gate: bool = True
    vt_enable_structured_action_dit: bool = True
    vt_enable_future_head: bool = False
    vt_num_future_tokens: int = 8
    vt_action_decoder_use_future_tokens: bool = False
    vt_detach_future_tokens: bool = True
    vt_detach_vlm_for_refiner: bool = True
    vt_action_clamp: float = 1.0
    vt_closed_loop_stage: int = 1
    vt_enable_tactile_refiner: bool = True
    vt_enable_execution_monitor: bool = False
    vt_enable_recovery_expert: bool = False
    vt_use_flux_teacher: bool = False
    vt_build_flux_batch: bool = False
    vt_flux_model_path: str = (
        "/HOME/sysu_xdliang/sysu_xdliang_1/HDD_POOL/linmin/models/FLUX.1-Fill-dev"
    )
    vt_flux_future_delta: int = 5
    vt_flux_resolution: int = 256
    vt_flux_mask_mode: str = "keep_reference"
    vt_loss_action: float = 1.0
    vt_loss_future: float = 0.1
    vt_loss_flux: float = 0.05
    vt_loss_contact: float = 0.2
    vt_loss_recovery: float = 0.01
    vt_loss_router: float = 0.05
    vt_loss_intent_diversity: float = 0.01

    use_visor: bool = False
    visor_flow_tau_split: float = 0.4
    """Flow-time split (GR00T: 0=noise, 1=clean). VISOR refines only when t >= split."""
    visor_history_vq_tokens: int = 2
    visor_vq_codebook_size: int = 64
    visor_vq_hidden_dim: int = 64
    visor_vq_commit_weight: float = 0.1
    visor_use_contact_rate_prior: bool = True
    visor_use_semantic_gate: bool = True
    visor_gate_components: tuple[str, ...] = ("right_hand",)
    visor_use_split_action_gates: bool = True
    """Flat VISOR: separate tactile gates for arm (EEF) and gripper action dims."""
    visor_arm_action_slice: tuple[int, int] = (1, 7)
    """Flat action indices [start, end) for EEF pos+rot (RoboCasa365 layout)."""
    visor_base_action_slice: tuple[int, int] = (7, 11)
    """Flat action indices [start, end) for base_motion."""
    visor_hand_action_slice: tuple[int, int] = (0, 1)
    """Flat action indices [start, end) for gripper_close."""
    visor_arm_action_dim: int = 6
    visor_base_action_dim: int = 4
    visor_hand_action_dim: int = 1
    visor_tactile_num_force: int = 2
    """Number of force channels in stacked tactile (RoboCasa: 2 pads; DexJoCo single: 4 fingers)."""
    visor_tactile_num_contact: int = 1
    """Number of contact channels in stacked tactile (DexJoCo: one per finger)."""
    visor_tactile_warmup_steps: int = 2000
    visor_aux_warmup_steps: int = 2000
    visor_aux_delay_steps: int = 500
    visor_gate_mode: str = "tactile_hand_only"
    visor_tactile_align_mode: str = "hold_last"
    visor_use_readout_fed_gates: bool = False
    visor_use_visual_supervision: bool = False
    visor_visual_waypoints: int = 8
    visor_visual_dim: int = 2
    visor_loss_weight_visual: float = 0.03
    visor_visual_vq_tokens: int = 1
    visor_visual_gt_level: str = "flow"
    visor_detach_tactile_for_gate: bool = True
    visor_iht_tokens: int = 2
    """Legacy; tactile IHT count is history_vq_tokens + 1 (instant only)."""
    visor_hidden_dim: int = 256
    visor_loss_weight_tactile: float = 0.1
    visor_contact_loss_weight: float = 1.0
    visor_use_tactile_supervision: bool = True
    tune_visor: bool = True
    decouple_base_arm: bool = True
    """When True, suppress base_motion flow loss, nav visual aux, and base gates."""

    # MoT joint training (shared DiT inpaint + action)
    use_joint_dual_branch: bool = False
    mot_inpaint_tokens: int = 4
    """Pooled VAE latent tokens per anchor/future image (must be a perfect square)."""
    joint_train_mode: str = "simultaneous"
    joint_flux_model_path: str = (
        "/HOME/sysu_xdliang/sysu_xdliang_1/HDD_POOL/linmin/models/FLUX.1-Fill-dev"
    )
    joint_flux_future_delta: int = 5
    joint_flux_resolution: int = 256
    joint_flux_mask_mode: str = "keep_reference"
    joint_flux_logit_mean: float = 0.0
    joint_flux_logit_std: float = 1.0
    joint_phase1_ratio: float = 0.2
    joint_alpha_phase1: float = 1.0
    joint_beta_phase1: float = 0.1
    joint_alpha_phase2: float = 0.2
    joint_beta_phase2: float = 2.0
    joint_visor_aux_delay_steps: int | None = None
    joint_visor_visual_weight_phase1: float = 1.0
    joint_visor_visual_weight_phase2: float = 0.15
    joint_visor_tactile_weight_phase1: float = 0.01
    joint_visor_tactile_weight_phase2: float = 0.02
    max_steps: int = 30000

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)

        # Ensures that all dataclass defaults (including those using default_factory)
        # are explicitly assigned to the instance, even if dataclasses initialization or subclassing
        # (PretrainedConfig) interferes with normal default injection.
        for f in self.__dataclass_fields__.values():
            if not hasattr(self, f.name):
                if f.default is not MISSING:
                    setattr(self, f.name, f.default)
                elif getattr(f, "default_factory", MISSING) is not MISSING:
                    setattr(self, f.name, f.default_factory())

    def to_filtered_dict(self, exclude_augment: bool = True) -> dict:
        """Return a dictionary representation of this config, optionally excluding augmentation keys."""
        if is_dataclass(self):
            cfg = asdict(self)
        else:
            cfg = dict(self.__dict__)

        if exclude_augment:
            exclude_keys = {
                "random_rotation_angle",
                "color_jitter_params",
                "use_albumentations_transforms",
                "formalize_language",
                "image_crop_size",
                "image_target_size",
                "shortest_image_edge",
                "crop_fraction",
            }
            cfg = {k: v for k, v in cfg.items() if k not in exclude_keys}

        return cfg

    def to_filtered_json(self, exclude_augment: bool = True, **kwargs) -> str:
        """Return a JSON string of this config, optionally excluding augmentation keys."""

        def default(o):
            if isinstance(o, (Path, torch.dtype, torch.device)):
                return str(o)
            if isinstance(o, Enum):
                return o.value
            return str(o)

        return json.dumps(
            self.to_filtered_dict(exclude_augment),
            indent=2,
            default=default,
            **kwargs,
        )


register_model_config("Gr00tN1d7", Gr00tN1d7Config)
