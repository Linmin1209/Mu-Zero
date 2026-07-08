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

import json
import logging
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoModel, AutoProcessor

from gr00t.configs.base_config import Config
from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
from gr00t.data.dataset.factory import DatasetFactory
from gr00t.experiment.dist_utils import run_or_wait_on_rank0
from gr00t.model.base.model_pipeline import ModelPipeline
from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7
from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor
from gr00t.model.modules.qwen3_motion import (
    ensure_motion_gate,
    finalize_motion_trainable,
    is_motion_missing_key,
    motion_config_from_model_config,
    resolve_motion_text_hidden_size,
)
from gr00t.model.registry import register_model


def _is_adaptive_action_head_key(key: str) -> bool:
    if not key.startswith("action_head."):
        return False
    if _is_component_factored_decoder_key(key):
        return False
    return (
        ".msat" in key
        or key.startswith("action_head.component_projectors")
        or key.startswith("action_head.component_inverse_projectors")
        or key.startswith("action_head.component_type_embed")
        or "msat_decode" in key
    )


def _is_component_factored_decoder_key(key: str) -> bool:
    return key.startswith(
        (
            "action_head.component_decoders",
            "action_head.extra_decoders",
            "action_head.component_lora_adapters",
            "action_head.extra_lora_adapters",
        )
    )


def _is_vt_closed_loop_key(key: str) -> bool:
    return key.startswith("action_head.vt_policy.") or key.startswith(
        "action_head.flux_teacher."
    )


def _is_visor_key(key: str) -> bool:
    return key.startswith("action_head.visor.") or key.startswith(
        "action_head.inpaint_expert."
    )


def _is_flat_action_decoder_key(key: str) -> bool:
    return key.startswith("action_head.action_decoder")


def _load_pretrained_state_dict(checkpoint_path: str) -> dict[str, torch.Tensor]:
    import json

    from safetensors.torch import load_file

    path = Path(checkpoint_path)
    index_file = path / "model.safetensors.index.json"
    if index_file.exists():
        index = json.loads(index_file.read_text())
        state: dict[str, torch.Tensor] = {}
        for shard in sorted(set(index["weight_map"].values())):
            state.update(load_file(str(path / shard)))
        return state
    single = path / "model.safetensors"
    if single.exists():
        return load_file(str(single))
    raise FileNotFoundError(f"No safetensors checkpoint found under {checkpoint_path}")


def _init_component_factored_decoders(model, checkpoint_path: str) -> None:
    action_head = getattr(model, "action_head", None)
    if action_head is None or not hasattr(
        action_head, "load_flat_decoder_into_component_decoders"
    ):
        return
    ckpt_state = _load_pretrained_state_dict(checkpoint_path)
    flat_state = {
        key.removeprefix("action_head.action_decoder."): value
        for key, value in ckpt_state.items()
        if key.startswith("action_head.action_decoder.")
    }
    action_head.load_flat_decoder_into_component_decoders(flat_state)


def _is_legacy_action_head_key(key: str) -> bool:
    if not key.startswith("action_head.") or _is_adaptive_action_head_key(key):
        return False
    shared_prefixes = (
        "action_head.state_encoder",
        "action_head.vlln",
        "action_head.vl_self_attention",
    )
    if any(key.startswith(prefix) for prefix in shared_prefixes):
        return False
    return True


# Convert tensors to lists for JSON serialization
def convert_tensors_to_lists(obj):
    """Recursively convert tensors to lists in nested dictionaries/lists."""
    if torch.is_tensor(obj) or isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_tensors_to_lists(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_tensors_to_lists(item) for item in obj]
    else:
        return obj


class Gr00tN1d7Pipeline(ModelPipeline):
    model_class = Gr00tN1d7
    processor_class = Gr00tN1d7Processor

    def __init__(self, config: Config, save_cfg_dir: Path):
        super().__init__(config)
        self.save_cfg_dir = save_cfg_dir

        # Build transformers loading kwargs from training config
        transformers_loading_kwargs = {
            "trust_remote_code": self.config.training.transformers_trust_remote_code,
            "local_files_only": self.config.training.transformers_local_files_only,
        }
        if self.model_config.model_revision is not None:
            transformers_loading_kwargs["revision"] = self.model_config.model_revision
        if self.config.training.transformers_cache_dir is not None:
            transformers_loading_kwargs["cache_dir"] = self.config.training.transformers_cache_dir
        if self.config.training.transformers_access_token is not None:
            transformers_loading_kwargs["token"] = self.config.training.transformers_access_token

        self.transformers_loading_kwargs = transformers_loading_kwargs

    @property
    def model_config(self):
        return self.config.model

    def setup(self):
        self.model = self._create_model()
        self.train_dataset, self.eval_dataset = self._create_dataset(self.save_cfg_dir)
        self.data_collator = self._create_collator()

    def _create_model(self):
        """Setup model with proper vocabulary expansion."""
        skip_weight_loading = getattr(self.config.training, "skip_weight_loading", False)
        if self.config.training.start_from_checkpoint is not None and not skip_weight_loading:
            checkpoint = self.config.training.start_from_checkpoint
            model_cfg = AutoConfig.from_pretrained(
                checkpoint,
                trust_remote_code=self.config.training.transformers_trust_remote_code,
                local_files_only=self.config.training.transformers_local_files_only,
            )
            # Checkpoint config.json stores hub id; override with local Cosmos path.
            model_cfg.model_name = self.model_config.model_name
            model_cfg.tune_llm = self.config.model.tune_llm
            model_cfg.tune_visual = self.config.model.tune_visual
            model_cfg.tune_projector = self.config.model.tune_projector
            model_cfg.tune_diffusion_model = self.config.model.tune_diffusion_model
            model_cfg.tune_vlln = self.config.model.tune_vlln
            model_cfg.state_dropout_prob = self.config.model.state_dropout_prob
            model_cfg.backbone_trainable_params_fp32 = (
                self.config.model.backbone_trainable_params_fp32
            )
            model_cfg.load_bf16 = self.config.model.load_bf16
            model_cfg.use_flash_attention = getattr(
                self.config.model, "use_flash_attention", True
            )
            for motion_field in (
                "use_motion",
                "motion_insert_layer",
                "motion_injection_point",
                "motion_d_hid",
                "motion_window",
                "motion_ext_chnls",
                "motion_int_chnls",
                "motion_corr_func",
                "motion_n_encoders",
                "motion_use_layerscale",
                "motion_layerscale_init",
                "motion_use_layernorm",
                "motion_use_syncbn",
                "motion_gradient_check",
                "motion_int_mode",
                "tune_motion",
                "motion_use_gating",
                "motion_gate_hidden",
                "motion_gate_init_bias",
                "motion_gate_mode",
                "motion_gate_g_min",
                "motion_gate_g_max",
                "motion_gate_lr_scale",
            ):
                if hasattr(self.config.model, motion_field):
                    setattr(model_cfg, motion_field, getattr(self.config.model, motion_field))
            for vt_field in (
                "use_vt_closed_loop",
                "decouple_base_arm",
                "vt_tactile_dim",
                "vt_tactile_history_len",
                "vt_tactile_num_tokens",
                "vt_tactile_use_pressure_map",
                "vt_num_intent_tokens",
                "vt_num_intent_phases",
                "vt_num_route_modes",
                "vt_num_global_intent_tokens",
                "vt_num_motion_intent_tokens",
                "vt_num_contact_intent_tokens",
                "vt_num_recovery_intent_tokens",
                "vt_tactile_refiner_layers",
                "vt_tactile_refiner_heads",
                "vt_enable_intent_adapter",
                "vt_enable_tactile_encoder",
                "vt_enable_contact_gate",
                "vt_enable_structured_action_dit",
                "vt_enable_future_head",
                "vt_num_future_tokens",
                "vt_action_decoder_use_future_tokens",
                "vt_detach_future_tokens",
                "vt_detach_vlm_for_refiner",
                "vt_action_clamp",
                "vt_closed_loop_stage",
                "vt_enable_tactile_refiner",
                "vt_enable_execution_monitor",
                "vt_enable_recovery_expert",
                "vt_use_flux_teacher",
                "vt_build_flux_batch",
                "vt_flux_model_path",
                "vt_flux_future_delta",
                "vt_flux_resolution",
                "vt_flux_mask_mode",
                "vt_loss_action",
                "vt_loss_future",
                "vt_loss_flux",
                "vt_loss_contact",
                "vt_loss_recovery",
                "vt_loss_router",
                "vt_loss_intent_diversity",
            ):
                if hasattr(self.config.model, vt_field):
                    setattr(model_cfg, vt_field, getattr(self.config.model, vt_field))
            for adaptive_field in (
                "use_adaptive_component_head",
                "use_component_factored_head",
                "use_visor",
                "visor_flow_tau_split",
                "visor_history_vq_tokens",
                "visor_vq_codebook_size",
                "visor_vq_hidden_dim",
                "visor_vq_commit_weight",
                "visor_use_contact_rate_prior",
                "visor_use_semantic_gate",
                "visor_use_split_action_gates",
                "visor_arm_action_slice",
                "visor_base_action_slice",
                "visor_hand_action_slice",
                "visor_arm_action_dim",
                "visor_base_action_dim",
                "visor_hand_action_dim",
                "visor_tactile_num_force",
                "visor_tactile_num_contact",
                "visor_gate_components",
                "visor_tactile_warmup_steps",
                "visor_aux_warmup_steps",
                "visor_aux_delay_steps",
                "visor_gate_mode",
                "visor_tactile_align_mode",
                "visor_use_readout_fed_gates",
                "visor_use_visual_supervision",
                "visor_visual_waypoints",
                "visor_visual_dim",
                "visor_loss_weight_visual",
                "visor_visual_vq_tokens",
                "visor_visual_gt_level",
                "visor_detach_tactile_for_gate",
                "visor_iht_tokens",
                "visor_hidden_dim",
                "visor_loss_weight_tactile",
                "visor_contact_loss_weight",
                "visor_use_tactile_supervision",
                "tune_visor",
                "component_projector_dims",
                "component_loss_weights",
                "component_msat_cfg",
                "component_action_key_order",
                "component_action_key_dims",
                "component_layout_embodiment_tag",
            ):
                if hasattr(self.config.model, adaptive_field):
                    setattr(model_cfg, adaptive_field, getattr(self.config.model, adaptive_field))
            for joint_field in (
                "use_joint_dual_branch",
                "decouple_base_arm",
                "mot_inpaint_tokens",
                "joint_train_mode",
                "joint_flux_model_path",
                "joint_flux_future_delta",
                "joint_flux_resolution",
                "joint_flux_mask_mode",
                "joint_flux_logit_mean",
                "joint_flux_logit_std",
                "joint_phase1_ratio",
                "joint_alpha_phase1",
                "joint_beta_phase1",
                "joint_alpha_phase2",
                "joint_beta_phase2",
                "joint_visor_aux_delay_steps",
                "joint_visor_visual_weight_phase1",
                "joint_visor_visual_weight_phase2",
                "joint_visor_tactile_weight_phase1",
                "joint_visor_tactile_weight_phase2",
                "max_steps",
            ):
                if hasattr(self.config.model, joint_field):
                    setattr(model_cfg, joint_field, getattr(self.config.model, joint_field))
            model_cfg.max_steps = self.config.training.max_steps
            model, loading_info = AutoModel.from_pretrained(
                checkpoint,
                config=model_cfg,
                transformers_loading_kwargs=self.transformers_loading_kwargs,
                output_loading_info=True,
                **self.transformers_loading_kwargs,
            )

            missing_keys = loading_info.get("missing_keys", [])
            mask_token_missing = any("mask_token" in key for key in missing_keys)
            if mask_token_missing and hasattr(model.action_head, "mask_token"):
                if model.action_head.mask_token is not None:
                    with torch.no_grad():
                        model.action_head.mask_token.data.copy_(
                            0.02 * torch.randn_like(model.action_head.mask_token)
                        )
                    logging.info("mask_token not in checkpoint - initialized")

            unexpected_keys = loading_info.get("unexpected_keys", [])
            mismatched_keys = loading_info.get("mismatched_keys", [])
            use_adaptive = getattr(model.config, "use_adaptive_component_head", False)
            use_factored = getattr(model.config, "use_component_factored_head", False)
            use_visor = getattr(model.config, "use_visor", False)
            use_vt = getattr(model.config, "use_vt_closed_loop", False)
            other_missing = [
                k
                for k in missing_keys
                if "mask_token" not in k
                and not is_motion_missing_key(k)
                and not (use_adaptive and _is_adaptive_action_head_key(k))
                and not (use_factored and _is_component_factored_decoder_key(k))
                and not (use_visor and _is_visor_key(k))
                and not (use_vt and _is_vt_closed_loop_key(k))
            ]
            if use_adaptive:
                unexpected_keys = [k for k in unexpected_keys if not _is_legacy_action_head_key(k)]
            elif use_factored:
                unexpected_keys = [
                    k
                    for k in unexpected_keys
                    if not _is_adaptive_action_head_key(k)
                    and not _is_flat_action_decoder_key(k)
                ]
            elif use_visor and not use_factored:
                unexpected_keys = [
                    k
                    for k in unexpected_keys
                    if not _is_adaptive_action_head_key(k)
                    and not _is_component_factored_decoder_key(k)
                ]
            elif getattr(model.config, "use_adaptive_component_head", False) is False:
                unexpected_keys = [
                    k for k in unexpected_keys if not _is_adaptive_action_head_key(k)
                ]
            if use_factored:
                _init_component_factored_decoders(
                    model, self.config.training.start_from_checkpoint
                )
            if getattr(model.config, "use_motion", False):
                visual = model.backbone.model.visual
                motion_block = getattr(visual, "motion_block", None)
                motion_cfg = motion_config_from_model_config(model.config)
                text_hidden_size = resolve_motion_text_hidden_size(model.backbone.model)
                if motion_block is None:
                    from gr00t.model.modules.qwen3_motion import install_motion_module

                    install_motion_module(
                        visual,
                        motion_cfg,
                        text_hidden_size=text_hidden_size,
                    )
                    model.backbone._convert_motion_bn_to_float()
                    visual._gr00t_tune_motion_only = (
                        getattr(model.config, "tune_motion", True)
                        and not getattr(model.config, "tune_visual", False)
                    )
                    logging.info(
                        "Installed MotionModule after checkpoint load (weights initialized)"
                    )
                ensure_motion_gate(
                    visual, motion_cfg, text_hidden_size=text_hidden_size
                )
                finalize_motion_trainable(
                    model.backbone,
                    tune_visual=getattr(model.config, "tune_visual", False),
                    tune_motion=getattr(model.config, "tune_motion", True),
                    trainable_fp32=getattr(
                        model.config, "backbone_trainable_params_fp32", True
                    ),
                )
            errors = []
            if other_missing:
                errors.append(f"Missing keys ({len(other_missing)}): {other_missing}")
            if unexpected_keys:
                errors.append(f"Unexpected keys ({len(unexpected_keys)}): {unexpected_keys}")
            if mismatched_keys:
                errors.append(f"Mismatched keys ({len(mismatched_keys)}): {mismatched_keys}")
            if errors:
                raise RuntimeError(
                    "Checkpoint weight mismatch for "
                    f"{self.config.training.start_from_checkpoint}:\n" + "\n".join(errors)
                )

        else:
            model = self.model_class(
                self.config.model,
                transformers_loading_kwargs=self.transformers_loading_kwargs,
            )
            if getattr(self.config.model, "use_joint_dual_branch", False):
                model.config.max_steps = self.config.training.max_steps

        logging.debug(f"Model Config: {model.config}")
        with run_or_wait_on_rank0(label="final_model_config.json write") as is_rank0:
            if is_rank0:
                with open(self.save_cfg_dir / "final_model_config.json", "w") as f:
                    f.write(model.config.to_filtered_json())
        # Print parameter statistics
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(f"Total parameters: {total_params:,}")
        logging.info(
            f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)"
        )
        logging.debug(f"Model architecture: {model}")

        return model

    def _get_statistics(
        self,
    ) -> dict[str, dict[str, dict[str, dict[str, list[float]]]]] | None:
        return None

    def _get_embodiment_id_mapping(self) -> dict[str, int]:
        return None

    def _joint_processor_kwargs(self) -> dict:
        mc = self.model_config
        kwargs: dict = {}
        if getattr(mc, "use_joint_dual_branch", False):
            kwargs.update(
                {
                    "joint_dual_branch": True,
                    "joint_flux_future_delta": mc.joint_flux_future_delta,
                    "joint_flux_resolution": mc.joint_flux_resolution,
                    "joint_flux_mask_mode": mc.joint_flux_mask_mode,
                    "joint_build_flux_batch": getattr(mc, "joint_train_mode", "simultaneous").lower()
                    not in ("gr00t_only", "visor_only"),
                    "decouple_base_arm": bool(getattr(mc, "decouple_base_arm", False)),
                    "visor_base_action_slice": getattr(mc, "visor_base_action_slice", (7, 11)),
                }
            )
        elif getattr(mc, "use_vt_closed_loop", False):
            kwargs["decouple_base_arm"] = bool(getattr(mc, "decouple_base_arm", False))
            kwargs["visor_base_action_slice"] = getattr(mc, "visor_base_action_slice", (7, 11))
            if getattr(mc, "vt_build_flux_batch", False) or getattr(mc, "vt_use_flux_teacher", False):
                kwargs["vt_build_flux_batch"] = True
                kwargs["joint_build_flux_batch"] = True
                kwargs["joint_flux_future_delta"] = int(getattr(mc, "vt_flux_future_delta", 5))
                kwargs["joint_flux_resolution"] = int(getattr(mc, "vt_flux_resolution", 256))
                kwargs["joint_flux_mask_mode"] = str(getattr(mc, "vt_flux_mask_mode", "keep_reference"))
            logging.info(
                "VT processor flags: vt_build_flux_batch=%s joint_build_flux_batch=%s",
                kwargs.get("vt_build_flux_batch", False),
                kwargs.get("joint_build_flux_batch", False),
            )
        return kwargs

    def _create_dataset(self, save_cfg_dir: Path):
        """Create appropriate dataset based on task and mode."""
        letter_box_transform = self.model_config.letter_box_transform
        logging.info("N1.7 letter_box_transform=%s", letter_box_transform)
        if self.config.training.start_from_checkpoint is not None:
            processor = AutoProcessor.from_pretrained(
                self.config.training.start_from_checkpoint,
                # Overrides
                modality_configs=self.config.data.modality_configs,
                use_percentiles=self.model_config.use_percentiles,
                image_crop_size=self.model_config.image_crop_size,
                image_target_size=self.model_config.image_target_size,
                random_rotation_angle=self.model_config.random_rotation_angle,
                color_jitter_params=self.model_config.color_jitter_params,
                model_name=self.model_config.model_name,
                model_type=self.model_config.backbone_model_type,
                formalize_language=self.model_config.formalize_language,
                apply_sincos_state_encoding=self.model_config.apply_sincos_state_encoding,
                max_action_horizon=self.model_config.action_horizon,
                use_albumentations=self.model_config.use_albumentations_transforms,
                extra_augmentation_config=self.model_config.extra_augmentation_config,
                shortest_image_edge=self.model_config.shortest_image_edge,
                crop_fraction=self.model_config.crop_fraction,
                letter_box_transform=letter_box_transform,
                transformers_loading_kwargs=self.transformers_loading_kwargs,
                use_alternate_vl_dit=self.model_config.use_alternate_vl_dit,
                use_relative_action=self.model_config.use_relative_action,
                use_adaptive_component_head=self.model_config.use_adaptive_component_head,
                component_projector_dims=self.model_config.component_projector_dims,
                # State augmentation overrides
                exclude_state=self.model_config.exclude_state,
                state_dropout_prob=self.model_config.state_dropout_prob,
                use_mean_std=self.model_config.use_mean_std,
                **self._joint_processor_kwargs(),
                **self.transformers_loading_kwargs,
            )
        else:
            processor = self.processor_class(
                modality_configs=self.config.data.modality_configs,
                use_percentiles=self.model_config.use_percentiles,
                statistics=self._get_statistics(),  # By default is None, so this will be computed and set later.
                embodiment_id_mapping=self._get_embodiment_id_mapping(),  # By default is None, so this will be set later.
                image_crop_size=self.model_config.image_crop_size,
                image_target_size=self.model_config.image_target_size,
                random_rotation_angle=self.model_config.random_rotation_angle,
                color_jitter_params=self.model_config.color_jitter_params,
                model_name=self.model_config.model_name,
                model_type=self.model_config.backbone_model_type,
                formalize_language=self.model_config.formalize_language,
                max_state_dim=self.model_config.max_state_dim,
                max_action_dim=self.model_config.max_action_dim,
                apply_sincos_state_encoding=self.model_config.apply_sincos_state_encoding,
                max_action_horizon=self.model_config.action_horizon,
                use_albumentations=self.model_config.use_albumentations_transforms,
                extra_augmentation_config=self.model_config.extra_augmentation_config,
                shortest_image_edge=self.model_config.shortest_image_edge,
                crop_fraction=self.model_config.crop_fraction,
                letter_box_transform=letter_box_transform,
                use_relative_action=self.model_config.use_relative_action,
                use_adaptive_component_head=self.model_config.use_adaptive_component_head,
                component_projector_dims=self.model_config.component_projector_dims,
                # State augmentation
                exclude_state=self.model_config.exclude_state,
                state_dropout_prob=self.model_config.state_dropout_prob,
                use_mean_std=self.model_config.use_mean_std,
                visor_visual_gt_level=getattr(
                    self.model_config, "visor_visual_gt_level", "flow"
                ),
                **self._joint_processor_kwargs(),
                transformers_loading_kwargs=self.transformers_loading_kwargs,
            )

        logging.debug(
            f"Processor configs for training: {json.dumps({k: str(v) for k, v in vars(processor).items()}, indent=2)}"
        )
        with run_or_wait_on_rank0(label="final_processor_config.json write") as is_rank0:
            if is_rank0:
                with open(self.save_cfg_dir / "final_processor_config.json", "w") as f:
                    json.dump({k: str(v) for k, v in vars(processor).items()}, f, indent=2)

        self.processor = processor
        if getattr(self.model_config, "use_joint_dual_branch", False):
            if not getattr(processor, "joint_dual_branch", False):
                raise RuntimeError(
                    "use_joint_dual_branch=True but processor.joint_dual_branch=False — "
                    "FLUX batch will never be built. Check Gr00tN1d7Processor.from_pretrained "
                    "override_keys include joint_dual_branch."
                )
            logging.info(
                "Joint processor flags: joint_dual_branch=%s joint_build_flux_batch=%s",
                processor.joint_dual_branch,
                getattr(processor, "joint_build_flux_batch", None),
            )
        dataset_factory = DatasetFactory(config=self.config)
        train_dataset, eval_dataset = dataset_factory.build(processor=self.processor)

        with run_or_wait_on_rank0(label="dataset_statistics.json write") as is_rank0:
            if is_rank0:
                stats = train_dataset.get_dataset_statistics()
                stats_dict = convert_tensors_to_lists(stats)
                with open(save_cfg_dir / "dataset_statistics.json", "w") as f:
                    json.dump(stats_dict, f, indent=2)
                logging.info("Saved dataset statistics for inference")

        return train_dataset, eval_dataset

    def _create_collator(self):
        data_collator = self.processor.collator
        return data_collator


register_model(Gr00tN1d7Config, Gr00tN1d7Pipeline)
