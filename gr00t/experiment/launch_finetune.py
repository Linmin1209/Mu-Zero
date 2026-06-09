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

# Launch finetuning for N1.7 on "single node".
# This script tries to provide a similar user experience as current OSS.
#
# RoboCasa365 (local LeRobot roots under robocasa365-datasets):
#   --robocasa365-root /path/to/robocasa365-datasets
#   --robocasa365-split pretrain|target|all
#   --robocasa365-category atomic|composite|all
#   --robocasa365-tasks TaskA,TaskB   (optional; default = all matching tasks)
#   --embodiment-tag ROBOCASA_PANDA_OMRON
#   --modality-config-path examples/RoboCasa365/robocasa365_config.py
#
# Pre-downloaded weights (no HuggingFace): HDD_POOL/linmin/models/{GR00T-N1.7-3B,Cosmos-Reason2-2B}

import json
import os
from pathlib import Path

# Must run before any other ``gr00t`` import (see gr00t/__init__.py patches).
_MODELS_ROOT = Path(
    os.environ.get(
        "GR00T_MODELS_ROOT",
        "/HOME/sysu_xdliang/sysu_xdliang_1/HDD_POOL/linmin/models",
    )
)
os.environ.setdefault("GR00T_MODELS_ROOT", str(_MODELS_ROOT))
os.environ["GROOT_PATCH_MISTRAL"] = "1"
os.environ["GROOT_HF_LOCAL_FIRST"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.setdefault("HF_HOME", str(_MODELS_ROOT / ".hf_cache"))

import tyro

from gr00t.configs.base_config import get_default_config
from gr00t.configs.finetune_config import FinetuneConfig
from gr00t.experiment.experiment import run
from gr00t.experiment.local_models import resolve_local_paths
from gr00t.experiment.robocasa365_datasets import (
    get_default_robocasa365_modality_config_path,
    resolve_robocasa365_dataset_paths,
)


# Make sure the user provided modality config is registered.
def load_modality_config(modality_config_path: str):
    import importlib
    import sys

    path = Path(modality_config_path)
    if path.exists() and path.suffix == ".py":
        sys.path.append(str(path.parent))
        importlib.import_module(path.stem)
        print(f"Loaded modality config: {path}")
    else:
        raise FileNotFoundError(f"Modality config path does not exist: {modality_config_path}")


if __name__ == "__main__":
    # Set LOGURU_LEVEL environment variable if not already set (default: INFO)
    if "LOGURU_LEVEL" not in os.environ:
        os.environ["LOGURU_LEVEL"] = "INFO"
    # Use tyro for clean CLI
    ft_config = tyro.cli(FinetuneConfig, description=__doc__)
    from gr00t.data.embodiment_tags import EmbodimentTag

    if not ft_config.base_model_path.strip():
        ft_config.base_model_path = str(
            Path(os.environ.get("GR00T_BASE_MODEL_PATH", resolve_local_paths()[0]))
        )
    gr00t_path, cosmos_path = resolve_local_paths(ft_config.base_model_path)
    ft_config.base_model_path = gr00t_path

    ft_config.embodiment_tag = EmbodimentTag.resolve(ft_config.embodiment_tag)
    embodiment_tag = ft_config.embodiment_tag.value

    if ft_config.robocasa365_root:
        if ft_config.modality_config_path is None:
            ft_config.modality_config_path = str(get_default_robocasa365_modality_config_path())
        dataset_paths = resolve_robocasa365_dataset_paths(
            root=ft_config.robocasa365_root,
            split=ft_config.robocasa365_split,
            category=ft_config.robocasa365_category,
            tasks=ft_config.robocasa365_tasks,
        )
        print(
            f"RoboCasa365: {len(dataset_paths)} lerobot dataset(s) "
            f"(split={ft_config.robocasa365_split}, category={ft_config.robocasa365_category})"
        )
        if ft_config.robocasa365_tasks:
            print(f"  tasks filter: {ft_config.robocasa365_tasks}")
        for p in dataset_paths[:5]:
            print(f"  - {p}")
        if len(dataset_paths) > 5:
            print(f"  ... and {len(dataset_paths) - 5} more")
    elif ft_config.dataset_path.strip():
        dataset_paths = [path for path in ft_config.dataset_path.split(os.pathsep) if path]
    else:
        raise ValueError(
            "Provide --dataset-path or --robocasa365-root for finetuning."
        )

    # all rank workers should register for the modality config
    if ft_config.modality_config_path is not None:
        load_modality_config(ft_config.modality_config_path)

    config = get_default_config().load_dict(
        {
            "data": {
                "download_cache": False,
                "datasets": [
                    {
                        "dataset_paths": dataset_paths,
                        "mix_ratio": 1.0,
                        "embodiment_tag": embodiment_tag,
                    }
                ],
            }
        }
    )
    config.load_config_path = None

    # overwrite with finetune config supplied by the user
    config.model.tune_llm = ft_config.tune_llm
    config.model.tune_visual = ft_config.tune_visual
    config.model.tune_projector = ft_config.tune_projector
    config.model.tune_diffusion_model = ft_config.tune_diffusion_model
    config.model.state_dropout_prob = ft_config.state_dropout_prob
    config.model.use_motion = ft_config.use_motion
    config.model.motion_insert_layer = ft_config.motion_insert_layer
    config.model.tune_motion = ft_config.tune_motion
    config.model.use_adaptive_component_head = ft_config.use_adaptive_component_head
    if ft_config.use_adaptive_component_head:
        component_dims = None
        if ft_config.modality_config_path:
            try:
                import importlib.util

                spec = importlib.util.spec_from_file_location(
                    "mod_cfg", ft_config.modality_config_path
                )
                if spec and spec.loader:
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)
                    component_dims = getattr(mod, "ROBOCASA365_COMPONENT_PROJECTOR_DIMS", None)
            except Exception:
                component_dims = None
        config.model.component_projector_dims = component_dims
        print("[i] AdaptiveEmbodimentActionHead enabled (component-level MSAT decoder)")
    if ft_config.use_motion:
        print(
            f"[i] STSS/MOSS enabled at vision layer {ft_config.motion_insert_layer}; "
            f"tune_motion={ft_config.tune_motion}"
        )
    if ft_config.gradient_checkpointing is None:
        config.training.gradient_checkpointing = ft_config.use_motion
    else:
        config.training.gradient_checkpointing = ft_config.gradient_checkpointing
    if config.training.gradient_checkpointing:
        print("[i] gradient_checkpointing enabled (reduces vision/motion activation memory)")
    config.model.random_rotation_angle = ft_config.random_rotation_angle
    config.model.color_jitter_params = ft_config.color_jitter_params
    if ft_config.extra_augmentation_config:
        config.model.extra_augmentation_config = json.loads(ft_config.extra_augmentation_config)
    else:
        config.model.extra_augmentation_config = None

    config.model.load_bf16 = False
    config.model.reproject_vision = False
    config.model.model_name = cosmos_path
    config.training.transformers_local_files_only = True
    config.training.start_from_checkpoint = gr00t_path
    print(f"Local GR00T base: {gr00t_path}")
    print(f"Local Cosmos VLM: {cosmos_path}")
    print("HuggingFace Hub disabled (offline local checkpoints only)")
    config.model.backbone_trainable_params_fp32 = True
    config.model.use_relative_action = True

    config.training.experiment_name = ft_config.experiment_name
    config.training.optim = "adamw_torch"
    config.training.global_batch_size = ft_config.global_batch_size
    config.training.dataloader_num_workers = ft_config.dataloader_num_workers
    config.training.learning_rate = ft_config.learning_rate
    config.training.gradient_accumulation_steps = ft_config.gradient_accumulation_steps
    config.training.output_dir = ft_config.output_dir
    config.training.save_steps = ft_config.save_steps
    config.training.save_total_limit = ft_config.save_total_limit
    config.training.num_gpus = ft_config.num_gpus
    config.training.use_wandb = ft_config.use_wandb
    config.training.max_steps = ft_config.max_steps
    config.training.weight_decay = ft_config.weight_decay
    config.training.warmup_ratio = ft_config.warmup_ratio
    config.training.wandb_project = ft_config.wandb_project

    config.data.shard_size = ft_config.shard_size
    config.data.episode_sampling_rate = ft_config.episode_sampling_rate
    config.data.num_shards_per_epoch = ft_config.num_shards_per_epoch

    from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS

    mod_cfg = MODALITY_CONFIGS.get(embodiment_tag)
    if mod_cfg and "video" in mod_cfg:
        min_vid_delta = min(mod_cfg["video"].delta_indices)
        if min_vid_delta < 0:
            config.data.allow_padding = True
            print(
                f"[i] video delta_indices min={min_vid_delta}; "
                "enabled data.allow_padding for episode start"
            )

    config.training.save_only_model = ft_config.save_only_model
    config.training.skip_weight_loading = ft_config.skip_weight_loading

    run(config)
