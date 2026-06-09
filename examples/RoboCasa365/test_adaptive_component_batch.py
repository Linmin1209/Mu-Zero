#!/usr/bin/env python3
"""Small-batch smoke test for AdaptiveEmbodimentActionHead data pipeline + forward."""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

_MODELS_ROOT = Path(
    os.environ.get(
        "GR00T_MODELS_ROOT",
        "/HOME/sysu_xdliang/sysu_xdliang_1/HDD_POOL/linmin/models",
    )
)
os.environ.setdefault("GR00T_MODELS_ROOT", str(_MODELS_ROOT))
os.environ.setdefault("GROOT_PATCH_MISTRAL", "1")
os.environ.setdefault("GROOT_HF_LOCAL_FIRST", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_modality_config(modality_config_path: Path) -> None:
    sys.path.insert(0, str(modality_config_path.parent))
    importlib.import_module(modality_config_path.stem)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--robocasa365-root",
        default=os.environ.get(
            "ROBOCASA365_ROOT",
            "/HOME/sysu_xdliang/sysu_xdliang_1/HDD_POOL/linmin/datasets/robocasa365-datasets",
        ),
    )
    parser.add_argument("--task", default="PickPlaceToasterToCounter")
    parser.add_argument(
        "--modality-config-path",
        default=str(PROJECT_ROOT / "examples" / "RoboCasa365" / "robocasa365_config.py"),
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--base-model-path",
        default=str(_MODELS_ROOT / "GR00T-N1.7-3B"),
    )
    parser.add_argument("--forward", action="store_true", help="Run one GPU forward+backward")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    modality_path = Path(args.modality_config_path)
    load_modality_config(modality_path)
    mod = importlib.import_module(modality_path.stem)
    component_dims = getattr(mod, "ROBOCASA365_COMPONENT_PROJECTOR_DIMS", None)

    from gr00t.configs.base_config import get_default_config
    from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.data.dataset.factory import DatasetFactory
    from gr00t.experiment.local_models import resolve_local_paths
    from gr00t.experiment.robocasa365_datasets import resolve_robocasa365_dataset_paths
    from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor

    embodiment_tag = EmbodimentTag.ROBOCASA_PANDA_OMRON.value
    dataset_paths = resolve_robocasa365_dataset_paths(
        root=args.robocasa365_root,
        split="pretrain",
        category="atomic",
        tasks=args.task,
    )
    print(f"[1/4] Dataset: {dataset_paths[0]}")

    gr00t_path, cosmos_path = resolve_local_paths(args.base_model_path)

    config = get_default_config().load_dict(
        {
            "data": {
                "download_cache": False,
                "shard_size": 64,
                "num_shards_per_epoch": 1,
                "episode_sampling_rate": 0.2,
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
    config.model.use_adaptive_component_head = True
    config.model.component_projector_dims = component_dims
    config.model.model_name = cosmos_path
    config.model.use_relative_action = True
    config.training.transformers_local_files_only = True
    config.validate()

    processor = Gr00tN1d7Processor(
        modality_configs=MODALITY_CONFIGS,
        model_name=cosmos_path,
        max_state_dim=config.model.max_state_dim,
        max_action_dim=config.model.max_action_dim,
        max_action_horizon=config.model.action_horizon,
        image_crop_size=list(config.model.image_crop_size or (230, 230)),
        image_target_size=list(config.model.image_target_size or (256, 256)),
        use_relative_action=True,
        use_adaptive_component_head=True,
        component_projector_dims=component_dims,
        transformers_loading_kwargs={"trust_remote_code": True, "local_files_only": True},
    )

    factory = DatasetFactory(config=config)
    print("[2/4] Building dataset (first shard load may take ~1 min)...")
    train_dataset, _ = factory.build(processor=processor)

    collator = processor.collator
    samples = []
    for sample in train_dataset:
        samples.append(sample)
        if len(samples) >= args.batch_size:
            break
    assert len(samples) >= args.batch_size

    print(f"[3/4] Collate batch_size={args.batch_size}...")
    batch = collator(samples[: args.batch_size])["inputs"]
    comp = batch["component_actions"]
    mask = batch["active_component_mask"]
    print(f"  active_component_mask:\n{mask}")
    for name, tensor in sorted(comp.items()):
        print(f"  {name}: shape={tuple(tensor.shape)}")
    assert "action" not in batch
    assert set(comp.keys()) == {"right_arm", "right_hand", "base"}
    print("  data pipeline OK")

    if not args.forward:
        print("[4/4] Skipped model forward (use --forward to enable)")
        print("=== DATA CHECKS PASSED ===")
        return

    import torch
    from transformers import AutoConfig, AutoModel

    if not torch.cuda.is_available():
        print("[4/4] No CUDA; skipping forward")
        return

    device = torch.device(args.device)
    print(f"[4/4] Forward+backward on {device}...")
    model_cfg = AutoConfig.from_pretrained(
        gr00t_path, trust_remote_code=True, local_files_only=True
    )
    model_cfg.model_name = cosmos_path
    model_cfg.use_adaptive_component_head = True
    model_cfg.component_projector_dims = component_dims
    model_cfg.use_relative_action = True
    model_cfg.tune_llm = False
    model_cfg.tune_visual = False

    model = AutoModel.from_pretrained(
        gr00t_path,
        config=model_cfg,
        trust_remote_code=True,
        local_files_only=True,
        ignore_mismatched_sizes=True,
    )
    model = model.to(device=device, dtype=torch.bfloat16)
    model.train()

    inputs = dict(batch)
    for key, val in list(inputs.items()):
        if torch.is_tensor(val):
            inputs[key] = val.to(device)
        elif isinstance(val, dict):
            inputs[key] = {k: v.to(device) for k, v in val.items()}

    out = model(inputs=inputs)
    loss = out["loss"]
    print(f"  loss={loss.item():.6f}")
    loss.backward()
    print("  backward OK")
    print("=== ALL CHECKS PASSED ===")


if __name__ == "__main__":
    main()
