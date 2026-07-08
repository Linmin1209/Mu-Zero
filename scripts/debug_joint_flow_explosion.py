#!/usr/bin/env python3
"""Debug VISOR flow_loss stability: 20-step forward/backward health check."""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
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

import torch

MOD_PATH = PROJECT / "examples/RoboCasa365/robocasa365_config_4frame.py"
sys.path.insert(0, str(MOD_PATH.parent))
importlib.import_module(MOD_PATH.stem)


def build_config(*, joint: bool):
    from gr00t.configs.base_config import get_default_config
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.experiment.robocasa365_datasets import resolve_robocasa365_dataset_paths

    dataset_paths = resolve_robocasa365_dataset_paths(
        root="/HOME/sysu_xdliang/sysu_xdliang_1/HDD_POOL/linmin/datasets/robocasa365-datasets",
        split="pretrain",
        category="atomic",
        tasks="PickPlaceToasterOvenToCounter",
    )

    config = get_default_config().load_dict(
        {
            "data": {
                "download_cache": False,
                "datasets": [
                    {
                        "dataset_paths": dataset_paths,
                        "mix_ratio": 1.0,
                        "embodiment_tag": EmbodimentTag.ROBOCASA_PANDA_OMRON.value,
                    }
                ],
            }
        }
    )
    config.model.use_visor = True
    config.model.use_joint_dual_branch = joint
    config.model.use_component_factored_head = False
    config.model.visor_gate_mode = "visual_manip_nav_tactile_hand"
    config.model.visor_use_visual_supervision = True
    config.model.visor_use_readout_fed_gates = True
    config.model.visor_visual_gt_level = "flow"
    config.model.visor_visual_dim = 2
    config.model.visor_aux_delay_steps = 6000
    config.model.visor_loss_weight_tactile = 0.01
    config.model.visor_loss_weight_visual = 0.15
    config.model.load_bf16 = True
    config.model.max_steps = 30
    config.training.max_steps = 30
    config.training.start_from_checkpoint = str(_MODELS_ROOT / "GR00T-N1.7-3B")
    config.training.transformers_local_files_only = True
    config.model.model_name = str(_MODELS_ROOT / "Cosmos-Reason2-2B")
    config.data.allow_padding = True
    config.data.shard_size = 512
    config.data.episode_sampling_rate = 1.0
    config.data.num_shards_per_epoch = 1
    return config


def run_steps(mode: str, *, steps: int = 20, backward: bool = True) -> int:
    from gr00t.configs.joint_finetune_config import joint_finetune_config_from_run
    from gr00t.experiment.joint_train.joint_model import JointDualBranchModel
    from gr00t.model import MODEL_REGISTRY

    joint = mode == "joint"
    config = build_config(joint=joint)
    save_cfg = PROJECT / f"output/debug_flow_{mode}/experiment_cfg"
    save_cfg.mkdir(parents=True, exist_ok=True)
    pipeline = MODEL_REGISTRY.get(type(config.model))(config, save_cfg)
    pipeline.setup()
    model = pipeline.return_model()
    if joint:
        model = JointDualBranchModel(model, joint_finetune_config_from_run(config))
    model = model.cuda().train()
    collator = pipeline.return_collator()
    train_dataset = pipeline.return_dataset()[0]

    params = model.gr00t.parameters() if joint else model.parameters()
    opt = None
    if backward:
        opt = torch.optim.AdamW(
            [p for p in params if p.requires_grad],
            lr=1e-4,
            weight_decay=1e-5,
        )

    bad_flow = 0
    bad_grad = 0
    it = iter(train_dataset)
    print(f"=== mode={mode} backward={backward} steps={steps} ===")
    for step in range(steps):
        if joint:
            model.set_global_step(step)
        batch = collator([next(it)])
        inputs = batch["inputs"]
        inputs = {k: v.cuda() if torch.is_tensor(v) else v for k, v in inputs.items()}

        if opt is not None:
            opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            out = model(inputs) if not joint else model(inputs)
            loss = out["loss"]
        flow = float(out.get("flow_loss", torch.tensor(float("nan"))))
        finite_flow = torch.isfinite(torch.tensor(flow))
        finite_loss = torch.isfinite(loss)
        if not finite_flow:
            bad_flow += 1

        grad_norm = float("nan")
        step_bad_grad = 0
        if backward and opt is not None and finite_loss:
            loss.backward()
            grad_sq = 0.0
            for p in params:
                if p.grad is None:
                    continue
                if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                    step_bad_grad += 1
                grad_sq += p.grad.detach().float().pow(2).sum().item()
            grad_norm = grad_sq**0.5
            if step_bad_grad:
                bad_grad += 1
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()

        ah = getattr(model.gr00t if joint else model, "action_head", None)
        vstep = int(ah._visor_train_step.item()) if ah else -1
        cs = float(out.get("visor_coupling_scale", torch.tensor(-1.0)))
        print(
            f"step={step:02d} visor_step={vstep} coupling_scale={cs:.5f} "
            f"flow={flow:.4g} finite_flow={finite_flow} grad_norm={grad_norm:.4g} bad_grad={step_bad_grad}"
        )

    print(f"SUMMARY mode={mode}: bad_flow={bad_flow}/{steps} bad_grad={bad_grad}/{steps}")
    return bad_flow


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("joint", "baseline", "both"), default="both")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--forward-only", action="store_true")
    args = parser.parse_args()

    backward = not args.forward_only
    modes = ["joint", "baseline"] if args.mode == "both" else [args.mode]
    totals = {}
    for mode in modes:
        totals[mode] = run_steps(mode, steps=args.steps, backward=backward)
    if len(totals) > 1:
        print(f"COMPARE bad_flow: {totals}")


if __name__ == "__main__":
    main()
