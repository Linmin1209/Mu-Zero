#!/usr/bin/env python3
"""Per-sample flow_loss in batch=8; plain vs joint collate."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
_MODELS_ROOT = Path(os.environ.get("GR00T_MODELS_ROOT", "/HOME/sysu_xdliang/sysu_xdliang_1/HDD_POOL/linmin/models"))
for k, v in {
    "GR00T_MODELS_ROOT": str(_MODELS_ROOT),
    "GROOT_PATCH_MISTRAL": "1",
    "GROOT_HF_LOCAL_FIRST": "1",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
}.items():
    os.environ.setdefault(k, v)

import torch
import torch.nn.functional as F

MOD_PATH = PROJECT / "examples/RoboCasa365/robocasa365_config_4frame.py"
sys.path.insert(0, str(MOD_PATH.parent))
importlib.import_module(MOD_PATH.stem)


def build_config(*, joint: bool):
    from gr00t.configs.base_config import get_default_config
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.experiment.robocasa365_datasets import resolve_robocasa365_dataset_paths

    paths = resolve_robocasa365_dataset_paths(
        root="/HOME/sysu_xdliang/sysu_xdliang_1/HDD_POOL/linmin/datasets/robocasa365-datasets",
        split="pretrain", category="atomic", tasks="PickPlaceToasterOvenToCounter",
    )
    c = get_default_config().load_dict({"data": {"download_cache": False, "datasets": [
        {"dataset_paths": paths, "mix_ratio": 1.0,
         "embodiment_tag": EmbodimentTag.ROBOCASA_PANDA_OMRON.value}]}})
    for a, v in {
        "use_visor": True, "use_joint_dual_branch": joint, "joint_train_mode": "gr00t_only",
        "use_motion": True, "tune_motion": True,
        "visor_gate_mode": "visual_manip_nav_tactile_hand",
        "visor_use_visual_supervision": True, "visor_use_readout_fed_gates": True,
        "visor_visual_gt_level": "flow", "visor_visual_dim": 2,
        "visor_aux_delay_steps": 6000, "load_bf16": True,
    }.items():
        setattr(c.model, a, v)
    c.training.start_from_checkpoint = str(_MODELS_ROOT / "GR00T-N1.7-3B")
    c.training.transformers_local_files_only = True
    c.model.model_name = str(_MODELS_ROOT / "Cosmos-Reason2-2B")
    c.data.allow_padding = True
    c.data.shard_size = 256
    c.data.seed = 42
    return c


@torch.no_grad()
def per_sample_flow(model, collator, raw_samples, seed=0):
    flows = []
    for i, r in enumerate(raw_samples):
        model.action_head._visor_train_step.zero_()
        inp = collator([r])["inputs"]
        inp = {k: v.cuda() if torch.is_tensor(v) else v for k, v in inp.items()}
        inp = {k: v for k, v in inp.items() if not str(k).startswith("flux_")}
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        model = model.cuda().train()
        model.gradient_checkpointing_enable()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(inp)
        flows.append(float(out["flow_loss"]))
    return flows


def main():
    from gr00t.model import MODEL_REGISTRY

    out = PROJECT / "output/diag_per_sample"
    out.mkdir(parents=True, exist_ok=True)

    for label, joint in [("plain", False), ("joint", True)]:
        c = build_config(joint=joint)
        save = out / label
        save.mkdir(parents=True, exist_ok=True)
        p = MODEL_REGISTRY.get(type(c.model))(c, save)
        p.setup()
        m = p.return_model()
        col = p.return_collator()
        raw = [next(iter(p.return_dataset()[0])) for _ in range(1)]  # wrong - need 8 same
        it = iter(p.return_dataset()[0])
        raw = [next(it) for _ in range(8)]
        singles = per_sample_flow(m, col, raw, seed=0)
        batched = per_sample_flow(m, col, raw, seed=0)  # wrong - need batched forward
        # batched forward
        m.action_head._visor_train_step.zero_()
        binp = col(raw)["inputs"]
        binp = {k: v.cuda() if torch.is_tensor(v) else v for k, v in binp.items()}
        binp = {k: v for k, v in binp.items() if not str(k).startswith("flux_")}
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        m = m.cuda().train()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            bf = float(m(binp)["flow_loss"])
        print(f"\n[{label}] per-sample (batch=1, seed=0): {[round(x,3) for x in singles]}")
        print(f"[{label}] mean={sum(singles)/8:.3f} min={min(singles):.3f} max={max(singles):.3f}")
        print(f"[{label}] batched batch=8 seed=0: flow={bf:.3f}")


if __name__ == "__main__":
    main()
