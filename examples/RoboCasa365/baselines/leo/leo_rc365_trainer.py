#!/usr/bin/env python3
"""LEO × RoboCasa365 multi-task LoRA trainer bridge.

Trains one checkpoint on manifest_target50.jsonl using LeoAgent + Vicuna LoRA
and a continuous 12D action head (default). Falls back to a lightweight ResNet
bridge when --bridge-only is set or LEO deps are unavailable.

Usage:
  python leo_rc365_trainer.py --manifest data/manifest_target50_pretrain.jsonl \\
    --leo-repo ../embodied-generalist --output-dir output/leo_rc365_target50_lora
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, random_split

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from leo_rc365_dataset import LeoRc365ManifestDataset  # noqa: E402


class MultiViewActionPolicy(nn.Module):
    """Lightweight bridge: 3-view CNN + proprio -> 12D Panda-Omron action."""

    def __init__(self, state_dim: int = 16, action_dim: int = 12, hidden: int = 1024):
        super().__init__()
        from torchvision.models import resnet18

        backbone = resnet18(weights="IMAGENET1K_V1")
        self.feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.state_mlp = nn.Sequential(
            nn.Linear(state_dim, hidden // 4),
            nn.ReLU(),
            nn.Linear(hidden // 4, hidden // 4),
        )
        self.head = nn.Sequential(
            nn.Linear(self.feat_dim * 3 + hidden // 4, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, images: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        b, n, c, h, w = images.shape
        feats = []
        for i in range(n):
            feats.append(self.backbone(images[:, i]))
        img_feat = torch.cat(feats, dim=-1)
        st = self.state_mlp(state)
        return self.head(torch.cat([img_feat, st], dim=-1))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--leo-repo", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--pretrained-ckpt", default="align")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--max-steps", type=int, default=30000)
    p.add_argument("--global-batch-size", type=int, default=32)
    p.add_argument("--num-gpus", type=int, default=1)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--video-backend", default="opencv")
    p.add_argument("--max-samples", type=int, default=0, help="0=all manifest rows")
    p.add_argument("--save-every", type=int, default=5000)
    p.add_argument(
        "--bridge-only",
        action="store_true",
        help="Use ResNet bridge instead of LeoAgent+LoRA (debug / no LEO stack).",
    )
    p.add_argument("--max-obj-len", type=int, default=4, help="Max 3D objects per sample (RoboCasa uses 1).")
    p.add_argument("--num-workers", type=int, default=4, help="DataLoader workers for LeRobot IO.")
    return p.parse_args()


def try_import_leo(leo_repo: Path) -> bool:
    sys.path.insert(0, str(leo_repo))
    try:
        from model.leo_agent import LeoAgent  # noqa: F401

        return True
    except Exception as exc:
        print(f"[w] LEO import failed ({exc}); using lightweight bridge policy.")
        return False


def save_checkpoint(
    output_dir: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    args: argparse.Namespace,
    cfg: dict,
    policy_tag: str,
) -> None:
    ckpt_dir = output_dir / f"checkpoint-{step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if hasattr(model, "learnable_state_dict"):
        torch.save(model.learnable_state_dict(), ckpt_dir / "pytorch_model.bin")
    else:
        torch.save(model.state_dict(), ckpt_dir / "pytorch_model.bin")
    torch.save(optimizer.state_dict(), ckpt_dir / "optimizer.pt")
    (output_dir / "train_config.json").write_text(
        json.dumps(
            {
                "manifest": str(args.manifest),
                "pretrained_ckpt": args.pretrained_ckpt,
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "max_steps": args.max_steps,
                "global_batch_size": args.global_batch_size,
                "step": step,
                "policy": policy_tag,
                "leo_config": cfg,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[i] Saved {ckpt_dir}")


def build_loaders(
    args: argparse.Namespace,
    cfg: dict,
    collate_fn=None,
    num_workers: int = 2,
) -> tuple[DataLoader, DataLoader, LeoRc365ManifestDataset]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = LeoRc365ManifestDataset(
        args.manifest,
        video_backend=args.video_backend,
        max_samples=args.max_samples,
        use_3d=bool(cfg.get("data", {}).get("use_3d", False)),
        require_3d=bool(cfg.get("data", {}).get("require_3d", False)),
        num_points=int(cfg.get("data", {}).get("num_points", 1024)),
        normalize_action=bool(cfg.get("data", {}).get("normalize_action", False)),
    )
    if len(dataset) == 0:
        raise RuntimeError("Empty manifest dataset")

    n_val = max(1, int(0.02 * len(dataset)))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )

    per_gpu_bs = max(1, args.global_batch_size // max(1, args.num_gpus))
    loader_kwargs = {
        "batch_size": per_gpu_bs,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    if collate_fn is not None:
        loader_kwargs["collate_fn"] = collate_fn

    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, dataset


def train_lightweight(args: argparse.Namespace, cfg: dict) -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[i] Device: {device} (lightweight bridge)")

    train_loader, val_loader, _ = build_loaders(args, cfg)
    model = MultiViewActionPolicy(
        action_dim=int(cfg.get("data", {}).get("action_dim", 12)),
        hidden=int(cfg.get("action_head", {}).get("hidden_dim", 1024)),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    criterion = nn.MSELoss()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    step = 0
    best_val = math.inf
    t0 = time.time()
    epoch = 0

    while step < args.max_steps:
        epoch += 1
        model.train()
        for batch in train_loader:
            images = batch["images"].to(device)
            state = batch["state"].to(device)
            action = batch["action"].to(device)

            pred = model(images, state)
            loss = criterion(pred, action)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            step += 1

            if step % 100 == 0:
                print(f"[train] step={step} loss={loss.item():.6f} epoch={epoch}")

            if step % args.save_every == 0 or step == args.max_steps:
                save_checkpoint(
                    args.output_dir, model, optimizer, step, args, cfg, "multiview_resnet_bridge"
                )

            if step >= args.max_steps:
                break

        model.eval()
        val_loss = 0.0
        n_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                images = batch["images"].to(device)
                state = batch["state"].to(device)
                action = batch["action"].to(device)
                pred = model(images, state)
                val_loss += criterion(pred, action).item()
                n_batches += 1
        val_loss /= max(1, n_batches)
        print(f"[val] epoch={epoch} step={step} val_mse={val_loss:.6f}")
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(
                args.output_dir, model, optimizer, step, args, cfg, "multiview_resnet_bridge"
            )
            (args.output_dir / "best_step.txt").write_text(str(step), encoding="utf-8")

    elapsed = time.time() - t0
    print(f"[i] Training done in {elapsed/60:.1f} min, best_val_mse={best_val:.6f}")
    return 0


def train_leo_lora(args: argparse.Namespace, cfg: dict) -> int:
    from accelerate import PartialState

    PartialState()  # required before LeoAgent (accelerate logging)

    from leo_rc365_sanitize import apply_leo_numeric_patches

    apply_leo_numeric_patches()
    print("[i] Applied LEO 3D numeric safety patches (pairwise + spatial-attn)")

    from leo_rc365_leo_agent import build_leo_agent_for_training
    from leo_rc365_leo_batch import collate_leo_batch
    from leo_rc365_leo_model import (
        build_leo_cfg,
        LeoRc365ActionModel,
        resolve_pretrained_ckpt,
    )

    use_3d_cfg = bool(cfg.get("data", {}).get("use_3d", False))
    if use_3d_cfg and not torch.cuda.is_available():
        raise RuntimeError(
            "LEO 3D (PointNet++) training requires CUDA; set data.use_3d=false or use --bridge-only."
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.num_gpus > 1:
        print("[w] LeoAgent training uses single-process single-GPU for now; "
              f"ignoring --num-gpus={args.num_gpus} beyond batch sizing.")
    print(f"[i] Device: {device} (LeoAgent + LoRA + action head)")

    use_3d = bool(cfg.get("data", {}).get("use_3d", False))
    max_obj_len = int(args.max_obj_len)

    def collate_fn(samples):
        return collate_leo_batch(samples, use_3d=use_3d, max_obj_len=max_obj_len)

    train_loader, val_loader, _ = build_loaders(
        args, cfg, collate_fn=collate_fn, num_workers=args.num_workers
    )

    leo_cfg = build_leo_cfg(
        args.leo_repo,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        clip_txt_guidance=False,
    )
    leo_agent = build_leo_agent_for_training(leo_cfg)
    leo_agent.get_learnable_named_params()
    if hasattr(leo_agent.llm_model, "gradient_checkpointing_enable") and not getattr(
        leo_agent, "_rc365_offload", False
    ):
        leo_agent.llm_model.gradient_checkpointing_enable()
        print("[i] Enabled LLM gradient checkpointing for memory savings.")
    elif getattr(leo_agent, "_rc365_offload", False):
        print("[i] LLM CPU offload active; gradient checkpointing disabled.")

    model = LeoRc365ActionModel(
        leo_agent,
        state_dim=16,
        action_dim=int(cfg.get("data", {}).get("action_dim", 12)),
        action_hidden=int(cfg.get("action_head", {}).get("hidden_dim", 1024)),
    )
    model.move_trainable_modules_to_cuda()

    ckpt_path = resolve_pretrained_ckpt(args.leo_repo, args.pretrained_ckpt)
    print(f"[i] Loading LEO pretrained weights from {ckpt_path}")
    missing, unexpected = model.load_pretrained_leo(ckpt_path, strict=False)
    if missing:
        print(f"[i] Pretrained load missing keys (expected for action head): {len(missing)}")
    if unexpected:
        print(f"[w] Pretrained unexpected keys: {unexpected[:8]}{'...' if len(unexpected) > 8 else ''}")

    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"[i] Trainable parameter tensors: {len(trainable)}")
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    step = 0
    best_val = math.inf
    t0 = time.time()
    epoch = 0

    skipped_batches = 0
    while step < args.max_steps:
        epoch += 1
        model.train()
        for batch in train_loader:
            try:
                out = model(batch)
            except (AssertionError, RuntimeError) as exc:
                skipped_batches += 1
                print(f"[w] skip batch at step={step + 1} ({type(exc).__name__}: {exc})")
                if skipped_batches <= 3 or skipped_batches % 50 == 0:
                    import traceback
                    traceback.print_exc()
                continue
            loss = out["loss"]
            if not torch.isfinite(loss):
                skipped_batches += 1
                print(f"[w] skip non-finite loss at step={step + 1}")
                continue
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 5.0)
            optimizer.step()
            step += 1

            if step % 100 == 0:
                print(f"[train] step={step} loss={loss.item():.6f} epoch={epoch}")

            if step % args.save_every == 0 or step == args.max_steps:
                save_checkpoint(
                    args.output_dir,
                    model,
                    optimizer,
                    step,
                    args,
                    cfg,
                    "leo_agent_lora_action_head",
                )

            if step >= args.max_steps:
                break

        model.eval()
        val_loss = 0.0
        n_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                out = model(batch)
                val_loss += out["loss"].item()
                n_batches += 1
        val_loss /= max(1, n_batches)
        print(f"[val] epoch={epoch} step={step} val_mse={val_loss:.6f}")
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(
                args.output_dir,
                model,
                optimizer,
                step,
                args,
                cfg,
                "leo_agent_lora_action_head",
            )
            (args.output_dir / "best_step.txt").write_text(str(step), encoding="utf-8")

    elapsed = time.time() - t0
    print(f"[i] LEO training done in {elapsed/60:.1f} min, best_val_mse={best_val:.6f}")
    return 0


def main() -> int:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text())

    summary_path = args.manifest.with_suffix(".summary.json")
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text())
        print(
            f"[i] Manifest frames: {summary.get('num_frames')} "
            f"tasks: {summary.get('tasks_with_data')}/50"
        )

    if bool(cfg.get("data", {}).get("normalize_action", False)):
        from leo_rc365_action_norm import summarize_action_stats

        action_norm = summarize_action_stats(args.manifest)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "action_norm.json").write_text(
            json.dumps(action_norm, indent=2), encoding="utf-8"
        )
        print(
            f"[i] Action normalization: per-task LeRobot stats "
            f"({action_norm['num_tasks']} tasks, dim={action_norm['action_dim']})"
        )

    if args.bridge_only:
        print("[i] --bridge-only: using lightweight ResNet trainer.")
        return train_lightweight(args, cfg)

    leo_ok = try_import_leo(args.leo_repo)
    if leo_ok:
        print("[i] LEO import OK — running LeoAgent + LoRA + action-head trainer.")
        return train_leo_lora(args, cfg)

    print("[i] Falling back to lightweight bridge trainer.")
    return train_lightweight(args, cfg)


if __name__ == "__main__":
    raise SystemExit(main())
