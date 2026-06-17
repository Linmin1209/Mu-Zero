#!/usr/bin/env python3
"""LEO × RoboCasa365 multi-task LoRA trainer bridge.

Trains one checkpoint on manifest_target50_pretrain.jsonl.
v1: multi-view ResNet + proprio action regressor (runs without full LEO stack).
v2: swap in LeoAgent + LoRA when LEO deps are installed (see setup_leo.sh).

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
        self.view_pool = nn.AdaptiveAvgPool1d(1)
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
        # images: B,3,C,H,W
        b, n, c, h, w = images.shape
        feats = []
        for i in range(n):
            x = images[:, i]
            feats.append(self.backbone(x))
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
) -> None:
    ckpt_dir = output_dir / f"checkpoint-{step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
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
                "policy": "multiview_resnet_bridge",
                "leo_config": cfg,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[i] Saved {ckpt_dir}")


def train_lightweight(args: argparse.Namespace, cfg: dict) -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[i] Device: {device}")

    dataset = LeoRc365ManifestDataset(
        args.manifest,
        video_backend=args.video_backend,
        max_samples=args.max_samples,
    )
    if len(dataset) == 0:
        print("[x] Empty manifest dataset")
        return 1

    n_val = max(1, int(0.02 * len(dataset)))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )

    per_gpu_bs = max(1, args.global_batch_size // max(1, args.num_gpus))
    train_loader = DataLoader(
        train_ds,
        batch_size=per_gpu_bs,
        shuffle=True,
        num_workers=2,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(val_ds, batch_size=per_gpu_bs, shuffle=False, num_workers=2)

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
                save_checkpoint(args.output_dir, model, optimizer, step, args, cfg)

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
            save_checkpoint(args.output_dir, model, optimizer, step, args, cfg)
            (args.output_dir / "best_step.txt").write_text(str(step), encoding="utf-8")

    elapsed = time.time() - t0
    print(f"[i] Training done in {elapsed/60:.1f} min, best_val_mse={best_val:.6f}")
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

    leo_ok = try_import_leo(args.leo_repo)
    if leo_ok:
        print("[i] LEO import OK — full LeoAgent+LoRA training not wired yet.")
        print("[i] Running lightweight bridge trainer (same manifest, action MSE).")

    return train_lightweight(args, cfg)


if __name__ == "__main__":
    raise SystemExit(main())
