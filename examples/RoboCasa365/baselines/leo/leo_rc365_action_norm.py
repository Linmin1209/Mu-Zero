#!/usr/bin/env python3
"""Action mean/std normalization from LeRobot meta/stats.json (per task root)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

_STD_EPS = 1e-3
_STATS_CACHE: dict[str, dict[str, np.ndarray]] = {}


def _stats_path(lerobot_root: Path | str) -> Path:
    return Path(lerobot_root).resolve() / "meta" / "stats.json"


def load_action_stats(lerobot_root: Path | str) -> dict[str, np.ndarray]:
    """Load 12D action mean/std for one LeRobot task root."""
    key = str(Path(lerobot_root).resolve())
    if key in _STATS_CACHE:
        return _STATS_CACHE[key]

    path = _stats_path(lerobot_root)
    if not path.is_file():
        raise FileNotFoundError(f"Missing LeRobot stats for action norm: {path}")

    raw = json.loads(path.read_text(encoding="utf-8"))
    if "action" not in raw:
        raise KeyError(f"No 'action' entry in {path}")

    action_stats = raw["action"]
    mean = np.asarray(action_stats["mean"], dtype=np.float32)
    std = np.asarray(action_stats["std"], dtype=np.float32)
    if mean.shape != std.shape:
        raise ValueError(f"Action mean/std shape mismatch in {path}: {mean.shape} vs {std.shape}")

    std = np.maximum(std, _STD_EPS)
    stats = {"mean": mean, "std": std}
    _STATS_CACHE[key] = stats
    return stats


def normalize_action(action: torch.Tensor, lerobot_root: Path | str) -> torch.Tensor:
    stats = load_action_stats(lerobot_root)
    mean = torch.from_numpy(stats["mean"]).to(action.device, dtype=action.dtype)
    std = torch.from_numpy(stats["std"]).to(action.device, dtype=action.dtype)
    return (action.float() - mean) / std


def denormalize_action(action: torch.Tensor, lerobot_root: Path | str) -> torch.Tensor:
    stats = load_action_stats(lerobot_root)
    mean = torch.from_numpy(stats["mean"]).to(action.device, dtype=action.dtype)
    std = torch.from_numpy(stats["std"]).to(action.device, dtype=action.dtype)
    return action.float() * std + mean


def collect_manifest_lerobot_roots(manifest_path: Path | str) -> list[str]:
    roots: list[str] = []
    seen: set[str] = set()
    with Path(manifest_path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            root = str(Path(json.loads(line)["lerobot_root"]).resolve())
            if root not in seen:
                seen.add(root)
                roots.append(root)
    return roots


def summarize_action_stats(manifest_path: Path | str) -> dict[str, Any]:
    """Aggregate per-task stats into a manifest-level summary (for checkpoint metadata)."""
    roots = collect_manifest_lerobot_roots(manifest_path)
    if not roots:
        return {"num_tasks": 0, "action_dim": 0}

    means, stds = [], []
    per_task: dict[str, dict[str, list[float]]] = {}
    for root in roots:
        stats = load_action_stats(root)
        means.append(stats["mean"])
        stds.append(stats["std"])
        per_task[root] = {
            "mean": stats["mean"].tolist(),
            "std": stats["std"].tolist(),
        }

    mean_stack = np.stack(means, axis=0)
    std_stack = np.stack(stds, axis=0)
    return {
        "mode": "per_task_lerobot_stats",
        "num_tasks": len(roots),
        "action_dim": int(mean_stack.shape[-1]),
        "mean_of_means": mean_stack.mean(axis=0).tolist(),
        "mean_of_stds": std_stack.mean(axis=0).tolist(),
        "per_task": per_task,
    }
