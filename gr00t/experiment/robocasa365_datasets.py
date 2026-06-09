# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve RoboCasa365 LeRobot dataset roots for GR00T finetuning."""

from __future__ import annotations

from pathlib import Path

DEFAULT_ROBOCASA365_ROOT = Path(
    "/HOME/sysu_xdliang/sysu_xdliang_1/HDD_POOL/linmin/datasets/robocasa365-datasets"
)
VALID_SPLITS = ("pretrain", "target", "all")
VALID_CATEGORIES = ("atomic", "composite", "all")


def get_default_robocasa365_modality_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "examples" / "RoboCasa365" / "robocasa365_config.py"


def _normalize_choice(value: str, valid: tuple[str, ...], name: str) -> str:
    normalized = value.strip().lower()
    if normalized not in valid:
        raise ValueError(f"Invalid {name}={value!r}. Choose from: {', '.join(valid)}")
    return normalized


def _parse_tasks(tasks: str | None) -> set[str] | None:
    if tasks is None or not str(tasks).strip():
        return None
    return {t.strip() for t in tasks.split(",") if t.strip()}


def _pick_lerobot_root(task_dir: Path, latest_snapshot_only: bool) -> Path | None:
    """Return .../<date>/lerobot if present."""
    candidates: list[Path] = []
    if (task_dir / "lerobot" / "meta" / "info.json").is_file():
        candidates.append(task_dir / "lerobot")
    for date_dir in sorted(task_dir.iterdir()):
        if not date_dir.is_dir():
            continue
        lerobot = date_dir / "lerobot"
        if (lerobot / "meta" / "info.json").is_file():
            candidates.append(lerobot)
    if not candidates:
        return None
    if latest_snapshot_only:
        return candidates[-1]
    return candidates[0]


def resolve_robocasa365_dataset_paths(
    root: str | Path,
    split: str = "all",
    category: str = "all",
    tasks: str | None = None,
    latest_snapshot_only: bool = True,
) -> list[str]:
    """
    Discover LeRobot dataset roots under the RoboCasa365 release layout:

        {root}/{pretrain|target}/{atomic|composite}/{TaskName}/{YYYYMMDD}/lerobot/

    Returns absolute paths to each ``lerobot`` directory (for ``dataset_path`` / pathsep lists).
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"robocasa365 root not found: {root}")

    split = _normalize_choice(split, VALID_SPLITS, "robocasa365_split")
    category = _normalize_choice(category, VALID_CATEGORIES, "robocasa365_category")
    task_filter = _parse_tasks(tasks)

    splits = ["pretrain", "target"] if split == "all" else [split]
    categories = ["atomic", "composite"] if category == "all" else [category]

    paths: list[str] = []
    missing_tasks: list[str] = []

    for sp in splits:
        for cat in categories:
            cat_dir = root / sp / cat
            if not cat_dir.is_dir():
                continue
            for task_dir in sorted(cat_dir.iterdir()):
                if not task_dir.is_dir() or task_dir.name.startswith("."):
                    continue
                if task_filter is not None and task_dir.name not in task_filter:
                    continue
                lerobot = _pick_lerobot_root(task_dir, latest_snapshot_only)
                if lerobot is None:
                    if task_filter is not None and task_dir.name in task_filter:
                        missing_tasks.append(f"{sp}/{cat}/{task_dir.name}")
                    continue
                paths.append(str(lerobot.resolve()))

    if task_filter:
        found_tasks = {Path(p).parent.parent.name for p in paths}
        for task in sorted(task_filter - found_tasks):
            missing_tasks.append(task)
        if missing_tasks:
            raise FileNotFoundError(
                "Requested RoboCasa365 task(s) missing or not extracted yet: "
                + ", ".join(missing_tasks)
            )

    if not paths:
        raise FileNotFoundError(
            f"No extracted lerobot datasets under {root} "
            f"(split={split}, category={category}). "
            "Ensure *.tar files are extracted to .../<date>/lerobot/."
        )

    return sorted(paths)


def format_dataset_paths(paths: list[str]) -> str:
    import os

    return os.pathsep.join(paths)
