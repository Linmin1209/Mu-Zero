#!/usr/bin/env python3
"""List RoboCasa365 lerobot roots that need gripper tactile labeling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

TACTILE_COLS = ("tactile.left", "tactile.right", "tactile.contact")


def pick_lerobot_root(task_dir: Path) -> Path | None:
    candidates: list[Path] = []
    if (task_dir / "lerobot" / "meta" / "info.json").is_file():
        candidates.append(task_dir / "lerobot")
    for date_dir in sorted(task_dir.iterdir()):
        if date_dir.is_dir() and (date_dir / "lerobot" / "meta" / "info.json").is_file():
            candidates.append(date_dir / "lerobot")
    if not candidates:
        return None
    return candidates[-1]


def is_fully_labeled(lerobot: Path) -> bool:
    info_path = lerobot / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    features = info.get("features", {})
    if not all(col in features for col in TACTILE_COLS):
        return False
    total = int(info.get("total_episodes", 0))
    summary_path = lerobot / "meta" / "haptic_gripper_label_summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text())
        if isinstance(summary, list) and len(summary) >= total:
            return True
    # Spot-check first parquet for column presence.
    parquet = next((lerobot / "data").rglob("*.parquet"), None)
    if parquet is None:
        return False
    import pandas as pd

    df = pd.read_parquet(parquet)
    return all(col in df.columns for col in TACTILE_COLS)


def discover(root: Path, splits: list[str]) -> tuple[list[Path], list[Path]]:
    needs: list[Path] = []
    complete: list[Path] = []
    for split in splits:
        split_dir = root / split
        if not split_dir.is_dir():
            continue
        for cat in ("atomic", "composite"):
            cat_dir = split_dir / cat
            if not cat_dir.is_dir():
                continue
            for task_dir in sorted(cat_dir.iterdir()):
                if not task_dir.is_dir() or task_dir.name.startswith("."):
                    continue
                lerobot = pick_lerobot_root(task_dir)
                if lerobot is None:
                    continue
                if not (lerobot / "extras").is_dir():
                    continue
                if is_fully_labeled(lerobot):
                    complete.append(lerobot)
                else:
                    needs.append(lerobot)
    return needs, complete


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(
            "/HOME/sysu_xdliang/sysu_xdliang_1/HDD_POOL/linmin/datasets/robocasa365-datasets"
        ),
    )
    parser.add_argument(
        "--splits",
        default="pretrain,target",
        help="Comma-separated splits to scan (default: pretrain,target)",
    )
    parser.add_argument(
        "--list-complete",
        action="store_true",
        help="Also print fully labeled dataset paths.",
    )
    args = parser.parse_args()
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    needs, complete = discover(args.root.resolve(), splits)
    print(f"needs_labeling={len(needs)} complete={len(complete)}")
    for path in needs:
        print(path)
    if args.list_complete:
        for path in complete:
            print(f"COMPLETE {path}")


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        pass
