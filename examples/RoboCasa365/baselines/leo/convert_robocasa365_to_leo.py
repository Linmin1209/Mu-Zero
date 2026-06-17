#!/usr/bin/env python3
"""Build LEO multi-task training manifest from RoboCasa365 LeRobot exports (target50).

Each line is one training frame with paths to RGB frames, language, and 12D action.
Used by finetune_leo_target50_lora.sh (LEO repo dataloader or bridge trainer).

Usage:
  python convert_robocasa365_to_leo.py \\
    --robocasa365-root /path/to/robocasa365-datasets \\
    --task-yaml ../../task_sets.yaml \\
    --output manifest_target50.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
RC365_DIR = SCRIPT_DIR.parent.parent
PROJECT_ROOT = RC365_DIR.parent.parent

sys.path.insert(0, str(PROJECT_ROOT))
from gr00t.experiment.robocasa365_datasets import resolve_robocasa365_dataset_paths  # noqa: E402

VIDEO_KEYS = [
    "observation.images.robot0_agentview_left",
    "observation.images.robot0_agentview_right",
    "observation.images.robot0_eye_in_hand",
]
STATE_KEYS = [
    "observation.state.base_position",
    "observation.state.base_rotation",
    "observation.state.end_effector_position_relative",
    "observation.state.end_effector_rotation_relative",
    "observation.state.gripper_qpos",
]
ACTION_KEYS = [
    "action.gripper_close",
    "action.end_effector_position",
    "action.end_effector_rotation",
    "action.base_motion",
    "action.control_mode",
]


def load_target50_tasks(task_yaml: Path) -> list[str]:
    cfg = yaml.safe_load(task_yaml.read_text())
    return (
        list(cfg["atomic_seen"])
        + list(cfg["composite_seen"])
        + list(cfg["composite_unseen"])
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--robocasa365-root",
        type=Path,
        default=Path(
            "/HOME/sysu_xdliang/sysu_xdliang_1/HDD_POOL/linmin/datasets/robocasa365-datasets"
        ),
    )
    parser.add_argument("--task-yaml", type=Path, default=RC365_DIR / "task_sets.yaml")
    parser.add_argument(
        "--split", default="pretrain", choices=["pretrain", "target", "all"]
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=SCRIPT_DIR / "data" / "manifest_target50_pretrain.jsonl",
    )
    parser.add_argument("--max-episodes-per-task", type=int, default=0, help="0=all")
    parser.add_argument("--stride", type=int, default=1, help="Frame subsample stride")
    args = parser.parse_args()

    tasks = load_target50_tasks(args.task_yaml)
    lerobot_roots: list[str] = []
    missing: list[str] = []
    for task in tasks:
        try:
            paths = resolve_robocasa365_dataset_paths(
                root=args.robocasa365_root,
                split=args.split,
                category="all",
                tasks=task,
            )
            lerobot_roots.extend(paths)
        except FileNotFoundError:
            missing.append(task)

    found_tasks = {Path(p).parent.parent.name for p in lerobot_roots}
    if missing:
        print(
            f"[w] No local LeRobot data for {len(missing)} tasks "
            f"(will train/eval on {len(found_tasks)} available):",
            ", ".join(missing[:6]),
            ("..." if len(missing) > 6 else ""),
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    n_lines = 0
    with args.output.open("w", encoding="utf-8") as out_f:
        for lerobot_root in lerobot_roots:
            root = Path(lerobot_root)
            task_name = root.parent.parent.name
            meta_info = json.loads((root / "meta" / "info.json").read_text())
            fps = meta_info.get("fps", 20)
            tasks_jsonl = root / "meta" / "tasks.jsonl"
            default_lang = task_name
            if tasks_jsonl.is_file():
                first = tasks_jsonl.read_text(encoding="utf-8").strip().split("\n")[0]
                default_lang = json.loads(first).get("task", default_lang)

            import pandas as pd  # noqa: WPS433

            data_glob = sorted((root / "data").glob("chunk-*/episode_*.parquet"))
            if args.max_episodes_per_task > 0:
                data_glob = data_glob[: args.max_episodes_per_task]

            for parquet_path in data_glob:
                ep_idx = int(parquet_path.stem.split("_")[-1])
                df = pd.read_parquet(parquet_path)
                for row_i in range(0, len(df), args.stride):
                    row = df.iloc[row_i]
                    frame = {
                        "task": task_name,
                        "lerobot_root": str(root),
                        "episode_index": ep_idx,
                        "frame_index": int(row.get("frame_index", row_i)),
                        "fps": fps,
                        "language": default_lang,
                        "video_keys": VIDEO_KEYS,
                        "state_keys": STATE_KEYS,
                        "action_keys": ACTION_KEYS,
                    }
                    for k in STATE_KEYS + ACTION_KEYS:
                        if k in row.index:
                            val = row[k]
                            frame[k.replace("observation.", "").replace("action.", "act_")] = (
                                val.tolist() if hasattr(val, "tolist") else val
                            )
                    out_f.write(json.dumps(frame, ensure_ascii=False) + "\n")
                    n_lines += 1

    summary = {
        "tasks_requested": len(tasks),
        "tasks_with_data": len(found_tasks),
        "missing_tasks": missing,
        "tasks_available_for_training": sorted(found_tasks),
        "manifest": str(args.output),
        "num_frames": n_lines,
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[i] Wrote {args.output} ({n_lines} frames)")
    print(f"[i] Summary: {summary_path}")
    print(f"[i] Tasks with data: {len(found_tasks)}/{len(tasks)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
