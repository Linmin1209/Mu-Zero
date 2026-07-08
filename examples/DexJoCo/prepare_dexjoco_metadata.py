#!/usr/bin/env python3
"""Prepare DexJoCo LeRobot v3 task roots for GR00T training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from dexjoco_tactile_schema import modality_tactile_mapping, parquet_column, tactile_keys

from gr00t.data.dataset.lerobot_v3 import (
    load_v3_episodes_metadata,
    load_v3_tasks_map,
    slim_episode_index_record,
)


SINGLE_ARM_TASKS = {
    "click_mouse",
    "fold_glasses",
    "hammer_nail",
    "pick_bucket",
    "pinch_tongs",
    "water_plant",
}
BIMANUAL_TASKS = {
    "bimanual_assembly",
    "bimanual_hanoi",
    "bimanual_microwave_cook",
    "bimanual_photograph",
    "bimanual_unlock_ipad",
}


def _load_task_registry(registry_path: Path) -> dict:
    with open(registry_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _video_keys_from_info(info: dict, task_name: str) -> dict[str, str]:
    features = info.get("features", {})
    video_keys = [k for k, v in features.items() if v.get("dtype") == "video"]
    short_to_original: dict[str, str] = {}
    if task_name in SINGLE_ARM_TASKS:
        if "observation.images.ego_right" in video_keys:
            short_to_original["base"] = "observation.images.ego_right"
        elif "observation.images.front" in video_keys:
            short_to_original["base"] = "observation.images.front"
        if "observation.images.wrist" in video_keys:
            short_to_original["wrist"] = "observation.images.wrist"
    else:
        mapping = {
            "base": "observation.images.ego",
            "wrist_left": "observation.images.wrist_left",
            "wrist_right": "observation.images.wrist_right",
        }
        for short, original in mapping.items():
            if original in video_keys:
                short_to_original[short] = original
    if not short_to_original:
        raise ValueError(f"No supported video keys in {video_keys}")
    return short_to_original


def _tactile_modality_from_info(info: dict, task_name: str) -> dict | None:
    features = info.get("features", {})
    dual_arm = task_name in BIMANUAL_TASKS
    if parquet_column("ff") in features or parquet_column("R_ff") in features:
        return modality_tactile_mapping(dual_arm=dual_arm)
    if all(k in features for k in ("tactile.left", "tactile.right", "tactile.contact")):
        return {
            "left": {"original_key": "tactile.left"},
            "right": {"original_key": "tactile.right"},
            "contact": {"original_key": "tactile.contact"},
        }
    return None


def build_modality_json(info: dict, task_name: str) -> dict:
    action_dim = int(info["features"]["action"]["shape"][0])
    state_dim = int(info["features"]["observation.state"]["shape"][0])
    video = {
        short: {"original_key": original}
        for short, original in _video_keys_from_info(info, task_name).items()
    }
    if action_dim == 22:
        out = {
            "state": {
                "tcp_pose": {"start": 0, "end": 7, "original_key": "observation.state"},
                "hand": {"start": 7, "end": 23, "original_key": "observation.state"},
            },
            "action": {
                "eef_rotvec": {"start": 0, "end": 6, "original_key": "action"},
                "hand": {"start": 6, "end": 22, "original_key": "action"},
            },
            "video": video,
            "annotation": {
                "human.task_description": {"original_key": "task_index"},
            },
        }
    elif action_dim == 44:
        out = {
            "state": {
                "right_tcp": {"start": 0, "end": 7, "original_key": "observation.state"},
                "left_tcp": {"start": 7, "end": 14, "original_key": "observation.state"},
                "right_hand": {"start": 14, "end": 30, "original_key": "observation.state"},
                "left_hand": {"start": 30, "end": 46, "original_key": "observation.state"},
            },
            "action": {
                "right_eef": {"start": 0, "end": 6, "original_key": "action"},
                "right_hand": {"start": 6, "end": 22, "original_key": "action"},
                "left_eef": {"start": 22, "end": 28, "original_key": "action"},
                "left_hand": {"start": 28, "end": 44, "original_key": "action"},
            },
            "video": video,
            "annotation": {
                "human.task_description": {"original_key": "task_index"},
            },
        }
    else:
        raise ValueError(f"Unsupported action dim {action_dim} for task {task_name}")

    tactile = _tactile_modality_from_info(info, task_name)
    if tactile is not None:
        out["tactile"] = tactile
    return out


def write_tasks_jsonl(task_root: Path, task_name: str, registry: dict) -> None:
    prompt = registry["tasks"][task_name]["prompt"]
    out = task_root / "meta" / "tasks.jsonl"
    out.write_text(json.dumps({"task_index": 0, "task": prompt}, ensure_ascii=False) + "\n")


def write_episodes_jsonl(task_root: Path) -> None:
    episodes = load_v3_episodes_metadata(task_root)
    out = task_root / "meta" / "episodes.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for row in episodes:
            slim = slim_episode_index_record(row)
            f.write(json.dumps(slim, ensure_ascii=False) + "\n")


def prepare_task(task_root: Path, task_name: str, registry: dict, force: bool = False) -> None:
    info_path = task_root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Missing {info_path}")
    info = json.loads(info_path.read_text())

    modality_path = task_root / "meta" / "modality.json"
    if force or not modality_path.exists():
        modality_path.write_text(
            json.dumps(build_modality_json(info, task_name), indent=2, ensure_ascii=False) + "\n"
        )

    tasks_jsonl = task_root / "meta" / "tasks.jsonl"
    if force or not tasks_jsonl.exists():
        write_tasks_jsonl(task_root, task_name, registry)

    episodes_jsonl = task_root / "meta" / "episodes.jsonl"
    episodes_parquet = list((task_root / "meta" / "episodes").rglob("*.parquet"))
    if episodes_parquet and (force or not episodes_jsonl.exists()):
        write_episodes_jsonl(task_root)

    # Validate tasks map loads
    _ = load_v3_tasks_map(task_root)
    print(f"[ok] prepared {task_root}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets-root", type=Path, required=True)
    parser.add_argument("--task", action="append", default=[], help="Task name (repeatable)")
    parser.add_argument("--all-tasks", action="store_true")
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path(__file__).resolve().parent / "task_registry.yaml",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    registry = _load_task_registry(args.registry)
    task_names = sorted(registry["tasks"].keys()) if args.all_tasks else args.task
    if not task_names:
        raise SystemExit("Provide --task NAME or --all-tasks")

    for task_name in task_names:
        task_root = args.datasets_root / task_name
        prepare_task(task_root, task_name, registry, force=args.force)


if __name__ == "__main__":
    main()
