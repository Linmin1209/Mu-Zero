#!/usr/bin/env python3
"""Generate per-finger Allegro tactile for DexJoCo LeRobot v3 datasets via MuJoCo replay.

Replays each episode's rotvec actions in the DexJoCo sim and writes tactile columns
into the parquet files. Updates meta/info.json and meta/modality.json.

Requirements:
  - DEXJOCo_ROOT on PYTHONPATH (mujoco + dexjoco package)
  - Existing LeRobot v3 task roots under --datasets-root

Example:
  DEXJOCo_ROOT=/path/to/dexjoco \\
  python examples/DexJoCo/generate_dexjoco_haptic_labels.py \\
    --datasets-root /path/to/dexjoco_lerobot_datasets \\
    --task water_plant
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from dexjoco_haptic_extraction import (
    extract_dexjoco_tactile_frame,
    rotvec_action_to_policy,
)
from dexjoco_tactile_schema import modality_tactile_mapping, parquet_column, tactile_feature_template, tactile_keys
from gr00t.data.dataset.lerobot_v3 import load_v3_episodes_metadata  # noqa: E402


def _load_registry(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _patch_info_json(info_path: Path, *, dual_arm: bool) -> None:
    info = json.loads(info_path.read_text(encoding="utf-8"))
    features = dict(info.get("features", {}))
    features.update(tactile_feature_template(dual_arm=dual_arm))
    info["features"] = features
    info_path.write_text(json.dumps(info, indent=4) + "\n", encoding="utf-8")


def _patch_modality_json(modality_path: Path, *, dual_arm: bool) -> None:
    modality = json.loads(modality_path.read_text(encoding="utf-8"))
    modality["tactile"] = modality_tactile_mapping(dual_arm=dual_arm)
    modality_path.write_text(json.dumps(modality, indent=2, ensure_ascii=False) + "\n")


def _ensure_dexjoco_import(dexjoco_root: Path) -> None:
    pkg = dexjoco_root / "dexjoco"
    if not (pkg / "dexjoco").is_dir():
        raise FileNotFoundError(f"DexJoCo package not found under {pkg}")
    if str(pkg) not in sys.path:
        sys.path.insert(0, str(pkg))


def _get_mujoco_model_data(env) -> tuple[Any, Any]:
    cur = env
    while cur is not None:
        if hasattr(cur, "model") and hasattr(cur, "data"):
            return cur.model, cur.data
        cur = getattr(cur, "env", None)
    raise RuntimeError("Could not find MuJoCo model/data on env wrapper chain")


def _replay_episode_tactile(
    *,
    task_name: str,
    actions_rotvec: np.ndarray,
    dexjoco_root: Path,
    dual_arm: bool,
    seed: int,
) -> list[dict]:
    _ensure_dexjoco_import(dexjoco_root)
    from dexjoco.tasks import CONFIG_MAPPING

    config = CONFIG_MAPPING[task_name]()
    env = config.get_environment(
        policy_mode=True,
        render_mode="rgb_array",
        randomize=False,
        seed=seed,
        randomize_dynamics=False,
    )
    rows: list[dict] = []
    try:
        env.reset(seed=seed)
        model, data = _get_mujoco_model_data(env)

        for t in range(len(actions_rotvec)):
            frame = extract_dexjoco_tactile_frame(model, data, dual_arm=dual_arm)
            rows.append(frame.as_parquet_values())
            policy_action = rotvec_action_to_policy(actions_rotvec[t], dual_arm=dual_arm)
            env.step(policy_action)
    finally:
        env.close()
    return rows


def _label_parquet(
    parquet_path: Path,
    *,
    task_name: str,
    dexjoco_root: Path,
    dual_arm: bool,
    dry_run: bool,
    max_episodes: int = 0,
) -> int:
    df = pd.read_parquet(parquet_path)
    if "episode_index" not in df.columns:
        raise KeyError(f"{parquet_path} missing episode_index")

    updated_frames = 0
    episode_indices = sorted(df["episode_index"].unique())
    if max_episodes and max_episodes > 0:
        episode_indices = episode_indices[:max_episodes]
    for ep_idx in episode_indices:
        ep_mask = df["episode_index"] == ep_idx
        ep_df = df.loc[ep_mask]
        actions = np.stack(ep_df["action"].to_list(), axis=0)
        tactile_rows = _replay_episode_tactile(
            task_name=task_name,
            actions_rotvec=actions,
            dexjoco_root=dexjoco_root,
            dual_arm=dual_arm,
            seed=int(ep_idx),
        )
        if len(tactile_rows) != len(ep_df):
            raise RuntimeError(
                f"Episode {ep_idx}: tactile rows {len(tactile_rows)} != frames {len(ep_df)}"
            )
        for col in tactile_keys(dual_arm=dual_arm):
            col_name = parquet_column(col)
            values = [row[col_name] for row in tactile_rows]
            df.loc[ep_mask, col_name] = values
        updated_frames += len(ep_df)
        contact_keys = [k for k in tactile_keys(dual_arm=dual_arm) if k.endswith("_contact")]
        n_contact = sum(
            1
            for r in tactile_rows
            for k in contact_keys
            if r[parquet_column(k)][0] > 0.5
        )
        contact_rate_val = n_contact / max(len(tactile_rows) * len(contact_keys), 1)
        print(f"  ep {ep_idx}: frames={len(ep_df)} finger_contact_rate={contact_rate_val:.3f}")

    if not dry_run:
        df.to_parquet(parquet_path, index=False)
    return updated_frames


def process_task(
    task_root: Path,
    *,
    task_name: str,
    dexjoco_root: Path,
    dual_arm: bool,
    dry_run: bool,
    max_episodes: int = 0,
) -> None:
    info_path = task_root / "meta" / "info.json"
    modality_path = task_root / "meta" / "modality.json"
    if not info_path.exists():
        raise FileNotFoundError(info_path)

    parquet_files = sorted((task_root / "data").rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet under {task_root / 'data'}")

    print(f"[i] {task_name}: {len(parquet_files)} parquet file(s)")
    total = 0
    for pq in parquet_files:
        print(f"  -> {pq.relative_to(task_root)}")
        total += _label_parquet(
            pq,
            task_name=task_name,
            dexjoco_root=dexjoco_root,
            dual_arm=dual_arm,
            dry_run=dry_run,
            max_episodes=max_episodes,
        )

    if not dry_run:
        _patch_info_json(info_path, dual_arm=dual_arm)
        if modality_path.exists():
            _patch_modality_json(modality_path, dual_arm=dual_arm)
    print(f"[ok] {task_name}: labeled {total} frames (dry_run={dry_run})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets-root", type=Path, required=True)
    parser.add_argument("--dexjoco-root", type=Path, default=None)
    parser.add_argument("--task", action="append", default=[])
    parser.add_argument("--all-tasks", action="store_true")
    parser.add_argument(
        "--registry",
        type=Path,
        default=SCRIPT_DIR / "task_registry.yaml",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-episodes", type=int, default=0, help="Debug: cap episodes per file")
    args = parser.parse_args()

    dexjoco_root = args.dexjoco_root or PROJECT_ROOT.parent / "dexjoco"
    registry = _load_registry(args.registry)
    task_names = sorted(registry["tasks"].keys()) if args.all_tasks else args.task
    if not task_names:
        raise SystemExit("Provide --task NAME or --all-tasks")

    for task_name in task_names:
        robot_type = registry["tasks"][task_name]["robot_type"]
        dual_arm = robot_type == "bimanual"
        task_root = args.datasets_root / task_name
        process_task(
            task_root,
            task_name=task_name,
            dexjoco_root=dexjoco_root,
            dual_arm=dual_arm,
            dry_run=args.dry_run,
            max_episodes=args.max_episodes,
        )


if __name__ == "__main__":
    main()
