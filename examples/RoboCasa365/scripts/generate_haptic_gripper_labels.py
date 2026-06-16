#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate real-sensor-aligned 1D tactile GT for RoboCasa365 LeRobot datasets.

Replays each episode from ``extras/episode_*/states.npz`` and writes:
``tactile.left``, ``tactile.right`` (pad normal force, N), ``tactile.contact`` (thresholded).

Example:
  MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 \\
  python examples/RoboCasa365/scripts/generate_haptic_gripper_labels.py \\
    --dataset /path/to/lerobot \\
    --output-dataset /path/to/lerobot_haptic \\
    --num-workers 4
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from haptic_gripper_extraction import (  # noqa: E402
    DEFAULT_FORCE_MAX_N,
    DEFAULT_FORCE_THRESHOLD_N,
    RealSensorTactileConfig,
    TACTILE_COLUMN_NAMES,
    TACTILE_FEATURE_TEMPLATE,
    extract_gripper_tactile_frame,
)


def _patch_info_json(info_path: Path) -> None:
    info = json.loads(info_path.read_text())
    features = dict(info.get("features", {}))
    features.update(TACTILE_FEATURE_TEMPLATE)
    info["features"] = features
    info_path.write_text(json.dumps(info, indent=4) + "\n")


def _patch_modality_json(modality_path: Path) -> None:
    modality = json.loads(modality_path.read_text())
    tactile_modality = modality.setdefault("tactile", {})
    tactile_modality["left"] = {"original_key": "tactile.left"}
    tactile_modality["right"] = {"original_key": "tactile.right"}
    tactile_modality["contact"] = {"original_key": "tactile.contact"}
    modality_path.write_text(json.dumps(modality, indent=4) + "\n")


def _prepare_output_dataset(
    input_dataset: Path,
    output_dataset: Path,
    overwrite: bool,
) -> None:
    if output_dataset.resolve() == input_dataset.resolve():
        return
    if output_dataset.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output dataset exists: {output_dataset}. Pass --overwrite to replace."
            )
        shutil.rmtree(output_dataset)
    shutil.copytree(
        input_dataset,
        output_dataset,
        ignore=shutil.ignore_patterns("data", "videos"),
    )
    (output_dataset / "data").mkdir(parents=True, exist_ok=True)
    videos_src = input_dataset / "videos"
    if videos_src.is_dir():
        if (output_dataset / "videos").exists():
            shutil.rmtree(output_dataset / "videos")
        shutil.copytree(videos_src, output_dataset / "videos", symlinks=True)
    _patch_info_json(output_dataset / "meta" / "info.json")
    _patch_modality_json(output_dataset / "meta" / "modality.json")


def _load_info(dataset: Path) -> dict[str, Any]:
    return json.loads((dataset / "meta" / "info.json").read_text())


def _episode_parquet_path(dataset: Path, episode_index: int, info: dict[str, Any]) -> Path:
    chunk = episode_index // int(info["chunks_size"])
    rel = info["data_path"].format(episode_chunk=chunk, episode_index=episode_index)
    return dataset / rel


def _create_env(dataset: Path, robocasa_root: Path):
    if str(robocasa_root) not in sys.path:
        sys.path.insert(0, str(robocasa_root))
    import robosuite  # noqa: WPS433
    import robocasa.utils.lerobot_utils as LU  # noqa: WPS433

    env_meta = LU.get_env_metadata(dataset)
    env_kwargs = dict(env_meta["env_kwargs"])
    env_kwargs["env_name"] = env_meta["env_name"]
    env_kwargs["has_renderer"] = False
    env_kwargs["has_offscreen_renderer"] = False
    env_kwargs["use_camera_obs"] = False
    env = robosuite.make(**env_kwargs)
    return env, LU


def _label_episode(
    *,
    input_dataset: Path,
    output_dataset: Path,
    episode_index: int,
    robocasa_root: Path,
    tactile_cfg: RealSensorTactileConfig,
) -> dict[str, Any]:
    from robocasa.scripts.dataset_scripts.playback_dataset import reset_to  # noqa: WPS433

    env, LU = _create_env(input_dataset, robocasa_root)
    try:
        info = _load_info(input_dataset)
        states = LU.get_episode_states(input_dataset, episode_index)
        initial_state = {
            "states": states[0],
            "model": LU.get_episode_model_xml(input_dataset, episode_index),
            "ep_meta": json.dumps(LU.get_episode_meta(input_dataset, episode_index)),
        }
        reset_to(env, initial_state)

        tactile_cols = {name: [] for name in TACTILE_COLUMN_NAMES}
        contact_frames = 0
        max_force = 0.0
        for state in states:
            reset_to(env, {"states": state})
            frame = extract_gripper_tactile_frame(env, tactile_cfg)
            for col, value in frame.as_parquet_values().items():
                tactile_cols[col].append(value)
            if frame.in_contact:
                contact_frames += 1
            max_force = max(max_force, frame.left_force, frame.right_force)

        src_parquet = _episode_parquet_path(input_dataset, episode_index, info)
        dst_parquet = _episode_parquet_path(output_dataset, episode_index, info)
        df = pd.read_parquet(src_parquet)
        if len(df) != len(states):
            raise ValueError(
                f"Episode {episode_index}: parquet rows={len(df)} != states={len(states)}"
            )
        for col, values in tactile_cols.items():
            df[col] = values

        dst_parquet.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(dst_parquet, index=False)

        return {
            "episode_index": episode_index,
            "num_frames": len(states),
            "contact_frame_rate": contact_frames / max(len(states), 1),
            "max_force_n": max_force,
            "parquet_path": str(dst_parquet),
        }
    finally:
        env.close()


def _worker_init(robocasa_root: str) -> None:
    root = Path(robocasa_root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def _worker_label_episode(payload: dict[str, Any]) -> dict[str, Any]:
    return _label_episode(
        input_dataset=Path(payload["input_dataset"]),
        output_dataset=Path(payload["output_dataset"]),
        episode_index=int(payload["episode_index"]),
        robocasa_root=Path(payload["robocasa_root"]),
        tactile_cfg=RealSensorTactileConfig(
            force_threshold_n=float(payload["force_threshold_n"]),
            force_max_n=float(payload["force_max_n"]),
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dataset", type=Path, default=None)
    parser.add_argument(
        "--robocasa-root",
        type=Path,
        default=Path(
            "/HOME/sysu_xdliang/sysu_xdliang_1/HDD_POOL/linmin/RLDX-1/external_dependencies/robocasa365"
        ),
    )
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--episode-start", type=int, default=0)
    parser.add_argument("--episode-end", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--force-threshold-n",
        type=float,
        default=DEFAULT_FORCE_THRESHOLD_N,
        help="Forces below this (N) are zeroed; contact uses the same threshold.",
    )
    parser.add_argument(
        "--force-max-n",
        type=float,
        default=DEFAULT_FORCE_MAX_N,
        help="Clip per-finger force to this maximum (N).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dataset = args.dataset.resolve()
    output_dataset = (args.output_dataset or args.dataset).resolve()

    if not (input_dataset / "meta" / "info.json").is_file():
        raise FileNotFoundError(f"Missing meta/info.json under {input_dataset}")
    if not (input_dataset / "extras").is_dir():
        raise FileNotFoundError(f"Missing extras/ under {input_dataset} (required for replay).")

    info = _load_info(input_dataset)
    total_episodes = int(info["total_episodes"])
    episode_end = args.episode_end if args.episode_end is not None else total_episodes
    episode_indices = list(range(args.episode_start, min(episode_end, total_episodes)))
    if not episode_indices:
        raise ValueError("No episodes selected.")

    print(f"[i] input dataset:  {input_dataset}")
    print(f"[i] output dataset: {output_dataset}")
    print(f"[i] episodes: {episode_indices[0]}..{episode_indices[-1]} ({len(episode_indices)} total)")
    print(f"[i] workers: {args.num_workers}")
    print(f"[i] robocasa root: {args.robocasa_root}")
    print(f"[i] columns: {list(TACTILE_COLUMN_NAMES)}")
    print(
        f"[i] force threshold={args.force_threshold_n}N max={args.force_max_n}N "
        "(pad-only, real-sensor profile)"
    )

    if output_dataset != input_dataset:
        print("[i] preparing output dataset tree + meta patches...")
        _prepare_output_dataset(input_dataset, output_dataset, args.overwrite)
    else:
        print("[i] in-place mode: patching meta/info.json + meta/modality.json")
        _patch_info_json(output_dataset / "meta" / "info.json")
        _patch_modality_json(output_dataset / "meta" / "modality.json")

    if args.dry_run:
        print("[i] dry-run complete.")
        return

    payloads = [
        {
            "input_dataset": str(input_dataset),
            "output_dataset": str(output_dataset),
            "episode_index": ep,
            "robocasa_root": str(args.robocasa_root.resolve()),
            "force_threshold_n": args.force_threshold_n,
            "force_max_n": args.force_max_n,
        }
        for ep in episode_indices
    ]

    results: list[dict[str, Any]] = []
    if args.num_workers <= 1:
        _worker_init(str(args.robocasa_root.resolve()))
        for payload in payloads:
            print(f"[i] labeling episode {payload['episode_index']}...")
            result = _worker_label_episode(payload)
            results.append(result)
            print(
                f"[i] episode {result['episode_index']}: frames={result['num_frames']} "
                f"contact_rate={result['contact_frame_rate']:.2%} "
                f"max_force={result['max_force_n']:.3f}N"
            )
    else:
        with ProcessPoolExecutor(
            max_workers=args.num_workers,
            initializer=_worker_init,
            initargs=(str(args.robocasa_root.resolve()),),
        ) as pool:
            futures = {pool.submit(_worker_label_episode, payload): payload for payload in payloads}
            for fut in as_completed(futures):
                payload = futures[fut]
                ep = payload["episode_index"]
                result = fut.result()
                results.append(result)
                print(
                    f"[i] episode {ep}: frames={result['num_frames']} "
                    f"contact_rate={result['contact_frame_rate']:.2%} "
                    f"max_force={result['max_force_n']:.3f}N"
                )

    results.sort(key=lambda r: r["episode_index"])
    summary_path = output_dataset / "meta" / "haptic_gripper_label_summary.json"
    summary_path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"[i] wrote {len(results)} labeled episodes")
    print(f"[i] summary: {summary_path}")


if __name__ == "__main__":
    main()
