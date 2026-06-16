#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate arm/base 2D trajectory labels for RoboCasa365 LeRobot datasets.

Replays each episode from ``extras/episode_*/states.npz`` (deterministic state playback),
projects end-effector and mobile-base reference points into each dataset camera using
MuJoCo ``cam_xpos`` / ``cam_xmat`` / ``cam_fovy``, and writes labels into parquet.

Projection follows ``annotate_sim.py``:

- **Extrinsic cameras** (agentview): current EEF/base at the anchor frame.
- **Eye-in-hand arm**: current EEF is nearly static in ego view; we additionally store
  **future** EEF world positions projected with the **anchor frame's** camera so wrist
  trajectories are dynamic (``trajectory.arm_future_uv.*``).

Parallelism: one MuJoCo env per worker process; episodes are sharded across workers.

Example:
  MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa \\
  python examples/RoboCasa365/scripts/generate_trajectory_labels.py \\
    --dataset /path/to/lerobot \\
    --output-dataset /path/to/lerobot_with_traj \\
    --num-workers 4 \\
    --robocasa-root /path/to/external_dependencies/robocasa365
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
from trajectory_projection import (  # noqa: E402
    DEFAULT_ARM_CAMERAS,
    DEFAULT_BASE_CAMERAS,
    DEFAULT_CAMERA_NAMES,
    DEFAULT_FUTURE_LENGTH,
    EYE_IN_HAND_CAMERA,
    arm_uv_variance_diagnostic,
    get_arm_base_world_points,
    project_frame_annotate_sim,
)


def _trajectory_feature_template(future_length: int) -> dict[str, dict[str, Any]]:
    return {
        "trajectory.arm_uv.{camera}": {
            "dtype": "float32",
            "shape": [2],
            "names": ["u", "v"],
        },
        "trajectory.base_uv.{camera}": {
            "dtype": "float32",
            "shape": [2],
            "names": ["u", "v"],
        },
        "trajectory.arm_visible.{camera}": {
            "dtype": "bool",
            "shape": [1],
            "names": None,
        },
        "trajectory.base_visible.{camera}": {
            "dtype": "bool",
            "shape": [1],
            "names": None,
        },
        "trajectory.arm_future_uv.{camera}": {
            "dtype": "float32",
            "shape": [future_length, 2],
            "names": ["step", "uv"],
        },
        "trajectory.arm_future_visible.{camera}": {
            "dtype": "bool",
            "shape": [future_length],
            "names": None,
        },
        "trajectory.base_future_uv.{camera}": {
            "dtype": "float32",
            "shape": [future_length, 2],
            "names": ["step", "uv"],
        },
        "trajectory.base_future_visible.{camera}": {
            "dtype": "bool",
            "shape": [future_length],
            "names": None,
        },
    }


def _trajectory_column_names(camera_names: tuple[str, ...], future_length: int) -> list[str]:
    cols: list[str] = []
    for camera in camera_names:
        cols.extend(
            [
                f"trajectory.arm_uv.{camera}",
                f"trajectory.base_uv.{camera}",
                f"trajectory.arm_visible.{camera}",
                f"trajectory.base_visible.{camera}",
                f"trajectory.arm_future_uv.{camera}",
                f"trajectory.arm_future_visible.{camera}",
                f"trajectory.base_future_uv.{camera}",
                f"trajectory.base_future_visible.{camera}",
            ]
        )
    return cols


def _patch_info_json(info_path: Path, camera_names: tuple[str, ...], future_length: int) -> None:
    info = json.loads(info_path.read_text())
    features = dict(info.get("features", {}))
    template = _trajectory_feature_template(future_length)
    for camera in camera_names:
        for template_key, spec in template.items():
            key = template_key.format(camera=camera)
            features[key] = spec
    info["features"] = features
    info_path.write_text(json.dumps(info, indent=4) + "\n")


def _patch_modality_json(modality_path: Path, camera_names: tuple[str, ...]) -> None:
    modality = json.loads(modality_path.read_text())
    traj_modality = modality.setdefault("trajectory", {})
    for camera in camera_names:
        traj_modality[f"arm_uv_{camera}"] = {
            "original_key": f"trajectory.arm_uv.{camera}",
        }
        traj_modality[f"base_uv_{camera}"] = {
            "original_key": f"trajectory.base_uv.{camera}",
        }
        traj_modality[f"arm_visible_{camera}"] = {
            "original_key": f"trajectory.arm_visible.{camera}",
        }
        traj_modality[f"base_visible_{camera}"] = {
            "original_key": f"trajectory.base_visible.{camera}",
        }
        traj_modality[f"arm_future_uv_{camera}"] = {
            "original_key": f"trajectory.arm_future_uv.{camera}",
        }
        traj_modality[f"arm_future_visible_{camera}"] = {
            "original_key": f"trajectory.arm_future_visible.{camera}",
        }
        traj_modality[f"base_future_uv_{camera}"] = {
            "original_key": f"trajectory.base_future_uv.{camera}",
        }
        traj_modality[f"base_future_visible_{camera}"] = {
            "original_key": f"trajectory.base_future_visible.{camera}",
        }
    modality_path.write_text(json.dumps(modality, indent=4) + "\n")


def _prepare_output_dataset(
    input_dataset: Path,
    output_dataset: Path,
    camera_names: tuple[str, ...],
    future_length: int,
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
    _patch_info_json(output_dataset / "meta" / "info.json", camera_names, future_length)
    _patch_modality_json(output_dataset / "meta" / "modality.json", camera_names)


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


def _serialize_traj_column(col_name: str, values: list) -> list:
    """Convert trajectory columns to parquet-friendly Python lists."""
    if "future_uv" in col_name or "future_visible" in col_name:
        out = []
        for v in values:
            if isinstance(v, np.ndarray):
                out.append(v.tolist())
            else:
                out.append(v)
        return out
    return values


def _label_episode(
    *,
    input_dataset: Path,
    output_dataset: Path,
    episode_index: int,
    all_cameras: tuple[str, ...],
    arm_cameras: tuple[str, ...],
    base_cameras: tuple[str, ...],
    image_width: int,
    image_height: int,
    future_length: int,
    robocasa_root: Path,
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

        arm_world_all = np.zeros((len(states), 3), dtype=np.float64)
        base_world_all = np.zeros((len(states), 3), dtype=np.float64)
        for frame_idx, state in enumerate(states):
            reset_to(env, {"states": state})
            arm_world, base_world = get_arm_base_world_points(env.sim, env.robots[0])
            arm_world_all[frame_idx] = arm_world
            base_world_all[frame_idx] = base_world

        traj_cols = {
            name: [] for name in _trajectory_column_names(all_cameras, future_length)
        }
        arm_uv_left: list[np.ndarray] = []
        arm_vis_left: list[bool] = []
        arm_uv_ego: list[np.ndarray] = []
        arm_vis_ego: list[bool] = []

        for frame_idx, state in enumerate(states):
            reset_to(env, {"states": state})
            projected = project_frame_annotate_sim(
                env.sim,
                anchor_frame_idx=frame_idx,
                arm_world_all=arm_world_all,
                base_world_all=base_world_all,
                width=image_width,
                height=image_height,
                arm_cameras=arm_cameras,
                base_cameras=base_cameras,
                all_cameras=all_cameras,
                future_length=future_length,
            )
            for camera in all_cameras:
                arm = projected["arm"][camera]
                base = projected["base"][camera]
                arm_future = projected["arm_future"][camera]
                base_future = projected["base_future"][camera]
                traj_cols[f"trajectory.arm_uv.{camera}"].append(arm.uv.astype(np.float32))
                traj_cols[f"trajectory.base_uv.{camera}"].append(base.uv.astype(np.float32))
                traj_cols[f"trajectory.arm_visible.{camera}"].append(np.array([arm.visible]))
                traj_cols[f"trajectory.base_visible.{camera}"].append(np.array([base.visible]))
                traj_cols[f"trajectory.arm_future_uv.{camera}"].append(
                    arm_future.uv.astype(np.float32)
                )
                traj_cols[f"trajectory.arm_future_visible.{camera}"].append(
                    arm_future.visible.astype(bool)
                )
                traj_cols[f"trajectory.base_future_uv.{camera}"].append(
                    base_future.uv.astype(np.float32)
                )
                traj_cols[f"trajectory.base_future_visible.{camera}"].append(
                    base_future.visible.astype(bool)
                )

            primary = arm_cameras[0]
            arm_uv_left.append(projected["arm"][primary].uv.copy())
            arm_vis_left.append(bool(projected["arm"][primary].visible))
            if EYE_IN_HAND_CAMERA in arm_cameras:
                fut = projected["arm_future"][EYE_IN_HAND_CAMERA]
                visible_slots = fut.visible
                if np.any(visible_slots):
                    first_slot = int(np.argmax(visible_slots))
                    arm_uv_ego.append(fut.uv[first_slot].copy())
                    arm_vis_ego.append(True)
                else:
                    arm_uv_ego.append(projected["arm"][EYE_IN_HAND_CAMERA].uv.copy())
                    arm_vis_ego.append(bool(projected["arm"][EYE_IN_HAND_CAMERA].visible))

        src_parquet = _episode_parquet_path(input_dataset, episode_index, info)
        dst_parquet = _episode_parquet_path(output_dataset, episode_index, info)
        df = pd.read_parquet(src_parquet)
        if len(df) != len(states):
            raise ValueError(
                f"Episode {episode_index}: parquet rows={len(df)} != states={len(states)}"
            )
        for col, values in traj_cols.items():
            df[col] = _serialize_traj_column(col, values)

        dst_parquet.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(dst_parquet, index=False)

        primary_cam = arm_cameras[0]
        visible_rate = float(
            np.mean([v[0] for v in traj_cols[f"trajectory.arm_visible.{primary_cam}"]])
        )
        arm_std_primary = arm_uv_variance_diagnostic(
            np.stack(arm_uv_left),
            np.array(arm_vis_left, dtype=bool),
        )
        ego_arm_std = 0.0
        ego_future_visible_rate = 0.0
        if EYE_IN_HAND_CAMERA in arm_cameras and arm_uv_ego:
            ego_arm_std = arm_uv_variance_diagnostic(
                np.stack(arm_uv_ego),
                np.array(arm_vis_ego, dtype=bool),
            )
            ego_future_visible_rate = float(
                np.mean(
                    [
                        np.any(v)
                        for v in traj_cols[
                            f"trajectory.arm_future_visible.{EYE_IN_HAND_CAMERA}"
                        ]
                    ]
                )
            )
        return {
            "episode_index": episode_index,
            "num_frames": len(states),
            "arm_cameras": list(arm_cameras),
            "base_cameras": list(base_cameras),
            "future_length": future_length,
            "projection_method": "annotate_sim_anchored_future",
            "arm_visible_rate_primary_cam": visible_rate,
            "arm_uv_std_primary_cam": arm_std_primary,
            "eye_in_hand_arm_future_visible_rate": ego_future_visible_rate,
            "eye_in_hand_arm_future_uv_std": ego_arm_std,
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
        all_cameras=tuple(payload["all_cameras"]),
        arm_cameras=tuple(payload["arm_cameras"]),
        base_cameras=tuple(payload["base_cameras"]),
        image_width=int(payload["image_width"]),
        image_height=int(payload["image_height"]),
        future_length=int(payload["future_length"]),
        robocasa_root=Path(payload["robocasa_root"]),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Input LeRobot dataset root (contains meta/, data/, extras/).",
    )
    parser.add_argument(
        "--output-dataset",
        type=Path,
        default=None,
        help="Output dataset root. Defaults to --dataset (in-place parquet update).",
    )
    parser.add_argument(
        "--robocasa-root",
        type=Path,
        default=Path("/HOME/sysu_xdliang/sysu_xdliang_1/HDD_POOL/linmin/RLDX-1/external_dependencies/robocasa365"),
        help="Path to robocasa365 package root (contains robocasa/).",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=None,
        help="Optional interpreter hint printed in logs (script runs in current interpreter).",
    )
    parser.add_argument(
        "--cameras",
        nargs="+",
        default=list(DEFAULT_CAMERA_NAMES),
        help="All MuJoCo camera columns to write in parquet (superset of arm/base cameras).",
    )
    parser.add_argument(
        "--arm-cameras",
        nargs="+",
        default=list(DEFAULT_ARM_CAMERAS),
        help="Cameras used for arm/EEF trajectory (includes eye_in_hand with future anchoring).",
    )
    parser.add_argument(
        "--base-cameras",
        nargs="+",
        default=list(DEFAULT_BASE_CAMERAS),
        help="Cameras used for mobile-base trajectory.",
    )
    parser.add_argument(
        "--future-length",
        type=int,
        default=DEFAULT_FUTURE_LENGTH,
        help="Future horizon (frames) for anchored ego-view arm/base trails.",
    )
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--episode-start", type=int, default=0)
    parser.add_argument("--episode-end", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print planned episodes; do not replay or write parquet.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dataset = args.dataset.resolve()
    output_dataset = (args.output_dataset or args.dataset).resolve()
    all_cameras = tuple(args.cameras)
    arm_cameras = tuple(args.arm_cameras)
    base_cameras = tuple(args.base_cameras)
    future_length = int(args.future_length)
    if not set(arm_cameras).issubset(all_cameras):
        raise ValueError("--arm-cameras must be a subset of --cameras")
    if not set(base_cameras).issubset(all_cameras):
        raise ValueError("--base-cameras must be a subset of --cameras")
    if future_length <= 0:
        raise ValueError("--future-length must be positive")

    if not (input_dataset / "meta" / "info.json").is_file():
        raise FileNotFoundError(f"Missing meta/info.json under {input_dataset}")
    if not (input_dataset / "extras").is_dir():
        raise FileNotFoundError(f"Missing extras/ under {input_dataset} (required for replay).")

    info = _load_info(input_dataset)
    total_episodes = int(info["total_episodes"])
    image_width = int(info["features"][f"observation.images.{all_cameras[0]}"]["shape"][1])
    image_height = int(info["features"][f"observation.images.{all_cameras[0]}"]["shape"][0])

    episode_end = args.episode_end if args.episode_end is not None else total_episodes
    episode_indices = list(range(args.episode_start, min(episode_end, total_episodes)))
    if not episode_indices:
        raise ValueError("No episodes selected.")

    print(f"[i] input dataset:  {input_dataset}")
    print(f"[i] output dataset: {output_dataset}")
    print(f"[i] episodes: {episode_indices[0]}..{episode_indices[-1]} ({len(episode_indices)} total)")
    print(f"[i] all cameras: {all_cameras}")
    print(f"[i] arm cameras: {arm_cameras} (annotate_sim anchored future on eye_in_hand)")
    print(f"[i] base cameras: {base_cameras}")
    print(f"[i] future length: {future_length}")
    print(f"[i] image size: {image_width}x{image_height}")
    print(f"[i] workers: {args.num_workers}")
    print(f"[i] robocasa root: {args.robocasa_root}")

    if output_dataset != input_dataset:
        print("[i] preparing output dataset tree + meta patches...")
        _prepare_output_dataset(
            input_dataset, output_dataset, all_cameras, future_length, args.overwrite
        )
    else:
        print("[i] in-place mode: patching meta/info.json + meta/modality.json")
        _patch_info_json(output_dataset / "meta" / "info.json", all_cameras, future_length)
        _patch_modality_json(output_dataset / "meta" / "modality.json", all_cameras)

    if args.dry_run:
        print("[i] dry-run complete.")
        return

    payloads = [
        {
            "input_dataset": str(input_dataset),
            "output_dataset": str(output_dataset),
            "episode_index": ep,
            "all_cameras": list(all_cameras),
            "arm_cameras": list(arm_cameras),
            "base_cameras": list(base_cameras),
            "image_width": image_width,
            "image_height": image_height,
            "future_length": future_length,
            "robocasa_root": str(args.robocasa_root.resolve()),
        }
        for ep in episode_indices
    ]

    results: list[dict[str, Any]] = []
    if args.num_workers <= 1:
        _worker_init(str(args.robocasa_root.resolve()))
        for payload in payloads:
            print(f"[i] labeling episode {payload['episode_index']}...")
            results.append(_worker_label_episode(payload))
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
                try:
                    result = fut.result()
                    results.append(result)
                    print(
                        f"[i] episode {ep}: frames={result['num_frames']} "
                        f"arm_visible(primary)={result['arm_visible_rate_primary_cam']:.2%} "
                        f"arm_uv_std(primary)={result['arm_uv_std_primary_cam']:.4f} "
                        f"ego_future_std={result['eye_in_hand_arm_future_uv_std']:.4f}"
                    )
                except Exception as exc:
                    print(f"[e] episode {ep} failed: {exc}")
                    raise

    results.sort(key=lambda r: r["episode_index"])
    summary_path = output_dataset / "meta" / "trajectory_label_summary.json"
    summary_path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"[i] wrote {len(results)} labeled episodes")
    print(f"[i] summary: {summary_path}")


if __name__ == "__main__":
    main()
