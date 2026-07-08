#!/usr/bin/env python3
"""Replay DexJoCo episodes in sim and export video + per-pad tactile overlay.

Use this before running generate_dexjoco_haptic_labels.py on full datasets:
  1) replay 1–2 episodes with the same tactile extraction as label generation
  2) compare dataset ego camera vs sim replay ego camera
  3) inspect force/contact traces aligned to frame index
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dexjoco_haptic_extraction import (  # noqa: E402
    extract_dexjoco_tactile_frame,
    rotvec_action_to_policy,
)
from dexjoco_tactile_schema import BIMANUAL_FORCE_KEYS, SINGLE_ARM_FORCE_KEYS, tactile_keys
from gr00t.data.dataset.lerobot_v3 import (  # noqa: E402
    load_v3_episodes_metadata,
    resolve_v3_data_parquet_path,
    resolve_v3_video_parquet_path,
)


def _load_registry(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


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


def _to_uint8_image(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img)
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        if arr.max() <= 1.0:
            arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    elif arr.shape[-1] == 4:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
    return arr


def _load_episode_actions(
    task_root: Path,
    info_meta: dict,
    episode_meta: dict,
) -> np.ndarray:
    parquet_path = resolve_v3_data_parquet_path(task_root, info_meta, episode_meta)
    df = pd.read_parquet(parquet_path)
    ep_idx = int(episode_meta["episode_index"])
    ep_df = df.loc[df["episode_index"] == ep_idx]
    if ep_df.empty:
        raise RuntimeError(f"Episode {ep_idx} not found in {parquet_path}")
    return np.stack(ep_df["action"].to_list(), axis=0)


def _load_dataset_video_frames(
    task_root: Path,
    info_meta: dict,
    episode_meta: dict,
    *,
    video_key: str,
) -> list[np.ndarray]:
    video_path = resolve_v3_video_parquet_path(task_root, info_meta, episode_meta, video_key)
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    start = int(episode_meta["dataset_from_index"])
    length = int(
        episode_meta.get(
            "length",
            int(episode_meta["dataset_to_index"]) - start,
        )
    )
    indices = list(range(start, start + length))
    try:
        from gr00t.utils.video_utils import get_frames_by_indices

        frames_arr = get_frames_by_indices(str(video_path), indices)
        return [_to_uint8_image(f) for f in frames_arr]
    except Exception as exc:
        print(f"[warn] torchcodec decode failed ({exc!r}); falling back to OpenCV")
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}") from exc
        frames: list[np.ndarray] = []
        try:
            for frame_idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError(
                        f"Could not read frame {frame_idx} from {video_path} "
                        f"(episode {episode_meta['episode_index']})"
                    ) from exc
                frames.append(frame)
        finally:
            cap.release()
        return frames


def _force_keys(*, dual_arm: bool) -> tuple[str, ...]:
    return BIMANUAL_FORCE_KEYS if dual_arm else SINGLE_ARM_FORCE_KEYS


def _draw_tactile_panel(
    forces: dict[str, float],
    *,
    dual_arm: bool,
    width: int,
    height: int,
    frame_idx: int,
    total_frames: int,
) -> np.ndarray:
    panel = np.full((height, width, 3), 32, dtype=np.uint8)
    keys = _force_keys(dual_arm=dual_arm)
    n = len(keys)
    left_pad, top_pad, bar_h, gap = 12, 28, 18, 6
    usable_h = height - top_pad - 8
    row_h = max(12, (usable_h - gap * (n - 1)) // n)
    max_force = 50.0

    cv2.putText(
        panel,
        f"tactile replay  frame {frame_idx}/{total_frames - 1}",
        (left_pad, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )

    bar_x0 = 120
    bar_x1 = width - 70
    for i, key in enumerate(keys):
        y0 = top_pad + i * (row_h + gap)
        y1 = y0 + bar_h
        val = float(forces.get(key, 0.0))
        contact = float(forces.get(f"{key}_contact", 0.0 if not key.endswith("_contact") else val))
        if key.endswith("_contact"):
            continue
        contact_key = f"{key}_contact"
        contact = float(forces.get(contact_key, 1.0 if val > 0 else 0.0))
        cv2.putText(
            panel,
            key[:12],
            (left_pad, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )
        cv2.rectangle(panel, (bar_x0, y0), (bar_x1, y1), (70, 70, 70), 1)
        fill_x = bar_x0 + int((bar_x1 - bar_x0) * min(val / max_force, 1.0))
        color = (80, 220, 120) if contact > 0.5 else (120, 120, 120)
        cv2.rectangle(panel, (bar_x0 + 1, y0 + 1), (fill_x, y1 - 1), color, -1)
        cv2.putText(
            panel,
            f"{val:4.1f}N",
            (bar_x1 + 8, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
    return panel


def _compose_frame(
    dataset_img: np.ndarray | None,
    sim_img: np.ndarray,
    forces: dict[str, float],
    *,
    dual_arm: bool,
    frame_idx: int,
    total_frames: int,
    panel_width: int = 1280,
    cam_size: int = 360,
) -> np.ndarray:
    sim_bgr = _to_uint8_image(sim_img)
    sim_bgr = cv2.resize(sim_bgr, (cam_size, cam_size), interpolation=cv2.INTER_AREA)

    if dataset_img is not None:
        ds_bgr = _to_uint8_image(dataset_img)
        ds_bgr = cv2.resize(ds_bgr, (cam_size, cam_size), interpolation=cv2.INTER_AREA)
        cv2.putText(ds_bgr, "dataset ego", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(sim_bgr, "sim replay ego", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        top = np.concatenate([ds_bgr, sim_bgr], axis=1)
    else:
        cv2.putText(sim_bgr, "sim replay ego", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        top = sim_bgr

    if top.shape[1] != panel_width:
        scale = panel_width / top.shape[1]
        top = cv2.resize(
            top,
            (panel_width, max(1, int(top.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )

    tactile_h = 220 if dual_arm else 180
    tactile = _draw_tactile_panel(
        forces,
        dual_arm=dual_arm,
        width=panel_width,
        height=tactile_h,
        frame_idx=frame_idx,
        total_frames=total_frames,
    )
    return np.concatenate([top, tactile], axis=0)


def replay_episode_with_viz(
    *,
    task_name: str,
    task_root: Path,
    dexjoco_root: Path,
    episode_meta: dict,
    camera_key: str,
    dual_arm: bool,
    output_mp4: Path,
    compare_dataset_video: bool,
    max_frames: int,
    fps: int,
) -> dict[str, float]:
    info_meta = json.loads((task_root / "meta" / "info.json").read_text(encoding="utf-8"))
    actions = _load_episode_actions(task_root, info_meta, episode_meta)
    dataset_frames: list[np.ndarray] | None = None
    if compare_dataset_video:
        video_key = f"observation.images.{camera_key}"
        dataset_frames = _load_dataset_video_frames(
            task_root, info_meta, episode_meta, video_key=video_key
        )
        if len(dataset_frames) != len(actions):
            raise RuntimeError(
                f"Episode {episode_meta['episode_index']}: dataset frames {len(dataset_frames)} "
                f"!= actions {len(actions)}"
            )

    total = len(actions)
    if max_frames > 0:
        total = min(total, max_frames)
        actions = actions[:total]
        if dataset_frames is not None:
            dataset_frames = dataset_frames[:total]

    _ensure_dexjoco_import(dexjoco_root)
    from dexjoco.tasks import CONFIG_MAPPING

    ep_idx = int(episode_meta["episode_index"])
    config = CONFIG_MAPPING[task_name]()
    env = config.get_environment(
        policy_mode=True,
        render_mode="rgb_array",
        randomize=False,
        seed=ep_idx,
        randomize_dynamics=False,
    )

    force_keys = _force_keys(dual_arm=dual_arm)
    peak_forces = {k: 0.0 for k in force_keys}
    contact_steps = 0

    output_mp4.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    try:
        obs, _ = env.reset(seed=ep_idx)
        model, data = _get_mujoco_model_data(env)

        for t in range(total):
            tactile = extract_dexjoco_tactile_frame(model, data, dual_arm=dual_arm)
            forces = tactile.values
            for k in force_keys:
                peak_forces[k] = max(peak_forces[k], float(forces.get(k, 0.0)))
            if any(float(forces.get(f"{k}_contact", 0.0)) > 0.5 for k in force_keys):
                contact_steps += 1

            sim_img = obs.get(camera_key)
            if sim_img is None:
                raise KeyError(
                    f"Camera '{camera_key}' missing in sim obs keys: {sorted(obs.keys())}"
                )
            ds_img = dataset_frames[t] if dataset_frames is not None else None
            composed = _compose_frame(
                ds_img,
                sim_img,
                forces,
                dual_arm=dual_arm,
                frame_idx=t,
                total_frames=total,
            )
            if writer is None:
                h, w = composed.shape[:2]
                writer = cv2.VideoWriter(
                    str(output_mp4),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (w, h),
                )
            writer.write(composed)

            policy_action = rotvec_action_to_policy(actions[t], dual_arm=dual_arm)
            obs, _, _, _, _ = env.step(policy_action)
            model, data = _get_mujoco_model_data(env)
    finally:
        if writer is not None:
            writer.release()
        env.close()

    contact_rate = contact_steps / max(total * len(force_keys), 1)
    return {
        "frames": float(total),
        "contact_step_rate": contact_rate,
        **{f"peak_{k}": v for k, v in peak_forces.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets-root", type=Path, required=True)
    parser.add_argument("--dexjoco-root", type=Path, default=None)
    parser.add_argument("--task", required=True)
    parser.add_argument(
        "--episodes",
        default="0,1",
        help="Comma-separated episode indices (default: 0,1)",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--compare-dataset-video",
        action="store_true",
        help="Show dataset ego camera beside sim replay for alignment check",
    )
    parser.add_argument("--max-frames", type=int, default=0, help="Cap frames per episode (0=all)")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--registry",
        type=Path,
        default=SCRIPT_DIR / "task_registry.yaml",
    )
    args = parser.parse_args()

    dexjoco_root = args.dexjoco_root or PROJECT_ROOT.parent / "dexjoco"
    registry = _load_registry(args.registry)
    if args.task not in registry["tasks"]:
        raise SystemExit(f"Unknown task {args.task}")
    task_cfg = registry["tasks"][args.task]
    dual_arm = task_cfg["robot_type"] == "bimanual"
    camera_key = task_cfg["camera_mapping"]["base"]

    task_root = args.datasets_root / args.task
    episodes_meta = load_v3_episodes_metadata(task_root)
    wanted = [int(x.strip()) for x in args.episodes.split(",") if x.strip()]
    meta_by_ep = {int(m["episode_index"]): m for m in episodes_meta}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for ep_idx in wanted:
        if ep_idx not in meta_by_ep:
            raise SystemExit(f"Episode {ep_idx} not found under {task_root}")
        out_mp4 = args.output_dir / f"{args.task}_ep{ep_idx:03d}_tactile_replay.mp4"
        print(f"[i] Replaying {args.task} ep {ep_idx} -> {out_mp4}")
        stats = replay_episode_with_viz(
            task_name=args.task,
            task_root=task_root,
            dexjoco_root=dexjoco_root,
            episode_meta=meta_by_ep[ep_idx],
            camera_key=camera_key,
            dual_arm=dual_arm,
            output_mp4=out_mp4,
            compare_dataset_video=args.compare_dataset_video,
            max_frames=args.max_frames,
            fps=args.fps,
        )
        print(
            f"[ok] ep {ep_idx}: frames={int(stats['frames'])} "
            f"any_pad_contact_rate={stats['contact_step_rate']:.3f}"
        )
        summary.append({"episode_index": ep_idx, "video": str(out_mp4), **stats})

    summary_path = args.output_dir / f"{args.task}_tactile_replay_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"[ok] summary -> {summary_path}")


if __name__ == "__main__":
    main()
