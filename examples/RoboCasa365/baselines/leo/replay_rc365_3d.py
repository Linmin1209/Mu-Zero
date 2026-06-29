#!/usr/bin/env python3
"""Replay RoboCasa365 MuJoCo states to export depth, camera params, and scene point clouds.

Requires lerobot ``extras/episode_*/states.npz`` (standard RC365 export).
Outputs per-frame bundles under ``data/leo_3d_cache/`` for LEO 3D branch training.

Usage:
  # Smoke: one task, 2 episodes, stride 4
  python replay_rc365_3d.py --tasks NavigateKitchen --max-episodes-per-task 2 --stride 4

  # Full target50 (slow; run on GPU node with MUJOCO_GL=egl)
  python replay_rc365_3d.py --from-manifest data/manifest_target50.jsonl
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
RC365_DIR = SCRIPT_DIR.parent.parent
PROJECT_ROOT = RC365_DIR.parent.parent
RC365_SCRIPTS = RC365_DIR / "scripts"

sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(RC365_SCRIPTS))

from leo_3d_utils import (  # noqa: E402
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    DEFAULT_CAMERAS,
    fuse_scene_pointcloud,
    pcd_frame_dir,
    save_frame_3d_bundle,
    unproject_depth_to_world,
)

logger = logging.getLogger(__name__)


def load_episode_states(lerobot_root: Path, episode_idx: int) -> np.ndarray:
    states_path = lerobot_root / "extras" / f"episode_{episode_idx:06d}" / "states.npz"
    if not states_path.is_file():
        raise FileNotFoundError(f"Missing states: {states_path}")
    return np.load(states_path)["states"]


def load_episode_model_xml(lerobot_root: Path, episode_idx: int) -> str:
    xml_path = lerobot_root / "extras" / f"episode_{episode_idx:06d}" / "model.xml.gz"
    if not xml_path.is_file():
        raise FileNotFoundError(f"Missing model xml: {xml_path}")
    with gzip.open(xml_path, "rb") as f:
        return f.read().decode("utf-8")


def load_episode_ep_meta(lerobot_root: Path, episode_idx: int) -> dict:
    meta_path = lerobot_root / "extras" / f"episode_{episode_idx:06d}" / "ep_meta.json"
    if not meta_path.is_file():
        return {}
    return json.loads(meta_path.read_text(encoding="utf-8"))


def create_replay_env(camera_names: list[str]):
    import robocasa  # noqa: F401
    import robosuite

    n = len(camera_names)
    return robosuite.make(
        env_name="Kitchen",
        robots="PandaOmron",
        has_renderer=False,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_camera_obs=True,
        camera_names=camera_names,
        camera_heights=CAMERA_HEIGHT,
        camera_widths=CAMERA_WIDTH,
        camera_depths=[True] * n,
    )


def reset_episode_env(env, lerobot_root: Path, episode_idx: int) -> None:
    states = load_episode_states(lerobot_root, episode_idx)
    model_xml = load_episode_model_xml(lerobot_root, episode_idx)
    ep_meta = load_episode_ep_meta(lerobot_root, episode_idx)

    if hasattr(env, "set_ep_meta"):
        env.set_ep_meta(ep_meta)
    elif hasattr(env, "set_attrs_from_ep_meta"):
        env.set_attrs_from_ep_meta(ep_meta)

    env.reset()
    xml = env.edit_model_xml(model_xml)
    env.reset_from_xml_string(xml)
    env.sim.reset()
    env.sim.set_state_from_flattened(states[0])
    env.sim.forward()
    if hasattr(env, "update_sites"):
        env.update_sites()
    if hasattr(env, "update_state"):
        env.update_state()
    return states


def render_rgb_depth(sim, camera_name: str) -> tuple[np.ndarray, np.ndarray]:
    rgb, depth = sim.render(
        camera_name=camera_name,
        width=CAMERA_WIDTH,
        height=CAMERA_HEIGHT,
        depth=True,
    )
    rgb = np.asarray(rgb)[::-1].copy()
    depth = np.asarray(depth)[::-1].copy()
    return rgb, depth


def replay_episode(
    env,
    *,
    task: str,
    lerobot_root: Path,
    episode_idx: int,
    pcd_root: Path,
    camera_names: list[str],
    stride: int,
    num_points: int,
    frame_indices: list[int] | None = None,
    save_depth_maps: bool = True,
    depth_pixel_stride: int = 2,
) -> int:
    from robosuite.utils.camera_utils import (
        get_camera_extrinsic_matrix,
        get_camera_intrinsic_matrix,
    )

    states = reset_episode_env(env, lerobot_root, episode_idx)
    n_frames = states.shape[0]
    if frame_indices is None:
        frame_indices = list(range(0, n_frames, stride))

    saved = 0
    for frame_idx in frame_indices:
        out_dir = pcd_frame_dir(pcd_root, task, episode_idx, frame_idx)
        if (out_dir / "scene_pcd.npz").is_file():
            saved += 1
            continue

        env.sim.set_state_from_flattened(states[frame_idx])
        env.sim.forward()

        view_chunks: list[np.ndarray] = []
        cameras_meta: dict[str, dict] = {}
        depths_out: dict[str, np.ndarray] = {}

        for cam_name in camera_names:
            rgb, depth = render_rgb_depth(env.sim, cam_name)
            intrinsic = get_camera_intrinsic_matrix(
                env.sim, cam_name, CAMERA_HEIGHT, CAMERA_WIDTH
            )
            extrinsic = get_camera_extrinsic_matrix(env.sim, cam_name)
            cam_id = env.sim.model.camera_name2id(cam_name)
            cam_pos = env.sim.data.cam_xpos[cam_id].copy().tolist()

            pts = unproject_depth_to_world(
                depth, rgb, intrinsic, extrinsic, stride=depth_pixel_stride
            )
            view_chunks.append(pts)
            cameras_meta[cam_name] = {
                "intrinsic": np.asarray(intrinsic).tolist(),
                "extrinsic_world_to_cam": np.asarray(extrinsic).tolist(),
                "camera_pos_world": cam_pos,
                "width": CAMERA_WIDTH,
                "height": CAMERA_HEIGHT,
            }
            if save_depth_maps:
                depths_out[cam_name] = depth

        scene_pcd, obj_loc = fuse_scene_pointcloud(
            view_chunks, num_points=num_points, seed=episode_idx * 100000 + frame_idx
        )

        anchor_pos = None
        anchor_quat = None
        try:
            grip_site = None
            for candidate in ["gripper0_right_grip_site", "robot0_grip_site"]:
                if candidate in env.sim.model.site_names:
                    grip_site = candidate
                    break
            if grip_site:
                sid = env.sim.model.site_name2id(grip_site)
                anchor_pos = env.sim.data.site_xpos[sid].copy()
        except Exception:
            pass

        save_frame_3d_bundle(
            out_dir,
            scene_pcd=scene_pcd,
            obj_loc=obj_loc,
            cameras=cameras_meta,
            depths=depths_out if save_depth_maps else None,
            anchor_pos=anchor_pos,
            anchor_quat=anchor_quat,
        )
        saved += 1

    return saved


def iter_manifest_jobs(manifest_path: Path) -> list[dict]:
    jobs: dict[tuple, dict] = {}
    with manifest_path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            key = (row["task"], row["lerobot_root"], int(row["episode_index"]))
            fi = int(row["frame_index"])
            if key not in jobs:
                jobs[key] = {
                    "task": row["task"],
                    "lerobot_root": row["lerobot_root"],
                    "episode_index": int(row["episode_index"]),
                    "frame_indices": [],
                }
            jobs[key]["frame_indices"].append(fi)
    for job in jobs.values():
        job["frame_indices"] = sorted(set(job["frame_indices"]))
    return list(jobs.values())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--robocasa365-root", type=Path, default=None)
    p.add_argument("--pcd-root", type=Path, default=SCRIPT_DIR / "data" / "leo_3d_cache")
    p.add_argument("--from-manifest", type=Path, default=None)
    p.add_argument("--tasks", type=str, default=None, help="Comma-separated task names")
    p.add_argument("--max-episodes-per-task", type=int, default=50)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--num-points", type=int, default=1024)
    p.add_argument("--no-save-depth", action="store_true")
    p.add_argument("--depth-pixel-stride", type=int, default=2)
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    args.pcd_root.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    camera_names = DEFAULT_CAMERAS
    logger.info("Creating Kitchen replay env (RGB-D, %s)...", camera_names)
    env = create_replay_env(camera_names)
    total_saved = 0

    if args.from_manifest:
        jobs = iter_manifest_jobs(args.from_manifest)
        if args.tasks:
            task_filter = {t.strip() for t in args.tasks.split(",") if t.strip()}
            jobs = [j for j in jobs if j["task"] in task_filter]
        logger.info("Manifest jobs: %d episode groups", len(jobs))
        for i, job in enumerate(jobs):
            logger.info(
                "[%d/%d] %s ep=%d frames=%d",
                i + 1,
                len(jobs),
                job["task"],
                job["episode_index"],
                len(job["frame_indices"]),
            )
            try:
                n = replay_episode(
                    env,
                    task=job["task"],
                    lerobot_root=Path(job["lerobot_root"]),
                    episode_idx=job["episode_index"],
                    pcd_root=args.pcd_root,
                    camera_names=camera_names,
                    stride=args.stride,
                    num_points=args.num_points,
                    frame_indices=job["frame_indices"],
                    save_depth_maps=not args.no_save_depth,
                    depth_pixel_stride=args.depth_pixel_stride,
                )
                total_saved += n
            except Exception as exc:
                logger.error("Failed %s ep %d: %s", job["task"], job["episode_index"], exc)
        logger.info("Done. %d frame bundles written under %s", total_saved, args.pcd_root)
        return 0

    # Task-based mode (for smoke / partial replay)
    if not args.tasks:
        logger.error("Provide --from-manifest or --tasks")
        return 1

    from convert_robocasa365_to_leo import (  # noqa: WPS433
        load_task_split_map,
        resolve_task_lerobot_roots,
    )

    task_yaml = RC365_DIR / "task_sets.yaml"
    task_split = load_task_split_map(task_yaml)
    root = args.robocasa365_root or Path(
        "/HOME/sysu_xdliang/sysu_xdliang_1/HDD_POOL/linmin/datasets/robocasa365-datasets"
    )
    for task in [t.strip() for t in args.tasks.split(",") if t.strip()]:
        roots = resolve_task_lerobot_roots(root, task, task_split.get(task))
        if not roots:
            logger.warning("No data for task %s", task)
            continue
        lerobot_root = Path(roots[-1])
        extras = sorted((lerobot_root / "extras").glob("episode_*"))[: args.max_episodes_per_task]
        for ep_dir in extras:
            ep_idx = int(ep_dir.name.split("_")[-1])
            logger.info("Replaying %s episode %d", task, ep_idx)
            n = replay_episode(
                env,
                task=task,
                lerobot_root=lerobot_root,
                episode_idx=ep_idx,
                pcd_root=args.pcd_root,
                camera_names=camera_names,
                stride=args.stride,
                num_points=args.num_points,
                save_depth_maps=not args.no_save_depth,
                depth_pixel_stride=args.depth_pixel_stride,
            )
            total_saved += n

    logger.info("Done. %d frame bundles under %s", total_saved, args.pcd_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
