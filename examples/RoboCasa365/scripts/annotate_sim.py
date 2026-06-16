#!/usr/bin/env python3
"""Simulator-based end-effector trajectory annotation.

Replays MuJoCo simulation states to obtain ground-truth EEF world
coordinates, then projects them onto camera image planes using camera
intrinsics/extrinsics. This replaces the VLM-based annotation approach
with exact physics-grounded labels.

Special handling for eye-in-hand camera:
  The wrist camera (robot0_eye_in_hand) moves with the robot, so its
  extrinsic matrix changes every frame. The EEF grip site is at the
  camera's mount point — projecting it yields a nearly-fixed pixel
  location (the EEF is always in the center of the wrist camera view).
  This is expected: for the wrist camera, we additionally record the
  camera pose so downstream code can reason about it.

Usage:
    # Annotate a single episode
    python annotate_sim.py --dataset-dir /home/ma-user/work/l30083605/Datasets/PandaOmron.CoolBakedCake --episodes 0

    # Annotate with visualization
    python annotate_sim.py --dataset-dir ... --episodes 0-2 --visualize

    # Only specific cameras
    python annotate_sim.py --dataset-dir ... --cameras robot0_agentview_left

    # Sparse annotation (every 10th frame)
    python annotate_sim.py --dataset-dir ... --sample-rate 10
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

logger = logging.getLogger(__name__)

# Default camera views in RoboCasa datasets
DEFAULT_CAMERAS = [
    "robot0_eye_in_hand",
    "robot0_agentview_left",
    "robot0_agentview_right",
    "robot0_agentview_front",
]

CAMERA_HEIGHT = 256
CAMERA_WIDTH = 256

# LeRobot dataset camera key prefix
CAMERA_KEY_PREFIX = "observation.images."

# Eye-in-hand camera names (these move with the robot)
EYE_IN_HAND_CAMERAS = {"robot0_eye_in_hand"}


def parse_episode_range(range_str: str) -> list[int]:
    """Parse episode range string like '0-5' or '0,3,7' or '0-5,8,10-12'."""
    episodes = []
    for part in range_str.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            episodes.extend(range(int(start), int(end) + 1))
        else:
            episodes.append(int(part))
    return sorted(set(episodes))


def get_episode_count(dataset_dir: Path) -> int:
    """Count episodes in a dataset based on extras directory."""
    extras_dir = dataset_dir / "extras"
    if not extras_dir.exists():
        return 0
    return len([d for d in extras_dir.iterdir() if d.is_dir() and d.name.startswith("episode_")])


def load_episode_states(dataset_dir: Path, episode_idx: int) -> np.ndarray:
    """Load MuJoCo states for an episode from states.npz."""
    states_path = dataset_dir / "extras" / f"episode_{episode_idx:06d}" / "states.npz"
    if not states_path.exists():
        raise FileNotFoundError(f"States file not found: {states_path}")
    data = np.load(states_path)
    return data["states"]


def load_episode_model_xml(dataset_dir: Path, episode_idx: int) -> str:
    """Load MuJoCo model XML for an episode from model.xml.gz."""
    xml_path = dataset_dir / "extras" / f"episode_{episode_idx:06d}" / "model.xml.gz"
    if not xml_path.exists():
        raise FileNotFoundError(f"Model XML not found: {xml_path}")
    with gzip.open(xml_path, "rb") as f:
        return f.read().decode("utf-8")


def create_env(camera_names: list[str]):
    """Create a robocasa Kitchen environment.

    Importing robocasa registers the Kitchen env with robosuite's
    factory.  Only passes parameters that Kitchen.__init__ accepts.
    """
    import robocasa  # noqa: F401 — registers Kitchen env
    import robosuite

    env = robosuite.make(
        env_name="Kitchen",
        robots="PandaOmron",
        has_renderer=False,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_camera_obs=True,
        camera_names=camera_names,
        camera_heights=CAMERA_HEIGHT,
        camera_widths=CAMERA_WIDTH,
    )
    return env


def load_episode_ep_meta(dataset_dir: Path, episode_idx: int) -> dict:
    """Load episode metadata from extras directory."""
    meta_path = dataset_dir / "extras" / f"episode_{episode_idx:06d}" / "ep_meta.json"
    if not meta_path.exists():
        return {}
    with open(meta_path) as f:
        return json.load(f)


def annotate_episode(
    env,
    dataset_dir: Path,
    episode_idx: int,
    camera_names: list[str],
    sample_rate: int = 1,
) -> dict[str, list[dict]]:
    """Annotate a single episode by replaying simulation states.

    Returns:
        Dict mapping camera_name -> list of annotation dicts per frame.
    """
    from robosuite.utils.camera_utils import (
        get_camera_extrinsic_matrix,
        get_camera_intrinsic_matrix,
        get_camera_transform_matrix,
        project_points_from_world_to_camera,
    )

    # Load states, model XML, and episode metadata
    states = load_episode_states(dataset_dir, episode_idx)
    model_xml = load_episode_model_xml(dataset_dir, episode_idx)
    ep_meta = load_episode_ep_meta(dataset_dir, episode_idx)

    # Reset environment following the same sequence as playback_dataset.py:
    # 1. Set ep_meta on the environment (layout, style, camera configs etc.)
    # 2. Call env.reset() for a soft reset
    # 3. Edit model XML and reset_from_xml_string for the episode's model
    # 4. sim.reset() and set_state_from_flattened for exact state
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

    # Update state after reset
    if hasattr(env, "update_sites"):
        env.update_sites()
    if hasattr(env, "update_state"):
        env.update_state()

    # Find the gripper site ID
    # PandaOmron uses "gripper0_right_grip_site"; other robots may use "robot0_grip_site"
    grip_site_name = None
    for candidate in ["gripper0_right_grip_site", "robot0_grip_site"]:
        if candidate in env.sim.model.site_names:
            grip_site_name = candidate
            break

    if grip_site_name is None:
        # Fallback: search for any grip site matching gripper* or robot* prefix
        for name in env.sim.model.site_names:
            if "grip_site" in name and ("gripper" in name or "robot" in name):
                grip_site_name = name
                break

    if grip_site_name is None:
        raise ValueError(
            f"Could not find grip site. Available: {env.sim.model.site_names[:20]}"
        )

    grip_site_id = env.sim.model.site_name2id(grip_site_name)
    logger.info(f"  Using grip site: {grip_site_name}")

    # Find the mobile base body for chassis trajectory annotation
    # PandaOmron uses "mobilebase0_wheeled_base"; other robots may differ
    base_body_name = None
    for candidate in ["mobilebase0_wheeled_base", "robot0_base"]:
        if candidate in env.sim.model.body_names:
            base_body_name = candidate
            break

    if base_body_name is None:
        for name in env.sim.model.body_names:
            if ("wheeled_base" in name or "mobile_base" in name) and "mobilebase" in name:
                base_body_name = name
                break

    base_body_id = None
    base_yaw_joint_id = None
    if base_body_name is not None:
        base_body_id = env.sim.model.body_name2id(base_body_name)
        logger.info(f"  Using base body: {base_body_name}")
        # Find the yaw joint for base rotation
        for jname in env.sim.model.joint_names:
            if "mobile" in jname and "yaw" in jname:
                base_yaw_joint_id = env.sim.model.joint_name2id(jname)
                break
    else:
        logger.warning("  No mobile base body found; skipping base trajectory annotation")

    n_frames = states.shape[0]
    frame_indices = list(range(0, n_frames, sample_rate))

    # Pre-collect world positions for ALL frames (needed for future trajectory
    # projection).  For eye-in-hand cameras the camera moves every frame, so
    # we must project future world positions using the *current* frame's camera
    # parameters — otherwise the future trajectory looks wrong.
    logger.info("  Pre-collecting world positions for all frames...")
    all_eef_pos = {}      # frame_idx -> eef_world_pos
    all_base_pos = {}     # frame_idx -> base_world_pos
    all_base_yaw = {}     # frame_idx -> base_yaw
    all_gripper = {}      # frame_idx -> gripper_open
    gripper_idx_map = env.robots[0]._ref_gripper_joint_pos_indexes

    for frame_idx in frame_indices:
        env.sim.set_state_from_flattened(states[frame_idx])
        env.sim.forward()
        all_eef_pos[frame_idx] = env.sim.data.site_xpos[grip_site_id].copy()
        if base_body_id is not None:
            all_base_pos[frame_idx] = env.sim.data.body_xpos[base_body_id].copy()
        if base_yaw_joint_id is not None:
            all_base_yaw[frame_idx] = float(env.sim.data.qpos[base_yaw_joint_id])
        if "right" in gripper_idx_map:
            gqpos = env.sim.data.qpos[gripper_idx_map["right"]]
        else:
            gqpos = env.sim.data.qpos[gripper_idx_map[next(iter(gripper_idx_map))]]
        all_gripper[frame_idx] = float(np.mean(gqpos))

    # Default future length for annotation (how many future frames to project)
    FUTURE_LENGTH = 50

    # Initialize per-camera annotation lists
    annotations = {cam: [] for cam in camera_names}

    for frame_idx in frame_indices:
        # Set sim state (needed for camera parameters)
        env.sim.set_state_from_flattened(states[frame_idx])
        env.sim.forward()

        eef_world_pos = all_eef_pos[frame_idx]
        base_world_pos = all_base_pos.get(frame_idx)
        base_yaw = all_base_yaw.get(frame_idx)
        gripper_open = all_gripper[frame_idx]

        # Determine future frame indices for trajectory projection
        future_frame_indices = [
            fi for fi in frame_indices
            if 0 < fi - frame_idx <= FUTURE_LENGTH * sample_rate
        ]

        for cam_name in camera_names:
            # Get camera transform for THIS frame — all future world positions
            # will be projected using this transform so the trajectory shows
            # where the robot will go relative to the current camera view.
            transform = get_camera_transform_matrix(
                env.sim, cam_name, CAMERA_HEIGHT, CAMERA_WIDTH
            )

            # --- Project current EEF position ---
            pixel = project_points_from_world_to_camera(
                points=eef_world_pos.reshape(1, 3),
                world_to_camera_transform=transform,
                camera_height=CAMERA_HEIGHT,
                camera_width=CAMERA_WIDTH,
            )[0]
            py, px = int(pixel[0]), int(pixel[1])

            eef_homo = np.append(eef_world_pos, 1.0)
            cam_coords = transform @ eef_homo
            depth = float(cam_coords[2])

            visible = (0 <= px < CAMERA_WIDTH) and (0 <= py < CAMERA_HEIGHT) and (depth > 0)

            ann = {
                "frame_index": frame_idx,
                "x": px if visible else -1,
                "y": py if visible else -1,
                "visible": visible,
                "eef_world_pos": eef_world_pos.tolist(),
                "gripper_open": gripper_open,
                "depth_in_camera": depth,
            }

            # --- Project current base position ---
            if base_world_pos is not None:
                base_pixel = project_points_from_world_to_camera(
                    points=base_world_pos.reshape(1, 3),
                    world_to_camera_transform=transform,
                    camera_height=CAMERA_HEIGHT,
                    camera_width=CAMERA_WIDTH,
                )[0]
                base_py, base_px = int(base_pixel[0]), int(base_pixel[1])

                base_homo = np.append(base_world_pos, 1.0)
                base_cam_coords = transform @ base_homo
                base_depth = float(base_cam_coords[2])

                base_visible = (
                    (0 <= base_px < CAMERA_WIDTH)
                    and (0 <= base_py < CAMERA_HEIGHT)
                    and (base_depth > 0)
                )

                ann["base_world_pos"] = base_world_pos.tolist()
                ann["base_x"] = base_px if base_visible else -1
                ann["base_y"] = base_py if base_visible else -1
                ann["base_visible"] = base_visible
                ann["base_depth_in_camera"] = base_depth
                if base_yaw is not None:
                    ann["base_yaw"] = base_yaw

            # --- Project future EEF trajectory using current camera ---
            future_eef_pixels = []
            for fi in future_frame_indices:
                fut_pos = all_eef_pos[fi]
                fut_pixel = project_points_from_world_to_camera(
                    points=fut_pos.reshape(1, 3),
                    world_to_camera_transform=transform,
                    camera_height=CAMERA_HEIGHT,
                    camera_width=CAMERA_WIDTH,
                )[0]
                fut_py, fut_px = int(fut_pixel[0]), int(fut_pixel[1])
                # Check visibility
                fut_homo = np.append(fut_pos, 1.0)
                fut_depth = float((transform @ fut_homo)[2])
                fut_visible = (
                    (0 <= fut_px < CAMERA_WIDTH)
                    and (0 <= fut_py < CAMERA_HEIGHT)
                    and (fut_depth > 0)
                )
                future_eef_pixels.append({
                    "frame_index": fi,
                    "x": fut_px if fut_visible else -1,
                    "y": fut_py if fut_visible else -1,
                    "visible": fut_visible,
                })
            ann["future_eef"] = future_eef_pixels

            # --- Project future base trajectory using current camera ---
            if base_world_pos is not None:
                future_base_pixels = []
                for fi in future_frame_indices:
                    fut_pos = all_base_pos.get(fi)
                    if fut_pos is None:
                        continue
                    fut_pixel = project_points_from_world_to_camera(
                        points=fut_pos.reshape(1, 3),
                        world_to_camera_transform=transform,
                        camera_height=CAMERA_HEIGHT,
                        camera_width=CAMERA_WIDTH,
                    )[0]
                    fut_py, fut_px = int(fut_pixel[0]), int(fut_pixel[1])
                    fut_homo = np.append(fut_pos, 1.0)
                    fut_depth = float((transform @ fut_homo)[2])
                    fut_visible = (
                        (0 <= fut_px < CAMERA_WIDTH)
                        and (0 <= fut_py < CAMERA_HEIGHT)
                        and (fut_depth > 0)
                    )
                    future_base_pixels.append({
                        "frame_index": fi,
                        "x": fut_px if fut_visible else -1,
                        "y": fut_py if fut_visible else -1,
                        "visible": fut_visible,
                    })
                ann["future_base"] = future_base_pixels

            # For eye-in-hand cameras, record camera pose for downstream use
            if cam_name in EYE_IN_HAND_CAMERAS:
                extrinsic = get_camera_extrinsic_matrix(env.sim, cam_name)
                intrinsic = get_camera_intrinsic_matrix(
                    env.sim, cam_name, CAMERA_HEIGHT, CAMERA_WIDTH
                )
                cam_pos = env.sim.data.cam_xpos[
                    env.sim.model.camera_name2id(cam_name)
                ].copy()
                ann["camera_extrinsic"] = extrinsic.tolist()
                ann["camera_intrinsic"] = intrinsic.tolist()
                ann["camera_pos"] = cam_pos.tolist()
                eef_cam = extrinsic @ eef_homo
                eef_cam_xyz = (eef_cam[:3] / eef_cam[3]).tolist()
                ann["eef_in_camera_frame"] = eef_cam_xyz
                if base_world_pos is not None:
                    base_cam = extrinsic @ base_homo
                    base_cam_xyz = (base_cam[:3] / base_cam[3]).tolist()
                    ann["base_in_camera_frame"] = base_cam_xyz

            annotations[cam_name].append(ann)

    return annotations


def save_annotations(
    annotations: dict[str, list[dict]],
    dataset_dir: Path,
    episode_idx: int,
    sample_rate: int,
    output_dir: Path,
) -> list[Path]:
    """Save annotation JSONs for each camera view."""
    dataset_name = dataset_dir.name
    saved_paths = []

    for cam_name, ann_list in annotations.items():
        ann_data = {
            "dataset": dataset_name,
            "episode_index": episode_idx,
            "camera_view": cam_name,
            "frame_sample_rate": sample_rate,
            "image_width": CAMERA_WIDTH,
            "image_height": CAMERA_HEIGHT,
            "method": "simulator_projection",
            "eye_in_hand": cam_name in EYE_IN_HAND_CAMERAS,
            "annotations": ann_list,
        }

        ann_path = output_dir / dataset_name / cam_name / f"episode_{episode_idx:06d}.json"
        ann_path.parent.mkdir(parents=True, exist_ok=True)
        with open(ann_path, "w") as f:
            json.dump(ann_data, f, indent=2)

        saved_paths.append(ann_path)
        visible_count = sum(1 for a in ann_list if a.get("visible", False))
        base_visible_count = sum(1 for a in ann_list if a.get("base_visible", False))
        logger.info(
            f"  Saved {cam_name}: {len(ann_list)} frames, "
            f"{visible_count} EEF visible, {base_visible_count} base visible -> {ann_path}"
        )

    return saved_paths


def visualize_episode_annotations(
    dataset_dir: Path,
    episode_idx: int,
    camera_names: list[str],
    output_dir: Path,
    sample_rate: int,
):
    """Visualize annotations by overlaying trajectory on video frames."""
    from utils.visualization import overlay_trajectory

    dataset_name = dataset_dir.name

    for cam_name in camera_names:
        ann_path = output_dir / dataset_name / cam_name / f"episode_{episode_idx:06d}.json"
        if not ann_path.exists():
            logger.warning(f"  Annotation not found: {ann_path}")
            continue

        with open(ann_path) as f:
            ann_data = json.load(f)

        video_path = (
            dataset_dir
            / "videos"
            / "chunk-000"
            / f"{CAMERA_KEY_PREFIX}{cam_name}"
            / f"episode_{episode_idx:06d}.mp4"
        )

        if not video_path.exists():
            logger.warning(f"  Video not found: {video_path}")
            continue

        vis_dir = output_dir / dataset_name / cam_name / "visualized"
        vis_path = vis_dir / f"episode_{episode_idx:06d}_annotated.mp4"

        overlay_trajectory(
            video_path=video_path,
            annotations=ann_data["annotations"],
            output_path=vis_path,
            sample_rate=sample_rate,
            show_base=True,
        )
        logger.info(f"  Visualized: {vis_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Simulator-based EEF trajectory annotation using MuJoCo replay"
    )
    parser.add_argument(
        "--dataset-dir", type=str, required=True,
        help="Path to dataset directory (e.g. /path/to/PandaOmron.CoolBakedCake)",
    )
    parser.add_argument(
        "--episodes", type=str, default=None,
        help="Episode range (e.g. '0-5', '0,3,7'). Default: all episodes.",
    )
    parser.add_argument(
        "--cameras", type=str, nargs="+", default=None,
        help="Camera view names (e.g. robot0_agentview_left). Default: all 3 views.",
    )
    parser.add_argument(
        "--sample-rate", type=int, default=1,
        help="Annotate every N-th frame (default: 1 = every frame).",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for annotation JSONs. Default: <dataset_dir>/../annotations_sim",
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="Also render annotated videos with trajectory overlay.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        logger.error(f"Dataset not found: {dataset_dir}")
        sys.exit(1)

    camera_names = args.cameras or DEFAULT_CAMERAS
    output_dir = Path(args.output_dir) if args.output_dir else dataset_dir.parent / "annotations_sim"

    # Create environment once (reused across episodes)
    logger.info("Creating robocasa environment...")
    env = create_env(camera_names=camera_names)
    logger.info("Environment created.")

    logger.info(f"\n{'=' * 60}")
    logger.info(f"Dataset:  {dataset_dir.name}")
    logger.info(f"Output:   {output_dir}")
    logger.info(f"Cameras:  {camera_names}")
    logger.info(f"{'=' * 60}")

    ep_count = get_episode_count(dataset_dir)
    if args.episodes:
        episode_indices = parse_episode_range(args.episodes)
    else:
        episode_indices = list(range(ep_count))

    valid_eps = [e for e in episode_indices if e < ep_count]
    logger.info(f"Episodes: {len(valid_eps)} of {ep_count}")

    for ep_idx in valid_eps:
        logger.info(f"\n  Episode {ep_idx}:")
        try:
            annotations = annotate_episode(
                env=env,
                dataset_dir=dataset_dir,
                episode_idx=ep_idx,
                camera_names=camera_names,
                sample_rate=args.sample_rate,
            )

            save_annotations(
                annotations=annotations,
                dataset_dir=dataset_dir,
                episode_idx=ep_idx,
                sample_rate=args.sample_rate,
                output_dir=output_dir,
            )

            if args.visualize:
                visualize_episode_annotations(
                    dataset_dir=dataset_dir,
                    episode_idx=ep_idx,
                    camera_names=camera_names,
                    output_dir=output_dir,
                    sample_rate=args.sample_rate,
                )

        except KeyboardInterrupt:
            logger.info("Interrupted by user.")
            sys.exit(0)
        except Exception as e:
            logger.error(f"  Failed to annotate episode {ep_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue

    logger.info("\nAnnotation completed!")


if __name__ == "__main__":
    main()