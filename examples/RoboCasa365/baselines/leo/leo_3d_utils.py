#!/usr/bin/env python3
"""Depth unprojection + scene point cloud helpers for LEO 3D branch."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_CAMERAS = [
    "robot0_agentview_left",
    "robot0_agentview_right",
    "robot0_eye_in_hand",
]
CAMERA_HEIGHT = 256
CAMERA_WIDTH = 256


def pcd_frame_dir(pcd_root: Path, task: str, episode_index: int, frame_index: int) -> Path:
    return pcd_root / task / f"ep_{episode_index:06d}" / f"frame_{frame_index:06d}"


def pcd_npz_path(pcd_root: Path, task: str, episode_index: int, frame_index: int) -> Path:
    return pcd_frame_dir(pcd_root, task, episode_index, frame_index) / "scene_pcd.npz"


def unproject_depth_to_world(
    depth: np.ndarray,
    rgb: np.ndarray,
    intrinsic: np.ndarray,
    world_to_cam: np.ndarray,
    depth_min: float = 0.05,
    depth_max: float = 3.5,
    stride: int = 2,
) -> np.ndarray:
    """Back-project depth map to world-frame RGBXYZ points (N, 6)."""
    h, w = depth.shape
    valid = (depth > depth_min) & (depth < depth_max) & np.isfinite(depth)
    if stride > 1:
        valid[::stride, :] &= True
        valid[:, ::stride] &= True

    v_idx, u_idx = np.where(valid)
    if len(v_idx) == 0:
        return np.zeros((0, 6), dtype=np.float32)

    z = depth[v_idx, u_idx].astype(np.float64)
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    x = (u_idx - cx) * z / fx
    y = (v_idx - cy) * z / fy
    cam_h = np.stack([x, y, z, np.ones_like(z)], axis=-1)
    cam_to_world = np.linalg.inv(world_to_cam)
    world = (cam_to_world @ cam_h.T).T[:, :3]

    colors = rgb[v_idx, u_idx].astype(np.float32)
    if colors.max() > 1.0:
        colors = colors / 255.0
    return np.concatenate([colors, world.astype(np.float32)], axis=1)


def fuse_scene_pointcloud(
    view_chunks: list[np.ndarray],
    num_points: int = 1024,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Fuse multi-view RGBXYZ into one LEO-style object point cloud."""
    rng = np.random.default_rng(seed)
    if not view_chunks or all(len(c) == 0 for c in view_chunks):
        obj_fts = np.zeros((num_points, 6), dtype=np.float32)
        return obj_fts, np.zeros(6, dtype=np.float32)

    pts = np.concatenate([c for c in view_chunks if len(c) > 0], axis=0)
    if len(pts) > num_points * 8:
        idx = rng.choice(len(pts), size=num_points * 8, replace=False)
        pts = pts[idx]

    xyz = pts[:, 3:6].astype(np.float32)
    center = xyz.mean(0)
    size = xyz.max(0) - xyz.min(0)
    obj_loc = np.concatenate([center, size]).astype(np.float32)

    pts = pts.copy()
    pts[:, 3:6] = xyz - center
    max_dist = float(np.sqrt((pts[:, 3:6] ** 2).sum(1)).max())
    if max_dist < 1e-6:
        max_dist = 1.0
    pts[:, 3:6] /= max_dist

    if len(pts) < num_points:
        idx = rng.choice(len(pts), size=num_points, replace=True)
    else:
        idx = rng.choice(len(pts), size=num_points, replace=False)
    return pts[idx].astype(np.float32), obj_loc


def save_frame_3d_bundle(
    out_dir: Path,
    *,
    scene_pcd: np.ndarray,
    obj_loc: np.ndarray,
    cameras: dict[str, Any],
    depths: dict[str, np.ndarray] | None = None,
    anchor_pos: np.ndarray | None = None,
    anchor_quat: np.ndarray | None = None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / "scene_pcd.npz"
    payload = {
        "obj_fts": scene_pcd.astype(np.float32),
        "obj_locs": obj_loc.astype(np.float32),
    }
    if anchor_pos is not None:
        payload["anchor_locs"] = anchor_pos.astype(np.float32)
    if anchor_quat is not None:
        payload["anchor_orientation"] = anchor_quat.astype(np.float32)
    np.savez_compressed(npz_path, **payload)

    import json

    (out_dir / "cameras.json").write_text(json.dumps(cameras, indent=2), encoding="utf-8")
    if depths:
        for cam_name, depth in depths.items():
            np.save(out_dir / f"depth_{cam_name}.npy", depth.astype(np.float32))
    return npz_path
