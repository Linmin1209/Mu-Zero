# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MuJoCo world -> camera -> normalized 2D projection for RoboCasa365 PandaOmron.

``robot0_eye_in_hand`` is rigidly attached to the gripper; projecting the *current*
EEF into the wrist camera yields a nearly static (u, v). Following ``annotate_sim.py``,
we pre-collect world positions and project **future** EEF/base points using the
**anchor frame's** camera extrinsics so ego-view arm trajectories are dynamic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


EYE_IN_HAND_CAMERA = "robot0_eye_in_hand"
EYE_IN_HAND_CAMERAS = frozenset({EYE_IN_HAND_CAMERA})

DEFAULT_CAMERA_NAMES: tuple[str, ...] = (
    "robot0_agentview_left",
    "robot0_agentview_right",
    EYE_IN_HAND_CAMERA,
)

# Arm labels on all views; eye_in_hand uses anchored-future projection.
DEFAULT_ARM_CAMERAS: tuple[str, ...] = DEFAULT_CAMERA_NAMES

DEFAULT_BASE_CAMERAS: tuple[str, ...] = DEFAULT_CAMERA_NAMES

DEFAULT_FUTURE_LENGTH = 50

INVALID_UV = np.array([-1.0, -1.0], dtype=np.float32)


@dataclass(frozen=True)
class ProjectionResult:
    uv: np.ndarray  # (2,) normalized in [0, 1], or INVALID_UV when not applicable
    visible: bool
    applicable: bool = True


@dataclass(frozen=True)
class FutureProjectionResult:
    uv: np.ndarray  # (future_length, 2)
    visible: np.ndarray  # (future_length,) bool


def world_to_camera(point_world: np.ndarray, cam_pos: np.ndarray, cam_mat: np.ndarray) -> np.ndarray:
    """Transform a world point into the MuJoCo camera frame (looks along -Z)."""
    rotation = cam_mat.reshape(3, 3)
    return rotation.T @ (point_world - cam_pos)


def project_point_to_normalized_uv(
    point_world: np.ndarray,
    cam_pos: np.ndarray,
    cam_mat: np.ndarray,
    fovy_deg: float,
    width: int,
    height: int,
) -> ProjectionResult:
    """Project a 3D world point to normalized image coordinates (u, v) in [0, 1]."""
    point_cam = world_to_camera(point_world, cam_pos, cam_mat)
    if point_cam[2] >= -1e-5:
        return ProjectionResult(uv=INVALID_UV.copy(), visible=False)

    focal = 0.5 * height / np.tan(np.deg2rad(fovy_deg) / 2.0)
    cx = width / 2.0
    cy = height / 2.0
    u = focal * point_cam[0] / (-point_cam[2]) + cx
    v = cy - focal * point_cam[1] / (-point_cam[2])

    visible = (0.0 <= u < width) and (0.0 <= v < height)
    uv = np.array([u / width, v / height], dtype=np.float32)
    return ProjectionResult(uv=uv, visible=visible)


def get_arm_base_world_points(sim, robot) -> tuple[np.ndarray, np.ndarray]:
    """Return (eef_world, base_world) for PandaOmron in **world** coordinates."""
    eef_site_id = robot.eef_site_id["right"]
    arm_world = sim.data.site_xpos[eef_site_id].copy()
    base_body_id = sim.model.body_name2id("mobilebase0_support")
    base_world = sim.data.body_xpos[base_body_id].copy()
    return arm_world, base_world


def _camera_projection(
    sim,
    point_world: np.ndarray,
    camera_name: str,
    width: int,
    height: int,
) -> ProjectionResult:
    cam_id = sim.model.camera_name2id(camera_name)
    cam_pos = sim.data.cam_xpos[cam_id].copy()
    cam_mat = sim.data.cam_xmat[cam_id].copy()
    fovy = float(sim.model.cam_fovy[cam_id])
    return project_point_to_normalized_uv(point_world, cam_pos, cam_mat, fovy, width, height)


def _camera_params(sim, camera_name: str) -> tuple[np.ndarray, np.ndarray, float]:
    cam_id = sim.model.camera_name2id(camera_name)
    return (
        sim.data.cam_xpos[cam_id].copy(),
        sim.data.cam_xmat[cam_id].copy(),
        float(sim.model.cam_fovy[cam_id]),
    )


def _project_with_camera_params(
    point_world: np.ndarray,
    cam_pos: np.ndarray,
    cam_mat: np.ndarray,
    fovy_deg: float,
    width: int,
    height: int,
) -> ProjectionResult:
    return project_point_to_normalized_uv(point_world, cam_pos, cam_mat, fovy_deg, width, height)


def project_future_points(
    world_points: np.ndarray,
    future_frame_indices: list[int],
    cam_pos: np.ndarray,
    cam_mat: np.ndarray,
    fovy_deg: float,
    width: int,
    height: int,
    future_length: int,
) -> FutureProjectionResult:
    """Project world points at future frames using a fixed anchor camera."""
    uv = np.full((future_length, 2), -1.0, dtype=np.float32)
    visible = np.zeros((future_length,), dtype=bool)
    for slot, fi in enumerate(future_frame_indices):
        if slot >= future_length:
            break
        result = _project_with_camera_params(
            world_points[fi], cam_pos, cam_mat, fovy_deg, width, height
        )
        uv[slot] = result.uv
        visible[slot] = result.visible
    return FutureProjectionResult(uv=uv, visible=visible)


def project_frame_annotate_sim(
    sim,
    anchor_frame_idx: int,
    arm_world_all: np.ndarray,
    base_world_all: np.ndarray,
    width: int,
    height: int,
    arm_cameras: tuple[str, ...] = DEFAULT_ARM_CAMERAS,
    base_cameras: tuple[str, ...] = DEFAULT_BASE_CAMERAS,
    all_cameras: tuple[str, ...] = DEFAULT_CAMERA_NAMES,
    future_length: int = DEFAULT_FUTURE_LENGTH,
) -> dict[str, dict[str, ProjectionResult | FutureProjectionResult]]:
    """Project arm/base at anchor frame using annotate_sim-style future anchoring.

    Extrinsic cameras: current EEF/base at the anchor frame.
    Eye-in-hand arm: current EEF is nearly static; ``arm_future_*`` holds the
    dynamic trail (future world points projected with the anchor camera).
    """
    n_frames = len(arm_world_all)
    arm_set = set(arm_cameras)
    base_set = set(base_cameras)
    future_frame_indices = [
        fi for fi in range(n_frames) if 0 < fi - anchor_frame_idx <= future_length
    ]

    out: dict[str, dict[str, ProjectionResult | FutureProjectionResult]] = {
        "arm": {},
        "arm_future": {},
        "base": {},
        "base_future": {},
    }

    for camera_name in all_cameras:
        cam_pos, cam_mat, fovy = _camera_params(sim, camera_name)

        if camera_name in arm_set:
            arm_point = arm_world_all[anchor_frame_idx]
            out["arm"][camera_name] = _project_with_camera_params(
                arm_point, cam_pos, cam_mat, fovy, width, height
            )
            out["arm_future"][camera_name] = project_future_points(
                arm_world_all,
                future_frame_indices,
                cam_pos,
                cam_mat,
                fovy,
                width,
                height,
                future_length,
            )
        else:
            out["arm"][camera_name] = ProjectionResult(
                uv=INVALID_UV.copy(), visible=False, applicable=False
            )
            out["arm_future"][camera_name] = FutureProjectionResult(
                uv=np.full((future_length, 2), -1.0, dtype=np.float32),
                visible=np.zeros((future_length,), dtype=bool),
            )

        if camera_name in base_set:
            base_point = base_world_all[anchor_frame_idx]
            out["base"][camera_name] = _project_with_camera_params(
                base_point, cam_pos, cam_mat, fovy, width, height
            )
            out["base_future"][camera_name] = project_future_points(
                base_world_all,
                future_frame_indices,
                cam_pos,
                cam_mat,
                fovy,
                width,
                height,
                future_length,
            )
        else:
            out["base"][camera_name] = ProjectionResult(
                uv=INVALID_UV.copy(), visible=False, applicable=False
            )
            out["base_future"][camera_name] = FutureProjectionResult(
                uv=np.full((future_length, 2), -1.0, dtype=np.float32),
                visible=np.zeros((future_length,), dtype=bool),
            )

    return out


def project_frame(
    sim,
    robot,
    width: int,
    height: int,
    arm_cameras: tuple[str, ...] = DEFAULT_ARM_CAMERAS,
    base_cameras: tuple[str, ...] = DEFAULT_BASE_CAMERAS,
    all_cameras: tuple[str, ...] = DEFAULT_CAMERA_NAMES,
) -> dict[str, dict[str, ProjectionResult]]:
    """Legacy same-frame projection (kept for backward compatibility)."""
    arm_world, base_world = get_arm_base_world_points(sim, robot)
    arm_set = set(arm_cameras)
    base_set = set(base_cameras)
    out: dict[str, dict[str, ProjectionResult]] = {"arm": {}, "base": {}}

    for camera_name in all_cameras:
        if camera_name in arm_set:
            out["arm"][camera_name] = _camera_projection(
                sim, arm_world, camera_name, width, height
            )
        else:
            out["arm"][camera_name] = ProjectionResult(
                uv=INVALID_UV.copy(),
                visible=False,
                applicable=False,
            )

        if camera_name in base_set:
            out["base"][camera_name] = _camera_projection(
                sim, base_world, camera_name, width, height
            )
        else:
            out["base"][camera_name] = ProjectionResult(
                uv=INVALID_UV.copy(),
                visible=False,
                applicable=False,
            )

    return out


def arm_uv_variance_diagnostic(
    arm_uv_series: np.ndarray,
    visible_mask: np.ndarray,
) -> float:
    """Return max std of u/v over visible frames (for static-trajectory sanity checks)."""
    if not np.any(visible_mask):
        return 0.0
    vals = arm_uv_series[visible_mask]
    if len(vals) < 2:
        return 0.0
    return float(np.max(np.std(vals, axis=0)))
