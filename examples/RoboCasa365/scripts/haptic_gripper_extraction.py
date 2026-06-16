"""Extract real-sensor-aligned 1D gripper tactile GT from RoboCasa365 replay states."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np

# Real FSR-style defaults: denoise floor + Panda-scale saturation.
DEFAULT_FORCE_THRESHOLD_N = 1.0
DEFAULT_FORCE_MAX_N = 100.0

FINGER1_PAD_GEOM_SUFFIX = "finger1_pad_collision"
FINGER2_PAD_GEOM_SUFFIX = "finger2_pad_collision"


@dataclass(frozen=True)
class RealSensorTactileConfig:
    force_threshold_n: float = DEFAULT_FORCE_THRESHOLD_N
    force_max_n: float = DEFAULT_FORCE_MAX_N


@dataclass(frozen=True)
class GripperTactileFrame:
    left_force: float
    right_force: float
    in_contact: bool

    def as_parquet_values(self) -> dict[str, Any]:
        return {
            "tactile.left": np.array([self.left_force], dtype=np.float32),
            "tactile.right": np.array([self.right_force], dtype=np.float32),
            "tactile.contact": np.array([self.in_contact], dtype=bool),
        }


def _pad_geom_names(env) -> tuple[set[str], set[str]]:
    robot = env.robots[0]
    if "right" not in robot.gripper:
        raise AttributeError("Expected robot.gripper['right'] for PandaOmron.")
    geoms = set(robot.gripper["right"].contact_geoms)
    finger1 = {g for g in geoms if FINGER1_PAD_GEOM_SUFFIX in g}
    finger2 = {g for g in geoms if FINGER2_PAD_GEOM_SUFFIX in g}
    if not finger1 or not finger2:
        raise RuntimeError(
            f"Missing pad collision geoms: finger1={finger1}, finger2={finger2}"
        )
    return finger1, finger2


def _denoise_and_clip(force: float, cfg: RealSensorTactileConfig) -> float:
    if force < cfg.force_threshold_n:
        return 0.0
    return float(min(force, cfg.force_max_n))


def extract_gripper_tactile_frame(
    env,
    cfg: RealSensorTactileConfig | None = None,
) -> GripperTactileFrame:
    """Read pad-only normal forces; contact follows real threshold logic."""
    cfg = cfg or RealSensorTactileConfig()
    sim = env.sim
    finger1_geoms, finger2_geoms = _pad_geom_names(env)
    pad_geoms = finger1_geoms | finger2_geoms

    left_raw = 0.0
    right_raw = 0.0
    force_buf = np.zeros(6, dtype=np.float64)

    for contact_idx in range(sim.data.ncon):
        contact = sim.data.contact[contact_idx]
        g1 = sim.model.geom_id2name(contact.geom1)
        g2 = sim.model.geom_id2name(contact.geom2)

        g1_is_pad = g1 in pad_geoms
        g2_is_pad = g2 in pad_geoms
        if not (g1_is_pad or g2_is_pad):
            continue
        if g1_is_pad and g2_is_pad:
            continue

        mujoco.mj_contactForce(sim.model._model, sim.data._data, contact_idx, force_buf)
        nf = max(0.0, float(force_buf[0]))

        active_geom = g1 if g1_is_pad else g2
        if active_geom in finger1_geoms:
            left_raw += nf
        elif active_geom in finger2_geoms:
            right_raw += nf

    left = _denoise_and_clip(left_raw, cfg)
    right = _denoise_and_clip(right_raw, cfg)
    in_contact = (left > 0.0) or (right > 0.0)
    return GripperTactileFrame(left_force=left, right_force=right, in_contact=in_contact)


# Backward-compatible alias used by older call sites.
extract_gripper_haptic_frame = extract_gripper_tactile_frame

TACTILE_FEATURE_TEMPLATE: dict[str, dict[str, Any]] = {
    "tactile.left": {
        "dtype": "float32",
        "shape": [1],
        "names": ["force"],
    },
    "tactile.right": {
        "dtype": "float32",
        "shape": [1],
        "names": ["force"],
    },
    "tactile.contact": {
        "dtype": "bool",
        "shape": [1],
        "names": None,
    },
}

HAPTIC_FEATURE_TEMPLATE = TACTILE_FEATURE_TEMPLATE
TACTILE_COLUMN_NAMES: tuple[str, ...] = tuple(TACTILE_FEATURE_TEMPLATE.keys())
HAPTIC_COLUMN_NAMES = TACTILE_COLUMN_NAMES
