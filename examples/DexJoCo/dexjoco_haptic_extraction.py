"""Per-pad Allegro tactile extraction from DexJoCo MuJoCo sim (fingertips + palm)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

from dexjoco_tactile_schema import (
    PAD_KEYS,
    modality_tactile_mapping,
    parquet_column,
    tactile_feature_template,
    tactile_keys,
)

DEFAULT_FORCE_THRESHOLD_N = 0.5
DEFAULT_FORCE_MAX_N = 50.0

TACTILE_FEATURE_TEMPLATE_SINGLE = tactile_feature_template(dual_arm=False)
TACTILE_FEATURE_TEMPLATE_BIMANUAL = tactile_feature_template(dual_arm=True)


@dataclass(frozen=True)
class DexJoCoTactileConfig:
    force_threshold_n: float = DEFAULT_FORCE_THRESHOLD_N
    force_max_n: float = DEFAULT_FORCE_MAX_N


@dataclass
class DexJoCoTactileFrame:
    """One timestep: per-pad normal force (N) + contact flags (fingertips + palm)."""

    values: dict[str, float] = field(default_factory=dict)
    dual_arm: bool = False

    @classmethod
    def zeros(cls, *, dual_arm: bool) -> DexJoCoTactileFrame:
        frame = cls(dual_arm=dual_arm)
        for key in tactile_keys(dual_arm=dual_arm):
            frame.values[key] = 0.0
        return frame

    def as_parquet_values(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, val in self.values.items():
            out[parquet_column(key)] = np.array([val], dtype=np.float32)
        return out


def rotvec_action_to_policy(action_rotvec: np.ndarray, *, dual_arm: bool) -> np.ndarray:
    """Convert LeRobot rotvec action (22/44) to policy env action (23/46, quat)."""
    action_rotvec = np.asarray(action_rotvec, dtype=np.float64)
    if dual_arm:
        r_xyz, r_rotvec, r_hand = action_rotvec[:3], action_rotvec[3:6], action_rotvec[6:22]
        l_xyz, l_rotvec, l_hand = action_rotvec[22:25], action_rotvec[25:28], action_rotvec[28:44]
        r_quat = R.from_rotvec(r_rotvec).as_quat(scalar_first=True)
        l_quat = R.from_rotvec(l_rotvec).as_quat(scalar_first=True)
        return np.concatenate([r_xyz, r_quat, r_hand, l_xyz, l_quat, l_hand])
    xyz, rotvec, hand = action_rotvec[:3], action_rotvec[3:6], action_rotvec[6:22]
    quat = R.from_rotvec(rotvec).as_quat(scalar_first=True)
    return np.concatenate([xyz, quat, hand])


def _classify_finger(text: str) -> str | None:
    name = text.lower()
    if "thumb" in name or "thumbtip" in name or "_thj" in name:
        return "thumb"
    if "fingertip_ff" in name or "_ff" in name or "ffj" in name:
        return "ff"
    if "fingertip_mf" in name or "_mf" in name or "mfj" in name:
        return "mf"
    if "fingertip_rf" in name or "_rf" in name or "rfj" in name:
        return "rf"
    return None


def _is_palm_pad(text: str) -> bool:
    name = text.lower()
    if "visual" in name:
        return False
    if "palm_collision" in name:
        return True
    if "allegro_palm" in name and "collision" in name:
        return True
    return "palm" in name and "collision" in name


def _is_fingertip_pad(text: str) -> bool:
    name = text.lower()
    if "visual" in name:
        return False
    return ("fingertip" in name or "thumbtip" in name) and (
        "collision" in name or "tip" in name
    )


def _matches_hand_side(geom_name: str, body_name: str, hand_side: str) -> bool:
    text = f"{geom_name} {body_name}".lower()
    if hand_side == "right":
        return "left" not in text or "allegro_right" in text or "_right" in text
    if hand_side == "left":
        return "left" in text or "allegro_left" in text or "_left" in text
    return True


def build_hand_pad_geom_map(model: mujoco.MjModel, hand_side: str) -> dict[str, set[int]]:
    """Map pad key (ff/mf/rf/thumb/palm) -> collision geom ids for one Allegro hand."""
    mapping: dict[str, set[int]] = {k: set() for k in PAD_KEYS}
    for gid in range(model.ngeom):
        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
        body_id = int(model.geom_bodyid[gid])
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        if not _matches_hand_side(geom_name, body_name, hand_side):
            continue
        text = f"{geom_name}/{body_name}"
        if _is_palm_pad(text):
            mapping["palm"].add(gid)
            continue
        if _is_fingertip_pad(text):
            finger = _classify_finger(text)
            if finger is not None:
                mapping[finger].add(gid)
    return mapping


def _normal_force_on_geom(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    geom_id: int,
) -> float:
    force_buf = np.zeros(6, dtype=np.float64)
    peak = 0.0
    for contact_idx in range(data.ncon):
        contact = data.contact[contact_idx]
        g1, g2 = int(contact.geom1), int(contact.geom2)
        if geom_id not in (g1, g2):
            continue
        mujoco.mj_contactForce(model, data, contact_idx, force_buf)
        peak = max(peak, max(0.0, float(force_buf[0])))
    return peak


def _pad_forces(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    pad_geoms: dict[str, set[int]],
    cfg: DexJoCoTactileConfig,
) -> dict[str, float]:
    forces: dict[str, float] = {}
    for pad, geom_ids in pad_geoms.items():
        raw = 0.0
        for gid in geom_ids:
            raw = max(raw, _normal_force_on_geom(model, data, gid))
        if raw < cfg.force_threshold_n:
            forces[pad] = 0.0
        else:
            forces[pad] = float(min(raw, cfg.force_max_n))
    return forces


def _prefix_pad_values(pads: dict[str, float], prefix: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for pad, val in pads.items():
        out[f"{prefix}_{pad}"] = val
        out[f"{prefix}_{pad}_contact"] = 1.0 if val > 0.0 else 0.0
    return out


def extract_dexjoco_tactile_frame(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    dual_arm: bool,
    cfg: DexJoCoTactileConfig | None = None,
) -> DexJoCoTactileFrame:
    """Per-pad normal force + contact (4 fingertips + palm, FSR-style thresholding)."""
    cfg = cfg or DexJoCoTactileConfig()
    frame = DexJoCoTactileFrame.zeros(dual_arm=dual_arm)

    if dual_arm:
        right = _pad_forces(model, data, build_hand_pad_geom_map(model, "right"), cfg)
        left = _pad_forces(model, data, build_hand_pad_geom_map(model, "left"), cfg)
        frame.values.update(_prefix_pad_values(right, "R"))
        frame.values.update(_prefix_pad_values(left, "L"))
    else:
        pads = _pad_forces(model, data, build_hand_pad_geom_map(model, "right"), cfg)
        for pad, val in pads.items():
            frame.values[pad] = val
            frame.values[f"{pad}_contact"] = 1.0 if val > 0.0 else 0.0

    return frame


TACTILE_COLUMN_NAMES = tuple(parquet_column(k) for k in tactile_keys(dual_arm=False))
TACTILE_FEATURE_TEMPLATE = TACTILE_FEATURE_TEMPLATE_SINGLE
