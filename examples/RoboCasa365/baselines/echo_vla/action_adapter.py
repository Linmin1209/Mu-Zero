#!/usr/bin/env python3
"""Convert between Echo VLA 10D actions and RoboCasa365 Panda-Omron 12D actions."""

from __future__ import annotations

import numpy as np

# Echo: [arm7, base3]  — arm = delta pos(3) + delta rot(3) + gripper(1) typical layout
ECHO_DIM = 10
RC365_DIM = 12


def echo_to_rc365(action10: np.ndarray, control_mode: float = 1.0) -> np.ndarray:
    """Map Echo 10D → RC365 12D [gripper, eef_pos(3), eef_rot(3), base(3), mode(1)]."""
    a = np.asarray(action10, dtype=np.float32).reshape(-1)
    if a.shape[0] < ECHO_DIM:
        a = np.pad(a, (0, ECHO_DIM - a.shape[0]))
    a = a[:ECHO_DIM]
    gripper = a[6:7]
    eef_pos = a[0:3]
    eef_rot = a[3:6]
    base = a[7:10]
    mode = np.array([control_mode], dtype=np.float32)
    return np.concatenate([gripper, eef_pos, eef_rot, base, mode], axis=0)


def rc365_to_echo(action12: np.ndarray) -> np.ndarray:
    """Map RC365 12D → Echo 10D (drops control_mode)."""
    a = np.asarray(action12, dtype=np.float32).reshape(-1)
    if a.shape[0] < RC365_DIM:
        a = np.pad(a, (0, RC365_DIM - a.shape[0]))
    a = a[:RC365_DIM]
    gripper = a[0:1]
    eef_pos = a[1:4]
    eef_rot = a[4:7]
    base = a[7:10]
    arm7 = np.concatenate([eef_pos, eef_rot, gripper], axis=0)
    return np.concatenate([arm7, base], axis=0)


def echo_chunk_to_rc365(actions: np.ndarray, control_mode: float = 1.0) -> np.ndarray:
    """(T, 10) → (T, 12)."""
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim == 1:
        return echo_to_rc365(actions, control_mode=control_mode)
    return np.stack([echo_to_rc365(a, control_mode=control_mode) for a in actions], axis=0)
