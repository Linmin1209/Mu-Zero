#!/usr/bin/env python3
"""PyTorch Dataset for LEO × RoboCasa365 manifest (multi-task LoRA training)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

VIDEO_KEYS = [
    "observation.images.robot0_agentview_left",
    "observation.images.robot0_agentview_right",
    "observation.images.robot0_eye_in_hand",
]
STATE_KEYS = [
    "observation.state.base_position",
    "observation.state.base_rotation",
    "observation.state.end_effector_position_relative",
    "observation.state.end_effector_rotation_relative",
    "observation.state.gripper_qpos",
]
ACTION_KEYS = [
    "action.gripper_close",
    "action.end_effector_position",
    "action.end_effector_rotation",
    "action.base_motion",
    "action.control_mode",
]


def _flatten_action(row: pd.Series) -> np.ndarray:
    parts: list[np.ndarray] = []
    for k in ACTION_KEYS:
        if k not in row.index:
            continue
        val = row[k]
        arr = np.asarray(val, dtype=np.float32).reshape(-1)
        parts.append(arr)
    if not parts:
        return np.zeros(12, dtype=np.float32)
    out = np.concatenate(parts)
    if out.shape[0] < 12:
        out = np.pad(out, (0, 12 - out.shape[0]))
    return out[:12].astype(np.float32)


def _flatten_state(row: pd.Series) -> np.ndarray:
    parts: list[np.ndarray] = []
    for k in STATE_KEYS:
        if k not in row.index:
            continue
        val = row[k]
        arr = np.asarray(val, dtype=np.float32).reshape(-1)
        parts.append(arr)
    if not parts:
        return np.zeros(16, dtype=np.float32)
    return np.concatenate(parts).astype(np.float32)


def _video_path(lerobot_root: Path, video_key: str, episode_index: int) -> Path:
    rel = video_key.replace("observation.images.", "")
    return lerobot_root / "videos" / "chunk-000" / rel / f"episode_{episode_index:06d}.mp4"


class LeoRc365ManifestDataset(Dataset):
    def __init__(
        self,
        manifest_path: Path | str,
        video_backend: str = "opencv",
        image_size: int = 224,
        max_samples: int = 0,
    ):
        self.manifest_path = Path(manifest_path)
        self.video_backend = video_backend
        self.image_size = image_size
        self.rows: list[dict[str, Any]] = []
        with self.manifest_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.rows.append(json.loads(line))
        if max_samples > 0:
            self.rows = self.rows[:max_samples]

        self._parquet_cache: dict[str, pd.DataFrame] = {}

    def __len__(self) -> int:
        return len(self.rows)

    def _load_frame_rgb(self, video_path: Path, frame_index: int) -> np.ndarray:
        from gr00t.utils.video_utils import get_frames_by_indices

        frames = get_frames_by_indices(
            str(video_path),
            [frame_index],
            video_backend=self.video_backend,
        )
        img = frames[0]
        if img.dtype != np.uint8:
            img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        return img

    def _get_parquet_row(self, lerobot_root: Path, episode_index: int, frame_index: int) -> pd.Series:
        key = f"{lerobot_root}:{episode_index}"
        if key not in self._parquet_cache:
            pq = lerobot_root / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet"
            if not pq.is_file():
                matches = sorted(lerobot_root.glob(f"data/chunk-*/episode_{episode_index:06d}.parquet"))
                pq = matches[0] if matches else pq
            self._parquet_cache[key] = pd.read_parquet(pq)
        df = self._parquet_cache[key]
        idx = min(frame_index, len(df) - 1)
        return df.iloc[idx]

    def __getitem__(self, index: int) -> dict[str, Any]:
        meta = self.rows[index]
        lerobot_root = Path(meta["lerobot_root"])
        ep = int(meta["episode_index"])
        fi = int(meta["frame_index"])

        images = []
        for vk in VIDEO_KEYS:
            vp = _video_path(lerobot_root, vk, ep)
            if vp.is_file():
                images.append(self._load_frame_rgb(vp, fi))
            else:
                images.append(np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8))

        row = self._get_parquet_row(lerobot_root, ep, fi)
        state = _flatten_state(row)
        action = _flatten_action(row)

        imgs = np.stack(images, axis=0).astype(np.float32) / 255.0
        imgs = torch.from_numpy(imgs).permute(0, 3, 1, 2)

        return {
            "images": imgs,
            "state": torch.from_numpy(state),
            "action": torch.from_numpy(action),
            "language": meta.get("language", meta.get("task", "")),
            "task": meta.get("task", ""),
        }
