#!/usr/bin/env python3
"""PyTorch Dataset for LEO × RoboCasa365 (manifest index + GR00T LeRobot loader)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from leo_rc365_action_norm import normalize_action
from leo_rc365_lerobot import RC365_VIDEO_KEYS, get_lerobot_loader
from leo_rc365_sanitize import sanitize_3d_numpy


class LeoRc365ManifestDataset(Dataset):
    """Flat manifest index; per-sample IO via CachedRoboCasaLeRobotLoader (LeRobot v2.1)."""

    def __init__(
        self,
        manifest_path: Path | str,
        video_backend: str = "opencv",  # kept for CLI compat; loader picks torchcodec or robocasa cv2
        image_size: int = 224,
        max_samples: int = 0,
        use_3d: bool = False,
        require_3d: bool = False,
        num_points: int = 1024,
        normalize_action: bool = False,
    ):
        self.manifest_path = Path(manifest_path)
        self.image_size = image_size
        self.use_3d = use_3d
        self.require_3d = require_3d
        self.num_points = num_points
        self.normalize_action = normalize_action
        self.rows: list[dict[str, Any]] = []
        with self.manifest_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if require_3d and not row.get("has_3d"):
                    continue
                self.rows.append(row)
        if max_samples > 0:
            self.rows = self.rows[:max_samples]

    def __len__(self) -> int:
        return len(self.rows)

    def _resize_image(self, img: np.ndarray) -> np.ndarray:
        from PIL import Image

        if img.shape[0] == self.image_size and img.shape[1] == self.image_size:
            return img.astype(np.uint8)
        return np.asarray(
            Image.fromarray(img.astype(np.uint8)).resize(
                (self.image_size, self.image_size), Image.BILINEAR
            )
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        meta = self.rows[index]
        lerobot_root = Path(meta["lerobot_root"])
        ep = int(meta["episode_index"])
        fi = int(meta["frame_index"])
        task = str(meta.get("task", ""))

        loader = get_lerobot_loader(lerobot_root)
        step = loader.load_step(ep, fi)

        images = [
            self._resize_image(img) for img in step["images_nhwc"]
        ]
        if len(images) != len(RC365_VIDEO_KEYS):
            raise RuntimeError(f"Expected {len(RC365_VIDEO_KEYS)} views, got {len(images)}")

        imgs = np.stack(images, axis=0).astype(np.float32) / 255.0
        imgs = torch.from_numpy(imgs).permute(0, 3, 1, 2)

        language = step.get("language") or meta.get("language") or task
        if isinstance(language, str) and language.isdigit():
            language = meta.get("language") or task

        action = torch.from_numpy(step["action"])
        if self.normalize_action:
            action = normalize_action(action, lerobot_root)

        sample: dict[str, Any] = {
            "images": imgs,
            "state": torch.from_numpy(step["state"]),
            "action": action,
            "language": language,
            "task": task,
            "lerobot_root": str(lerobot_root),
            "num_points": self.num_points,
        }

        if self.use_3d:
            pcd_path = meta.get("pcd3d_path")
            if pcd_path and Path(pcd_path).is_file():
                pcd = np.load(pcd_path)
                cleaned = sanitize_3d_numpy(
                    pcd["obj_fts"],
                    pcd["obj_locs"],
                    anchor_locs=pcd["anchor_locs"] if "anchor_locs" in pcd else None,
                    anchor_orientation=pcd["anchor_orientation"] if "anchor_orientation" in pcd else None,
                )
                sample["obj_fts"] = torch.from_numpy(cleaned["obj_fts"])
                sample["obj_locs"] = torch.from_numpy(cleaned["obj_locs"])
                sample["anchor_locs"] = torch.from_numpy(
                    cleaned["anchor_locs"]
                    if cleaned["anchor_locs"] is not None
                    else np.zeros(3, dtype=np.float32)
                )
                sample["anchor_orientation"] = torch.from_numpy(
                    cleaned["anchor_orientation"]
                    if cleaned["anchor_orientation"] is not None
                    else np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
                )
                sample["has_3d"] = bool(cleaned["obj_masks"].any())
            else:
                sample["has_3d"] = False

        return sample
