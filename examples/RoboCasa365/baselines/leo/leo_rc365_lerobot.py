#!/usr/bin/env python3
"""GR00T LeRobotEpisodeLoader helpers for LEO × RoboCasa365."""

from __future__ import annotations

import json
import os
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.types import ModalityConfig

# RoboCasa365 Panda-Omron modality keys (see meta/modality.json per task).
RC365_VIDEO_KEYS = [
    "robot0_agentview_left",
    "robot0_agentview_right",
    "robot0_eye_in_hand",
]
RC365_STATE_KEYS = [
    "base_position",
    "base_rotation",
    "end_effector_position_relative",
    "end_effector_rotation_relative",
    "gripper_qpos",
]
RC365_ACTION_KEYS = [
    "base_motion",
    "control_mode",
    "end_effector_position",
    "end_effector_rotation",
    "gripper_close",
]
RC365_LANGUAGE_KEY = "annotation.human.task_description"


def rc365_modality_configs() -> dict[str, ModalityConfig]:
    return {
        "video": ModalityConfig(delta_indices=[0], modality_keys=list(RC365_VIDEO_KEYS)),
        "state": ModalityConfig(delta_indices=[0], modality_keys=list(RC365_STATE_KEYS)),
        "action": ModalityConfig(delta_indices=[0], modality_keys=list(RC365_ACTION_KEYS)),
        "language": ModalityConfig(delta_indices=[0], modality_keys=[RC365_LANGUAGE_KEY]),
    }


class _LRUCache:
    def __init__(self, maxsize: int = 32):
        self.maxsize = maxsize
        self._data: OrderedDict[Any, Any] = OrderedDict()

    def get(self, key: Any):
        return self._data.get(key)

    def put(self, key: Any, value: Any) -> None:
        if key in self._data:
            self._data.move_to_end(key)
        self._data[key] = value
        while len(self._data) > self.maxsize:
            self._data.popitem(last=False)


def _robocasa_python() -> str:
    return os.environ.get(
        "ROBOCASA_PYTHON",
        "/XYAIFS00/sysu_xdliang_1/miniconda3/envs/robocasa/bin/python",
    )


class _RobocasaVideoDaemon:
    """Long-lived robocasa Python subprocess to avoid ~1s startup per decode."""

    _WORKER_SCRIPT = r"""
import json, sys, cv2, numpy as np, tempfile, os
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    req = json.loads(line)
    paths = req["paths"]
    indices = np.asarray(req["indices"], dtype=np.int64)
    result = {}
    for key, path in paths.items():
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            result[key] = np.zeros((len(indices), 1, 1, 3), dtype=np.uint8)
            continue
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok or frame is None:
                frames.append(np.zeros((1, 1, 3), dtype=np.uint8))
            else:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        result[key] = np.stack(frames, axis=0)
    fd, out_path = tempfile.mkstemp(suffix=".npz")
    os.close(fd)
    np.savez(out_path, **result)
    print(out_path, flush=True)
"""

    def __init__(self, python_bin: str):
        import subprocess

        self._python_bin = python_bin
        self._proc = subprocess.Popen(
            [python_bin, "-c", self._WORKER_SCRIPT],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )

    def decode(self, video_paths: dict[str, str], indices: np.ndarray) -> dict[str, np.ndarray]:
        if self._proc.poll() is not None:
            self.__init__(self._python_bin)
        req = json.dumps({"paths": video_paths, "indices": indices.tolist()})
        assert self._proc.stdin is not None and self._proc.stdout is not None
        self._proc.stdin.write(req + "\n")
        self._proc.stdin.flush()
        out_path = self._proc.stdout.readline().strip()
        if not out_path or not Path(out_path).is_file():
            return {}
        try:
            data = np.load(out_path)
            return {k: data[k] for k in data.files}
        finally:
            Path(out_path).unlink(missing_ok=True)


_VIDEO_DAEMON: _RobocasaVideoDaemon | None = None
_VIDEO_FRAME_CACHE = _LRUCache(maxsize=512)


def _get_video_daemon() -> _RobocasaVideoDaemon | None:
    global _VIDEO_DAEMON
    python_bin = _robocasa_python()
    if not Path(python_bin).is_file():
        return None
    if _VIDEO_DAEMON is None or _VIDEO_DAEMON._proc.poll() is not None:
        _VIDEO_DAEMON = _RobocasaVideoDaemon(python_bin)
    return _VIDEO_DAEMON


def decode_videos_subprocess(
    video_paths: dict[str, str],
    indices: np.ndarray,
    *,
    python_bin: str | None = None,
) -> dict[str, np.ndarray]:
    """Decode frames for multiple cameras (persistent robocasa cv2 worker)."""
    idx_list = tuple(int(i) for i in indices.tolist())
    cache_key = (tuple(sorted(video_paths.items())), idx_list)
    cached = _VIDEO_FRAME_CACHE.get(cache_key)
    if cached is not None:
        return cached

    daemon = _get_video_daemon()
    if daemon is not None:
        out = daemon.decode(video_paths, indices)
        if out:
            _VIDEO_FRAME_CACHE.put(cache_key, out)
            return out

    # One-shot fallback if daemon failed to start.
    import subprocess
    import tempfile

    python_bin = python_bin or _robocasa_python()
    if not Path(python_bin).is_file():
        return {}

    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
        out_path = tmp.name

    script = (
        "import cv2, json, numpy as np\n"
        f"paths = json.loads({json.dumps(json.dumps(video_paths))})\n"
        f"indices = np.asarray({indices.tolist()}, dtype=np.int64)\n"
        f"out_path = {out_path!r}\n"
        "result = {}\n"
        "for key, path in paths.items():\n"
        "    cap = cv2.VideoCapture(path)\n"
        "    if not cap.isOpened():\n"
        "        result[key] = np.zeros((len(indices), 1, 1, 3), dtype=np.uint8)\n"
        "        continue\n"
        "    frames = []\n"
        "    for idx in indices:\n"
        "        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))\n"
        "        ok, frame = cap.read()\n"
        "        if not ok or frame is None:\n"
        "            frames.append(np.zeros((1, 1, 3), dtype=np.uint8))\n"
        "        else:\n"
        "            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))\n"
        "    cap.release()\n"
        "    result[key] = np.stack(frames, axis=0)\n"
        "np.savez_compressed(out_path, **result)\n"
    )
    proc = subprocess.run([python_bin, "-c", script], capture_output=True, check=False)
    try:
        if proc.returncode != 0:
            return {}
        data = np.load(out_path)
        out = {k: data[k] for k in data.files}
        _VIDEO_FRAME_CACHE.put(cache_key, out)
        return out
    finally:
        Path(out_path).unlink(missing_ok=True)


def _try_torchcodec_frames(video_path: str, indices: np.ndarray) -> np.ndarray | None:
    try:
        from gr00t.utils.video_utils import get_frames_by_indices

        return get_frames_by_indices(video_path, indices)
    except Exception:
        return None


class CachedRoboCasaLeRobotLoader(LeRobotEpisodeLoader):
    """LeRobot loader with episode parquet cache and fast video decode fallback."""

    def __init__(self, dataset_path: str | Path, *, parquet_cache_size: int = 32):
        super().__init__(dataset_path, rc365_modality_configs())
        self._parquet_cache = _LRUCache(parquet_cache_size)

    def _load_parquet_data(self, episode_index: int) -> pd.DataFrame:
        chunk_idx = episode_index // self.chunk_size
        parquet_filename = self.data_path_pattern.format(
            episode_chunk=chunk_idx, episode_index=episode_index
        )
        original_df = pd.read_parquet(self.dataset_path / parquet_filename)
        loaded_df = pd.DataFrame()

        ann_col = "annotation.human.task_description"
        if ann_col in original_df.columns:
            loaded_df[f"language.{RC365_LANGUAGE_KEY}"] = original_df[ann_col].apply(
                lambda x: self.tasks_map[int(x)]
            )

        for modality_type in ("state", "action"):
            joint_groups_df = self._extract_joint_groups(
                original_df,
                self.modality_configs[modality_type].modality_keys,
                modality_type,
            )
            for joint_group in joint_groups_df.columns:
                loaded_df[f"{modality_type}.{joint_group}"] = joint_groups_df[joint_group]

        return loaded_df

    def _load_parquet_data_cached(self, episode_index: int) -> pd.DataFrame:
        cached = self._parquet_cache.get(episode_index)
        if cached is not None:
            return cached
        df = self._load_parquet_data(episode_index)
        self._parquet_cache.put(episode_index, df)
        return df

    def _video_paths_for_episode(self, episode_index: int) -> dict[str, str]:
        if not self.video_path_pattern or "video" not in self.modality_configs:
            return {}
        chunk_idx = episode_index // self.chunk_size
        paths: dict[str, str] = {}
        for image_key in self.modality_configs["video"].modality_keys:
            meta_key = self._video_key_mapping.get(image_key, image_key)
            original_key = self.modality_meta["video"][meta_key].get(
                "original_key", f"observation.images.{meta_key}"
            )
            video_filename = self.video_path_pattern.format(
                episode_chunk=chunk_idx,
                video_key=original_key,
                episode_index=episode_index,
            )
            paths[image_key] = str(self.dataset_path / video_filename)
        return paths

    def _load_video_data(self, episode_index: int, indices: np.ndarray) -> dict[str, np.ndarray]:
        paths = self._video_paths_for_episode(episode_index)
        if not paths:
            return {}

        out: dict[str, np.ndarray] = {}
        missing: dict[str, str] = {}
        for image_key, video_path in paths.items():
            frames = _try_torchcodec_frames(video_path, indices)
            if frames is not None and len(frames) == len(indices):
                out[image_key] = frames
            else:
                missing[image_key] = video_path

        if missing:
            decoded = decode_videos_subprocess(missing, indices)
            for key, frames in decoded.items():
                out[key] = frames

        for key in paths:
            if key not in out:
                n = len(indices)
                out[key] = np.zeros((n, 1, 1, 3), dtype=np.uint8)
        return out

    def load_step(self, episode_index: int, frame_index: int) -> dict[str, Any]:
        """Load a single frame using LeRobot parquet + video paths."""
        df = self._load_parquet_data_cached(episode_index)
        fi = min(int(frame_index), len(df) - 1)
        row = df.iloc[fi]

        state = np.concatenate(
            [np.asarray(row[f"state.{k}"], dtype=np.float32).reshape(-1) for k in RC365_STATE_KEYS]
        )
        action = np.concatenate(
            [np.asarray(row[f"action.{k}"], dtype=np.float32).reshape(-1) for k in RC365_ACTION_KEYS]
        )

        lang_col = f"language.{RC365_LANGUAGE_KEY}"
        language = str(row[lang_col]) if lang_col in row.index else ""

        video_data = self._load_video_data(episode_index, np.asarray([fi], dtype=np.int64))
        images_nhwc = []
        for key in RC365_VIDEO_KEYS:
            frames = video_data.get(key)
            if frames is None or len(frames) == 0:
                images_nhwc.append(np.zeros((224, 224, 3), dtype=np.uint8))
            else:
                images_nhwc.append(np.asarray(frames[0], dtype=np.uint8))

        return {
            "images_nhwc": images_nhwc,
            "state": state.astype(np.float32),
            "action": action.astype(np.float32),
            "language": language,
        }


_LOADER_CACHE: dict[str, CachedRoboCasaLeRobotLoader] = {}


def get_lerobot_loader(lerobot_root: str | Path) -> CachedRoboCasaLeRobotLoader:
    key = str(Path(lerobot_root).resolve())
    if key not in _LOADER_CACHE:
        _LOADER_CACHE[key] = CachedRoboCasaLeRobotLoader(key)
    return _LOADER_CACHE[key]
