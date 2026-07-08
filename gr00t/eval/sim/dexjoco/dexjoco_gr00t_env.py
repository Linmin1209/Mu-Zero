"""DexJoCo MuJoCo env wrapper for GR00T policy server eval."""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R


def _ensure_dexjoco_on_path(dexjoco_root: str | Path) -> None:
    root = Path(dexjoco_root).resolve()
    pkg = root / "dexjoco"
    if not (pkg / "dexjoco").is_dir():
        raise FileNotFoundError(
            f"DexJoCo package not found under {pkg}. Set DEXJOCo_ROOT to the repo root."
        )
    if str(pkg) not in sys.path:
        sys.path.insert(0, str(pkg))


class DexJoCoGr00tEnv:
    """Wrap DexJoCo sim + convert obs/actions for GR00T PolicyClient."""

    def __init__(
        self,
        *,
        dexjoco_root: str | Path,
        env_name: str,
        camera_mapping: dict[str, str],
        prompt: str,
        dual_arm: bool,
        seed: int = 0,
        rand_full: bool = False,
        render_mode: str = "rgb_array",
    ):
        _ensure_dexjoco_on_path(dexjoco_root)
        from dexjoco.tasks import CONFIG_MAPPING

        self.env_name = env_name
        self.camera_mapping = camera_mapping
        self.prompt = prompt
        self.dual_arm = dual_arm
        self.seed = seed
        self.rand_full = rand_full
        self.render_mode = render_mode

        config = CONFIG_MAPPING[env_name]()
        self.env = config.get_environment(
            policy_mode=True,
            render_mode=render_mode,
            randomize=rand_full,
            seed=seed,
            randomize_dynamics=False,
        )
        self._done = False
        self._success = False
        self._raw_obs: dict[str, Any] = {}

    def close(self) -> None:
        if self.env is not None:
            self.env.close()
            self.env = None

    def reset(self) -> dict[str, Any]:
        obs, _ = self.env.reset()
        self._done = False
        self._success = False
        self._update_raw_obs(obs)
        return self.build_gr00t_observation()

    def step(self, action_rotvec: np.ndarray) -> tuple[dict[str, Any], dict[str, Any]]:
        env_action = self._rotvec_action_to_env(action_rotvec)
        obs, _reward, terminated, _truncated, info = self.env.step(env_action)
        self._done = bool(terminated)
        self._success = bool(info.get("succeed", False))
        self._update_raw_obs(obs)
        return self.build_gr00t_observation(), info

    @property
    def is_done(self) -> bool:
        return self._done

    @property
    def is_success(self) -> bool:
        return self._success

    def get_raw_cameras(self) -> dict[str, np.ndarray]:
        return copy.deepcopy(self._raw_obs)

    def build_gr00t_observation(self) -> dict[str, Any]:
        """Build single-sample GR00T obs dict (batch added by caller)."""
        video: dict[str, np.ndarray] = {}
        for policy_key, env_key in self.camera_mapping.items():
            img = self._raw_obs[env_key]
            if img.dtype != np.uint8:
                img = np.clip(img * 255.0, 0, 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
            video[policy_key] = img[None, None, ...]  # (B=1, T=1, H, W, C)

        state_vec = self._raw_obs["state"]
        if self.dual_arm:
            proprio = state_vec[:46].astype(np.float32)
            state = {
                "right_tcp": proprio[0:7][None, None, :],
                "left_tcp": proprio[7:14][None, None, :],
                "right_hand": proprio[14:30][None, None, :],
                "left_hand": proprio[30:46][None, None, :],
            }
        else:
            proprio = state_vec[:23].astype(np.float32)
            state = {
                "tcp_pose": proprio[0:7][None, None, :],
                "hand": proprio[7:23][None, None, :],
            }

        return {
            "video": video,
            "state": state,
            "language": {
                "annotation.human.task_description": [[self.prompt]],
            },
        }

    def _rotvec_action_to_env(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float64)
        if self.dual_arm:
            r_xyz, r_rotvec, r_hand = action[:3], action[3:6], action[6:22]
            l_xyz, l_rotvec, l_hand = action[22:25], action[25:28], action[28:44]
            r_quat = R.from_rotvec(r_rotvec).as_quat(scalar_first=True)
            l_quat = R.from_rotvec(l_rotvec).as_quat(scalar_first=True)
            return np.concatenate([r_xyz, r_quat, l_xyz, l_quat, r_hand, l_hand])
        xyz, rotvec, hand = action[:3], action[3:6], action[6:22]
        quat = R.from_rotvec(rotvec).as_quat(scalar_first=True)
        return np.concatenate([xyz, quat, hand])

    def flatten_gr00t_action_chunk(self, action_dict: dict[str, Any]) -> list[np.ndarray]:
        """Expand GR00T action dict (B,T,D per key) into rotvec steps."""
        if self.dual_arm:
            keys = ("right_eef", "right_hand", "left_eef", "left_hand")
        else:
            keys = ("eef_rotvec", "hand")

        horizon = int(action_dict[keys[0]].shape[1])
        steps: list[np.ndarray] = []
        for t in range(horizon):
            parts = [np.asarray(action_dict[k][0, t], dtype=np.float64) for k in keys]
            steps.append(np.concatenate(parts, axis=-1))
        return steps

    def _update_raw_obs(self, env_obs: dict[str, Any]) -> None:
        self._raw_obs = {}
        for _policy_key, env_key in self.camera_mapping.items():
            self._raw_obs[env_key] = np.asarray(env_obs["images"][env_key])
        self._raw_obs["state"] = np.asarray(env_obs["state"], dtype=np.float64)
