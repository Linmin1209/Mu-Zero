"""Add online gripper tactile keys to gym observations for policy eval."""

from __future__ import annotations

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from gr00t.eval.sim.robocasa365.tactile_from_sim import (
    TACTILE_OBS_KEYS,
    extract_gripper_tactile_frame,
)


def _find_robosuite_env(gym_env) -> object:
    env = gym_env
    while env is not None:
        if hasattr(env, "sim") and hasattr(env, "robots"):
            return env
        env = getattr(env, "env", None)
    raise RuntimeError("Could not find robosuite env for tactile extraction")


class TactileObservationWrapper(gym.Wrapper):
    """Inject ``tactile.left/right/contact`` from MuJoCo contact forces each step."""

    def __init__(self, env):
        super().__init__(env)
        self._robosuite_env = _find_robosuite_env(env.unwrapped)
        tactile_space = spaces.Dict(
            {
                key: spaces.Box(low=0.0, high=100.0, shape=(1,), dtype=np.float32)
                for key in TACTILE_OBS_KEYS
            }
        )
        if isinstance(self.observation_space, spaces.Dict):
            obs_space = dict(self.observation_space.spaces)
            obs_space.update(tactile_space.spaces)
            self.observation_space = spaces.Dict(obs_space)
        else:
            raise TypeError(f"Unsupported observation space: {type(self.observation_space)}")

    def _append_tactile(self, obs: dict) -> dict:
        tactile = extract_gripper_tactile_frame(self._robosuite_env).as_obs_arrays()
        obs = dict(obs)
        for key in TACTILE_OBS_KEYS:
            obs[key] = tactile[key]
        return obs

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        return self._append_tactile(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        return self._append_tactile(obs), reward, terminated, truncated, info
