# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Partial-chunk closed-loop inference runner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class ClosedLoopInferenceConfig:
    action_horizon: int = 16
    execute_steps: int = 4
    recovery_threshold: float = 0.5
    use_recovery: bool = True
    use_flux: bool = False


class ClosedLoopInferenceRunner:
    """Execute ``execute_steps`` per policy call, then re-observe."""

    def __init__(
        self,
        policy_fn: Callable[[Any], dict[str, Any]],
        env_step_fn: Callable[[Any], None],
        get_obs_fn: Callable[[], Any],
        cfg: ClosedLoopInferenceConfig | None = None,
    ):
        self.policy_fn = policy_fn
        self.env_step_fn = env_step_fn
        self.get_obs_fn = get_obs_fn
        self.cfg = cfg or ClosedLoopInferenceConfig()

    def run_until_done(self, max_steps: int = 10_000) -> dict[str, Any]:
        done = False
        step = 0
        last_outputs: dict[str, Any] = {}
        while not done and step < max_steps:
            obs = self.get_obs_fn()
            outputs = self.policy_fn(obs)
            last_outputs = outputs
            chunk = outputs["action"]
            k = min(self.cfg.execute_steps, chunk.shape[1])
            for i in range(k):
                self.env_step_fn(chunk[:, i])
                step += 1
            if self.cfg.use_recovery:
                gate = float(outputs.get("monitor", {}).get("recovery_gate", 0.0))
                if hasattr(gate, "mean"):
                    gate = float(gate.mean())
                if gate > self.cfg.recovery_threshold:
                    continue
            done = obs.get("done", False) if isinstance(obs, dict) else False
        return {"steps": step, "last_outputs": last_outputs}
