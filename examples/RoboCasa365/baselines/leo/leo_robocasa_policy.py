#!/usr/bin/env python3
"""LEO LoRA policy rollout on RoboCasa365 sim (Panda-Omron).

Loads checkpoint from finetune_leo_target50_lora.sh and runs gym rollouts.
Integrates with eval_robocasa365.sh-style policy server when ready.

Until LEO inference is wired, raises NotImplementedError with clear message.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def run_leo_sim_rollouts(
    model_path: Path | str,
    task_name: str,
    split: str = "pretrain",
    n_episodes: int = 50,
    max_episode_steps: int = 720,
    seed: int | None = 0,
) -> dict[str, Any]:
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

    model_path = Path(model_path)
    if not model_path.is_dir():
        raise FileNotFoundError(f"LEO checkpoint dir not found: {model_path}")

    ckpt_marker = model_path / "train_config.json"
    if not ckpt_marker.is_file():
        raise FileNotFoundError(
            f"Expected trained checkpoint at {model_path} (missing train_config.json). "
            "Run finetune_leo_target50_lora.sh first."
        )

    # TODO: load LEO + LoRA + action head; decode 12D actions per step
    raise NotImplementedError(
        "LEO sim rollout pending: implement load_leo_policy() and step loop. "
        "Use eval_robocasa365.sh two-process layout (policy server + sim client) "
        "once leo_policy_server.py is added."
    )
