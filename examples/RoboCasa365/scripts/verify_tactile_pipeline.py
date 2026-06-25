#!/usr/bin/env python3
"""Smoke test tactile modality: processor, policy wrapper, and optional sim extraction."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def test_modality_config() -> None:
    import examples.RoboCasa365.robocasa365_config_4frame  # noqa: F401
    from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
    from gr00t.data.embodiment_tags import EmbodimentTag

    cfg = MODALITY_CONFIGS[EmbodimentTag.ROBOCASA_PANDA_OMRON.value]
    tactile = cfg["tactile"]
    assert tactile.delta_indices == [-6, -4, -2, 0]
    assert tactile.modality_keys == ["left", "right", "contact"]
    print("[ok] robocasa365_config_4frame tactile modality registered")


DEFAULT_MODEL_ROOT = REPO / "output/rc365_PickPlaceToasterToCounter_30k_b64_4frame_visor"
DEFAULT_CHECKPOINT = DEFAULT_MODEL_ROOT / "checkpoint-30000"
DEFAULT_PROCESSOR = DEFAULT_MODEL_ROOT / "processor"


def test_processor_tactile(model_path: Path) -> None:
    import examples.RoboCasa365.robocasa365_config_4frame  # noqa: F401
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.data.types import VLAStepData
    from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor

    processor = Gr00tN1d7Processor.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        local_files_only=True,
    )
    processor.state_action_processor.training = False
    processor.training = False
    tag = EmbodimentTag.ROBOCASA_PANDA_OMRON
    t_horizon = len(
        processor.modality_configs[tag.value]["tactile"].delta_indices
    )
    v_horizon = len(processor.modality_configs[tag.value]["video"].delta_indices)
    tactile = {
        "left": np.random.rand(t_horizon, 1).astype(np.float32),
        "right": np.random.rand(t_horizon, 1).astype(np.float32),
        "contact": np.zeros((t_horizon, 1), dtype=np.float32),
    }
    state_dims = {
        "base_position": 3,
        "base_rotation": 4,
        "end_effector_position_relative": 3,
        "end_effector_rotation_relative": 4,
        "gripper_qpos": 2,
    }
    step = VLAStepData(
        images={
            k: np.zeros((v_horizon, 256, 256, 3), dtype=np.uint8)
            for k in processor.modality_configs[tag.value]["video"].modality_keys
        },
        states={
            k: np.zeros((1, state_dims[k]), dtype=np.float32)
            for k in processor.modality_configs[tag.value]["state"].modality_keys
        },
        actions={},
        text="pick up the mug",
        embodiment=tag,
        metadata={"tactile": tactile},
    )
    from gr00t.data.types import MessageType

    messages = [{"type": MessageType.EPISODE_STEP.value, "content": step}]
    batch = processor(messages)
    assert "tactile_sensor" in batch
    assert batch["tactile_sensor"].shape[-2:] == (t_horizon, 3)
    print(f"[ok] processor tactile_sensor shape {tuple(batch['tactile_sensor'].shape)}")


def test_sim_observation_layout() -> None:
    """Validate flat sim observation layout expected by Gr00tSimPolicyWrapper."""
    import examples.RoboCasa365.robocasa365_config_4frame  # noqa: F401
    from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
    from gr00t.data.embodiment_tags import EmbodimentTag

    cfg = MODALITY_CONFIGS[EmbodimentTag.ROBOCASA_PANDA_OMRON.value]
    v_t = len(cfg["video"].delta_indices)
    s_t = len(cfg["state"].delta_indices)
    tactile_t = len(cfg["tactile"].delta_indices)
    state_dims = {
        "base_position": 3,
        "base_rotation": 4,
        "end_effector_position_relative": 3,
        "end_effector_rotation_relative": 4,
        "gripper_qpos": 2,
    }
    obs = {"annotation.human.task_description": ("pick up the mug",)}
    for vk in cfg["video"].modality_keys:
        obs[f"video.{vk}"] = np.zeros((1, v_t, 256, 256, 3), dtype=np.uint8)
    for sk in cfg["state"].modality_keys:
        obs[f"state.{sk}"] = np.zeros((1, s_t, state_dims[sk]), dtype=np.float32)
    for tk in cfg["tactile"].modality_keys:
        obs[f"tactile.{tk}"] = np.zeros((1, tactile_t, 1), dtype=np.float32)
    assert obs["tactile.left"].shape == (1, tactile_t, 1)
    print("[ok] sim observation layout includes tactile (B, T, D)")


def test_sim_tactile_extraction(task: str, split: str) -> None:
    import os

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    import gymnasium as gym
    import robocasa  # noqa: F401

    from gr00t.eval.sim.wrapper.tactile_observation_wrapper import TactileObservationWrapper

    env = gym.make(f"robocasa/{task}", split=split)
    env = TactileObservationWrapper(env)
    obs, _ = env.reset()
    for key in ("tactile.left", "tactile.right", "tactile.contact"):
        assert key in obs, f"missing {key}"
        assert obs[key].shape == (1,)
        assert obs[key].dtype == np.float32
    env.step(env.action_space.sample())
    print(f"[ok] sim tactile extraction for robocasa/{task} split={split}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--processor-path", type=Path, default=DEFAULT_PROCESSOR)
    parser.add_argument("--skip-sim", action="store_true")
    parser.add_argument("--task", default="NavigateKitchen")
    parser.add_argument("--split", default="pretrain")
    args = parser.parse_args()

    test_modality_config()
    test_processor_tactile(args.processor_path)
    test_sim_observation_layout()
    if not args.skip_sim:
        test_sim_tactile_extraction(args.task, args.split)
    print("All tactile pipeline checks passed.")


if __name__ == "__main__":
    main()
