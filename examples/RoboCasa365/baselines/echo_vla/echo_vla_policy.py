#!/usr/bin/env python3
"""GR00T BasePolicy wrapper for Echo VLA checkpoints (PI0.5 / DDPM)."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

from gr00t.data.types import ModalityConfig
from gr00t.policy.policy import BasePolicy

SCRIPT_DIR = Path(__file__).resolve().parent
from action_adapter import echo_chunk_to_rc365  # noqa: E402


class EchoVlaGr00tPolicy(BasePolicy):
    """Loads Echo agent from checkpoint; exposes GR00T policy API for rollout_policy."""

    def __init__(
        self,
        echo_repo: Path | str,
        checkpoint_dir: Path | str,
        config_name: str = "robocasa_config_pi05.yaml",
        device: str = "cuda",
        strict: bool = False,
    ):
        self.echo_repo = Path(echo_repo)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.config_name = config_name
        self.device = device
        self.strict = strict
        self._agent = None
        self._cfg = None

    def _ensure_agent(self) -> None:
        if self._agent is not None:
            return
        if not self.echo_repo.is_dir():
            raise FileNotFoundError(f"ECHO_VLA_REPO not found: {self.echo_repo}")
        if not self.checkpoint_dir.is_dir():
            raise FileNotFoundError(f"Checkpoint dir not found: {self.checkpoint_dir}")

        sys.path.insert(0, str(self.echo_repo))
        os.chdir(self.echo_repo)

        import hydra
        from hydra import compose, initialize_config_dir
        import torch

        cfg_dir = self.echo_repo / "configs"
        with initialize_config_dir(config_dir=str(cfg_dir), version_base=None):
            cfg = compose(config_name=self.config_name.replace(".yaml", ""))
        cfg.eval_dir = str(self.checkpoint_dir)
        cfg.device = self.device

        self._cfg = cfg
        agent = hydra.utils.instantiate(cfg.agents)
        ckpt = self._find_checkpoint(self.checkpoint_dir)
        state = torch.load(ckpt, map_location=self.device)
        agent.load_state_dict(state, strict=False)
        agent.eval()
        agent.to(self.device)
        self._agent = agent

    @staticmethod
    def _find_checkpoint(model_dir: Path) -> Path:
        for name in ("best_val_model.pth", "last_model.pth"):
            p = model_dir / name
            if p.is_file():
                return p
        matches = sorted(model_dir.glob("best_val_model*.pth"))
        if matches:
            return matches[0]
        matches = sorted(model_dir.glob("*.pth"))
        if matches:
            return matches[0]
        raise FileNotFoundError(f"No .pth checkpoint in {model_dir}")

    def get_modality_config(self) -> dict[str, ModalityConfig]:
        self._ensure_agent()
        return {}

    def get_action(
        self,
        observation: dict[str, Any],
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._ensure_agent()
        agent = self._agent

        # Build Echo obs dict from GR00T sim observation keys
        obs = self._observation_to_echo(observation)
        with __import__("torch").no_grad():
            action_chunk = agent.predict(obs)
        if hasattr(action_chunk, "cpu"):
            action_chunk = action_chunk.cpu().numpy()
        action_chunk = np.asarray(action_chunk, dtype=np.float32)
        if action_chunk.ndim == 1:
            action_chunk = action_chunk[None, :]
        rc365_chunk = echo_chunk_to_rc365(action_chunk)
        return {"action": rc365_chunk}

    def _observation_to_echo(self, observation: dict[str, Any]) -> dict[str, Any]:
        """Map GR00T/RoboCasa365 obs to Echo agent input (minimal v1: RGB + state + lang)."""
        out: dict[str, Any] = {}
        for k, v in observation.items():
            out[k] = v
        if "language" not in out and "annotation.human.task_description" in out:
            lang = out["annotation.human.task_description"]
            if isinstance(lang, (list, tuple)):
                lang = lang[0]
            out["language"] = lang
        return out

    def reset(self, options: dict[str, Any] | None = None) -> None:
        if self._agent is not None and hasattr(self._agent, "reset"):
            self._agent.reset()
