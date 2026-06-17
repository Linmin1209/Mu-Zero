#!/usr/bin/env python3
"""Start GR00T PolicyServer wrapping an Echo VLA checkpoint for RoboCasa365 rollout."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from gr00t.policy.server_client import PolicyServer
import tyro

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(SCRIPT_DIR))

from echo_vla_policy import EchoVlaGr00tPolicy  # noqa: E402


@dataclass
class EchoVlaServerConfig:
    model_path: str
    echo_repo: str = os.environ.get(
        "ECHO_VLA_REPO",
        "/HOME/sysu_xdliang/sysu_xdliang_1/HDD_POOL/linmin/UR-manipulation-modelscope/Echo_VLA",
    )
    config_name: str = os.environ.get("ECHO_VLA_CONFIG", "robocasa_config_pi05.yaml")
    device: str = "cuda"
    host: str = "0.0.0.0"
    port: int = int(os.environ.get("SERVER_PORT", "5560"))


def main(cfg: EchoVlaServerConfig) -> None:
    print("[i] Echo VLA → GR00T PolicyServer")
    print(f"    echo_repo={cfg.echo_repo}")
    print(f"    model_path={cfg.model_path}")
    print(f"    port={cfg.port}")
    policy = EchoVlaGr00tPolicy(
        echo_repo=cfg.echo_repo,
        checkpoint_dir=cfg.model_path,
        config_name=cfg.config_name,
        device=cfg.device,
        strict=False,
    )
    server = PolicyServer(policy, host=cfg.host, port=cfg.port)
    server.run()


if __name__ == "__main__":
    main(tyro.cli(EchoVlaServerConfig))
