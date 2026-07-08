#!/usr/bin/env python3
"""Evaluate a GR00T checkpoint on DexJoCo MuJoCo tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from gr00t.eval.sim.dexjoco.dexjoco_gr00t_env import DexJoCoGr00tEnv
from gr00t.policy.server_client import PolicyClient


def _load_task(registry_path: Path, task_name: str) -> dict:
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    if task_name not in registry["tasks"]:
        raise KeyError(f"Unknown task {task_name}")
    return registry["tasks"][task_name]


def run_episode(
    env: DexJoCoGr00tEnv,
    client: PolicyClient,
    *,
    max_steps: int,
) -> tuple[bool, int]:
    obs = env.reset()
    action_buffer: list[np.ndarray] = []
    step = 0
    while step < max_steps and not env.is_done:
        if not action_buffer:
            action_dict, _info = client.get_action(obs)
            action_buffer = env.flatten_gr00t_action_chunk(action_dict)
        action = action_buffer.pop(0)
        obs, _info = env.step(action)
        step += 1
    return env.is_success, step


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dexjoco-root", type=Path, required=True)
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--registry", type=Path, default=None)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--n-episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rand-full", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    registry_path = args.registry or (
        Path(__file__).resolve().parents[3] / "examples" / "DexJoCo" / "task_registry.yaml"
    )
    task_cfg = _load_task(registry_path, args.task)
    dual_arm = task_cfg["robot_type"] == "bimanual"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    client = PolicyClient(host=args.host, port=args.port, strict=False)

    successes = 0
    results = []
    try:
        for ep in range(args.n_episodes):
            env = DexJoCoGr00tEnv(
                dexjoco_root=args.dexjoco_root,
                env_name=args.task,
                camera_mapping=task_cfg["camera_mapping"],
                prompt=task_cfg["prompt"],
                dual_arm=dual_arm,
                seed=args.seed + ep,
                rand_full=args.rand_full,
            )
            try:
                ok, steps = run_episode(env, client, max_steps=args.max_steps)
            finally:
                env.close()
            successes += int(ok)
            results.append({"episode": ep, "success": ok, "steps": steps})
            print(f"[ep {ep}] success={ok} steps={steps}")
    finally:
        client.close()

    summary = {
        "task": args.task,
        "n_episodes": args.n_episodes,
        "success_rate": successes / max(args.n_episodes, 1),
        "seed": args.seed,
        "rand_full": args.rand_full,
        "results": results,
    }
    out_path = args.output_dir / f"{args.task}_eval.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"[ok] success_rate={summary['success_rate']:.3f} -> {out_path}")


if __name__ == "__main__":
    main()
