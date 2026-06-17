#!/usr/bin/env python3
"""Evaluate Echo VLA checkpoint on RoboCasa365 target50 (50 tasks × N episodes).

Uses GR00T rollout_policy + Echo PolicyServer (run_echo_vla_server.py).
Output: summary_shard0of1.csv (same format as GR00T / LEO / DynaMem).
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
RC365_DIR = SCRIPT_DIR.parent.parent
PROJECT_ROOT = RC365_DIR.parent.parent
TASK_SETS_YAML = RC365_DIR / "task_sets.yaml"
COMPAT_YAML = SCRIPT_DIR / "task_compatibility.yaml"


def load_task_list(task_set: str, task_sets_path: Path) -> list[str]:
    cfg = yaml.safe_load(task_sets_path.read_text())
    if task_set == "target50":
        return (
            list(cfg["atomic_seen"])
            + list(cfg["composite_seen"])
            + list(cfg["composite_unseen"])
        )
    return list(cfg[task_set])


def load_compat() -> dict:
    if COMPAT_YAML.is_file():
        return yaml.safe_load(COMPAT_YAML.read_text()) or {}
    return {}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--echo-repo", type=Path, default=Path(os.environ.get(
        "ECHO_VLA_REPO",
        "/HOME/sysu_xdliang/sysu_xdliang_1/HDD_POOL/linmin/UR-manipulation-modelscope/Echo_VLA",
    )))
    p.add_argument("--task-set", default="target50")
    p.add_argument("--tasks", default="")
    p.add_argument("--split", default="pretrain", choices=["pretrain", "target"])
    p.add_argument("--task-yaml", type=Path, default=TASK_SETS_YAML)
    p.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "output" / "robocasa365_eval_echo_vla")
    p.add_argument("--n-episodes", type=int, default=50)
    p.add_argument("--max-episode-steps", type=int, default=720)
    p.add_argument("--n-action-steps", type=int, default=16)
    p.add_argument("--server-port", type=int, default=int(os.environ.get("SERVER_PORT", "5560")))
    p.add_argument("--server-device", default="cuda")
    p.add_argument("--skip-sim", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    tasks = load_task_list(args.task_set, args.task_yaml)
    if args.tasks.strip():
        wanted = {t.strip() for t in args.tasks.split(",") if t.strip()}
        tasks = [t for t in tasks if t in wanted]

    compat = load_compat()
    na_set = set(compat.get("na") or [])

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir / f"echo_vla_{args.task_set}_{args.split}_exp{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "summary_shard0of1.csv"

    py365 = os.environ.get(
        "PY365",
        str(PROJECT_ROOT / "gr00t/eval/sim/robocasa365/robocasa365_uv/.venv/bin/python"),
    )
    gr00t_py = os.environ.get("GR00T_PYTHON", str(PROJECT_ROOT / ".venv/bin/python"))
    server_proc = None

    if not args.skip_sim:
        if not Path(py365).is_file():
            print(f"[x] RoboCasa365 python not found: {py365}")
            return 1
        server_cmd = [
            gr00t_py, "-u", str(SCRIPT_DIR / "run_echo_vla_server.py"),
            "--model-path", str(args.model_path),
            "--echo-repo", str(args.echo_repo),
            "--port", str(args.server_port),
            "--device", args.server_device,
        ]
        print("[i] Starting Echo VLA policy server:", " ".join(server_cmd))
        server_proc = subprocess.Popen(server_cmd, cwd=str(PROJECT_ROOT))
        time.sleep(15)

    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

    with summary_path.open("w", newline="", encoding="utf-8") as sf:
        writer = csv.writer(sf)
        writer.writerow(["task", "success_rate", "log_file", "status"])
        rates: list[float] = []

        for task in tasks:
            log_file = run_dir / task / "eval.log"
            log_file.parent.mkdir(parents=True, exist_ok=True)
            rate_str = ""
            status = "ok"

            if task in na_set:
                status = "na"
                log_file.write_text(f"task={task}\nstatus=na (no Echo mapping)\n", encoding="utf-8")
            elif args.skip_sim:
                status = "skipped"
            else:
                rollout_cmd = [
                    py365, "-m", "gr00t.eval.rollout_policy",
                    "--env-name", f"robocasa/{task}",
                    "--robocasa-split", args.split,
                    "--n-episodes", str(args.n_episodes),
                    "--max-episode-steps", str(args.max_episode_steps),
                    "--n-action-steps", str(args.n_action_steps),
                    "--policy-client-host", "127.0.0.1",
                    "--policy-client-port", str(args.server_port),
                    "--seed", str(args.seed),
                    "--output-dir", str(log_file.parent),
                ]
                print(f"[i] Rollout: {task}")
                rc = subprocess.run(rollout_cmd, cwd=str(PROJECT_ROOT))
                if rc.returncode != 0:
                    status = "error"
                else:
                    metrics = log_file.parent / "metrics.json"
                    if metrics.is_file():
                        import json
                        m = json.loads(metrics.read_text())
                        sr = m.get("success_rate", m.get("mean_success", 0.0))
                        rate_str = f"{float(sr):.4f}"
                        rates.append(float(sr))

            writer.writerow([task, rate_str, str(log_file), status])

    if server_proc is not None:
        server_proc.terminate()
        try:
            server_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server_proc.kill()

    if rates:
        print(f"[i] Mean success over {len(rates)} tasks: {sum(rates)/len(rates):.4f}")
    print(f"[i] Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
