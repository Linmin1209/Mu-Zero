#!/usr/bin/env python3
"""Evaluate one LEO LoRA checkpoint on RoboCasa365 target50 (50 tasks × N episodes).

Output format matches GR00T / DynaMem: summary_shard0of1.csv

Usage:
  python eval_leo_robocasa365.py \\
    --model-path output/leo_rc365_target50_lora \\
    --task-set target50 --split pretrain --n-episodes 50
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
import sys

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
RC365_DIR = SCRIPT_DIR.parent.parent
PROJECT_ROOT = RC365_DIR.parent.parent
TASK_SETS_YAML = RC365_DIR / "task_sets.yaml"


def load_task_list(task_set: str, task_sets_path: Path) -> list[str]:
    cfg = yaml.safe_load(task_sets_path.read_text())
    if task_set == "target50":
        return (
            list(cfg["atomic_seen"])
            + list(cfg["composite_seen"])
            + list(cfg["composite_unseen"])
        )
    if task_set not in cfg:
        raise KeyError(f"Unknown task-set {task_set!r}")
    return list(cfg[task_set])


def write_task_log(log_path: Path, task: str, result: dict | None) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "baseline=LEO (multi-task LoRA, single checkpoint)",
        f"task={task}",
        "",
    ]
    if result is not None:
        lines.extend(
            [
                f"n_episodes={result['n_episodes']}",
                f"success_rate={result['success_rate']:.4f}",
                f"successes={sum(result['successes'])}/{len(result['successes'])}",
            ]
        )
    else:
        lines.append("eval skipped (policy server not ready)")
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--task-set", default="target50")
    parser.add_argument("--tasks", default="", help="Comma-separated filter")
    parser.add_argument("--split", default="pretrain", choices=["pretrain", "target"])
    parser.add_argument("--task-yaml", type=Path, default=TASK_SETS_YAML)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "output" / "robocasa365_eval_leo",
    )
    parser.add_argument("--n-episodes", type=int, default=50)
    parser.add_argument("--max-episode-steps", type=int, default=720)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-sim", action="store_true")
    args = parser.parse_args()

    tasks = load_task_list(args.task_set, args.task_yaml)
    if args.tasks.strip():
        wanted = {t.strip() for t in args.tasks.split(",") if t.strip()}
        tasks = [t for t in tasks if t in wanted]

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir / f"leo_{args.task_set}_{args.split}_exp{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "summary_shard0of1.csv"

    if not args.skip_sim:
        try:
            from leo_robocasa_policy import run_leo_sim_rollouts  # noqa: WPS433
        except ImportError as e:
            print(f"[x] Policy module not ready: {e}")
            print("    Complete leo_robocasa_policy.py after LoRA training.")
            return 1

    with summary_path.open("w", newline="", encoding="utf-8") as sf:
        writer = csv.writer(sf)
        writer.writerow(["task", "success_rate", "log_file"])
        rates: list[float] = []

        for task in tasks:
            log_file = run_dir / task / "eval.log"
            result = None
            rate_str = ""

            if not args.skip_sim:
                print(f"[i] Sim eval: {task}")
                result = run_leo_sim_rollouts(
                    model_path=args.model_path,
                    task_name=task,
                    split=args.split,
                    n_episodes=args.n_episodes,
                    max_episode_steps=args.max_episode_steps,
                    seed=args.seed,
                )
                rate_str = f"{result['success_rate']:.4f}"
                rates.append(result["success_rate"])

            write_task_log(log_file, task, result)
            writer.writerow([task, rate_str, str(log_file)])

    if rates:
        mean_rate = sum(rates) / len(rates)
        print(f"[i] Mean success over {len(rates)} tasks: {mean_rate:.4f}")
    print(f"[i] Wrote {summary_path}")
    print(f"[i] Run dir: {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
