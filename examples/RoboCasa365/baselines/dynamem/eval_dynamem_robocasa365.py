#!/usr/bin/env python3
"""Generate DynaMem baseline row for each RoboCasa365 benchmark task.

Runs a RoboCasa365 sim (Panda-Omron) decomposed baseline for tasks in scope:
  - navigate_only: oracle base navigation (NavigateKitchen)
  - ovmm: navigate -> pick -> navigate -> place oracle pipeline

Tasks outside DynaMem scope leave success_rate empty in the summary CSV.

Usage:
  python eval_dynamem_robocasa365.py --task-set target50 --split pretrain
  python eval_dynamem_robocasa365.py --tasks NavigateKitchen --n-episodes 10
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
COMPAT_YAML = SCRIPT_DIR / "task_compatibility.yaml"

SIM_STATUSES = frozenset({"navigate_only", "ovmm"})


def load_task_list(task_set: str, task_sets_path: Path) -> list[str]:
    cfg = yaml.safe_load(task_sets_path.read_text())
    if task_set == "target50":
        return (
            list(cfg["atomic_seen"])
            + list(cfg["composite_seen"])
            + list(cfg["composite_unseen"])
        )
    if task_set not in cfg:
        raise KeyError(f"Unknown task-set {task_set!r}; see {task_sets_path}")
    return list(cfg[task_set])


def load_compatibility(path: Path) -> dict:
    data = yaml.safe_load(path.read_text())
    return data.get("tasks", {})


def should_run_sim(status: str, skip_sim: bool) -> bool:
    return not skip_sim and status in SIM_STATUSES


def format_rate(rate: float | None) -> str:
    if rate is None:
        return ""
    return f"{rate:.4f}"


def write_task_log(
    log_path: Path,
    task: str,
    meta: dict,
    status: str,
    result: dict | None,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "baseline=DynaMem (RoboCasa365 sim decomposed oracle)",
        f"task={task}",
        f"status={status}",
        f"dynamem_query={meta.get('dynamem_query', '')}",
        f"notes={meta.get('notes', '')}",
        "",
        "Embodiment: Panda-Omron (RoboCasa365 default sim).",
        "Does NOT use stretch_ai / Stretch hardware / AnyGrasp.",
        "Pipeline: navigate -> pick -> navigate -> place (ovmm) or navigate-only.",
        "",
    ]
    if result is not None:
        lines.extend(
            [
                f"n_episodes={result['n_episodes']}",
                f"success_rate={result['success_rate']:.4f}",
                f"successes={sum(result['successes'])}/{len(result['successes'])}",
                f"mean_episode_length={sum(result['episode_lengths']) / max(len(result['episode_lengths']), 1):.1f}",
            ]
        )
    else:
        lines.append("success rate: NA (out of DynaMem scope on RoboCasa365)")
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-set", default="target50")
    parser.add_argument("--tasks", default="", help="Comma-separated task filter")
    parser.add_argument("--split", default="pretrain", choices=["pretrain", "target"])
    parser.add_argument("--task-yaml", type=Path, default=TASK_SETS_YAML)
    parser.add_argument("--compat-yaml", type=Path, default=COMPAT_YAML)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "output" / "robocasa365_eval_dynamem",
    )
    parser.add_argument("--n-episodes", type=int, default=50)
    parser.add_argument("--max-episode-steps", type=int, default=720)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--skip-sim",
        action="store_true",
        help="Only write compatibility matrix without running RoboCasa sim",
    )
    args = parser.parse_args()

    tasks = load_task_list(args.task_set, args.task_yaml)
    if args.tasks.strip():
        wanted = {t.strip() for t in args.tasks.split(",") if t.strip()}
        tasks = [t for t in tasks if t in wanted]

    compat = load_compatibility(args.compat_yaml)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir / f"dynamem_{args.task_set}_{args.split}_exp{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    summary_path = run_dir / "summary_shard0of1.csv"
    matrix_path = run_dir / "compatibility_matrix.csv"

    counts = {"na": 0, "navigate_only": 0, "ovmm": 0, "composite_na": 0, "unknown": 0}
    sim_evaluated = 0

    if not args.skip_sim:
        from dynamem_robocasa_sim import run_dynamem_sim_rollouts

    with summary_path.open("w", newline="", encoding="utf-8") as sf, matrix_path.open(
        "w", newline="", encoding="utf-8"
    ) as mf:
        summary_writer = csv.writer(sf)
        matrix_writer = csv.writer(mf)
        summary_writer.writerow(["task", "success_rate", "log_file"])
        matrix_writer.writerow(
            ["task", "status", "success_rate", "dynamem_query", "log_file", "notes"]
        )

        for task in tasks:
            meta = compat.get(task, {})
            status = meta.get("status", "unknown")
            counts[status if status in counts else "unknown"] += 1

            task_dir = run_dir / task
            log_file = task_dir / "eval.log"
            result = None
            rate_str = ""

            if should_run_sim(status, args.skip_sim):
                print(f"[i] Sim eval: {task} ({status})")
                result = run_dynamem_sim_rollouts(
                    task_name=task,
                    compat_status=status,
                    split=args.split,
                    n_episodes=args.n_episodes,
                    max_episode_steps=args.max_episode_steps,
                    seed=args.seed,
                )
                rate_str = format_rate(result["success_rate"])
                sim_evaluated += 1

            write_task_log(log_file, task, meta, status, result)
            summary_writer.writerow([task, rate_str, str(log_file)])
            matrix_writer.writerow(
                [
                    task,
                    status,
                    rate_str,
                    meta.get("dynamem_query", ""),
                    str(log_file),
                    meta.get("notes", ""),
                ]
            )

    readme = run_dir / "README.txt"
    readme.write_text(
        "\n".join(
            [
                "DynaMem baseline matrix for RoboCasa365",
                f"task_set={args.task_set} split={args.split} n_episodes={args.n_episodes}",
                "",
                "RoboCasa365 sim (Panda-Omron) decomposed oracle baseline.",
                "In-scope tasks (navigate_only + ovmm) have numeric success_rate.",
                "Out-of-scope tasks leave success_rate empty.",
                "",
                "Counts by compatibility status:",
                *(f"  {k}: {v}" for k, v in sorted(counts.items())),
                f"  sim_evaluated: {sim_evaluated}",
                "",
                "Upstream DynaMem paper: https://dynamem.github.io/",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"[i] Wrote {summary_path}")
    print(f"[i] Wrote {matrix_path}")
    print(f"[i] Run dir: {run_dir}")
    print(f"[i] Status counts: {counts}")
    print(f"[i] Sim evaluated: {sim_evaluated}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
