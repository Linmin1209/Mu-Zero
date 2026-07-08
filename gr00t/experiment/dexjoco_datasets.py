"""Resolve DexJoCo LeRobot v3 dataset roots for GR00T finetuning."""

from __future__ import annotations

from pathlib import Path

import yaml


DEFAULT_REGISTRY = Path(__file__).resolve().parents[2] / "examples" / "DexJoCo" / "task_registry.yaml"


def _load_registry(registry_path: Path | None = None) -> dict:
    path = registry_path or DEFAULT_REGISTRY
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_dexjoco_dataset_paths(
    root: str | Path,
    *,
    tasks: str | None = None,
    robot_type: str = "all",
    registry_path: Path | None = None,
) -> tuple[list[str], str]:
    """Return (dataset_paths, embodiment_tag_value)."""
    root = Path(root)
    registry = _load_registry(registry_path)
    known_tasks = registry["tasks"]

    if tasks:
        task_names = [t.strip() for t in tasks.split(",") if t.strip()]
    elif robot_type == "single_arm":
        task_names = list(registry["single_arm_tasks"])
    elif robot_type == "bimanual":
        task_names = list(registry["bimanual_tasks"])
    else:
        task_names = sorted(known_tasks.keys())

    paths: list[str] = []
    robot_types: set[str] = set()
    for task_name in task_names:
        if task_name not in known_tasks:
            raise ValueError(f"Unknown DexJoCo task: {task_name}")
        task_root = root / task_name
        if not (task_root / "meta" / "info.json").exists():
            raise FileNotFoundError(f"Missing DexJoCo task root: {task_root}")
        if not (task_root / "meta" / "modality.json").exists():
            raise FileNotFoundError(
                f"Missing {task_root}/meta/modality.json — run "
                "python examples/DexJoCo/prepare_dexjoco_metadata.py first."
            )
        paths.append(str(task_root.resolve()))
        robot_types.add(known_tasks[task_name]["robot_type"])

    if len(robot_types) != 1:
        raise ValueError(
            f"Mixed robot types in one run: {robot_types}. "
            "Use --dexjoco-robot-type single_arm|bimanual or filter --dexjoco-tasks."
        )
    embodiment = (
        "dexjoco_single_arm" if robot_types.pop() == "single_arm" else "dexjoco_bimanual"
    )
    return paths, embodiment


def get_default_dexjoco_modality_config_path(robot_type: str) -> Path:
    base = Path(__file__).resolve().parents[2] / "examples" / "DexJoCo"
    if robot_type == "single_arm":
        return base / "dexjoco_single_arm_config.py"
    if robot_type == "bimanual":
        return base / "dexjoco_bimanual_config.py"
    raise ValueError(f"Unknown robot_type: {robot_type}")
