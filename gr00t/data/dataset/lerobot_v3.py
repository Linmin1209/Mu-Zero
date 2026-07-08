"""Helpers for LeRobot v3.0 datasets (DexJoCo layout)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def is_lerobot_v3(info_meta: dict[str, Any]) -> bool:
    version = str(info_meta.get("codebase_version", ""))
    return version.startswith("v3")


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).replace("\\", "/") for c in df.columns]
    return df


def _pick_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str:
    for name in candidates:
        if name in df.columns:
            return name
    raise KeyError(f"None of {candidates} found in columns: {list(df.columns)}")


def load_v3_episodes_metadata(dataset_path: Path) -> list[dict[str, Any]]:
    episodes_dir = dataset_path / "meta" / "episodes"
    frames: list[pd.DataFrame] = []
    if episodes_dir.is_dir():
        for path in sorted(episodes_dir.rglob("*.parquet")):
            frames.append(_flatten_columns(pd.read_parquet(path)))
    if frames:
        table = pd.concat(frames, ignore_index=True)
        ep_col = _pick_column(table, ("episode_index", "index"))
        table = table.sort_values(ep_col)
        records = table.to_dict(orient="records")
        return [_normalize_episode_record(rec) for rec in records]

    episodes_jsonl = dataset_path / "meta" / "episodes.jsonl"
    if episodes_jsonl.exists():
        records = [json.loads(line) for line in episodes_jsonl.read_text().splitlines() if line.strip()]
        return [_normalize_episode_record(rec) for rec in records]

    raise FileNotFoundError(
        f"No v3 episode metadata under {episodes_dir} or {episodes_jsonl}. "
        "Run: python examples/DexJoCo/prepare_dexjoco_metadata.py --task-root <task>"
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            return [_json_safe(v) for v in value.tolist()]
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def slim_episode_index_record(record: dict[str, Any]) -> dict[str, Any]:
    """Keep only fields needed by GR00T v3 loaders (drop per-episode stats blobs)."""
    normalized = _normalize_episode_record(record)
    keep = (
        "episode_index",
        "length",
        "data_chunk_index",
        "data_file_index",
        "dataset_from_index",
        "dataset_to_index",
        "tasks",
    )
    return {k: normalized[k] for k in keep if k in normalized}


def _normalize_episode_record(record: dict[str, Any]) -> dict[str, Any]:
    out = {str(k): _json_safe(v) for k, v in record.items()}
    if "length" not in out and "dataset_to_index" in out and "dataset_from_index" in out:
        out["length"] = int(out["dataset_to_index"]) - int(out["dataset_from_index"]) + 1
    chunk = out.get("data/chunk_index", out.get("chunk_index", out.get("data_chunk_index")))
    file_idx = out.get("data/file_index", out.get("file_index", out.get("data_file_index")))
    if chunk is not None:
        out["data_chunk_index"] = int(chunk)
    if file_idx is not None:
        out["data_file_index"] = int(file_idx)
    if "dataset_from_index" in out:
        out["dataset_from_index"] = int(out["dataset_from_index"])
    if "dataset_to_index" in out:
        out["dataset_to_index"] = int(out["dataset_to_index"])
    if "episode_index" not in out and "index" in out:
        out["episode_index"] = int(out["index"])
    return out


def load_v3_tasks_map(dataset_path: Path) -> dict[int, str]:
    tasks_jsonl = dataset_path / "meta" / "tasks.jsonl"
    if tasks_jsonl.exists():
        tasks: dict[int, str] = {}
        for line in tasks_jsonl.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            tasks[int(row["task_index"])] = row["task"]
        return tasks

    tasks_parquet = dataset_path / "meta" / "tasks.parquet"
    if tasks_parquet.exists():
        table = _flatten_columns(pd.read_parquet(tasks_parquet))
        idx_col = _pick_column(table, ("task_index", "index"))
        task_col = _pick_column(table, ("task", "prompt", "language_instruction"))
        return {int(row[idx_col]): str(row[task_col]) for _, row in table.iterrows()}

    raise FileNotFoundError(
        f"Missing {tasks_jsonl} and {tasks_parquet}. "
        "Run examples/DexJoCo/prepare_dexjoco_metadata.py first."
    )


def resolve_v3_data_parquet_path(
    dataset_path: Path, info_meta: dict[str, Any], episode_meta: dict[str, Any]
) -> Path:
    chunk_index = int(
        episode_meta.get(
            "data_chunk_index",
            episode_meta.get("data/chunk_index", episode_meta.get("chunk_index", 0)),
        )
    )
    file_index = int(
        episode_meta.get(
            "data_file_index",
            episode_meta.get("data/file_index", episode_meta.get("file_index", 0)),
        )
    )
    rel = info_meta["data_path"].format(chunk_index=chunk_index, file_index=file_index)
    return dataset_path / rel


def resolve_v3_video_parquet_path(
    dataset_path: Path,
    info_meta: dict[str, Any],
    episode_meta: dict[str, Any],
    video_key: str,
) -> Path:
    chunk_index = int(
        episode_meta.get(
            "data_chunk_index",
            episode_meta.get("data/chunk_index", episode_meta.get("chunk_index", 0)),
        )
    )
    file_index = int(
        episode_meta.get(
            "data_file_index",
            episode_meta.get("data/file_index", episode_meta.get("file_index", 0)),
        )
    )
    rel = info_meta["video_path"].format(
        video_key=video_key, chunk_index=chunk_index, file_index=file_index
    )
    return dataset_path / rel
