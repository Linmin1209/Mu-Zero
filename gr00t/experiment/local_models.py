# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Use pre-downloaded checkpoints under HDD_POOL/linmin/models (no HuggingFace Hub)."""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_MODELS_ROOT = Path(
    "/HOME/sysu_xdliang/sysu_xdliang_1/HDD_POOL/linmin/models"
)
DEFAULT_GR00T_BASE = DEFAULT_MODELS_ROOT / "GR00T-N1.7-3B"
DEFAULT_COSMOS = DEFAULT_MODELS_ROOT / "Cosmos-Reason2-2B"


def bootstrap_offline_hub(models_root: Path | None = None) -> Path:
    """
    Set env vars before ``import gr00t`` so HF helpers stay offline/local-first.

    Call at the top of ``launch_finetune.py`` (before other gr00t imports).
    """
    root = Path(models_root or os.environ.get("GR00T_MODELS_ROOT", DEFAULT_MODELS_ROOT))
    os.environ.setdefault("GR00T_MODELS_ROOT", str(root))
    os.environ["GROOT_PATCH_MISTRAL"] = "1"
    os.environ["GROOT_HF_LOCAL_FIRST"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ.setdefault("HF_HOME", str(root / ".hf_cache"))
    return root


def resolve_local_paths(
    base_model_path: str | None = None,
    cosmos_model_path: str | None = None,
    models_root: Path | None = None,
) -> tuple[str, str]:
    """Return absolute local paths for GR00T base + Cosmos VLM."""
    root = Path(models_root or os.environ.get("GR00T_MODELS_ROOT", DEFAULT_MODELS_ROOT))
    gr00t = Path(
        base_model_path
        or os.environ.get("GR00T_BASE_MODEL_PATH", DEFAULT_GR00T_BASE)
    )
    cosmos = Path(
        cosmos_model_path
        or os.environ.get("GR00T_COSMOS_MODEL_PATH", DEFAULT_COSMOS)
    )
    if not gr00t.is_dir():
        raise FileNotFoundError(f"GR00T base model not found: {gr00t}")
    if not (gr00t / "config.json").is_file():
        raise FileNotFoundError(f"Missing config.json under {gr00t}")
    if not cosmos.is_dir():
        raise FileNotFoundError(f"Cosmos VLM not found: {cosmos}")
    if not (cosmos / "config.json").is_file():
        raise FileNotFoundError(f"Missing config.json under {cosmos}")
    return str(gr00t.resolve()), str(cosmos.resolve())
