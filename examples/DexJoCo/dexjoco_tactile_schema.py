"""Per-pad Allegro tactile schema for DexJoCo (4 fingertips + palm per hand)."""

from __future__ import annotations

from typing import Any

# One tactile pad per Allegro link group: 4 fingertips + palm.
PAD_KEYS = ("ff", "mf", "rf", "thumb", "palm")

# Stacked vector layout: [all forces..., all contacts...]
SINGLE_ARM_FORCE_KEYS = PAD_KEYS
SINGLE_ARM_CONTACT_KEYS = tuple(f"{k}_contact" for k in PAD_KEYS)
SINGLE_ARM_TACTILE_KEYS = SINGLE_ARM_FORCE_KEYS + SINGLE_ARM_CONTACT_KEYS

BIMANUAL_FORCE_KEYS = tuple(f"R_{k}" for k in PAD_KEYS) + tuple(f"L_{k}" for k in PAD_KEYS)
BIMANUAL_CONTACT_KEYS = tuple(f"R_{k}_contact" for k in PAD_KEYS) + tuple(
    f"L_{k}_contact" for k in PAD_KEYS
)
BIMANUAL_TACTILE_KEYS = BIMANUAL_FORCE_KEYS + BIMANUAL_CONTACT_KEYS


def tactile_keys(*, dual_arm: bool) -> tuple[str, ...]:
    return BIMANUAL_TACTILE_KEYS if dual_arm else SINGLE_ARM_TACTILE_KEYS


def tactile_num_force(*, dual_arm: bool) -> int:
    return 10 if dual_arm else 5


def tactile_num_contact(*, dual_arm: bool) -> int:
    return 10 if dual_arm else 5


def tactile_dim(*, dual_arm: bool) -> int:
    return tactile_num_force(dual_arm=dual_arm) + tactile_num_contact(dual_arm=dual_arm)


def parquet_column(key: str) -> str:
    return f"tactile.{key}"


def tactile_feature_template(*, dual_arm: bool) -> dict[str, dict[str, Any]]:
    features: dict[str, dict[str, Any]] = {}
    for key in tactile_keys(dual_arm=dual_arm):
        col = parquet_column(key)
        if key.endswith("_contact"):
            features[col] = {"dtype": "float32", "shape": [1], "names": ["contact"]}
        else:
            features[col] = {"dtype": "float32", "shape": [1], "names": ["force_n"]}
    return features


def modality_tactile_mapping(*, dual_arm: bool) -> dict[str, dict[str, str]]:
    return {key: {"original_key": parquet_column(key)} for key in tactile_keys(dual_arm=dual_arm)}


def stack_frame_values(frame: dict[str, float], *, dual_arm: bool) -> list[float]:
    """Order matches modality_keys → VISOR stacked vector."""
    keys = tactile_keys(dual_arm=dual_arm)
    return [float(frame[k]) for k in keys]
