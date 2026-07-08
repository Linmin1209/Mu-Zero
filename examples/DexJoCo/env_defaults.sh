#!/usr/bin/env bash
# Portable path defaults for DexJoCo + GR00T workflows.
#
# Expected sibling layout:
#   <parent>/
#     Isaac-GR00T/
#     datasets/DexJoCo-Datasets-LeRobot/dexjoco_lerobot_datasets/
#     dexjoco/
#     models/GR00T-N1.7-3B/
#
# Override via env or copy local_paths.example.sh -> local_paths.sh

: "${SCRIPT_DIR:?Set SCRIPT_DIR before sourcing env_defaults.sh}"
: "${PROJECT_ROOT:?Set PROJECT_ROOT before sourcing env_defaults.sh}"

if [[ -f "$SCRIPT_DIR/local_paths.sh" ]]; then
  # shellcheck source=/dev/null
  source "$SCRIPT_DIR/local_paths.sh"
fi

export DEXJOCo_ROOT="${DEXJOCo_ROOT:-$PROJECT_ROOT/../dexjoco}"
export DEXJOCo_DATASETS_ROOT="${DEXJOCo_DATASETS_ROOT:-$PROJECT_ROOT/../datasets/DexJoCo-Datasets-LeRobot/dexjoco_lerobot_datasets}"
export GR00T_MODELS_ROOT="${GR00T_MODELS_ROOT:-$PROJECT_ROOT/../models}"
export GR00T_BASE_MODEL="${GR00T_BASE_MODEL:-$GR00T_MODELS_ROOT/GR00T-N1.7-3B}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export GROOT_PATCH_MISTRAL="${GROOT_PATCH_MISTRAL:-1}"
export GROOT_HF_LOCAL_FIRST="${GROOT_HF_LOCAL_FIRST:-1}"
export NO_ALBUMENTATIONS_UPDATE="${NO_ALBUMENTATIONS_UPDATE:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
