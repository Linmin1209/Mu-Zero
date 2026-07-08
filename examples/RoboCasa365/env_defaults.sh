#!/usr/bin/env bash
# Portable path/env defaults for RoboCasa365 finetune scripts.
#
# Expected layout (sibling dirs, all paths relative to repo root):
#   <parent>/
#     Isaac-GR00T/          <- PROJECT_ROOT
#     datasets/robocasa365-datasets/
#     models/GR00T-N1.7-3B, Cosmos-Reason2-2B/
#
# Override any variable via env, or copy local_paths.example.sh -> local_paths.sh

: "${SCRIPT_DIR:?Set SCRIPT_DIR before sourcing env_defaults.sh}"
: "${PROJECT_ROOT:?Set PROJECT_ROOT before sourcing env_defaults.sh}"

if [[ -f "$SCRIPT_DIR/local_paths.sh" ]]; then
  # shellcheck source=/dev/null
  source "$SCRIPT_DIR/local_paths.sh"
fi

export ROBOCASA365_ROOT="${ROBOCASA365_ROOT:-$PROJECT_ROOT/../datasets/robocasa365-datasets}"
export GR00T_MODELS_ROOT="${GR00T_MODELS_ROOT:-$PROJECT_ROOT/../models}"
export GR00T_BASE_MODEL="${GR00T_BASE_MODEL:-$GR00T_MODELS_ROOT/GR00T-N1.7-3B}"
export GR00T_COSMOS_MODEL_PATH="${GR00T_COSMOS_MODEL_PATH:-$GR00T_MODELS_ROOT/Cosmos-Reason2-2B}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export GROOT_PATCH_MISTRAL="${GROOT_PATCH_MISTRAL:-1}"
export GROOT_HF_LOCAL_FIRST="${GROOT_HF_LOCAL_FIRST:-1}"
export NO_ALBUMENTATIONS_UPDATE="${NO_ALBUMENTATIONS_UPDATE:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

# MOSS fusion gate (MotionFusionGate): text-only task-conditional gating by default.
# g = g_min + (g_max - g_min) * sigmoid(MLP(LayerNorm(text_ctx)))
# Defaults: mode=text_only, g=[0,0.8], init_bias=0 -> initial g≈0.4
export MOTION_USE_GATING="${MOTION_USE_GATING:-1}"
export MOTION_GATE_MODE="${MOTION_GATE_MODE:-text_only}"
export MOTION_GATE_INIT_BIAS="${MOTION_GATE_INIT_BIAS:-0.0}"
export MOTION_GATE_G_MIN="${MOTION_GATE_G_MIN:-0.0}"
export MOTION_GATE_G_MAX="${MOTION_GATE_G_MAX:-0.8}"
export MOTION_GATE_LR_SCALE="${MOTION_GATE_LR_SCALE:-5.0}"
export MOTION_GATE_HIDDEN="${MOTION_GATE_HIDDEN:-256}"

# Append MOSS gating CLI flags to a MOTION_ARGS bash array (nameref).
append_motion_gate_cli_args() {
  local -n _motion_args=$1
  if [[ "${MOTION_USE_GATING}" == "1" ]]; then
    _motion_args+=(--motion-use-gating)
  else
    _motion_args+=(--no-motion-use-gating)
  fi
  _motion_args+=(--motion-gate-mode "${MOTION_GATE_MODE}")
  _motion_args+=(--motion-gate-init-bias "${MOTION_GATE_INIT_BIAS}")
  _motion_args+=(--motion-gate-g-min "${MOTION_GATE_G_MIN}")
  _motion_args+=(--motion-gate-g-max "${MOTION_GATE_G_MAX}")
  _motion_args+=(--motion-gate-lr-scale "${MOTION_GATE_LR_SCALE}")
  if [[ -n "${MOTION_GATE_HIDDEN}" ]]; then
    _motion_args+=(--motion-gate-hidden "${MOTION_GATE_HIDDEN}")
  fi
}
