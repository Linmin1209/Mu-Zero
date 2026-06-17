#!/usr/bin/env bash
# Evaluate Echo VLA checkpoint on RoboCasa365 target50 (50 tasks × 50 episodes).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
PY365_DEFAULT="$PROJECT_ROOT/gr00t/eval/sim/robocasa365/robocasa365_uv/.venv/bin/python"
RLDX_PY365="/HOME/sysu_xdliang/sysu_xdliang_1/HDD_POOL/linmin/RLDX-1/rldx/eval/sim/robocasa365/robocasa365_uv/.venv/bin/python"
PY365="${PY365:-$PY365_DEFAULT}"
if [[ ! -x "$PY365" && -x "$RLDX_PY365" ]]; then
  PY365="$RLDX_PY365"
fi

ECHO_VLA_REPO="${ECHO_VLA_REPO:-/HOME/sysu_xdliang/sysu_xdliang_1/HDD_POOL/linmin/UR-manipulation-modelscope/Echo_VLA}"
MODEL_PATH="${MODEL_PATH:-}"
TASK_SET="${TASK_SET:-target50}"
SPLIT="${SPLIT:-pretrain}"
TASKS_FILTER="${TASKS:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/output/robocasa365_eval_echo_vla}"
N_EPISODES="${N_EPISODES:-50}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-720}"
N_ACTION_STEPS="${N_ACTION_STEPS:-16}"
SEED="${SEED:-0}"
SERVER_PORT="${SERVER_PORT:-5560}"
SKIP_SIM="${SKIP_SIM:-0}"

export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"
export PY365
export ECHO_VLA_REPO

cd "$PROJECT_ROOT"

if [[ ! -x "$PY365" ]]; then
  echo "[x] RoboCasa365 python not found. Run: bash examples/RoboCasa365/setup_eval.sh"
  exit 1
fi

if [[ -z "$MODEL_PATH" || ! -d "$MODEL_PATH" ]]; then
  echo "[x] Set MODEL_PATH to Echo checkpoint dir (eval_dir with best_val_model*.pth)"
  exit 1
fi

if [[ ! -d "$ECHO_VLA_REPO" ]]; then
  echo "[x] Echo VLA repo not found: $ECHO_VLA_REPO"
  exit 1
fi

echo "[i] Echo VLA target50 eval"
echo "[i] MODEL_PATH=$MODEL_PATH"
echo "[i] ECHO_VLA_REPO=$ECHO_VLA_REPO"
echo "[i] TASK_SET=$TASK_SET SPLIT=$SPLIT N_EPISODES=$N_EPISODES"

EXTRA=()
if [[ -n "$TASKS_FILTER" ]]; then
  EXTRA+=(--tasks "$TASKS_FILTER")
fi
if [[ "$SKIP_SIM" == "1" ]]; then
  EXTRA+=(--skip-sim)
fi

"$PY365" -u "$SCRIPT_DIR/eval_echo_vla_robocasa365.py" \
  --model-path "$MODEL_PATH" \
  --echo-repo "$ECHO_VLA_REPO" \
  --task-set "$TASK_SET" \
  --split "$SPLIT" \
  --output-dir "$OUTPUT_ROOT" \
  --n-episodes "$N_EPISODES" \
  --max-episode-steps "$MAX_EPISODE_STEPS" \
  --n-action-steps "$N_ACTION_STEPS" \
  --server-port "$SERVER_PORT" \
  --seed "$SEED" \
  "${EXTRA[@]}"
