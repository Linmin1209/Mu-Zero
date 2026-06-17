#!/usr/bin/env bash
# DynaMem baseline matrix for RoboCasa365 benchmark tasks (RoboCasa365 sim, Panda-Omron).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
PY365_DEFAULT="$PROJECT_ROOT/gr00t/eval/sim/robocasa365/robocasa365_uv/.venv/bin/python"
RLDX_PY365="/HOME/sysu_xdliang/sysu_xdliang_1/HDD_POOL/linmin/RLDX-1/rldx/eval/sim/robocasa365/robocasa365_uv/.venv/bin/python"
PY365="${PY365:-$PY365_DEFAULT}"
if [[ ! -x "$PY365" && -x "$RLDX_PY365" ]]; then
  PY365="$RLDX_PY365"
fi

TASK_SET="${TASK_SET:-target50}"
SPLIT="${SPLIT:-pretrain}"
TASKS_FILTER="${TASKS:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/output/robocasa365_eval_dynamem}"
N_EPISODES="${N_EPISODES:-50}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-720}"
SEED="${SEED:-0}"
SKIP_SIM="${SKIP_SIM:-0}"

# Headless nodes: prefer osmesa; set MUJOCO_GL=egl if EGL libs are installed.
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"
# Manip controller: oracle (default, sim snap pick/place) or osc (delta arm only)
export DYNAMEM_MANIP_MODE="${DYNAMEM_MANIP_MODE:-oracle}"

cd "$PROJECT_ROOT"

if [[ ! -x "$PY365" ]]; then
  echo "[x] RoboCasa365 python not found: $PY365"
  echo "    Run: bash examples/RoboCasa365/setup_eval.sh"
  exit 1
fi

echo "[i] DynaMem RoboCasa365 sim baseline (Panda-Omron)"
echo "[i] Python: $PY365"
echo "[i] MUJOCO_GL=$MUJOCO_GL"
echo "[i] DYNAMEM_MANIP_MODE=$DYNAMEM_MANIP_MODE"

EXTRA=()
if [[ -n "$TASKS_FILTER" ]]; then
  EXTRA+=(--tasks "$TASKS_FILTER")
fi
if [[ "$SKIP_SIM" == "1" ]]; then
  EXTRA+=(--skip-sim)
fi

"$PY365" -u "$SCRIPT_DIR/eval_dynamem_robocasa365.py" \
  --task-set "$TASK_SET" \
  --split "$SPLIT" \
  --output-dir "$OUTPUT_ROOT" \
  --n-episodes "$N_EPISODES" \
  --max-episode-steps "$MAX_EPISODE_STEPS" \
  --seed "$SEED" \
  "${EXTRA[@]}"
