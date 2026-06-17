#!/usr/bin/env bash
# Evaluate one LEO LoRA ckpt on RoboCasa365 target50 (50 tasks × 50 episodes).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
PY365_DEFAULT="$PROJECT_ROOT/gr00t/eval/sim/robocasa365/robocasa365_uv/.venv/bin/python"
RLDX_PY365="/HOME/sysu_xdliang/sysu_xdliang_1/HDD_POOL/linmin/RLDX-1/rldx/eval/sim/robocasa365/robocasa365_uv/.venv/bin/python"
PY365="${PY365:-$PY365_DEFAULT}"
if [[ ! -x "$PY365" && -x "$RLDX_PY365" ]]; then
  PY365="$RLDX_PY365"
fi

MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/output/leo_rc365_target50_lora}"
TASK_SET="${TASK_SET:-target50}"
SPLIT="${SPLIT:-pretrain}"
TASKS_FILTER="${TASKS:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/output/robocasa365_eval_leo}"
N_EPISODES="${N_EPISODES:-50}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-720}"
SEED="${SEED:-0}"
SKIP_SIM="${SKIP_SIM:-0}"

export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"

cd "$PROJECT_ROOT"

if [[ ! -x "$PY365" ]]; then
  echo "[x] RoboCasa365 python not found. Run: bash examples/RoboCasa365/setup_eval.sh"
  exit 1
fi

if [[ ! -d "$MODEL_PATH" ]]; then
  echo "[x] LEO checkpoint not found: $MODEL_PATH"
  echo "    Train first: bash examples/RoboCasa365/baselines/leo/finetune_leo_target50_lora.sh"
  exit 1
fi

echo "[i] LEO target50 eval (one LoRA ckpt)"
echo "[i] MODEL_PATH=$MODEL_PATH"
echo "[i] TASK_SET=$TASK_SET SPLIT=$SPLIT N_EPISODES=$N_EPISODES"

EXTRA=()
if [[ -n "$TASKS_FILTER" ]]; then
  EXTRA+=(--tasks "$TASKS_FILTER")
fi
if [[ "$SKIP_SIM" == "1" ]]; then
  EXTRA+=(--skip-sim)
fi

"$PY365" -u "$SCRIPT_DIR/eval_leo_robocasa365.py" \
  --model-path "$MODEL_PATH" \
  --task-set "$TASK_SET" \
  --split "$SPLIT" \
  --output-dir "$OUTPUT_ROOT" \
  --n-episodes "$N_EPISODES" \
  --max-episode-steps "$MAX_EPISODE_STEPS" \
  --seed "$SEED" \
  "${EXTRA[@]}"
