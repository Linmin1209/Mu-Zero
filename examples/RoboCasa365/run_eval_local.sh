#!/usr/bin/env bash
# RoboCasa365 sim eval with a local GR00T finetune checkpoint.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"

CKPT="${GR00T_CKPT:-/HOME/sysu_xdliang/sysu_xdliang_1/HDD_POOL/linmin/RLDX-1/output/checkpoint-10}"
PY365="${PY365:-$PROJECT_REPO/gr00t/eval/sim/robocasa365/robocasa365_uv/.venv/bin/python}"
RLDX_PY365="/HOME/sysu_xdliang/sysu_xdliang_1/HDD_POOL/linmin/RLDX-1/rldx/eval/sim/robocasa365/robocasa365_uv/.venv/bin/python"
if [[ ! -x "$PY365" && -x "$RLDX_PY365" ]]; then
  PY365="$RLDX_PY365"
fi

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export NO_ALBUMENTATIONS_UPDATE=1

cd "$PROJECT_REPO"
bash "$SCRIPT_DIR/eval_robocasa365.sh" \
  --model-path "$CKPT" \
  --task-set "${TASK_SET:-atomic_seen}" \
  --split "${SPLIT:-pretrain}" \
  --n-episodes "${N_EPISODES:-10}" \
  --n-envs "${N_ENVS:-1}" \
  --python "$PY365" \
  "$@"
