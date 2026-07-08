#!/usr/bin/env bash
# 50-episode RoboCasa365 sim eval for the finetuned PickPlaceToasterOvenToCounter checkpoint.
#
# Multi-process layout (same as eval_robocasa365.sh):
#   1) GR00T policy server  (GPU, .venv)
#   2) rollout_policy       (sim venv, AsyncVectorEnv with N_ENVS worker processes)
#
# Usage:
#   # Run immediately (checkpoint must exist, GPU free):
#   bash examples/RoboCasa365/run_eval_PickPlaceToasterOvenToCounter_50ep_parallel.sh
#
#   # Wait for training to finish, then eval:
#   WAIT_FOR_TRAIN=1 bash examples/RoboCasa365/run_eval_PickPlaceToasterOvenToCounter_50ep_parallel.sh
#
# Tunables:
#   CKPT=... N_ENVS=5 N_EPISODES=50 CUDA_VISIBLE_DEVICES=0 WAIT_FOR_TRAIN=1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"

TRAIN_OUT="${TRAIN_OUT:-$PROJECT_REPO/output/rc365_PickPlaceToasterOvenToCounter_30k_visor_flat_moss_b16_1gpu}"
CKPT="${CKPT:-$TRAIN_OUT/checkpoint-30000}"
TASK="${TASK:-PickPlaceToasterOvenToCounter}"
SPLIT="${SPLIT:-pretrain}"
N_EPISODES="${N_EPISODES:-50}"
N_ENVS="${N_ENVS:-5}"
N_ACTION_STEPS="${N_ACTION_STEPS:-40}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-500}"
SERVER_DEVICE="${SERVER_DEVICE:-cuda:0}"
WAIT_FOR_TRAIN="${WAIT_FOR_TRAIN:-0}"
POLL_SEC="${POLL_SEC:-30}"
SERVER_WARMUP_SEC="${SERVER_WARMUP_SEC:-600}"

PY365="${PY365:-/HOME/sysu_xdliang/sysu_xdliang_1/HDD_POOL/linmin/RLDX-1/rldx/eval/sim/robocasa365/robocasa365_uv/.venv/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_REPO/output/robocasa365_eval}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export NO_ALBUMENTATIONS_UPDATE=1
export USE_TASK_HORIZON=0

LOG_DIR="$OUTPUT_ROOT/PickPlaceToasterOvenToCounter_50ep_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
MAIN_LOG="$LOG_DIR/run_eval.log"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "$MAIN_LOG"
}

wait_for_checkpoint() {
  log "Waiting for checkpoint: $CKPT"
  while [[ ! -d "$CKPT" ]]; do
    sleep "$POLL_SEC"
  done
  log "Checkpoint found: $CKPT"
}

wait_for_gpu() {
  log "Waiting for finetune process to exit and GPU to free..."
  local train_pat="output/rc365_PickPlaceToasterOvenToCounter_30k_visor_flat_moss_b16_1gpu"
  while pgrep -f "$train_pat" >/dev/null 2>&1; do
    sleep "$POLL_SEC"
  done
  # Extra buffer for CUDA context teardown.
  sleep 15
  log "GPU should be free; starting eval."
}

if [[ "$WAIT_FOR_TRAIN" == "1" ]]; then
  wait_for_checkpoint
  wait_for_gpu
elif [[ ! -d "$CKPT" ]]; then
  echo "[x] Checkpoint not found: $CKPT"
  echo "[x] Set WAIT_FOR_TRAIN=1 or pass CKPT=... to an existing checkpoint."
  exit 1
fi

log "Eval config:"
log "  checkpoint=$CKPT"
log "  task=$TASK split=$SPLIT episodes=$N_EPISODES n_envs=$N_ENVS action_steps=$N_ACTION_STEPS"
log "  max_episode_steps=$MAX_EPISODE_STEPS server_device=$SERVER_DEVICE"
log "  python365=$PY365"
log "  log_dir=$LOG_DIR"

cd "$PROJECT_REPO"
bash "$SCRIPT_DIR/eval_robocasa365.sh" \
  --model-path "$CKPT" \
  --task-set atomic_seen \
  --tasks "$TASK" \
  --split "$SPLIT" \
  --n-episodes "$N_EPISODES" \
  --n-envs "$N_ENVS" \
  --n-action-steps "$N_ACTION_STEPS" \
  --max-episode-steps "$MAX_EPISODE_STEPS" \
  --server-device "$SERVER_DEVICE" \
  --server-warmup-sec "$SERVER_WARMUP_SEC" \
  --python "$PY365" \
  --output-root "$OUTPUT_ROOT" \
  2>&1 | tee -a "$MAIN_LOG"

log "Eval finished. See summary CSV under $OUTPUT_ROOT/checkpoint-30000_*"
