#!/usr/bin/env bash
# Finetune GR00T N1.7 on RoboCasa365 PickPlaceToasterOvenToCounter (30k steps)
# VISOR v4.2b: online optical-flow visual GT + triple gates (no offline labels).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
# shellcheck source=examples/RoboCasa365/env_defaults.sh
source "$SCRIPT_DIR/env_defaults.sh"

ROBOCASA365_SPLIT="${ROBOCASA365_SPLIT:-pretrain}"
ROBOCASA365_CATEGORY="${ROBOCASA365_CATEGORY:-atomic}"
ROBOCASA365_TASKS="${ROBOCASA365_TASKS:-PickPlaceToasterOvenToCounter}"
MODALITY_CONFIG="${MODALITY_CONFIG:-$SCRIPT_DIR/robocasa365_config_4frame.py}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/output/rc365_PickPlaceToasterOvenToCounter_30k_visor_v42_flow}"
LOG_FILE="${LOG_FILE:-$OUTPUT_DIR/train.log}"

NUM_GPUS="${NUM_GPUS:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
MAX_STEPS=30000
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-16}"
SAVE_STEPS="${SAVE_STEPS:-5000}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-10}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-8}"

USE_MOTION="${USE_MOTION:-1}"
MOTION_INSERT_LAYER="${MOTION_INSERT_LAYER:-9}"
TUNE_MOTION="${TUNE_MOTION:-1}"

cd "$PROJECT_ROOT"
mkdir -p "$OUTPUT_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "[i] dataset: $ROBOCASA365_ROOT"
echo "[i] base model: $GR00T_BASE_MODEL"
echo "[i] Task: $ROBOCASA365_TASKS"
echo "[i] VISOR v4.2b gate_mode=visual_manip_nav_tactile_hand"
echo "[i] visual_gt=online optical flow (video_future_*), visual_dim=2"
echo "[i] output: $OUTPUT_DIR"

EXTRA=(--robocasa365-tasks "$ROBOCASA365_TASKS")
VISOR_ARGS=(
  --use-visor
  --no-use-component-factored-head
  --visor-gate-mode visual_manip_nav_tactile_hand
  --visor-use-visual-supervision
  --visor-use-readout-fed-gates
  --visor-visual-gt-level flow
  --visor-visual-dim 2
  --visor-tactile-align-mode hold_last
)
MOTION_ARGS=()
if [[ "$USE_MOTION" == "1" ]]; then
  MOTION_ARGS+=(--use-motion --motion-insert-layer "$MOTION_INSERT_LAYER")
  if [[ "$TUNE_MOTION" == "1" ]]; then
    MOTION_ARGS+=(--tune-motion)
  else
    MOTION_ARGS+=(--no-tune-motion)
  fi
fi

run_train() {
  if [[ "$NUM_GPUS" -eq 1 ]]; then
    CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" .venv/bin/python -u gr00t/experiment/launch_finetune.py \
      --base-model-path "$GR00T_BASE_MODEL" \
      --robocasa365-root "$ROBOCASA365_ROOT" \
      --robocasa365-split "$ROBOCASA365_SPLIT" \
      --robocasa365-category "$ROBOCASA365_CATEGORY" \
      "${EXTRA[@]}" \
      --embodiment-tag ROBOCASA_PANDA_OMRON \
      --modality-config-path "$MODALITY_CONFIG" \
      --num-gpus 1 \
      --output-dir "$OUTPUT_DIR" \
      --max-steps "$MAX_STEPS" \
      --global-batch-size "$GLOBAL_BATCH_SIZE" \
      --save-steps "$SAVE_STEPS" \
      --save-total-limit "$SAVE_TOTAL_LIMIT" \
      --dataloader-num-workers "$DATALOADER_NUM_WORKERS" \
      "${VISOR_ARGS[@]}" \
      "${MOTION_ARGS[@]}"
  else
    CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" .venv/bin/python -u -m torch.distributed.run \
      --nproc_per_node="$NUM_GPUS" \
      --master_port="${MASTER_PORT:-29500}" \
      gr00t/experiment/launch_finetune.py \
      --base-model-path "$GR00T_BASE_MODEL" \
      --robocasa365-root "$ROBOCASA365_ROOT" \
      --robocasa365-split "$ROBOCASA365_SPLIT" \
      --robocasa365-category "$ROBOCASA365_CATEGORY" \
      "${EXTRA[@]}" \
      --embodiment-tag ROBOCASA_PANDA_OMRON \
      --modality-config-path "$MODALITY_CONFIG" \
      --num-gpus "$NUM_GPUS" \
      --output-dir "$OUTPUT_DIR" \
      --max-steps "$MAX_STEPS" \
      --global-batch-size "$GLOBAL_BATCH_SIZE" \
      --save-steps "$SAVE_STEPS" \
      --save-total-limit "$SAVE_TOTAL_LIMIT" \
      --dataloader-num-workers "$DATALOADER_NUM_WORKERS" \
      "${VISOR_ARGS[@]}" \
      "${MOTION_ARGS[@]}"
  fi
}

run_train
