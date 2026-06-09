#!/usr/bin/env bash
# Finetune GR00T N1.7 on RoboCasa365 NavigateKitchen (30k steps) with STSS/MOSS.
# Video: 4-frame history delta_indices [-6, -4, -2, 0] (see robocasa365_config_4frame.py).
# Default: 2-GPU distributed training (override via NUM_GPUS / CUDA_VISIBLE_DEVICES).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
# shellcheck source=examples/RoboCasa365/env_defaults.sh
source "$SCRIPT_DIR/env_defaults.sh"

ROBOCASA365_SPLIT="${ROBOCASA365_SPLIT:-pretrain}"
ROBOCASA365_CATEGORY="${ROBOCASA365_CATEGORY:-atomic}"
ROBOCASA365_TASKS="${ROBOCASA365_TASKS:-NavigateKitchen}"
MODALITY_CONFIG="${MODALITY_CONFIG:-$SCRIPT_DIR/robocasa365_config_4frame.py}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/output/rc365_NavigateKitchen_30k_b64_4frame_motion}"
LOG_FILE="${LOG_FILE:-$OUTPUT_DIR/train.log}"

NUM_GPUS="${NUM_GPUS:-2}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
MAX_STEPS=30000
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
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
echo "[i] Split/category: $ROBOCASA365_SPLIT / $ROBOCASA365_CATEGORY"
echo "[i] modality config: $MODALITY_CONFIG (video delta [-6,-4,-2,0])"
echo "[i] use_motion=$USE_MOTION motion_insert_layer=$MOTION_INSERT_LAYER tune_motion=$TUNE_MOTION"
echo "[i] max_steps=$MAX_STEPS global_batch_size=$GLOBAL_BATCH_SIZE num_gpus=$NUM_GPUS cuda=$CUDA_VISIBLE_DEVICES"
echo "[i] output: $OUTPUT_DIR"
echo "[i] log: $LOG_FILE"

EXTRA=(--robocasa365-tasks "$ROBOCASA365_TASKS")
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
      "${MOTION_ARGS[@]}"
  else
    CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" .venv/bin/python -u -m torch.distributed.run \
      --nproc_per_node="$NUM_GPUS" \
      --master_port="${MASTER_PORT:-29501}" \
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
      "${MOTION_ARGS[@]}"
  fi
}

run_train
