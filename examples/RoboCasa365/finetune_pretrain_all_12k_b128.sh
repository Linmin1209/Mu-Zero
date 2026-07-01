#!/usr/bin/env bash
# Finetune GR00T N1.7 on all RoboCasa365 pretrain tasks (atomic + composite).
# Default: flat VISOR + MOSS (no component-factored head — better for multitask pretrain).
#
# Defaults: 120k steps, global batch 128, 8 GPUs (per-GPU batch 16).
# Effective sample budget: 120000 × 128 = 15,360,000 samples.
# Checkpoints saved every 10k steps (SAVE_STEPS=10000).
#
# Prerequisites for VISOR (once per machine / dataset refresh):
#   SPLITS=pretrain bash examples/RoboCasa365/run_haptic_labels_all_missing.sh
#
# IO: shard_size=4096 + dataloader_workers=2 reduces HDD thrashing on 301-dataset mix.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash examples/RoboCasa365/finetune_pretrain_all_12k_b128.sh
#
# Optional env overrides:
#   ROBOCASA365_ROOT, GR00T_BASE_MODEL, OUTPUT_DIR, MAX_STEPS, GLOBAL_BATCH_SIZE,
#   NUM_GPUS, CUDA_VISIBLE_DEVICES, MASTER_PORT, DATALOADER_NUM_WORKERS,
#   NUM_SHARDS_PER_EPOCH, SHARD_SIZE, SAVE_STEPS, ROBOCASA365_TASKS,
#   USE_VISOR, USE_VISOR_COMPONENT_FACTORED, USE_MOTION, ...
#
# VISOR on flat head (default): --use-visor only; Scheme B split gates (arm EEF + gripper) enabled by default
# VISOR + component-factored (single-task): USE_VISOR_COMPONENT_FACTORED=1
#
# Disable VISOR or MOSS:
#   USE_VISOR=0 bash ...          # MOSS only (no tactile)
#   USE_MOTION=0 bash ...         # VISOR only
#   USE_VISOR=0 USE_MOTION=0 bash ...  # vanilla flat head
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
# shellcheck source=examples/RoboCasa365/env_defaults.sh
source "$SCRIPT_DIR/env_defaults.sh"

ROBOCASA365_SPLIT="${ROBOCASA365_SPLIT:-pretrain}"
ROBOCASA365_CATEGORY="${ROBOCASA365_CATEGORY:-all}"
ROBOCASA365_TASKS="${ROBOCASA365_TASKS:-}"

USE_VISOR="${USE_VISOR:-1}"
USE_VISOR_COMPONENT_FACTORED="${USE_VISOR_COMPONENT_FACTORED:-0}"
USE_MOTION="${USE_MOTION:-1}"
MOTION_INSERT_LAYER="${MOTION_INSERT_LAYER:-9}"
TUNE_MOTION="${TUNE_MOTION:-1}"
MOTION_USE_GATING="${MOTION_USE_GATING:-1}"

if [[ "$USE_VISOR" == "1" ]]; then
  MODALITY_CONFIG="${MODALITY_CONFIG:-$SCRIPT_DIR/robocasa365_config_4frame.py}"
elif [[ "$USE_MOTION" == "1" ]]; then
  MODALITY_CONFIG="${MODALITY_CONFIG:-$SCRIPT_DIR/robocasa365_config_4frame_no_tactile.py}"
else
  MODALITY_CONFIG="${MODALITY_CONFIG:-$SCRIPT_DIR/robocasa365_config.py}"
fi

if [[ "$USE_VISOR" == "1" && "$USE_MOTION" == "1" ]]; then
  OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/output/rc365_pretrain_all_120k_visor_flat_moss_b128_4frame_8gpu}"
elif [[ "$USE_VISOR" == "1" ]]; then
  OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/output/rc365_pretrain_all_120k_visor_b128_4frame_8gpu}"
elif [[ "$USE_MOTION" == "1" ]]; then
  OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/output/rc365_pretrain_all_120k_moss_b128_4frame_8gpu}"
else
  OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/output/rc365_pretrain_all_120k_vanilla_b128_8gpu}"
fi
LOG_FILE="${LOG_FILE:-$OUTPUT_DIR/train.log}"

NUM_GPUS="${NUM_GPUS:-8}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MAX_STEPS="${MAX_STEPS:-120000}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-128}"
SAVE_STEPS="${SAVE_STEPS:-10000}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-13}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-2}"
SHARD_SIZE="${SHARD_SIZE:-4096}"
LOAD_BF16="${LOAD_BF16:-1}"
OPTIM="${OPTIM:-adamw_torch_fused}"
DATALOADER_PREFETCH_FACTOR="${DATALOADER_PREFETCH_FACTOR:-2}"
NUM_SHARDS_PER_EPOCH="${NUM_SHARDS_PER_EPOCH:-100000}"
MASTER_PORT="${MASTER_PORT:-29500}"

if (( GLOBAL_BATCH_SIZE % NUM_GPUS != 0 )); then
  echo "[e] GLOBAL_BATCH_SIZE ($GLOBAL_BATCH_SIZE) must be divisible by NUM_GPUS ($NUM_GPUS)" >&2
  exit 1
fi

cd "$PROJECT_ROOT"
mkdir -p "$OUTPUT_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "[i] dataset: $ROBOCASA365_ROOT"
echo "[i] base model: $GR00T_BASE_MODEL"
echo "[i] split/category: $ROBOCASA365_SPLIT / $ROBOCASA365_CATEGORY"
if [[ -n "$ROBOCASA365_TASKS" ]]; then
  echo "[i] task filter: $ROBOCASA365_TASKS"
else
  echo "[i] task filter: (none — all pretrain atomic + composite tasks)"
fi
echo "[i] modality config: $MODALITY_CONFIG"
if [[ "$USE_VISOR" == "1" ]]; then
  if [[ "$USE_VISOR_COMPONENT_FACTORED" == "1" ]]; then
    echo "[i] head: component-factored + VISOR (T-Rex sensor-only)"
  else
    echo "[i] head: flat action_decoder + VISOR (T-Rex sensor-only, multitask)"
  fi
else
  echo "[i] head: native flat action_decoder"
fi
echo "[i] use_visor=$USE_VISOR use_motion=$USE_MOTION motion_insert_layer=$MOTION_INSERT_LAYER tune_motion=$TUNE_MOTION motion_use_gating=$MOTION_USE_GATING"
echo "[i] max_steps=$MAX_STEPS global_batch_size=$GLOBAL_BATCH_SIZE num_gpus=$NUM_GPUS cuda_visible=$CUDA_VISIBLE_DEVICES"
echo "[i] per_device_batch_size=$((GLOBAL_BATCH_SIZE / NUM_GPUS)) dataloader_workers=$DATALOADER_NUM_WORKERS prefetch=$DATALOADER_PREFETCH_FACTOR"
echo "[i] load_bf16=$LOAD_BF16 optim=$OPTIM"
echo "[i] shard_size=$SHARD_SIZE num_shards_per_epoch=$NUM_SHARDS_PER_EPOCH save_steps=$SAVE_STEPS"
echo "[i] effective_samples=$((MAX_STEPS * GLOBAL_BATCH_SIZE))"
echo "[i] output: $OUTPUT_DIR"
echo "[i] log: $LOG_FILE"
if [[ "$USE_VISOR" == "1" ]]; then
  echo "[i] VISOR requires tactile.* columns in all parquets — run run_haptic_labels_all_missing.sh if missing"
fi

EXTRA=()
if [[ -n "$ROBOCASA365_TASKS" ]]; then
  EXTRA+=(--robocasa365-tasks "$ROBOCASA365_TASKS")
fi

VISOR_ARGS=()
if [[ "$USE_VISOR" == "1" ]]; then
  VISOR_ARGS+=(--use-visor)
  if [[ "$USE_VISOR_COMPONENT_FACTORED" == "1" ]]; then
    VISOR_ARGS+=(--use-component-factored-head)
  fi
fi

MOTION_ARGS=()
if [[ "$USE_MOTION" == "1" ]]; then
  MOTION_ARGS+=(--use-motion --motion-insert-layer "$MOTION_INSERT_LAYER")
  if [[ "$TUNE_MOTION" == "1" ]]; then
    MOTION_ARGS+=(--tune-motion)
  else
    MOTION_ARGS+=(--no-tune-motion)
  fi
  if [[ "$MOTION_USE_GATING" == "1" ]]; then
    MOTION_ARGS+=(--motion-use-gating)
  else
    MOTION_ARGS+=(--no-motion-use-gating)
  fi
fi

COMMON_ARGS=(
  --base-model-path "$GR00T_BASE_MODEL"
  --robocasa365-root "$ROBOCASA365_ROOT"
  --robocasa365-split "$ROBOCASA365_SPLIT"
  --robocasa365-category "$ROBOCASA365_CATEGORY"
  "${EXTRA[@]}"
  --embodiment-tag ROBOCASA_PANDA_OMRON
  --modality-config-path "$MODALITY_CONFIG"
  --output-dir "$OUTPUT_DIR"
  --max-steps "$MAX_STEPS"
  --global-batch-size "$GLOBAL_BATCH_SIZE"
  --save-steps "$SAVE_STEPS"
  --save-total-limit "$SAVE_TOTAL_LIMIT"
  --dataloader-num-workers "$DATALOADER_NUM_WORKERS"
  --shard-size "$SHARD_SIZE"
  --num-shards-per-epoch "$NUM_SHARDS_PER_EPOCH"
  --optim "$OPTIM"
  --dataloader-prefetch-factor "$DATALOADER_PREFETCH_FACTOR"
  "${VISOR_ARGS[@]}"
  "${MOTION_ARGS[@]}"
)
if [[ "$LOAD_BF16" == "1" ]]; then
  COMMON_ARGS+=(--load-bf16)
else
  COMMON_ARGS+=(--no-load-bf16)
fi

if [[ "$NUM_GPUS" -eq 1 ]]; then
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" .venv/bin/python -u gr00t/experiment/launch_finetune.py \
    "${COMMON_ARGS[@]}" \
    --num-gpus 1
else
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" .venv/bin/python -u -m torch.distributed.run \
    --nproc_per_node="$NUM_GPUS" \
    --master_port="$MASTER_PORT" \
    gr00t/experiment/launch_finetune.py \
    "${COMMON_ARGS[@]}" \
    --num-gpus "$NUM_GPUS"
fi
