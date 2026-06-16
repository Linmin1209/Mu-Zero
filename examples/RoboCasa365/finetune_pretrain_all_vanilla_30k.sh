#!/usr/bin/env bash
# Finetune vanilla GR00T N1.7 on all RoboCasa365 pretrain data (atomic + composite).
# No VISOR / motion / adaptive-component heads — standard robocasa365_config.py only.
#
# Defaults (4× A800): batch 128 / 120k steps (RoboCasa365 multitask ref).
# Do NOT raise batch to 256 on this 301-dataset mix — step time ~3× slower, total wall time worse.
# To finish sooner: lower MAX_STEPS (e.g. 60000 @ b128 ≈ 17h, half the sample budget).
# IO: shard_size=4096 + dataloader_workers=2 reduces HDD thrashing on 301-dataset mix.
# Speed: load_bf16 + adamw_torch_fused enabled by default in launch_finetune (FlashAttention-2).
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash examples/RoboCasa365/finetune_pretrain_all_vanilla_30k.sh
#
# Optional env overrides:
#   ROBOCASA365_ROOT, GR00T_BASE_MODEL, OUTPUT_DIR, MAX_STEPS, GLOBAL_BATCH_SIZE,
#   NUM_GPUS, CUDA_VISIBLE_DEVICES, MASTER_PORT, DATALOADER_NUM_WORKERS,
#   NUM_SHARDS_PER_EPOCH, SHARD_SIZE
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
# shellcheck source=examples/RoboCasa365/env_defaults.sh
source "$SCRIPT_DIR/env_defaults.sh"

ROBOCASA365_SPLIT="${ROBOCASA365_SPLIT:-pretrain}"
ROBOCASA365_CATEGORY="${ROBOCASA365_CATEGORY:-all}"
ROBOCASA365_TASKS="${ROBOCASA365_TASKS:-}"
MODALITY_CONFIG="${MODALITY_CONFIG:-$SCRIPT_DIR/robocasa365_config.py}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/output/rc365_pretrain_all_120k_vanilla_n17_b128_4gpu}"
LOG_FILE="${LOG_FILE:-$OUTPUT_DIR/train.log}"

NUM_GPUS="${NUM_GPUS:-4}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MAX_STEPS="${MAX_STEPS:-120000}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-128}"
SAVE_STEPS="${SAVE_STEPS:-10000}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-5}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-2}"
SHARD_SIZE="${SHARD_SIZE:-4096}"
LOAD_BF16="${LOAD_BF16:-1}"
OPTIM="${OPTIM:-adamw_torch_fused}"
VIDEO_BACKEND="${VIDEO_BACKEND:-torchcodec}"
DATALOADER_PREFETCH_FACTOR="${DATALOADER_PREFETCH_FACTOR:-2}"
NUM_SHARDS_PER_EPOCH="${NUM_SHARDS_PER_EPOCH:-100000}"
MASTER_PORT="${MASTER_PORT:-29500}"

cd "$PROJECT_ROOT"
mkdir -p "$OUTPUT_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "[i] dataset: $ROBOCASA365_ROOT"
echo "[i] base model: $GR00T_BASE_MODEL"
echo "[i] split/category: $ROBOCASA365_SPLIT / $ROBOCASA365_CATEGORY (all pretrain tasks when tasks empty)"
if [[ -n "$ROBOCASA365_TASKS" ]]; then
  echo "[i] task filter: $ROBOCASA365_TASKS"
else
  echo "[i] task filter: (none — all tasks under split/category)"
fi
echo "[i] modality config: $MODALITY_CONFIG (vanilla N1.7, single-frame video)"
echo "[i] max_steps=$MAX_STEPS global_batch_size=$GLOBAL_BATCH_SIZE num_gpus=$NUM_GPUS cuda_visible=$CUDA_VISIBLE_DEVICES"
echo "[i] per_device_batch_size=$((GLOBAL_BATCH_SIZE / NUM_GPUS)) dataloader_workers=$DATALOADER_NUM_WORKERS prefetch=$DATALOADER_PREFETCH_FACTOR"
echo "[i] load_bf16=$LOAD_BF16 optim=$OPTIM video_backend=$VIDEO_BACKEND"
echo "[i] shard_size=$SHARD_SIZE num_shards_per_epoch=$NUM_SHARDS_PER_EPOCH"
echo "[i] effective_samples=$((MAX_STEPS * GLOBAL_BATCH_SIZE)) (official multitask ref: 120000*128=15360000)"
echo "[i] output: $OUTPUT_DIR"
echo "[i] log: $LOG_FILE"

EXTRA=()
if [[ -n "$ROBOCASA365_TASKS" ]]; then
  EXTRA+=(--robocasa365-tasks "$ROBOCASA365_TASKS")
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
  --video-backend "$VIDEO_BACKEND"
  --dataloader-prefetch-factor "$DATALOADER_PREFETCH_FACTOR"
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
