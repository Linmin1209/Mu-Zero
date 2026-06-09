#!/usr/bin/env bash
# Finetune GR00T N1.7 on RoboCasa365 (single task, multi-task, atomic/composite, pretrain/target).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Defaults (override via env or CLI args below)
export ROBOCASA365_ROOT="${ROBOCASA365_ROOT:-/HOME/sysu_xdliang/sysu_xdliang_1/HDD_POOL/linmin/datasets/robocasa365-datasets}"
export GR00T_MODELS_ROOT="${GR00T_MODELS_ROOT:-/HOME/sysu_xdliang/sysu_xdliang_1/HDD_POOL/linmin/models}"
export GR00T_BASE_MODEL="${GR00T_BASE_MODEL:-$GR00T_MODELS_ROOT/GR00T-N1.7-3B}"
export GR00T_COSMOS_MODEL_PATH="${GR00T_COSMOS_MODEL_PATH:-$GR00T_MODELS_ROOT/Cosmos-Reason2-2B}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export GROOT_PATCH_MISTRAL=1
export GROOT_HF_LOCAL_FIRST=1

ROBOCASA365_SPLIT="${ROBOCASA365_SPLIT:-pretrain}"
ROBOCASA365_CATEGORY="${ROBOCASA365_CATEGORY:-atomic}"
ROBOCASA365_TASKS="${ROBOCASA365_TASKS:-}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/gr00t_robocasa365_finetune}"
NUM_GPUS="${NUM_GPUS:-1}"
MAX_STEPS="${MAX_STEPS:-30000}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"

usage() {
  cat <<EOF
Usage: bash examples/RoboCasa365/finetune_robocasa365.sh [options]

Options (env vars):
  ROBOCASA365_ROOT      Dataset root (default: \$ROBOCASA365_ROOT)
  ROBOCASA365_SPLIT     pretrain | target | all
  ROBOCASA365_CATEGORY  atomic | composite | all
  ROBOCASA365_TASKS     Comma-separated task names (empty = all under split/category)
  GR00T_BASE_MODEL      Base checkpoint path
  OUTPUT_DIR            Training output dir
  NUM_GPUS              GPU count (default: 1)
  MAX_STEPS             Training steps
  GLOBAL_BATCH_SIZE     Global batch size

Examples:
  # Single atomic pretrain task
  ROBOCASA365_SPLIT=pretrain ROBOCASA365_CATEGORY=atomic \\
    ROBOCASA365_TASKS=CloseElectricKettleLid OUTPUT_DIR=/tmp/rc365_one \\
    bash examples/RoboCasa365/finetune_robocasa365.sh

  # All pretrain composite tasks
  ROBOCASA365_SPLIT=pretrain ROBOCASA365_CATEGORY=composite \\
    OUTPUT_DIR=/tmp/rc365_pretrain_composite \\
    bash examples/RoboCasa365/finetune_robocasa365.sh

  # All target atomic (50-task eval split subset)
  ROBOCASA365_SPLIT=target ROBOCASA365_CATEGORY=atomic \\
    OUTPUT_DIR=/tmp/rc365_target_atomic \\
    bash examples/RoboCasa365/finetune_robocasa365.sh
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

cd "$PROJECT_ROOT"

EXTRA=()
if [[ -n "$ROBOCASA365_TASKS" ]]; then
  EXTRA+=(--robocasa365-tasks "$ROBOCASA365_TASKS")
fi

if [[ "$NUM_GPUS" -eq 1 ]]; then
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" .venv/bin/python gr00t/experiment/launch_finetune.py \
    --base-model-path "$GR00T_BASE_MODEL" \
    --robocasa365-root "$ROBOCASA365_ROOT" \
    --robocasa365-split "$ROBOCASA365_SPLIT" \
    --robocasa365-category "$ROBOCASA365_CATEGORY" \
    "${EXTRA[@]}" \
    --embodiment-tag ROBOCASA_PANDA_OMRON \
    --modality-config-path "$SCRIPT_DIR/robocasa365_config.py" \
    --num-gpus 1 \
    --output-dir "$OUTPUT_DIR" \
    --max-steps "$MAX_STEPS" \
    --global-batch-size "$GLOBAL_BATCH_SIZE" \
    --save-steps 5000 \
    --save-total-limit 5 \
    --dataloader-num-workers 8
else
  .venv/bin/python -m torch.distributed.run --nproc_per_node="$NUM_GPUS" --master_port=29500 \
    gr00t/experiment/launch_finetune.py \
    --base-model-path "$GR00T_BASE_MODEL" \
    --robocasa365-root "$ROBOCASA365_ROOT" \
    --robocasa365-split "$ROBOCASA365_SPLIT" \
    --robocasa365-category "$ROBOCASA365_CATEGORY" \
    "${EXTRA[@]}" \
    --embodiment-tag ROBOCASA_PANDA_OMRON \
    --modality-config-path "$SCRIPT_DIR/robocasa365_config.py" \
    --num-gpus "$NUM_GPUS" \
    --output-dir "$OUTPUT_DIR" \
    --max-steps "$MAX_STEPS" \
    --global-batch-size "$GLOBAL_BATCH_SIZE" \
    --save-steps 1000 \
    --save-total-limit 5 \
    --dataloader-num-workers 4
fi
