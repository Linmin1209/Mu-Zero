#!/usr/bin/env bash
# Finetune GR00T on DexJoCo LeRobot v3 datasets.
#
# Prerequisites (once per task / after download):
#   python examples/DexJoCo/prepare_dexjoco_metadata.py \
#     --datasets-root "$DEXJOCo_DATASETS_ROOT" --all-tasks
#
# Optional VISOR tactile labels (MuJoCo replay, slow):
#   python examples/DexJoCo/generate_dexjoco_haptic_labels.py \
#     --datasets-root "$DEXJOCo_DATASETS_ROOT" --task water_plant
#   python examples/DexJoCo/prepare_dexjoco_metadata.py \
#     --datasets-root "$DEXJOCo_DATASETS_ROOT" --task water_plant --force
#   MODALITY_CONFIG=.../dexjoco_single_arm_visor_config.py USE_VISOR=1 \
#   VISOR_TACTILE_NUM_FORCE=5 VISOR_TACTILE_NUM_CONTACT=5 \
#   VISOR_ARM_ACTION_SLICE=0,6 VISOR_HAND_ACTION_SLICE=6,22 \
#   VISOR_ARM_ACTION_DIM=6 VISOR_HAND_ACTION_DIM=16 ...
#   VISOR_TACTILE_NUM_FORCE=5 VISOR_TACTILE_NUM_CONTACT=5 \
#   VISOR_ARM_ACTION_SLICE=0,6 VISOR_HAND_ACTION_SLICE=6,22 \
#   VISOR_ARM_ACTION_DIM=6 VISOR_HAND_ACTION_DIM=16 ...
#
# Single-task example:
#   DEXJOCo_TASKS=water_plant bash examples/DexJoCo/finetune_dexjoco.sh
#
# All single-arm tasks:
#   DEXJOCo_ROBOT_TYPE=single_arm bash examples/DexJoCo/finetune_dexjoco.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
# shellcheck source=examples/DexJoCo/env_defaults.sh
source "$SCRIPT_DIR/env_defaults.sh"

DEXJOCo_ROBOT_TYPE="${DEXJOCo_ROBOT_TYPE:-single_arm}"
DEXJOCo_TASKS="${DEXJOCo_TASKS:-}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/output/dexjoco_${DEXJOCo_ROBOT_TYPE}_gr00t}"
MODALITY_CONFIG="${MODALITY_CONFIG:-$SCRIPT_DIR/dexjoco_${DEXJOCo_ROBOT_TYPE}_config.py}"
NUM_GPUS="${NUM_GPUS:-1}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-8}"
MAX_STEPS="${MAX_STEPS:-20000}"
SAVE_STEPS="${SAVE_STEPS:-5000}"
USE_VISOR="${USE_VISOR:-0}"
USE_MOTION="${USE_MOTION:-0}"
MASTER_PORT="${MASTER_PORT:-29600}"

cd "$PROJECT_ROOT"
mkdir -p "$OUTPUT_DIR"

echo "[i] dexjoco root: $DEXJOCo_ROOT"
echo "[i] datasets root: $DEXJOCo_DATASETS_ROOT"
echo "[i] robot type: $DEXJOCo_ROBOT_TYPE"
echo "[i] base model: $GR00T_BASE_MODEL"
echo "[i] output: $OUTPUT_DIR"

EXTRA=()
if [[ -n "$DEXJOCo_TASKS" ]]; then
  EXTRA+=(--dexjoco-tasks "$DEXJOCo_TASKS")
fi

VISOR_ARGS=()
if [[ "$USE_VISOR" == "1" ]]; then
  VISOR_ARGS+=(--use-visor)
fi

COMMON_ARGS=(
  --base-model-path "$GR00T_BASE_MODEL"
  --dexjoco-root "$DEXJOCo_DATASETS_ROOT"
  --dexjoco-robot-type "$DEXJOCo_ROBOT_TYPE"
  --modality-config-path "$MODALITY_CONFIG"
  --output-dir "$OUTPUT_DIR"
  --max-steps "$MAX_STEPS"
  --global-batch-size "$GLOBAL_BATCH_SIZE"
  --save-steps "$SAVE_STEPS"
  "${EXTRA[@]}"
  "${VISOR_ARGS[@]}"
)

if [[ "$NUM_GPUS" -eq 1 ]]; then
  .venv/bin/python -u gr00t/experiment/launch_finetune.py \
    "${COMMON_ARGS[@]}" \
    --num-gpus 1
else
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" \
    .venv/bin/python -u -m torch.distributed.run \
    --nproc_per_node="$NUM_GPUS" \
    --master_port="$MASTER_PORT" \
    gr00t/experiment/launch_finetune.py \
    "${COMMON_ARGS[@]}" \
    --num-gpus "$NUM_GPUS"
fi
