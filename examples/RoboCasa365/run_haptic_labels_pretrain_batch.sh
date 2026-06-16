#!/usr/bin/env bash
# Batch-generate gripper haptic GT for pretrain tasks used in VISOR MVP.
#
# Tasks:
#   - PickPlaceToasterToCounter  (atomic, 105 ep)
#   - NavigateKitchen            (atomic, 503 ep)
#   - DeliverStraw               (composite, 104 ep)
#
# Output: by default patches ``lerobot/`` in-place (parquet + meta/info.json + modality.json).
# Set INPLACE=0 to write sibling ``lerobot_haptic/`` instead.
#
# Usage:
#   NUM_WORKERS=4 CUDA_VISIBLE_DEVICES=0 bash examples/RoboCasa365/run_haptic_labels_pretrain_batch.sh
#   TASKS=PickPlaceToasterToCounter bash ...   # subset
#   INPLACE=0 bash ...                        # separate output dir
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
# shellcheck source=examples/RoboCasa365/env_defaults.sh
source "$SCRIPT_DIR/env_defaults.sh"

NUM_WORKERS="${NUM_WORKERS:-4}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"
TASKS="${TASKS:-PickPlaceToasterToCounter,NavigateKitchen,DeliverStraw}"
INPLACE="${INPLACE:-1}"
LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/output/haptic_label_gen}"

export MUJOCO_GL=egl
export CUDA_VISIBLE_DEVICES
export MUJOCO_EGL_DEVICE_ID

mkdir -p "$LOG_DIR"
BATCH_LOG="$LOG_DIR/batch_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$BATCH_LOG") 2>&1

echo "[i] ROBOCASA365_ROOT=$ROBOCASA365_ROOT"
echo "[i] workers=$NUM_WORKERS cuda=$CUDA_VISIBLE_DEVICES inplace=$INPLACE log=$BATCH_LOG"

run_task() {
  local name="$1"
  local rel="$2"
  local input="$ROBOCASA365_ROOT/$rel/lerobot"
  local output="${ROBOCASA365_ROOT}/${rel}/lerobot_haptic"

  if [[ ! -d "$input/meta" ]]; then
    echo "[e] missing dataset: $input" >&2
    return 1
  fi

  echo ""
  echo "========== $name =========="
  echo "[i] dataset: $input"
  if [[ "$INPLACE" == "1" ]]; then
    echo "[i] mode: in-place (patch parquet + meta under lerobot/)"
    DATASET="$input" NUM_WORKERS="$NUM_WORKERS" \
      bash "$SCRIPT_DIR/run_haptic_label_generation.sh"
  else
    echo "[i] mode: copy-out -> $output"
    DATASET="$input" OUTPUT_DATASET="$output" NUM_WORKERS="$NUM_WORKERS" \
      bash "$SCRIPT_DIR/run_haptic_label_generation.sh"
  fi
}

IFS=',' read -ra TASK_ARR <<< "$TASKS"
for task in "${TASK_ARR[@]}"; do
  task="$(echo "$task" | xargs)"
  case "$task" in
    PickPlaceToasterToCounter)
      run_task "$task" "pretrain/atomic/PickPlaceToasterToCounter/20250819"
      ;;
    NavigateKitchen)
      run_task "$task" "pretrain/atomic/NavigateKitchen/20250821"
      ;;
    DeliverStraw)
      run_task "$task" "pretrain/composite/DeliverStraw/20250723"
      ;;
    *)
      echo "[e] unknown task: $task (use PickPlaceToasterToCounter, NavigateKitchen, DeliverStraw)" >&2
      exit 1
      ;;
  esac
done

echo ""
echo "[i] all requested tasks finished. log: $BATCH_LOG"
