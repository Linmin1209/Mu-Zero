#!/usr/bin/env bash
# Batch-generate arm/base 2D trajectory labels for pretrain tasks used in current finetune.
#
# Tasks (pretrain split, matches finetune scripts):
#   - PickPlaceToasterToCounter  (atomic, 105 ep)
#   - NavigateKitchen            (atomic, 503 ep)
#   - DeliverStraw               (composite, 104 ep)
#
# Output: sibling ``lerobot_traj_v2/`` (annotate_sim future columns) next to ``lerobot/``.
# Legacy ``lerobot_traj/`` used old per-frame projection without arm_future_uv.
#
# Usage:
#   NUM_WORKERS=4 bash examples/RoboCasa365/run_trajectory_labels_pretrain_batch.sh
#   TASKS=PickPlaceToasterToCounter bash ...
#   TRAJ_SUFFIX=lerobot_traj_v2 bash ...   # default: lerobot_traj_v2
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
# shellcheck source=examples/RoboCasa365/env_defaults.sh
source "$SCRIPT_DIR/env_defaults.sh"

NUM_WORKERS="${NUM_WORKERS:-4}"
TASKS="${TASKS:-PickPlaceToasterToCounter,NavigateKitchen,DeliverStraw}"
TRAJ_SUFFIX="${TRAJ_SUFFIX:-lerobot_traj_v2}"
LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/output/trajectory_label_gen}"

# Headless GL defaults to OSMesa via run_trajectory_label_generation.sh (MUJOCO_GL=osmesa).

mkdir -p "$LOG_DIR"
BATCH_LOG="$LOG_DIR/batch_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$BATCH_LOG") 2>&1

echo "[i] ROBOCASA365_ROOT=$ROBOCASA365_ROOT"
echo "[i] workers=$NUM_WORKERS MUJOCO_GL=${MUJOCO_GL:-osmesa} log=$BATCH_LOG"

run_task() {
  local name="$1"
  local rel="$2"
  local input="$ROBOCASA365_ROOT/$rel/lerobot"
  local output="$ROBOCASA365_ROOT/$rel/$TRAJ_SUFFIX"

  if [[ ! -d "$input/meta" ]]; then
    echo "[e] missing dataset: $input" >&2
    return 1
  fi

  echo ""
  echo "========== $name =========="
  echo "[i] input:  $input"
  echo "[i] output: $output"
  DATASET="$input" OUTPUT_DATASET="$output" NUM_WORKERS="$NUM_WORKERS" \
    bash "$SCRIPT_DIR/run_trajectory_label_generation.sh"
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
