#!/usr/bin/env bash
# Batch-generate gripper tactile GT for all RoboCasa365 lerobot datasets missing labels.
#
# Scans pretrain + target for lerobot/ roots without complete tactile.left/right/contact.
# Patches parquet + meta in-place (same as run_haptic_labels_pretrain_batch.sh).
#
# Parallelism (defaults tuned for ~56 CPU / 4 GPU / 240GB RAM):
#   DATASET_JOBS=4   — concurrent datasets (one slot per GPU by default)
#   NUM_WORKERS=14   — episode workers per dataset (4×14 ≈ 56 processes)
#   NUM_GPUS=4       — round-robin CUDA_VISIBLE_DEVICES 0..N-1
#
# Prerequisites (once per machine):
#   source /app/bin/proxy.sh
#   /app/ubuntu/bin/apt-get update
#   /app/ubuntu/bin/apt-get install -y libegl1 libegl1-mesa libgl1 libgl1-mesa-glx \
#     libglib2.0-0 libgles2 libgles2-mesa libosmesa6 libglew-dev libglfw3
#
# Usage:
#   bash examples/RoboCasa365/run_haptic_labels_all_missing.sh
#   DATASET_JOBS=4 NUM_WORKERS=14 bash ...
#   SPLITS=pretrain bash ...
#   DRY_RUN=1 bash ...
#   MAX_DATASETS=5 bash ...
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
# shellcheck source=examples/RoboCasa365/env_defaults.sh
source "$SCRIPT_DIR/env_defaults.sh"

NUM_WORKERS="${NUM_WORKERS:-14}"
DATASET_JOBS="${DATASET_JOBS:-4}"
NUM_GPUS="${NUM_GPUS:-4}"
STAGGER_SEC="${STAGGER_SEC:-20}"
SPLITS="${SPLITS:-pretrain,target}"
DRY_RUN="${DRY_RUN:-0}"
MAX_DATASETS="${MAX_DATASETS:-0}"
LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/output/haptic_label_gen}"
DISCOVER_PY="$SCRIPT_DIR/scripts/discover_haptic_label_datasets.py"

export MUJOCO_GL=egl

mkdir -p "$LOG_DIR"
BATCH_LOG="$LOG_DIR/all_missing_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$BATCH_LOG") 2>&1

echo "[i] ROBOCASA365_ROOT=$ROBOCASA365_ROOT"
echo "[i] splits=$SPLITS dataset_jobs=$DATASET_JOBS workers_per_dataset=$NUM_WORKERS num_gpus=$NUM_GPUS inplace=1"
echo "[i] approx_parallel_episode_workers=$((DATASET_JOBS * NUM_WORKERS))"
echo "[i] log=$BATCH_LOG task_log_dir=$LOG_DIR"

mapfile -t PENDING < <(
  "$PROJECT_ROOT/.venv/bin/python" "$DISCOVER_PY" --root "$ROBOCASA365_ROOT" --splits "$SPLITS"
)

SUMMARY="${PENDING[0]}"
PENDING=("${PENDING[@]:1}")
echo "[i] $SUMMARY pending_count=${#PENDING[@]}"

if [[ "${MAX_DATASETS}" -gt 0 && "${#PENDING[@]}" -gt "${MAX_DATASETS}" ]]; then
  PENDING=("${PENDING[@]:0:MAX_DATASETS}")
  echo "[i] MAX_DATASETS=$MAX_DATASETS — limiting run"
fi

if [[ "${#PENDING[@]}" -eq 0 ]]; then
  echo "[i] nothing to do."
  exit 0
fi

if [[ "$DRY_RUN" == "1" ]]; then
  echo "[i] DRY_RUN — datasets that would be labeled:"
  printf '  %s\n' "${PENDING[@]}"
  exit 0
fi

label_one_dataset() {
  local dataset="$1"
  local gpu="$2"
  local task_name
  task_name="$(basename "$(dirname "$(dirname "$dataset")")")"
  local task_log="$LOG_DIR/task_${task_name}.log"
  # Stagger EGL/MuJoCo init across GPUs to reduce driver startup races.
  sleep $((gpu * STAGGER_SEC))
  echo "[i] START gpu=$gpu task=$task_name dataset=$dataset"
  if CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_EGL_DEVICE_ID="$gpu" NUM_WORKERS="$NUM_WORKERS" \
    DATASET="$dataset" \
    bash "$SCRIPT_DIR/run_haptic_label_generation.sh" >>"$task_log" 2>&1; then
    echo "[i] OK gpu=$gpu task=$task_name"
    return 0
  fi
  echo "[e] FAILED gpu=$gpu task=$task_name (see $task_log)" >&2
  return 1
}

running=0
gpu_round=0
total="${#PENDING[@]}"
idx=0
failed=0

for dataset in "${PENDING[@]}"; do
  idx=$((idx + 1))
  while (( running >= DATASET_JOBS )); do
    wait -n || true
    running=$((running - 1))
  done

  gpu=$((gpu_round % NUM_GPUS))
  gpu_round=$((gpu_round + 1))
  task_name="$(basename "$(dirname "$(dirname "$dataset")")")"
  echo ""
  echo "========== queue [$idx/$total] $task_name (gpu=$gpu) =========="

  label_one_dataset "$dataset" "$gpu" &
  running=$((running + 1))
done

echo "[i] waiting for ${running} in-flight dataset jobs..."
while (( running > 0 )); do
  wait -n || true
  running=$((running - 1))
done

echo ""
echo "[i] batch finished. log: $BATCH_LOG"
