#!/usr/bin/env bash
# Quick MOSS visualization test (synthetic by default).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export GROOT_PATCH_MISTRAL=1
export GROOT_HF_LOCAL_FIRST=1
export PYTHONUNBUFFERED=1

MODE="${MODE:-checkpoint}"
CHECKPOINT="${CHECKPOINT:-$PROJECT_ROOT/output/rc365_PickPlaceToasterToCounter_30k_b64_4frame_motion/checkpoint-30000}"
VIDEO="${VIDEO:-}"
DATASET_ROOT="${DATASET_ROOT:-/HOME/sysu_xdliang/sysu_xdliang_1/HDD_POOL/linmin/datasets/robocasa365-datasets}"
TASK="${TASK:-PickPlaceToasterToCounter}"
EPISODE="${EPISODE:-0}"
STEP="${STEP:-100}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/output/moss_visualize_test}"
DEVICE="${DEVICE:-cuda}"
NUM_FRAMES="${NUM_FRAMES:-4}"
DISPLAY_VIEW="${DISPLAY_VIEW:-robot0_eye_in_hand}"

cd "$PROJECT_ROOT"

ARGS=(
  --mode "$MODE"
  --checkpoint "$CHECKPOINT"
  --output-dir "$OUTPUT_DIR"
  --device "$DEVICE"
  --num-frames "$NUM_FRAMES"
  --display-view "$DISPLAY_VIEW"
)
if [[ -n "$VIDEO" ]]; then
  ARGS+=(--video "$VIDEO")
elif [[ -n "$DATASET_ROOT" ]]; then
  ARGS+=(--dataset-root "$DATASET_ROOT" --task "$TASK" --episode "$EPISODE" --step "$STEP")
fi

echo "[i] MOSS visualize test mode=$MODE output=$OUTPUT_DIR"
exec .venv/bin/python -u examples/RoboCasa365/visualize_moss_test.py "${ARGS[@]}"
