#!/usr/bin/env bash
# Generate L1 pool visual_future labels for VISOR v4.2 (manip + nav streams).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
# shellcheck source=examples/RoboCasa365/env_defaults.sh
source "$SCRIPT_DIR/../env_defaults.sh"

DATASET="${DATASET:-$ROBOCASA365_ROOT/pretrain/atomic/PickPlaceToasterToCounter/20250804/lerobot}"
OUTPUT_DATASET="${OUTPUT_DATASET:-${DATASET}_visual_future}"
DEVICE="${DEVICE:-cuda:0}"
MAX_EPISODES="${MAX_EPISODES:-}"
OVERWRITE="${OVERWRITE:-0}"

cd "$PROJECT_ROOT"
ARGS=(
  --dataset "$DATASET"
  --output-dataset "$OUTPUT_DATASET"
  --device "$DEVICE"
)
if [[ -n "$MAX_EPISODES" ]]; then
  ARGS+=(--max-episodes "$MAX_EPISODES")
fi
if [[ "$OVERWRITE" == "1" ]]; then
  ARGS+=(--overwrite)
fi

echo "[i] input:  $DATASET"
echo "[i] output: $OUTPUT_DATASET"
echo "[i] device: $DEVICE"

.venv/bin/python -u examples/RoboCasa365/scripts/generate_visual_future_labels.py "${ARGS[@]}"
