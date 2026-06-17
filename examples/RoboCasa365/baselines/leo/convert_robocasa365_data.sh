#!/usr/bin/env bash
# Build LEO multi-task manifest for all 50 target tasks (pretrain split).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
ROBOCASA365_ROOT="${ROBOCASA365_ROOT:-/HOME/sysu_xdliang/sysu_xdliang_1/HDD_POOL/linmin/datasets/robocasa365-datasets}"
SPLIT="${SPLIT:-pretrain}"
OUTPUT="${OUTPUT:-$SCRIPT_DIR/data/manifest_target50_${SPLIT}.jsonl}"
STRIDE="${STRIDE:-2}"
MAX_EP_PER_TASK="${MAX_EP_PER_TASK:-0}"

PYTHON="${PYTHON:-$PROJECT_ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON=python3
fi

cd "$PROJECT_ROOT"
"$PYTHON" -u "$SCRIPT_DIR/convert_robocasa365_to_leo.py" \
  --robocasa365-root "$ROBOCASA365_ROOT" \
  --split "$SPLIT" \
  --output "$OUTPUT" \
  --stride "$STRIDE" \
  --max-episodes-per-task "$MAX_EP_PER_TASK"

echo "[i] Manifest ready: $OUTPUT"
