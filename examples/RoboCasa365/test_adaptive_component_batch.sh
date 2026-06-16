#!/usr/bin/env bash
# Small-batch smoke test: adaptive component head data + optional forward.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

export ROBOCASA365_ROOT="${ROBOCASA365_ROOT:-/HOME/sysu_xdliang/sysu_xdliang_1/HDD_POOL/linmin/datasets/robocasa365-datasets}"
export GR00T_MODELS_ROOT="${GR00T_MODELS_ROOT:-/HOME/sysu_xdliang/sysu_xdliang_1/HDD_POOL/linmin/models}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

FORWARD="${FORWARD:-1}"
EXTRA=()
if [[ "$FORWARD" == "1" ]]; then
  EXTRA+=(--forward)
fi

.venv/bin/python -u "$SCRIPT_DIR/test_adaptive_component_batch.py" \
  --batch-size 2 \
  --modality-config-path "$SCRIPT_DIR/robocasa365_config.py" \
  "${EXTRA[@]}"
