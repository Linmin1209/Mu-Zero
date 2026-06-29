#!/usr/bin/env bash
# Build LEO multi-task manifest for all 50 target tasks.
# - atomic_seen + composite_seen  -> pretrain split
# - composite_unseen (16 tasks)   -> target split
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
ROBOCASA365_ROOT="${ROBOCASA365_ROOT:-/HOME/sysu_xdliang/sysu_xdliang_1/HDD_POOL/linmin/datasets/robocasa365-datasets}"
SPLIT="${SPLIT:-target50}"
OUTPUT="${OUTPUT:-$SCRIPT_DIR/data/manifest_target50.jsonl}"
STRIDE="${STRIDE:-2}"
MAX_EP_PER_TASK="${MAX_EP_PER_TASK:-50}"

PYTHON="${PYTHON:-$PROJECT_ROOT/.venv/bin/python}"
PY365_DEFAULT="$PROJECT_ROOT/gr00t/eval/sim/robocasa365/robocasa365_uv/.venv/bin/python"
RLDX_PY365="/HOME/sysu_xdliang/sysu_xdliang_1/HDD_POOL/linmin/RLDX-1/rldx/eval/sim/robocasa365/robocasa365_uv/.venv/bin/python"
REPLAY_PYTHON="${REPLAY_PYTHON:-}"
if [[ -z "$REPLAY_PYTHON" ]]; then
  if [[ -x "$RLDX_PY365" ]]; then
    REPLAY_PYTHON="$RLDX_PY365"
  elif [[ -x "$PY365_DEFAULT" ]]; then
    REPLAY_PYTHON="$PY365_DEFAULT"
  else
    REPLAY_PYTHON="$PYTHON"
  fi
fi

if [[ ! -x "$PYTHON" ]]; then
  PYTHON=python3
fi

LINK_3D="${LINK_3D:-1}"
PCD_ROOT="${PCD_ROOT:-$SCRIPT_DIR/data/leo_3d_cache}"

echo "[i] ROBOCASA365_ROOT=$ROBOCASA365_ROOT"
echo "[i] SPLIT=$SPLIT STRIDE=$STRIDE MAX_EP_PER_TASK=$MAX_EP_PER_TASK"
echo "[i] OUTPUT=$OUTPUT"
echo "[i] LINK_3D=$LINK_3D PCD_ROOT=$PCD_ROOT"

cd "$PROJECT_ROOT"
EXTRA_3D=()
if [[ "$LINK_3D" == "1" ]]; then
  EXTRA_3D=(--pcd-root "$PCD_ROOT" --link-3d)
fi
"$PYTHON" -u "$SCRIPT_DIR/convert_robocasa365_to_leo.py" \
  --robocasa365-root "$ROBOCASA365_ROOT" \
  --split "$SPLIT" \
  --output "$OUTPUT" \
  --stride "$STRIDE" \
  --max-episodes-per-task "$MAX_EP_PER_TASK" \
  "${EXTRA_3D[@]}"

echo "[i] Manifest ready: $OUTPUT"
echo "[i] Summary: ${OUTPUT%.jsonl}.summary.json"

# Backward-compatible symlink for finetune script / yaml
ln -sfn "$(basename "$OUTPUT")" "$SCRIPT_DIR/data/manifest_target50_pretrain.jsonl"
ln -sfn "$(basename "${OUTPUT%.jsonl}.summary.json")" "$SCRIPT_DIR/data/manifest_target50_pretrain.summary.json"
