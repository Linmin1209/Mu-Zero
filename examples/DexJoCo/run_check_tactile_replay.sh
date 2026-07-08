#!/usr/bin/env bash
# Quick tactile sanity check: replay 1–2 DexJoCo episodes with video + tactile overlay.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
# shellcheck source=examples/DexJoCo/env_defaults.sh
source "$SCRIPT_DIR/env_defaults.sh"

TASK="${TASK:-bimanual_assembly}"
EPISODES="${EPISODES:-0,1}"
MAX_FRAMES="${MAX_FRAMES:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/output/dexjoco_tactile_check/${TASK}_$(date +%Y%m%d_%H%M%S)}"

PY365="$PROJECT_ROOT/.venv/bin/python"
RLDX_PY365="/HOME/sysu_xdliang/sysu_xdliang_1/HDD_POOL/linmin/RLDX-1/rldx/eval/sim/robocasa365/robocasa365_uv/.venv/bin/python"
if [[ ! -x "$PY365" ]]; then
  PY365="$RLDX_PY365"
fi

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
GR00T_SITE="$PROJECT_ROOT/.venv/lib/python3.10/site-packages"
export PYTHONPATH="$SCRIPT_DIR:$PROJECT_ROOT:$GR00T_SITE${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$OUTPUT_DIR"
echo "[i] task=$TASK episodes=$EPISODES datasets=$DEXJOCo_DATASETS_ROOT"
echo "[i] dexjoco=$DEXJOCo_ROOT output=$OUTPUT_DIR"

"$PY365" -u "$SCRIPT_DIR/visualize_dexjoco_tactile_replay.py" \
  --datasets-root "$DEXJOCo_DATASETS_ROOT" \
  --dexjoco-root "$DEXJOCo_ROOT" \
  --task "$TASK" \
  --episodes "$EPISODES" \
  --output-dir "$OUTPUT_DIR" \
  --compare-dataset-video \
  ${MAX_FRAMES:+--max-frames "$MAX_FRAMES"} \
  "$@"

echo "[ok] Videos written under $OUTPUT_DIR"
