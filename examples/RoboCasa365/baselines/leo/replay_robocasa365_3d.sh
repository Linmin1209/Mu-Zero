#!/usr/bin/env bash
# Replay RoboCasa365 sim states -> depth + camera params + fused scene point clouds.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
MANIFEST="${MANIFEST:-$SCRIPT_DIR/data/manifest_target50.jsonl}"
PCD_ROOT="${PCD_ROOT:-$SCRIPT_DIR/data/leo_3d_cache}"
PY365_DEFAULT="$PROJECT_ROOT/gr00t/eval/sim/robocasa365/robocasa365_uv/.venv/bin/python"
RLDX_PY365="/HOME/sysu_xdliang/sysu_xdliang_1/HDD_POOL/linmin/RLDX-1/rldx/eval/sim/robocasa365/robocasa365_uv/.venv/bin/python"
PYTHON="${REPLAY_PYTHON:-}"
if [[ -z "$PYTHON" ]]; then
  if [[ -x "$RLDX_PY365" ]]; then
    PYTHON="$RLDX_PY365"
  elif [[ -x "$PY365_DEFAULT" ]]; then
    PYTHON="$PY365_DEFAULT"
  else
    PYTHON="${PYTHON:-$PROJECT_ROOT/.venv/bin/python}"
  fi
fi

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export ROBOSUITE_NO_MACRO_CHECK="${ROBOSUITE_NO_MACRO_CHECK:-1}"

if [[ ! -x "$PYTHON" ]]; then
  echo "[x] RoboCasa365 python not found. Run: bash examples/RoboCasa365/setup_eval.sh"
  exit 1
fi

echo "[i] PYTHON=$PYTHON"
echo "[i] MANIFEST=$MANIFEST"
echo "[i] PCD_ROOT=$PCD_ROOT"
echo "[i] MUJOCO_GL=$MUJOCO_GL"

cd "$PROJECT_ROOT"
"$PYTHON" -u "$SCRIPT_DIR/replay_rc365_3d.py" \
  --from-manifest "$MANIFEST" \
  --pcd-root "$PCD_ROOT" \
  "$@"

echo "[i] Rebuilding manifest with 3D paths ..."
CONVERT_PY="${CONVERT_PY:-$PROJECT_ROOT/.venv/bin/python}"
if [[ ! -x "$CONVERT_PY" ]]; then
  CONVERT_PY=python3
fi
"$CONVERT_PY" -u "$SCRIPT_DIR/convert_robocasa365_to_leo.py" \
  --output "$MANIFEST" \
  --split target50 \
  --stride "${STRIDE:-2}" \
  --max-episodes-per-task "${MAX_EP_PER_TASK:-50}" \
  --pcd-root "$PCD_ROOT" \
  --link-3d

echo "[i] 3D replay + manifest link done."
