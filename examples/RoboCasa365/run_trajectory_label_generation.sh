#!/usr/bin/env bash
# Generate arm/base 2D trajectory labels for a RoboCasa365 LeRobot dataset.
#
# Requires robocasa365 sim stack (robosuite + robocasa). Default uses RLDX eval venv.
# Headless MuJoCo: default OSMesa (CPU offscreen). Override with MUJOCO_GL=egl if EGL works.
#
# OSMesa system deps (once):
#   apt-get install -y libosmesa6 libgl1 libglib2.0-0
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
# shellcheck source=examples/RoboCasa365/env_defaults.sh
source "$SCRIPT_DIR/env_defaults.sh"

DATASET="${DATASET:-}"
OUTPUT_DATASET="${OUTPUT_DATASET:-}"
ROBOCASA_ROOT="${ROBOCASA_ROOT:-/HOME/sysu_xdliang/sysu_xdliang_1/HDD_POOL/linmin/RLDX-1/external_dependencies/robocasa365}"
PYTHON="${PYTHON:-/HOME/sysu_xdliang/sysu_xdliang_1/HDD_POOL/linmin/RLDX-1/rldx/eval/sim/robocasa365/robocasa365_uv/.venv/bin/python}"
NUM_WORKERS="${NUM_WORKERS:-1}"
EPISODE_START="${EPISODE_START:-0}"
EPISODE_END="${EPISODE_END:-}"
MUJOCO_GL="${MUJOCO_GL:-osmesa}"

if [[ -z "$DATASET" ]]; then
  echo "Usage: DATASET=/path/to/lerobot [OUTPUT_DATASET=...] [NUM_WORKERS=4] $0" >&2
  exit 1
fi

export MUJOCO_GL
if [[ "$MUJOCO_GL" == "osmesa" ]]; then
  export PYOPENGL_PLATFORM=osmesa
  unset MUJOCO_EGL_DEVICE_ID 2>/dev/null || true
  GL_EXTRA="osmesa (CPU offscreen)"
else
  export PYOPENGL_PLATFORM=egl
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
  export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"
  GL_EXTRA="egl cuda=${CUDA_VISIBLE_DEVICES} egl_device=${MUJOCO_EGL_DEVICE_ID}"
fi

ARGS=(
  --dataset "$DATASET"
  --robocasa-root "$ROBOCASA_ROOT"
  --num-workers "$NUM_WORKERS"
  --episode-start "$EPISODE_START"
)
if [[ -n "$OUTPUT_DATASET" ]]; then
  ARGS+=(--output-dataset "$OUTPUT_DATASET" --overwrite)
fi
if [[ -n "$EPISODE_END" ]]; then
  ARGS+=(--episode-end "$EPISODE_END")
fi

if [[ -n "${FUTURE_LENGTH:-}" ]]; then
  ARGS+=(--future-length "$FUTURE_LENGTH")
fi

echo "[i] python=$PYTHON"
echo "[i] dataset=$DATASET"
echo "[i] output=${OUTPUT_DATASET:-$DATASET (in-place)}"
echo "[i] workers=$NUM_WORKERS headless_gl=$GL_EXTRA"

exec "$PYTHON" -u "$SCRIPT_DIR/scripts/generate_trajectory_labels.py" "${ARGS[@]}"
