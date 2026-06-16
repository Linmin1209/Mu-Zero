#!/usr/bin/env bash
# Generate gripper-only haptic GT labels for a RoboCasa365 LeRobot dataset.
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
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"

if [[ -z "$DATASET" ]]; then
  echo "Usage: DATASET=/path/to/lerobot [OUTPUT_DATASET=...] [NUM_WORKERS=4] $0" >&2
  exit 1
fi

export MUJOCO_GL=egl
export CUDA_VISIBLE_DEVICES
export MUJOCO_EGL_DEVICE_ID

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

echo "[i] python=$PYTHON"
echo "[i] dataset=$DATASET"
echo "[i] output=${OUTPUT_DATASET:-$DATASET (in-place)}"
echo "[i] workers=$NUM_WORKERS cuda=$CUDA_VISIBLE_DEVICES egl=$MUJOCO_EGL_DEVICE_ID"

exec "$PYTHON" -u "$SCRIPT_DIR/scripts/generate_haptic_gripper_labels.py" "${ARGS[@]}"
