#!/usr/bin/env bash
# Overlay annotate_sim-style trajectory GT on PickPlace videos.
#
# Requires lerobot_traj* parquet with trajectory.arm_future_uv.* columns
# (from generate_trajectory_labels.py / run_trajectory_label_generation.sh).
#
# Usage:
#   bash examples/RoboCasa365/run_trajectory_visualize_pickplace.sh
#   TRAJ_ROOT=.../lerobot_traj_v2 EPISODES="0 3" bash ...
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
# shellcheck source=examples/RoboCasa365/env_defaults.sh
source "$SCRIPT_DIR/env_defaults.sh"

TASK_REL="${TASK_REL:-pretrain/atomic/PickPlaceToasterToCounter/20250819}"
LEROBOT_ROOT="${LEROBOT_ROOT:-$ROBOCASA365_ROOT/$TASK_REL/lerobot}"
TRAJ_ROOT="${TRAJ_ROOT:-$LEROBOT_ROOT}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/output/traj_viz_pickplace_annotate_sim}"
EPISODES="${EPISODES:-0 3}"
TRAIL_LEN="${TRAIL_LEN:-40}"
PYTHON="${PYTHON:-$PROJECT_ROOT/.venv/bin/python}"

if [[ ! -d "$TRAJ_ROOT/meta" ]]; then
  echo "[e] missing traj dataset: $TRAJ_ROOT" >&2
  echo "    run labeling first, e.g.:" >&2
  echo "    DATASET=$LEROBOT_ROOT OUTPUT_DATASET=$TRAJ_ROOT NUM_WORKERS=4 \\" >&2
  echo "      bash examples/RoboCasa365/run_trajectory_label_generation.sh" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "[i] lerobot (video): $LEROBOT_ROOT"
echo "[i] traj (parquet):  $TRAJ_ROOT"
echo "[i] output:          $OUTPUT_DIR"
echo "[i] episodes:        $EPISODES"

"$PYTHON" -u "$SCRIPT_DIR/scripts/visualize_trajectory_labels.py" \
  --lerobot-root "$LEROBOT_ROOT" \
  --traj-root "$TRAJ_ROOT" \
  --episodes $EPISODES \
  --trail-len "$TRAIL_LEN" \
  --export-mp4 \
  --output-dir "$OUTPUT_DIR"

echo "[i] done. open overlay mp4 / png under: $OUTPUT_DIR"
