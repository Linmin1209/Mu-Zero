#!/usr/bin/env bash
# Parallel full 3D replay: up to 50 task workers (manifest-aligned).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
MANIFEST="${MANIFEST:-$SCRIPT_DIR/data/manifest_target50.jsonl}"
PCD_ROOT="${PCD_ROOT:-$SCRIPT_DIR/data/leo_3d_cache}"
LOG_DIR="${LOG_DIR:-$PCD_ROOT/_logs}"
PY365_DEFAULT="$PROJECT_ROOT/gr00t/eval/sim/robocasa365/robocasa365_uv/.venv/bin/python"
RLDX_PY365="/HOME/sysu_xdliang/sysu_xdliang_1/HDD_POOL/linmin/RLDX-1/rldx/eval/sim/robocasa365/robocasa365_uv/.venv/bin/python"
PYTHON="${REPLAY_PYTHON:-}"
if [[ -z "$PYTHON" ]]; then
  if [[ -x "$RLDX_PY365" ]]; then
    PYTHON="$RLDX_PY365"
  elif [[ -x "$PY365_DEFAULT" ]]; then
    PYTHON="$PY365_DEFAULT"
  else
    echo "[x] RoboCasa365 python not found"
    exit 1
  fi
fi

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export ROBOSUITE_NO_MACRO_CHECK="${ROBOSUITE_NO_MACRO_CHECK:-1}"

MAX_PARALLEL="${MAX_PARALLEL:-50}"
SAVE_DEPTH="${SAVE_DEPTH:-1}"

NUM_GPUS="${NUM_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
if [[ "$NUM_GPUS" -lt 1 ]]; then NUM_GPUS=1; fi

mkdir -p "$LOG_DIR" "$PCD_ROOT"
cd "$PROJECT_ROOT"

TASKS=()
while IFS= read -r t; do
  [[ -n "$t" ]] && TASKS+=("$t")
done < <(
  python3 - <<'PY' "$SCRIPT_DIR/../../task_sets.yaml"
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
for key in ("atomic_seen", "composite_seen", "composite_unseen"):
    for t in cfg.get(key, []):
        print(t)
PY
)

WORKER="$LOG_DIR/_run_task.sh"
{
  echo '#!/usr/bin/env bash'
  echo 'set -euo pipefail'
  echo 'task="$1"'
  echo 'gpu="$2"'
  echo "log=\"$LOG_DIR/\${task}.log\""
  echo 'echo "[start] $task gpu=$gpu -> $log"'
  echo "export MUJOCO_GL=\"$MUJOCO_GL\""
  echo "export PYOPENGL_PLATFORM=\"$PYOPENGL_PLATFORM\""
  echo "export ROBOSUITE_NO_MACRO_CHECK=\"$ROBOSUITE_NO_MACRO_CHECK\""
  echo -n "CUDA_VISIBLE_DEVICES=\"\$gpu\" \"$PYTHON\" -u \"$SCRIPT_DIR/replay_rc365_3d.py\" --from-manifest \"$MANIFEST\" --tasks \"\$task\" --pcd-root \"$PCD_ROOT\""
  if [[ "$SAVE_DEPTH" != "1" ]]; then
    echo -n ' --no-save-depth'
  fi
  echo ' >"$log" 2>&1'
  echo 'echo "[done] $task"'
} >"$WORKER"
chmod +x "$WORKER"

echo "[i] PYTHON=$PYTHON"
echo "[i] MANIFEST=$MANIFEST"
echo "[i] PCD_ROOT=$PCD_ROOT"
echo "[i] LOG_DIR=$LOG_DIR"
echo "[i] Tasks=${#TASKS[@]} MAX_PARALLEL=$MAX_PARALLEL NUM_GPUS=$NUM_GPUS"

idx=0
for task in "${TASKS[@]}"; do
  while [[ "$(jobs -pr | wc -l)" -ge "$MAX_PARALLEL" ]]; do
    sleep 2
  done
  gpu=$((idx % NUM_GPUS))
  echo "[launch] $task gpu=$gpu"
  bash "$WORKER" "$task" "$gpu" &
  idx=$((idx + 1))
done

wait || true
echo "[i] All task workers finished. Linking manifest ..."
CONVERT_PY="${CONVERT_PY:-$PROJECT_ROOT/.venv/bin/python}"
if [[ ! -x "$CONVERT_PY" ]]; then CONVERT_PY=python3; fi
"$CONVERT_PY" -u "$SCRIPT_DIR/convert_robocasa365_to_leo.py" \
  --output "$MANIFEST" \
  --split target50 \
  --stride "${STRIDE:-2}" \
  --max-episodes-per-task "${MAX_EP_PER_TASK:-50}" \
  --pcd-root "$PCD_ROOT" \
  --link-3d

echo "[i] Parallel 3D replay complete."
