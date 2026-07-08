#!/usr/bin/env bash
# Evaluate a GR00T DexJoCo checkpoint in MuJoCo sim.
#
# Usage:
#   bash examples/DexJoCo/eval_dexjoco.sh \
#     --model-path output/dexjoco_single_arm_gr00t/checkpoint-5000 \
#     --task water_plant
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
# shellcheck source=examples/DexJoCo/env_defaults.sh
source "$SCRIPT_DIR/env_defaults.sh"

MODEL_PATH=""
TASK="water_plant"
N_EPISODES=20
MAX_STEPS=600
SEED=0
RAND_FULL=0
SERVER_HOST="127.0.0.1"
SERVER_PORT="${SERVER_PORT:-5555}"
SERVER_DEVICE="cuda"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/output/dexjoco_eval}"
EMBODIMENT_TAG=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-path) MODEL_PATH="$2"; shift 2 ;;
    --task) TASK="$2"; shift 2 ;;
    --n-episodes) N_EPISODES="$2"; shift 2 ;;
    --max-steps) MAX_STEPS="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --rand-full) RAND_FULL=1; shift ;;
    --server-port) SERVER_PORT="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    *)
      echo "[x] Unknown arg: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "$MODEL_PATH" ]]; then
  echo "[x] --model-path is required" >&2
  exit 1
fi

REGISTRY="$SCRIPT_DIR/task_registry.yaml"
ROBOT_TYPE="$(python3 - <<PY
import yaml
from pathlib import Path
reg = yaml.safe_load(Path("$REGISTRY").read_text())
print(reg["tasks"]["$TASK"]["robot_type"])
PY
)"
if [[ "$ROBOT_TYPE" == "single_arm" ]]; then
  MODALITY_CONFIG="$SCRIPT_DIR/dexjoco_single_arm_config.py"
  EMBODIMENT_TAG="${EMBODIMENT_TAG:-DEXJOCo_SINGLE_ARM}"
else
  MODALITY_CONFIG="$SCRIPT_DIR/dexjoco_bimanual_config.py"
  EMBODIMENT_TAG="${EMBODIMENT_TAG:-DEXJOCo_BIMANUAL}"
fi

RUN_DIR="$OUTPUT_ROOT/${TASK}_$(basename "$MODEL_PATH")"
mkdir -p "$RUN_DIR"
SERVER_LOG="$RUN_DIR/gr00t_server.log"

echo "[i] Starting GR00T server on port $SERVER_PORT ..."
(
  cd "$PROJECT_ROOT"
  .venv/bin/python -u gr00t/eval/run_gr00t_server.py \
    --model-path "$MODEL_PATH" \
    --embodiment-tag "$EMBODIMENT_TAG" \
    --modality-config-path "$MODALITY_CONFIG" \
    --host "$SERVER_HOST" \
    --port "$SERVER_PORT" \
    --device "$SERVER_DEVICE" \
    >"$SERVER_LOG" 2>&1
) &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT

sleep 15

EVAL_ARGS=(
  --dexjoco-root "$DEXJOCo_ROOT"
  --task "$TASK"
  --registry "$REGISTRY"
  --host "$SERVER_HOST"
  --port "$SERVER_PORT"
  --n-episodes "$N_EPISODES"
  --max-steps "$MAX_STEPS"
  --seed "$SEED"
  --output-dir "$RUN_DIR"
)
if [[ "$RAND_FULL" == "1" ]]; then
  EVAL_ARGS+=(--rand-full)
fi

(
  cd "$PROJECT_ROOT"
  PYTHONPATH="$DEXJOCo_ROOT/dexjoco:${PYTHONPATH:-}" \
    .venv/bin/python -u gr00t/eval/sim/dexjoco/eval_dexjoco_gr00t.py \
    "${EVAL_ARGS[@]}"
)

echo "[ok] Eval complete -> $RUN_DIR"
