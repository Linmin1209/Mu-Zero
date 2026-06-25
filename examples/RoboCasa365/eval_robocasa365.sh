#!/usr/bin/env bash
set -euo pipefail

# RoboCasa365 benchmark eval for Isaac-GR00T finetuned checkpoints.
#
# Usage:
#   bash examples/RoboCasa365/eval_robocasa365.sh \
#     --model-path /path/to/checkpoint-5000 \
#     --task-set atomic_seen \
#     --split pretrain \
#     --tasks NavigateKitchen

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"
PY365_DEFAULT="$PROJECT_REPO/gr00t/eval/sim/robocasa365/robocasa365_uv/.venv/bin/python"
RLDX_PY365="/HOME/sysu_xdliang/sysu_xdliang_1/HDD_POOL/linmin/RLDX-1/rldx/eval/sim/robocasa365/robocasa365_uv/.venv/bin/python"
GR00T_PYTHON="${GR00T_PYTHON:-$PROJECT_REPO/.venv/bin/python}"

MODEL_PATH=""
TASK_SET="atomic_seen"
TASKS_FILTER=""
SPLIT="pretrain"
N_EPISODES=50
N_ENVS=5
N_ACTION_STEPS=40
MAX_EPISODE_STEPS=720
EMBODIMENT_TAG="ROBOCASA_PANDA_OMRON"
SERVER_HOST="127.0.0.1"
SERVER_BIND_HOST="127.0.0.1"
SERVER_PORT="${SERVER_PORT:-5555}"
SERVER_DEVICE="cuda"
PY365="${PY365:-$PY365_DEFAULT}"
OUTPUT_ROOT="$PROJECT_REPO/output/robocasa365_eval"
SERVER_WARMUP_SEC="${SERVER_WARMUP_SEC:-600}"
TASK_YAML="$SCRIPT_DIR/task_sets.yaml"
NUM_SHARDS=1
SHARD_INDEX=0
USER_SET_NUM_SHARDS=0
USER_SET_SHARD_INDEX=0
USER_SET_SERVER_PORT=0
USE_TASK_HORIZON="${USE_TASK_HORIZON:-1}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-path) MODEL_PATH="$2"; shift 2 ;;
    --task-set) TASK_SET="$2"; shift 2 ;;
    --tasks) TASKS_FILTER="$2"; shift 2 ;;
    --split) SPLIT="$2"; shift 2 ;;
    --n-episodes) N_EPISODES="$2"; shift 2 ;;
    --n-envs) N_ENVS="$2"; shift 2 ;;
    --n-action-steps) N_ACTION_STEPS="$2"; shift 2 ;;
    --max-episode-steps) MAX_EPISODE_STEPS="$2"; shift 2 ;;
    --embodiment-tag) EMBODIMENT_TAG="$2"; shift 2 ;;
    --server-host) SERVER_HOST="$2"; shift 2 ;;
    --server-bind-host) SERVER_BIND_HOST="$2"; shift 2 ;;
    --server-port) SERVER_PORT="$2"; USER_SET_SERVER_PORT=1; shift 2 ;;
    --server-device) SERVER_DEVICE="$2"; shift 2 ;;
    --python) PY365="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --server-warmup-sec) SERVER_WARMUP_SEC="$2"; shift 2 ;;
    --task-yaml) TASK_YAML="$2"; shift 2 ;;
    --num-shards) NUM_SHARDS="$2"; USER_SET_NUM_SHARDS=1; shift 2 ;;
    --shard-index) SHARD_INDEX="$2"; USER_SET_SHARD_INDEX=1; shift 2 ;;
    *)
      echo "[x] Unknown argument: $1"
      exit 1
      ;;
  esac
done

bool_to_int() {
  case "${1,,}" in
    1|true|yes|y|on) echo 1 ;;
    *) echo 0 ;;
  esac
}

USE_TASK_HORIZON_INT="$(bool_to_int "$USE_TASK_HORIZON")"

if [[ -z "$MODEL_PATH" ]]; then
  echo "[x] --model-path is required (finetune checkpoint dir, e.g. .../checkpoint-5000)"
  exit 1
fi

if [[ "$SPLIT" != "pretrain" && "$SPLIT" != "target" ]]; then
  echo "[x] --split must be pretrain or target"
  exit 1
fi

if [[ ! -x "$PY365" ]]; then
  if [[ -x "$RLDX_PY365" ]]; then
    PY365="$RLDX_PY365"
  else
    echo "[x] Sim python not found: $PY365"
    echo "[x] Run: bash examples/RoboCasa365/setup_eval.sh"
    exit 1
  fi
fi

if [[ ! -x "$GR00T_PYTHON" ]]; then
  echo "[x] GR00T python not found: $GR00T_PYTHON"
  exit 1
fi

cd "$PROJECT_REPO"
export NO_ALBUMENTATIONS_UPDATE=1
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export GROOT_PATCH_MISTRAL="${GROOT_PATCH_MISTRAL:-1}"
export GROOT_HF_LOCAL_FIRST="${GROOT_HF_LOCAL_FIRST:-1}"

GR00T_SITE="$PROJECT_REPO/.venv/lib/python3.10/site-packages"
SIM_SITE="$(cd "$(dirname "$PY365")/../lib/python3.10/site-packages" && pwd)"
if [[ ! -d "$GR00T_SITE/tyro" ]]; then
  echo "[x] Main .venv missing tyro. Run: uv sync (or use project .venv)"
  exit 1
fi
# PROJECT_REPO: import gr00t package; SIM_SITE first for numpy/mujoco pins.
export PYTHONPATH="$PROJECT_REPO:$SIM_SITE:$GR00T_SITE${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
unset VIRTUAL_ENV

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export GR00T_POLICY_CLIENT_TIMEOUT_MS="${GR00T_POLICY_CLIENT_TIMEOUT_MS:-120000}"

load_tasks_from_yaml_section() {
  local yaml_file="$1"
  local section="$2"
  awk -v section="$section" '
    /^[A-Za-z0-9_]+:[[:space:]]*$/ {
      key=$1
      sub(":", "", key)
      in_section = (key == section)
      next
    }
    in_section && /^[[:space:]]*-[[:space:]]*/ {
      line=$0
      sub(/^[[:space:]]*-[[:space:]]*/, "", line)
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", line)
      if (length(line) > 0) print line
    }
  ' "$yaml_file"
}

load_task_horizons_from_yaml() {
  local yaml_file="$1"
  awk '
    /^task_horizons:[[:space:]]*$/ { in_section = 1; next }
    in_section && /^[A-Za-z0-9_]+:[[:space:]]*$/ { exit }
    in_section && /^[[:space:]]+[A-Za-z0-9_]+:[[:space:]]*[0-9]+[[:space:]]*$/ {
      line = $0
      sub(/^[[:space:]]+/, "", line)
      split(line, kv, ":")
      key = kv[1]
      val = kv[2]
      gsub(/[[:space:]]/, "", val)
      print key "," val
    }
  ' "$yaml_file"
}

TASKS=()
if [[ -f "$TASK_YAML" && ( "$TASK_SET" == "atomic_seen" || "$TASK_SET" == "composite_seen" || "$TASK_SET" == "composite_unseen" || "$TASK_SET" == "target50" ) ]]; then
  if [[ "$TASK_SET" == "target50" ]]; then
    mapfile -t TASKS < <(
      {
        load_tasks_from_yaml_section "$TASK_YAML" "atomic_seen"
        load_tasks_from_yaml_section "$TASK_YAML" "composite_seen"
        load_tasks_from_yaml_section "$TASK_YAML" "composite_unseen"
      } | awk 'NF' | awk '!seen[$0]++'
    )
  else
    mapfile -t TASKS < <(load_tasks_from_yaml_section "$TASK_YAML" "$TASK_SET")
  fi
else
  echo "[x] Unknown task set: $TASK_SET"
  exit 1
fi

if [[ -n "$TASKS_FILTER" ]]; then
  IFS=',' read -ra WANT <<< "$TASKS_FILTER"
  FILTERED=()
  for task in "${TASKS[@]}"; do
    for w in "${WANT[@]}"; do
      if [[ "$task" == "${w// /}" ]]; then
        FILTERED+=("$task")
        break
      fi
    done
  done
  TASKS=("${FILTERED[@]}")
fi

if [[ ${#TASKS[@]} -eq 0 ]]; then
  echo "[x] No tasks selected"
  exit 1
fi

if [[ "$USER_SET_SHARD_INDEX" -eq 0 && -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  SHARD_INDEX="$SLURM_ARRAY_TASK_ID"
fi
if [[ "$USER_SET_NUM_SHARDS" -eq 0 ]]; then
  if [[ -n "${SLURM_ARRAY_TASK_COUNT:-}" ]]; then
    NUM_SHARDS="$SLURM_ARRAY_TASK_COUNT"
  fi
fi

SELECTED_TASKS=()
for idx in "${!TASKS[@]}"; do
  if (( idx % NUM_SHARDS == SHARD_INDEX )); then
    SELECTED_TASKS+=("${TASKS[$idx]}")
  fi
done

declare -A TASK_HORIZONS=()
if (( USE_TASK_HORIZON_INT == 1 )) && [[ -f "$TASK_YAML" ]]; then
  while IFS=, read -r task_name task_horizon; do
    if [[ -n "$task_name" && "$task_horizon" =~ ^[0-9]+$ ]]; then
      TASK_HORIZONS["$task_name"]="$task_horizon"
    fi
  done < <(load_task_horizons_from_yaml "$TASK_YAML")
fi

RUN_ID="${SLURM_ARRAY_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"
EXP_NAME="$(basename "$MODEL_PATH")_${TASK_SET}_${SPLIT}_exp${RUN_ID}"
RUN_DIR="$OUTPUT_ROOT/$EXP_NAME"
SHARD_TAG="shard${SHARD_INDEX}of${NUM_SHARDS}"
SERVER_LOG="$RUN_DIR/server_${SHARD_TAG}.log"
mkdir -p "$RUN_DIR"

echo "[i] Model: $MODEL_PATH"
echo "[i] Task set: $TASK_SET (${#SELECTED_TASKS[@]} tasks on shard $SHARD_INDEX/$NUM_SHARDS)"
echo "[i] Split: $SPLIT | action steps: $N_ACTION_STEPS | embodiment: $EMBODIMENT_TAG"
echo "[i] Output: $RUN_DIR"

print_log_tail() {
  local log_path="$1"
  [[ -f "$log_path" ]] && tail -n 40 "$log_path"
}

pick_free_port() {
  "$GR00T_PYTHON" - <<'PY'
import socket
s = socket.socket()
s.bind(("127.0.0.1", 0))
print(s.getsockname()[1])
s.close()
PY
}

port_in_use() {
  "$GR00T_PYTHON" - <<PY
import socket
port = int("$1")
s = socket.socket()
try:
    s.bind(("127.0.0.1", port))
except OSError:
    print("yes")
else:
    print("no")
finally:
    s.close()
PY
}

wait_for_server_ready() {
  local log_path="$1"
  local pid="$2"
  local timeout_sec="$3"
  local elapsed=0
  while (( elapsed < timeout_sec )); do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "[x] Server process $pid exited before becoming ready"
      return 1
    fi
    if [[ -f "$log_path" ]] && grep -Eq "Server ready|Server is ready and listening on" "$log_path"; then
      return 0
    fi
    sleep 5
    elapsed=$((elapsed + 5))
  done
  echo "[x] Server did not become ready within ${timeout_sec}s"
  return 1
}

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" || true
  fi
}
trap cleanup EXIT

echo "[i] Preflight sim imports..."
if ! "$PY365" -c "
import gr00t
import numpy
import gymnasium
import robocasa
from gr00t.eval.rollout_policy import create_eval_env
print('gr00t', gr00t.__file__)
print('numpy', numpy.__version__, 'OK')
"; then
  echo "[x] Rollout import preflight failed (check PYTHONPATH / gr00t install)"
  exit 1
fi

if [[ "$(port_in_use "$SERVER_PORT")" == "yes" ]]; then
  OLD_PORT="$SERVER_PORT"
  SERVER_PORT="$(pick_free_port)"
  echo "[i] Port $OLD_PORT busy; using free port $SERVER_PORT for GR00T policy server"
elif [[ "$USER_SET_SERVER_PORT" -eq 0 ]]; then
  SERVER_PORT="$(pick_free_port)"
  echo "[i] Auto-selected policy server port $SERVER_PORT (avoid stale servers on 5555)"
fi
echo "$SERVER_PORT" > "$RUN_DIR/server_port.txt"

# Ensure finetuned RoboCasa365 modality keys are registered before server load.
"$GR00T_PYTHON" -c "import examples.RoboCasa365.robocasa365_config_4frame  # noqa: F401" 2>/dev/null || \
"$GR00T_PYTHON" -c "import examples.RoboCasa365.robocasa365_config  # noqa: F401" 2>/dev/null || true

"$GR00T_PYTHON" -u gr00t/eval/run_gr00t_server.py \
  --model-path "$MODEL_PATH" \
  --embodiment-tag "$EMBODIMENT_TAG" \
  --use-sim-policy-wrapper \
  --host "$SERVER_BIND_HOST" \
  --port "$SERVER_PORT" \
  --device "$SERVER_DEVICE" \
  > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

echo "[i] Policy server pid=$SERVER_PID port=$SERVER_PORT (warmup up to ${SERVER_WARMUP_SEC}s)"
if ! wait_for_server_ready "$SERVER_LOG" "$SERVER_PID" "$SERVER_WARMUP_SEC"; then
  print_log_tail "$SERVER_LOG"
  exit 1
fi
echo "[i] Policy server ready on ${SERVER_HOST}:${SERVER_PORT}"

SUMMARY_CSV="$RUN_DIR/summary_${SHARD_TAG}.csv"
echo "task,success_rate,log_file" > "$SUMMARY_CSV"

for task in "${SELECTED_TASKS[@]}"; do
  env_name="robocasa/${task}"
  task_dir="$RUN_DIR/$task"
  log_file="$task_dir/eval.log"
  task_max_episode_steps="$MAX_EPISODE_STEPS"
  if (( USE_TASK_HORIZON_INT == 1 )) && [[ -n "${TASK_HORIZONS[$task]:-}" ]]; then
    task_max_episode_steps="${TASK_HORIZONS[$task]}"
  fi

  echo "[i] Evaluating $env_name (horizon=$task_max_episode_steps) ..."
  mkdir -p "$task_dir/videos"
  if ! "$PY365" -u -m gr00t.eval.rollout_policy \
    --n-episodes "$N_EPISODES" \
    --model-path "$MODEL_PATH" \
    --policy-client-host "$SERVER_HOST" \
    --policy-client-port "$SERVER_PORT" \
    --max-episode-steps "$task_max_episode_steps" \
    --env-name "$env_name" \
    --n-action-steps "$N_ACTION_STEPS" \
    --n-envs "$N_ENVS" \
    --robocasa-split "$SPLIT" \
    --video-dir "$task_dir/videos" \
    > "$log_file" 2>&1; then
    echo "[x] Rollout failed: $env_name"
    print_log_tail "$log_file"
    exit 1
  fi

  rate="$("$PY365" - <<PY
import re
p = r"$log_file"
rate = "NA"
with open(p, encoding="utf-8") as f:
    for line in f:
        m = re.search(r"success rate:\s*([0-9.]+)", line, re.I)
        if m:
            rate = m.group(1)
print(rate)
PY
)"
  echo "${task},${rate},${log_file}" >> "$SUMMARY_CSV"
  echo "[i] $task success_rate=$rate"
done

echo "[i] Summary: $SUMMARY_CSV"
cat "$SUMMARY_CSV"
