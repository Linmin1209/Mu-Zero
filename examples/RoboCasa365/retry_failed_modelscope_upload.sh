#!/usr/bin/env bash
# Retry only tasks marked FAIL in upload_progress.txt.
#
# Usage:
#   source /app/bin/proxy.sh
#   MS_TOKEN=ms-xxx bash examples/RoboCasa365/retry_failed_modelscope_upload.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

ROBOCASA365_ROOT="${ROBOCASA365_ROOT:-$PROJECT_ROOT/../datasets/robocasa365-datasets}"
MODELSCOPE_REPO="${MODELSCOPE_REPO:-Twilighted/Robocasa365-tactile}"
MODELSCOPE_TOKEN="${MODELSCOPE_TOKEN:-${MS_TOKEN:-}}"
MAX_WORKERS="${MAX_WORKERS:-8}"
LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/output/modelscope_upload}"
PROGRESS_FILE="${PROGRESS_FILE:-$LOG_DIR/upload_progress.txt}"

if [[ -z "$MODELSCOPE_TOKEN" ]]; then
  echo "Set MODELSCOPE_TOKEN or MS_TOKEN." >&2
  exit 1
fi

mkdir -p "$LOG_DIR"
touch "$PROGRESS_FILE"

mapfile -t FAIL_TASKS < <(grep '^FAIL ' "$PROGRESS_FILE" | awk '{print $2}' | sort -u)
if [[ ${#FAIL_TASKS[@]} -eq 0 ]]; then
  echo "[i] No FAIL entries in $PROGRESS_FILE"
  exit 0
fi

echo "[i] Retrying ${#FAIL_TASKS[@]} failed task(s) -> $MODELSCOPE_REPO"

retry_one() {
  local path_in_repo="$1"
  local local_path="$ROBOCASA365_ROOT/$path_in_repo"
  if [[ ! -d "$local_path" ]]; then
    echo "[x] missing local dir: $local_path" >&2
    return 1
  fi
  if grep -Fxq "OK $path_in_repo" "$PROGRESS_FILE"; then
    echo "[skip] already OK: $path_in_repo"
    return 0
  fi

  grep -vF "FAIL $path_in_repo" "$PROGRESS_FILE" >"${PROGRESS_FILE}.tmp" \
    && mv "${PROGRESS_FILE}.tmp" "$PROGRESS_FILE"

  local task_log="$LOG_DIR/task_retry_${path_in_repo//\//_}.log"
  echo "[retry] $path_in_repo"
  if modelscope upload "$MODELSCOPE_REPO" "$local_path" "$path_in_repo" \
    --repo-type dataset \
    --token "$MODELSCOPE_TOKEN" \
    --max-workers "$MAX_WORKERS" \
    --exclude "**/._____temp/**" "**/.msc" "**/.mv" \
    --commit-message "retry ${path_in_repo} (tactile labels)" \
    >"$task_log" 2>&1; then
    echo "OK $path_in_repo" >>"$PROGRESS_FILE"
    echo "[ok] $path_in_repo"
    return 0
  fi
  echo "FAIL $path_in_repo" >>"$PROGRESS_FILE"
  echo "[fail] $path_in_repo (see $task_log)" >&2
  return 1
}

ok=0
fail=0
for path in "${FAIL_TASKS[@]}"; do
  if retry_one "$path"; then
    ok=$((ok + 1))
  else
    fail=$((fail + 1))
  fi
  sleep 2
done

echo "[i] retry finished: ok=$ok fail=$fail progress=$PROGRESS_FILE"
