#!/usr/bin/env bash
# Upload robocasa365-datasets to ModelScope (chunked per task; avoids 50k files/dir limit).
#
# Usage:
#   source /app/bin/proxy.sh
#   MODELSCOPE_TOKEN=ms-xxx bash examples/RoboCasa365/upload_robocasa365_to_modelscope.sh
#
# Optional:
#   ROBOCASA365_ROOT=/path/to/robocasa365-datasets
#   MODELSCOPE_REPO=Twilighted/Robocasa365-tactile
#   PARALLEL_TASKS=2
#   FORCE_REUPLOAD=1
#   USE_UPLOAD_CACHE=0   # disable skip-cache; required for LFS blob fix
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

ROBOCASA365_ROOT="${ROBOCASA365_ROOT:-$PROJECT_ROOT/../datasets/robocasa365-datasets}"
MODELSCOPE_REPO="${MODELSCOPE_REPO:-Twilighted/Robocasa365-tactile}"
MODELSCOPE_TOKEN="${MODELSCOPE_TOKEN:-${MS_TOKEN:-}}"
PARALLEL_TASKS="${PARALLEL_TASKS:-2}"
MAX_WORKERS="${MAX_WORKERS:-16}"
FORCE_REUPLOAD="${FORCE_REUPLOAD:-0}"
USE_UPLOAD_CACHE="${USE_UPLOAD_CACHE:-1}"
LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/output/modelscope_upload}"
PROGRESS_FILE="${PROGRESS_FILE:-$LOG_DIR/upload_progress.txt}"
UPLOAD_EXCLUDES=(
  "**/._____temp/**"
  "**/.msc"
  "**/.mv"
  "**/.ms_upload_cache/**"
  "**/.ms_upload_cache"
)

if [[ -z "$MODELSCOPE_TOKEN" ]]; then
  echo "Set MODELSCOPE_TOKEN (ModelScope access token)." >&2
  exit 1
fi

mkdir -p "$LOG_DIR"
touch "$PROGRESS_FILE"

if [[ "$USE_UPLOAD_CACHE" == "0" ]]; then
  export UPLOAD_USE_CACHE=false
  echo "[i] UPLOAD_USE_CACHE=false (force re-upload LFS blobs)"
else
  export UPLOAD_USE_CACHE=true
fi

upload_one() {
  local local_path="$1"
  local path_in_repo="$2"
  if [[ "$FORCE_REUPLOAD" != "1" ]] && grep -Fxq "OK $path_in_repo" "$PROGRESS_FILE"; then
    echo "[skip] $path_in_repo"
    return 0
  fi
  local task_log="$LOG_DIR/task_${path_in_repo//\//_}.log"
  echo "[upload] $path_in_repo"
  if modelscope upload "$MODELSCOPE_REPO" "$local_path" "$path_in_repo" \
    --repo-type dataset \
    --token "$MODELSCOPE_TOKEN" \
    --max-workers "$MAX_WORKERS" \
    --exclude "${UPLOAD_EXCLUDES[@]}" \
    --commit-message "fix LFS blob: reupload ${path_in_repo}" \
    >"$task_log" 2>&1; then
    echo "OK $path_in_repo" >>"$PROGRESS_FILE"
    echo "[ok] $path_in_repo"
    return 0
  fi
  echo "FAIL $path_in_repo" >>"$PROGRESS_FILE"
  echo "[fail] $path_in_repo (see $task_log)" >&2
  return 1
}

export -f upload_one
export MODELSCOPE_REPO MODELSCOPE_TOKEN MAX_WORKERS LOG_DIR PROGRESS_FILE FORCE_REUPLOAD

# Root metadata (small files only).
for f in README.md .gitattributes; do
  if [[ -f "$ROBOCASA365_ROOT/$f" ]]; then
    if [[ "$FORCE_REUPLOAD" == "1" ]] || ! grep -Fxq "OK $f" "$PROGRESS_FILE"; then
      modelscope upload "$MODELSCOPE_REPO" "$ROBOCASA365_ROOT/$f" "$f" \
        --repo-type dataset --token "$MODELSCOPE_TOKEN" \
        --exclude "${UPLOAD_EXCLUDES[@]}" \
        --commit-message "reupload $f" \
        >>"$LOG_DIR/upload_root.log" 2>&1 && echo "OK $f" >>"$PROGRESS_FILE"
    fi
  fi
done

TASK_LIST="$(mktemp)"
for split in pretrain target; do
  for cat in atomic composite; do
  base="$ROBOCASA365_ROOT/$split/$cat"
  [[ -d "$base" ]] || continue
  for task_path in "$base"/*; do
    [[ -d "$task_path" ]] || continue
    echo "$task_path $split/$cat/$(basename "$task_path")" >>"$TASK_LIST"
  done
  done
done

total=$(wc -l <"$TASK_LIST")
echo "[i] repo=$MODELSCOPE_REPO root=$ROBOCASA365_ROOT tasks=$total parallel=$PARALLEL_TASKS"

if command -v parallel >/dev/null 2>&1; then
  parallel -j "$PARALLEL_TASKS" --colsep ' ' upload_one {1} {2} :::: "$TASK_LIST"
else
  while read -r local_path path_in_repo; do
    upload_one "$local_path" "$path_in_repo" || true
  done <"$TASK_LIST"
fi

rm -f "$TASK_LIST"
echo "[i] done. progress: $PROGRESS_FILE"
