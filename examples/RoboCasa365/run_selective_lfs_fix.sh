#!/usr/bin/env bash
# Scan for missing LFS blobs and re-upload only broken tasks.
#
# Usage:
#   source /app/bin/proxy.sh
#   MODELSCOPE_TOKEN=ms-xxx bash examples/RoboCasa365/run_selective_lfs_fix.sh
#
# Optional:
#   SCAN_ONLY=1          # only scan, write broken_tasks.txt
#   UPLOAD_ONLY=1        # upload from existing broken_tasks.txt
#   SCAN_WORKERS=8 UPLOAD_WORKERS=2
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/output/modelscope_upload}"
MODELSCOPE_TOKEN="${MODELSCOPE_TOKEN:-${MS_TOKEN:-}}"
ROBOCASA365_ROOT="${ROBOCASA365_ROOT:-$PROJECT_ROOT/../datasets/robocasa365-datasets}"

if [[ -z "$MODELSCOPE_TOKEN" ]]; then
  echo "Set MODELSCOPE_TOKEN" >&2
  exit 1
fi

mkdir -p "$LOG_DIR"
export UPLOAD_USE_CACHE=false
export PYTHONUNBUFFERED=1

ARGS=(
  --root "$ROBOCASA365_ROOT"
  --token "$MODELSCOPE_TOKEN"
  --log-dir "$LOG_DIR"
  --scan-workers "${SCAN_WORKERS:-8}"
  --upload-workers "${UPLOAD_WORKERS:-2}"
  --max-workers "${MAX_WORKERS:-16}"
)
[[ "${SCAN_ONLY:-0}" == "1" ]] && ARGS+=(--scan-only)
[[ "${UPLOAD_ONLY:-0}" == "1" ]] && ARGS+=(--upload-only)

LOG="$LOG_DIR/selective_lfs_fix_$(date +%Y%m%d_%H%M%S).log"
echo "[i] log=$LOG"
python3 "$SCRIPT_DIR/fix_broken_lfs_upload.py" "${ARGS[@]}" 2>&1 | tee "$LOG"
