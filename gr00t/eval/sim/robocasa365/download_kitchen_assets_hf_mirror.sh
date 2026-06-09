#!/usr/bin/env bash
# Download twilighted/Robocasa365-Assets via hf-mirror and extract into robocasa365
# models/assets/ (same layout as download_kitchen_assets.py).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_REPO="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
ROBOCASA365_REPO="${ROBOCASA365_REPO:-$PROJECT_REPO/external_dependencies/robocasa365}"
UV_ENV="${UV_ENV:-$SCRIPT_DIR/robocasa365_uv}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_ASSETS_REPO="${HF_ASSETS_REPO:-twilighted/Robocasa365-Assets}"

echo "HF mirror: HF_ENDPOINT=$HF_ENDPOINT"
echo "HF repo:   HF_ASSETS_REPO=$HF_ASSETS_REPO"

if [[ ! -f "$UV_ENV/.venv/bin/python" ]]; then
  echo "robocasa365 venv not found at $UV_ENV/.venv — run setup_RoboCasa365.sh first (or only pip install robocasa365)."
  echo "Attempting with system python3..."
  PYTHON=python3
else
  PYTHON="$UV_ENV/.venv/bin/python"
fi

# Prefer new `hf` CLI; fall back to huggingface_hub Python API inside the script.
if ! command -v hf >/dev/null 2>&1 && ! command -v huggingface-cli >/dev/null 2>&1; then
  echo "Installing huggingface_hub (provides hf / huggingface-cli)..."
  "$PYTHON" -m pip install -q "huggingface_hub>=0.26"
fi

cd "$ROBOCASA365_REPO"
export PYTHONPATH="${ROBOCASA365_REPO}:${PYTHONPATH:-}"

# Default: snapshot whole repo (avoids 404 on guessed zip paths like fixtures.zip)
if [[ $# -eq 0 ]]; then
  set -- --snapshot --types all
fi

"$PYTHON" "$SCRIPT_DIR/download_kitchen_assets_hf_mirror.py" "$@"
