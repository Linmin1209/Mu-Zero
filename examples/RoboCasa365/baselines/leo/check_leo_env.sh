#!/usr/bin/env bash
# Verify leo conda env can import LeoAgent (after deps installed).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
LEO_REPO="${LEO_REPO:-$PROJECT_ROOT/../embodied-generalist}"
PYTHON="${PYTHON:-/XYAIFS00/sysu_xdliang_1/miniconda3/envs/leo/bin/python}"

echo "[i] PYTHON=$PYTHON"
echo "[i] LEO_REPO=$LEO_REPO"

"$PYTHON" - <<PY
import sys
sys.path.insert(0, "$LEO_REPO")
from model.leo_agent import LeoAgent
print("[i] LeoAgent import OK")
PY

ls -lh "$LEO_REPO/checkpoints/align.pth" "$LEO_REPO/checkpoints/sft_noact.pth" 2>/dev/null || true
ls -lh "$LEO_REPO/weights/vicuna-7b/config.json" 2>/dev/null || echo "[w] Vicuna-7B not downloaded yet"
