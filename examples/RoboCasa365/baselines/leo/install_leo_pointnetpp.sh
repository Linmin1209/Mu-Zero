#!/usr/bin/env bash
# Build PointNet++ CUDA extension required by LEO (leo conda env).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
LEO_REPO="${LEO_REPO:-$PROJECT_ROOT/../embodied-generalist}"
PYTHON="${PYTHON:-/XYAIFS00/sysu_xdliang_1/miniconda3/envs/leo/bin/python}"

if [[ ! -x "$PYTHON" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON=python3
  else
    echo "[x] leo env python not found: $PYTHON"
    exit 1
  fi
fi

PNPP_DIR="$LEO_REPO/model/pointnetpp"
if [[ ! -d "$PNPP_DIR" ]]; then
  echo "[x] PointNet++ source not found: $PNPP_DIR"
  exit 1
fi

echo "[i] Building PointNet++ with $PYTHON"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-7.0;7.5;8.0;8.6;8.9;9.0+PTX}"

cd "$PNPP_DIR"
rm -rf build dist *.egg-info
"$PYTHON" setup.py install

echo "[i] Sanity check ..."
"$PYTHON" -c "import pointnet2._ext; print('pointnet2._ext OK')"
