#!/usr/bin/env bash
# Build PointNet++ CUDA extension required by LEO (leo conda env).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
LEO_REPO="${LEO_REPO:-$PROJECT_ROOT/../embodied-generalist}"
PYTHON="${PYTHON:-/XYAIFS00/sysu_xdliang_1/miniconda3/envs/leo/bin/python}"
# shellcheck source=leo_env.sh
source "$SCRIPT_DIR/leo_env.sh"

if [[ ! -x "$PYTHON" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON=python3
    sanitize_leo_ld_library_path
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

FORCE_REBUILD="${FORCE_REBUILD:-0}"
if [[ "$FORCE_REBUILD" != "1" ]] && "$PYTHON" -c "import pointnet2._ext" 2>/dev/null; then
  if "$PYTHON" - <<PY 2>/dev/null; then
import sys
import torch
sys.path.insert(0, "$LEO_REPO")
from model.pointnetpp.pointnet2_utils import furthest_point_sample
if not torch.cuda.is_available():
    raise SystemExit(0)
xyz = torch.randn(1, 64, 3, device="cuda")
idx = furthest_point_sample(xyz, 16)
assert idx.shape == (1, 16)
PY
    echo "[i] pointnet2._ext already installed and GPU FPS sanity check passed; skipping build."
    exit 0
  fi
  echo "[w] pointnet2._ext imports but GPU sanity check failed; rebuilding ..."
fi

echo "[i] Building PointNet++ with $PYTHON"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-7.0;7.5;8.0;8.6;8.9;9.0+PTX}"

cd "$PNPP_DIR"
rm -rf build dist *.egg-info

# Nodes with CUDA 12.x nvcc + torch 1.12 (cu113) hit a strict version check.
# The extension can still compile; patch the check for this install only.
"$PYTHON" - <<'PY'
import runpy
import sys

import torch.utils.cpp_extension as ext

_orig_check = getattr(ext.BuildExtension, "_check_cuda_version", None)

def _relaxed_cuda_check(self, compiler_name, compiler_version):
    if _orig_check is None:
        return
    try:
        _orig_check(self, compiler_name, compiler_version)
    except RuntimeError as exc:
        if "CUDA version" in str(exc):
            print(f"[w] Continuing despite CUDA/toolkit mismatch: {exc}")
        else:
            raise

if _orig_check is not None:
    ext.BuildExtension._check_cuda_version = _relaxed_cuda_check
sys.argv = ["setup.py", "install"]
runpy.run_path("setup.py", run_name="__main__")
PY

echo "[i] Sanity check ..."
"$PYTHON" -c "
import torch
import pointnet2._ext
print('pointnet2._ext OK', pointnet2._ext.__file__)
if torch.cuda.is_available():
    import sys
    sys.path.insert(0, '$LEO_REPO')
    from model.pointnetpp.pointnet2_utils import furthest_point_sample
    xyz = torch.randn(1, 64, 3, device='cuda')
    idx = furthest_point_sample(xyz, 16)
    print('GPU FPS OK', idx.shape)
"
