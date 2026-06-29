#!/usr/bin/env bash
# Upgrade leo conda env PyTorch for NVIDIA H100 (sm_90).
# torch 1.12+cu113 only ships sm_37..sm_86; H100 needs torch>=2.0 with CUDA 11.8+.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
LEO_REPO="${LEO_REPO:-$PROJECT_ROOT/../embodied-generalist}"
PYTHON="${PYTHON:-/XYAIFS00/sysu_xdliang_1/miniconda3/envs/leo/bin/python}"

# Default: torch 2.1.2 + cu121 (stable for LEO stack + H100 sm_90)
TORCH_VERSION="${TORCH_VERSION:-2.1.2}"
TV_VERSION="${TV_VERSION:-0.16.2}"
CUDA_WHEEL="${CUDA_WHEEL:-cu121}"
PYTORCH_INDEX="${PYTORCH_INDEX:-https://download.pytorch.org/whl/${CUDA_WHEEL}}"

if [[ ! -x "$PYTHON" ]]; then
  echo "[x] leo python not found: $PYTHON"
  echo "    conda create -n leo python=3.10 -y && conda activate leo"
  exit 1
fi

echo "[i] Upgrading leo env PyTorch for H100 (sm_90)"
echo "[i] PYTHON=$PYTHON"
echo "[i] target: torch==${TORCH_VERSION} torchvision==${TV_VERSION} (${CUDA_WHEEL})"
echo "[i] index:  $PYTORCH_INDEX"

"$PYTHON" -c "import torch; print('[i] before:', torch.__version__, torch.version.cuda)" || true

# Optional: use cluster proxy when available
if [[ -f /app/bin/proxy.sh ]]; then
  # shellcheck source=/dev/null
  source /app/bin/proxy.sh
fi

"$PYTHON" -m pip install -U pip wheel
"$PYTHON" -m pip install "setuptools==80.9.0"
"$PYTHON" -m pip uninstall -y torch torchvision torchaudio cudatoolkit 2>/dev/null || true
"$PYTHON" -m pip install "torch==${TORCH_VERSION}" "torchvision==${TV_VERSION}" --index-url "$PYTORCH_INDEX"

echo "[i] Verifying install ..."
"$PYTHON" - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda runtime", torch.version.cuda)
print("cuda available", torch.cuda.is_available())
arch = torch.cuda.get_arch_list() if hasattr(torch.cuda, "get_arch_list") else []
print("compiled arch list", arch)
if "sm_90" in arch or "9.0" in str(arch):
    print("[ok] sm_90 (H100) in arch list")
else:
  # cu121 wheels often report compute_90 in list differently
    if any("90" in a for a in arch):
        print("[ok] H100 arch present")
    else:
        print("[w] sm_90 not listed; if wheel is cu121 torch>=2.1, H100 should still work on node")
PY

echo "[i] Rebuilding PointNet++ for new torch/CUDA ..."
export FORCE_REBUILD=1
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-7.0;8.0;8.6;8.9;9.0}"
bash "$SCRIPT_DIR/install_leo_pointnetpp.sh"

echo "[i] Done. On H100 node, verify with:"
echo "  source $SCRIPT_DIR/leo_env.sh"
echo "  $PYTHON -c \"import torch; print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))\""
