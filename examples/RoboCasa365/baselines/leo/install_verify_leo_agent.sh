#!/usr/bin/env bash
# Full LeoAgent install + verification (run on GPU node with proxy).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
LEO_REPO="${LEO_REPO:-$PROJECT_ROOT/../embodied-generalist}"
PYTHON="${PYTHON:-/XYAIFS00/sysu_xdliang_1/miniconda3/envs/leo/bin/python}"
PIP="${PIP:-/XYAIFS00/sysu_xdliang_1/miniconda3/envs/leo/bin/pip}"
# shellcheck source=leo_env.sh
source "$SCRIPT_DIR/leo_env.sh"
WHEEL_DIR="${WHEEL_DIR:-$LEO_REPO/wheels}"

echo "=== LEO LeoAgent install/verify ==="
echo "[i] PYTHON=$PYTHON"
echo "[i] LEO_REPO=$LEO_REPO"

# 1) Ensure PyTorch 1.12.1 (LEO official version; NOT 2.x)
TORCH_VER=$("$PYTHON" -c "import torch; print(torch.__version__)" 2>/dev/null || echo "missing")
echo "[i] Current torch: $TORCH_VER"
if [[ "$TORCH_VER" != 1.12.1* ]]; then
  echo "[i] Installing PyTorch 1.12.1+cu113 ..."
  mkdir -p "$WHEEL_DIR"
  TORCH_WHL="$WHEEL_DIR/torch-1.12.1+cu113-cp39-cp39-linux_x86_64.whl"
  TV_WHL="$WHEEL_DIR/torchvision-0.13.1+cu113-cp39-cp39-linux_x86_64.whl"
  if [[ ! -f "$TORCH_WHL" ]]; then
    source /app/bin/proxy.sh 2>/dev/null || true
    wget -c --tries=0 --read-timeout=120 -O "$TORCH_WHL" \
      "https://download.pytorch.org/whl/cu113/torch-1.12.1%2Bcu113-cp39-cp39-linux_x86_64.whl"
  fi
  if [[ ! -f "$TV_WHL" ]]; then
    source /app/bin/proxy.sh 2>/dev/null || true
    wget -c --tries=0 --read-timeout=120 -O "$TV_WHL" \
      "https://download.pytorch.org/whl/cu113/torchvision-0.13.1%2Bcu113-cp39-cp39-linux_x86_64.whl"
  fi
  "$PIP" install "$TORCH_WHL" "$TV_WHL"
fi

# 2) Python deps
"$PIP" install -r "$LEO_REPO/requirements.txt" peft==0.5.0 --no-deps 2>/dev/null || \
  "$PIP" install -r "$LEO_REPO/requirements.txt"

# 2b) ConvNeXt/CLIP 2D backbone weights (local HF cache under HDD_POOL/linmin/models)
bash "$SCRIPT_DIR/download_leo_vision2d.sh"

# 3) PointNet++ CUDA ext
bash "$SCRIPT_DIR/install_leo_pointnetpp.sh"

# 4) PointNext prebuilt .so (optional)
PN_BATCH="$LEO_REPO/model/pointnext/cpp/pointnet2_batch"
if [[ ! -f "$PN_BATCH/pointnet2_batch_cuda.cpython-39-x86_64-linux-gnu.so" ]]; then
  source /app/bin/proxy.sh 2>/dev/null || true
  export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
  hf download huangjy-pku/LEO_data pointnet2_batch_cuda.cpython-39-x86_64-linux-gnu.so \
    --repo-type dataset --local-dir /tmp/leo_so_dl || true
  cp -f /tmp/leo_so_dl/pointnet2_batch_cuda.cpython-39-x86_64-linux-gnu.so "$PN_BATCH/" 2>/dev/null || true
fi

# 5) Verify imports + optional LeoAgent init (offline local weights; no proxy)
source "$SCRIPT_DIR/leo_env.sh"
export PYTHONPATH="$LEO_REPO${PYTHONPATH:+:$PYTHONPATH}"
export LEO_REPO
"$PYTHON" - <<PY
import os
import sys

leo_repo = os.environ["LEO_REPO"]
sys.path.insert(0, leo_repo)

import torch
print(f"[i] torch {torch.__version__} cuda={torch.cuda.is_available()}")

import pointnet2._ext
print("[i] pointnet2._ext OK")

from model.leo_agent import LeoAgent
print("[i] LeoAgent import OK")

from accelerate import Accelerator
Accelerator()  # required by accelerate.logging before LeoAgent __init__

from hydra import compose, initialize_config_dir

with initialize_config_dir(config_dir=f"{leo_repo}/configs", version_base=None):
    cfg = compose(config_name="default", overrides=["llm=vicuna7b", "vision3d=ose3d_pointnetpp"])
cfg.llm.cfg_path = f"{leo_repo}/weights/vicuna-7b"
cfg.vision3d.backbone.path = f"{leo_repo}/checkpoints/pointnetpp_vil3dref.pth"

print("[i] Building LeoAgent (loads Vicuna-7B, may take ~2 min)...")
agent = LeoAgent(cfg)
print(f"[i] LeoAgent built on {agent.device}")
print("[i] VERIFY OK")
PY

echo "[i] All checks passed."
