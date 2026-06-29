#!/usr/bin/env bash
# Shared env fixes for the leo conda env (torch 2.x + cu121 on H100, or legacy 1.12 on V100).
# Source this from any LEO install/verify script.

# System PyTorch 2.x on LD_LIBRARY_PATH (e.g. /usr/local/lib/python3.10/dist-packages/torch/lib)
# causes torch 1.12 in the leo env to segfault on import. We must:
#   1) drop foreign torch / torch_tensorrt lib dirs
#   2) prepend this env's torch/lib so extensions (pointnet2._ext) find libc10.so
sanitize_leo_ld_library_path() {
  local python_bin="${PYTHON:-/XYAIFS00/sysu_xdliang_1/miniconda3/envs/leo/bin/python}"
  local leo_torch_lib=""
  if [[ -x "$python_bin" ]]; then
    leo_torch_lib=$("$python_bin" -c "import site, os; print(os.path.join(site.getsitepackages()[0], 'torch', 'lib'))" 2>/dev/null || true)
  fi
  if [[ -z "$leo_torch_lib" || ! -d "$leo_torch_lib" ]]; then
    leo_torch_lib="/XYAIFS00/sysu_xdliang_1/miniconda3/envs/leo/lib/python3.9/site-packages/torch/lib"
  fi

  local cleaned="" part
  if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
    IFS=':' read -ra _ld_parts <<< "$LD_LIBRARY_PATH"
    for part in "${_ld_parts[@]}"; do
      [[ -z "$part" ]] && continue
      [[ "$part" == "$leo_torch_lib" ]] && continue
      if [[ "$part" == *"/torch/lib"* ]] || [[ "$part" == *"/torch_tensorrt/"* ]]; then
        continue
      fi
      cleaned="${cleaned:+$cleaned:}$part"
    done
  fi
  export LD_LIBRARY_PATH="${leo_torch_lib}${cleaned:+:$cleaned}"
}

# Scripts should set PYTHON before sourcing; apply LD fix once PYTHON is known.
PYTHON="${PYTHON:-/XYAIFS00/sysu_xdliang_1/miniconda3/envs/leo/bin/python}"
sanitize_leo_ld_library_path

# Local model / HF cache (convnext_laion2b, timm pretrained, etc.)
LEO_MODELS_ROOT="${LEO_MODELS_ROOT:-/HOME/sysu_xdliang/sysu_xdliang_1/HDD_POOL/linmin/models}"
export HF_HOME="${HF_HOME:-$LEO_MODELS_ROOT/.hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TORCH_HOME="${TORCH_HOME:-$LEO_MODELS_ROOT/.torch_cache}"
export TIMM_HOME="${TIMM_HOME:-$LEO_MODELS_ROOT/.timm_cache}"

# Prefer local CLIP/ConvNeXt weights; avoid broken cluster proxy hitting huggingface.co.
leo_configure_hf_offline() {
  local clip_bin="$LEO_MODELS_ROOT/CLIP-convnext_base_laion2b/open_clip_pytorch_model.bin"
  local clip_hub="$HF_HUB_CACHE/models--laion--CLIP-convnext_base_w-laion2B-s13B-b82K"
  if [[ -f "$clip_bin" ]] || [[ -d "$clip_hub/snapshots" ]]; then
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy NO_PROXY no_proxy
    echo "[i] leo_env: using local HF cache (offline), HF_HOME=$HF_HOME" >&2
  else
    unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
    export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
    echo "[i] leo_env: HF_ENDPOINT=$HF_ENDPOINT (CLIP not cached locally yet)" >&2
  fi
}

leo_configure_hf_offline

# OpenCV video decode lives in robocasa env (leo env lacks libGL / cv2).
export ROBOCASA_PYTHON="${ROBOCASA_PYTHON:-/XYAIFS00/sysu_xdliang_1/miniconda3/envs/robocasa/bin/python}"
