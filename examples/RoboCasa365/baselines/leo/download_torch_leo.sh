#!/usr/bin/env bash
# Resume-download PyTorch 1.12.1+cu113 wheels for LEO (curl -C -, multi-mirror).
set -euo pipefail

source /app/bin/proxy.sh 2>/dev/null || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
WHEEL_DIR="${WHEEL_DIR:-$PROJECT_ROOT/../embodied-generalist/wheels}"
mkdir -p "$WHEEL_DIR"

TORCH_WHL="$WHEEL_DIR/torch-1.12.1+cu113-cp39-cp39-linux_x86_64.whl"
TV_WHL="$WHEEL_DIR/torchvision-0.13.1+cu113-cp39-cp39-linux_x86_64.whl"
TORCH_SIZE=1837736693
TV_SIZE=23394075

download_file() {
  local dest="$1" size="$2"
  shift 2
  local urls=("$@")

  if [[ -f "$dest" ]] && [[ "$(stat -c '%s' "$dest")" -ge "$size" ]]; then
    echo "[i] Already complete: $dest"
    return 0
  fi

  for url in "${urls[@]}"; do
    local cur=0
    [[ -f "$dest" ]] && cur=$(stat -c '%s' "$dest")
    echo "[i] Resuming $dest from $url ($cur/$size)"

    if curl -fL --connect-timeout 60 --max-time 0 -C - \
      --retry 30 --retry-delay 10 --retry-all-errors \
      --speed-limit 1024 --speed-time 60 \
      -o "$dest" "$url" 2>/dev/null; then
      :
    else
      # HTTP 416 etc.: drop partial and retry once without resume
      echo "[w] Resume failed for $url — re-downloading from scratch"
      rm -f "$dest"
      if ! curl -fL --connect-timeout 60 --max-time 0 \
        --retry 30 --retry-delay 10 --retry-all-errors \
        --speed-limit 1024 --speed-time 60 \
        -o "$dest" "$url"; then
        echo "[w] curl failed for $url"
        continue
      fi
    fi

    if [[ -f "$dest" ]]; then
      local got
      got=$(stat -c '%s' "$dest")
      if [[ "$got" -ge "$((size - 4096))" ]]; then
        echo "[i] Done: $dest ($got bytes)"
        return 0
      fi
      echo "[w] Incomplete ($got/$size), trying next mirror..."
    else
      echo "[w] curl failed for $url"
    fi
  done
  echo "[x] All mirrors failed for $dest"
  return 1
}

download_file "$TORCH_WHL" "$TORCH_SIZE" \
  "https://download.pytorch.org/whl/cu113/torch-1.12.1%2Bcu113-cp39-cp39-linux_x86_64.whl" \
  "https://mirrors.aliyun.com/pytorch-wheels/cu113/torch-1.12.1%2Bcu113-cp39-cp39-linux_x86_64.whl"

download_file "$TV_WHL" "$TV_SIZE" \
  "https://mirrors.aliyun.com/pytorch-wheels/cu113/torchvision-0.13.1%2Bcu113-cp39-cp39-linux_x86_64.whl" \
  "https://download.pytorch.org/whl/cu113/torchvision-0.13.1%2Bcu113-cp39-cp39-linux_x86_64.whl"

ls -lh "$TORCH_WHL" "$TV_WHL"
