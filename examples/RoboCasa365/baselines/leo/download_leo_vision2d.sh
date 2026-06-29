#!/usr/bin/env bash
# Download LEO 2D backbone weights (timm convnext_base_laion2b -> open_clip_pytorch_model.bin).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_ROOT="${LEO_MODELS_ROOT:-/HOME/sysu_xdliang/sysu_xdliang_1/HDD_POOL/linmin/models}"
CLIP_REPO="laion/CLIP-convnext_base_w-laion2B-s13B-b82K"
CLIP_FILE="open_clip_pytorch_model.bin"
CLIP_DIR="$MODELS_ROOT/CLIP-convnext_base_laion2b"
HF_HOME="${HF_HOME:-$MODELS_ROOT/.hf_cache}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
EXPECTED_SIZE=717619969

source /app/bin/proxy.sh 2>/dev/null || true
mkdir -p "$CLIP_DIR" "$HF_HOME/hub"

if command -v hf >/dev/null 2>&1; then
  HF_DL=(hf download)
elif command -v huggingface-cli >/dev/null 2>&1; then
  HF_DL=(huggingface-cli download)
else
  echo "[x] Install huggingface_hub (hf CLI)"
  exit 1
fi

echo "[i] MODELS_ROOT=$MODELS_ROOT"
echo "[i] HF_HOME=$HF_HOME"

# Reuse an existing complete download from the default HF cache.
DEFAULT_HUB="${HOME}/.cache/huggingface/hub"
SRC_HUB="$DEFAULT_HUB/models--laion--CLIP-convnext_base_w-laion2B-s13B-b82K"
DST_HUB="$HF_HOME/hub/models--laion--CLIP-convnext_base_w-laion2B-s13B-b82K"

if [[ ! -d "$DST_HUB" && -d "$SRC_HUB" ]]; then
  blob=$(find "$SRC_HUB/blobs" -type f 2>/dev/null | head -1 || true)
  if [[ -n "$blob" ]] && [[ "$(stat -c '%s' "$blob")" -ge "$EXPECTED_SIZE" ]]; then
    echo "[i] Migrating complete CLIP hub cache -> $DST_HUB"
    cp -a "$SRC_HUB" "$DST_HUB"
  fi
fi

if [[ ! -f "$CLIP_DIR/$CLIP_FILE" ]]; then
  blob=""
  if [[ -d "$DST_HUB/blobs" ]]; then
    for f in "$DST_HUB/blobs"/*; do
      [[ -f "$f" ]] || continue
      if [[ "$(stat -c '%s' "$f")" -ge "$EXPECTED_SIZE" ]]; then
        blob="$f"
        break
      fi
    done
  fi
  if [[ -n "$blob" ]]; then
    echo "[i] Copying $CLIP_FILE from local HF hub cache"
    cp -f "$blob" "$CLIP_DIR/$CLIP_FILE"
  else
    echo "[i] Downloading $CLIP_REPO/$CLIP_FILE (~685 MiB, resumable) ..."
    export HF_HOME HF_HUB_CACHE="$HF_HOME/hub" HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-3600}"
    "${HF_DL[@]}" "$CLIP_REPO" "$CLIP_FILE" --local-dir "$CLIP_DIR" --max-workers 2
  fi
fi

size=$(stat -c '%s' "$CLIP_DIR/$CLIP_FILE" 2>/dev/null || echo 0)
if [[ "$size" -lt "$EXPECTED_SIZE" ]]; then
  echo "[x] Incomplete $CLIP_FILE ($size bytes, expected >= $EXPECTED_SIZE)"
  exit 1
fi

echo "[i] CLIP convnext ready:"
ls -lh "$CLIP_DIR/$CLIP_FILE"
echo "[i] HF hub cache: $DST_HUB"
echo "[i] Use HF_HOME=$HF_HOME (set automatically by leo_env.sh)"
