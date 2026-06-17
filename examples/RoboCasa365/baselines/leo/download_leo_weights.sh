#!/usr/bin/env bash
# Download LEO align.pth and sft_noact.pth to LEO_REPO/checkpoints/
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
LEO_REPO="${LEO_REPO:-$PROJECT_ROOT/../embodied-generalist}"
CKPT_DIR="${CKPT_DIR:-$LEO_REPO/checkpoints}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
REPO_ID="huangjy-pku/LEO_data"
EXPECTED_SIZE=198175787

if [[ ! -d "$LEO_REPO" ]]; then
  echo "[x] LEO repo not found: $LEO_REPO"
  echo "    Set LEO_REPO or run: bash $SCRIPT_DIR/setup_leo.sh"
  exit 1
fi

mkdir -p "$CKPT_DIR"
echo "[i] LEO_REPO=$LEO_REPO"
echo "[i] CKPT_DIR=$CKPT_DIR"
echo "[i] HF_ENDPOINT=$HF_ENDPOINT"

export HF_ENDPOINT HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-1800}"

download_one() {
  local f="$1"
  local dest="$CKPT_DIR/$f"

  if [[ -f "$dest" ]] && [[ "$(stat -c '%s' "$dest")" == "$EXPECTED_SIZE" ]]; then
    echo "[i] Already complete: $dest"
    return 0
  fi

  echo "[i] Downloading $f ..."
  if command -v hf >/dev/null 2>&1; then
    if hf download "$REPO_ID" "$f" --repo-type dataset --local-dir "$CKPT_DIR" 2>/dev/null; then
      :
    elif hf download "$REPO_ID" "$f" --repo-type dataset --local-dir "$CKPT_DIR" --force-download; then
      :
    else
      echo "[!] hf download failed, trying wget resume ..."
      local url
      url="$(python3 - <<PY
from huggingface_hub import hf_hub_url
print(hf_hub_url("$REPO_ID", "$f", repo_type="dataset", endpoint="$HF_ENDPOINT"))
PY
)"
      wget -c --tries=0 --read-timeout=120 --timeout=120 -O "$dest" "$url"
    fi
  elif command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli download "$REPO_ID" "$f" --repo-type dataset --local-dir "$CKPT_DIR" || {
      local url
      url="$(python3 - <<PY
from huggingface_hub import hf_hub_url
print(hf_hub_url("$REPO_ID", "$f", repo_type="dataset", endpoint="$HF_ENDPOINT"))
PY
)"
      wget -c --tries=0 --read-timeout=120 --timeout=120 -O "$dest" "$url"
    }
  else
    echo "[x] Install huggingface_hub or wget"
    exit 1
  fi

  if [[ ! -f "$dest" ]] || [[ "$(stat -c '%s' "$dest")" != "$EXPECTED_SIZE" ]]; then
    echo "[x] Incomplete download: $dest"
    exit 1
  fi
}

for f in align.pth sft_noact.pth; do
  download_one "$f"
done

echo "[i] Done:"
ls -lh "$CKPT_DIR"/align.pth "$CKPT_DIR"/sft_noact.pth
