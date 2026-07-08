#!/usr/bin/env bash
# Verify ModelScope dataset download (avoid 0-byte files).
# Usage on the download machine:
#   bash verify_modelscope_download.sh
set -euo pipefail

TOKEN="${MODELSCOPE_TOKEN:-ms-f6c0ff61-7b21-4122-bf35-8e64619e36ad}"
REPO="Twilighted/Robocasa365-tactile"
ENDPOINT="${MODELSCOPE_DOMAIN:-https://www.modelscope.cn}"
FILE="pretrain/composite/AddSugarCubes/20250804/lerobot/meta/info.json"
DL_DIR="${1:-./ms_download_test}"

echo "[i] endpoint=$ENDPOINT"
echo "[i] clearing cache for $REPO"
rm -rf "$HOME/.cache/modelscope/hub/Twilighted___Robocasa365-tactile" 2>/dev/null || true
rm -rf "$DL_DIR"
mkdir -p "$DL_DIR"

echo "[i] downloading $FILE"
modelscope download --dataset "$REPO" "$FILE" \
  --token "$TOKEN" \
  --endpoint "$ENDPOINT" \
  --local_dir "$DL_DIR"

TARGET="$DL_DIR/$FILE"
if [[ ! -f "$TARGET" ]]; then
  echo "[fail] file not found at expected path: $TARGET"
  find "$DL_DIR" -name info.json -exec stat -c '%s %n' {} \;
  exit 1
fi

SIZE=$(stat -c%s "$TARGET")
echo "[i] downloaded size: $SIZE bytes"
if [[ "$SIZE" -lt 1000 ]]; then
  echo "[fail] file too small (expected ~5181 bytes). Check:"
  echo "  1) export MODELSCOPE_DOMAIN=www.modelscope.cn  (NOT modelscope.ai)"
  echo "  2) network access to www.modelscope.cn"
  echo "  3) upgrade modelscope: pip install -U modelscope"
  exit 1
fi
echo "[ok] download verified"
