#!/usr/bin/env bash
# One-time: convert upstream LFS pointers to regular git blobs so Mu-Zero push
# does not require GitHub LFS (avoids GH008 / lfs.locksverify errors).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "[x] Working tree not clean. Commit or stash changes first."
  exit 1
fi

if ! command -v git-lfs >/dev/null 2>&1; then
  echo "[x] git-lfs not found. Install git-lfs first."
  exit 1
fi

echo "[i] Rewriting history: export LFS -> regular git blobs (~586MB local cache)."
echo "[i] Paths: demo_data, media/*.gif, deployment wheels."
echo "[i] Fetching any missing LFS blobs from origin first ..."
git lfs fetch origin --all

echo "[i] This creates a new history; you must force-push afterward."

git lfs migrate export \
  --include="media/*.gif,demo_data/**/*.mp4,demo_data/**/*.parquet,scripts/deployment/dgpu/wheels/*.whl,scripts/deployment/orin/wheels/*.whl,scripts/deployment/spark/wheels/*.whl" \
  --everything

echo "[ok] LFS export done. Next:"
echo "  bash scripts/push_to_mu_zero.sh"
