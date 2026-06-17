#!/usr/bin/env bash
# Push Isaac-GR00T (Mu-Zero fork) to https://github.com/Linmin1209/Mu-Zero.git
#
# Routine code pushes: git only (no LFS re-upload).
# First-time / GH008: set MU_ZERO_PUSH_LFS=1 once, or script auto-retries LFS on GH008.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

REMOTE="${MU_ZERO_REMOTE:-mu-zero}"
BRANCH="${MU_ZERO_BRANCH:-main}"
FORCE="${MU_ZERO_FORCE:-1}"
MAX_RETRIES="${MU_ZERO_MAX_RETRIES:-3}"
PUSH_LFS="${MU_ZERO_PUSH_LFS:-0}"
GITHUB_TOKEN="${MU_ZERO_GITHUB_TOKEN:-${GITHUB_TOKEN:-}}"

if [[ -n "$GITHUB_TOKEN" ]]; then
  MU_ZERO_USE_HTTPS=1
fi

if [[ "${MU_ZERO_USE_HTTPS:-0}" == "1" ]]; then
  if [[ -n "$GITHUB_TOKEN" ]]; then
    REMOTE_URL="${MU_ZERO_REMOTE_URL:-https://x-access-token:${GITHUB_TOKEN}@github.com/Linmin1209/Mu-Zero.git}"
  else
    REMOTE_URL="${MU_ZERO_REMOTE_URL:-https://github.com/Linmin1209/Mu-Zero.git}"
  fi
else
  REMOTE_URL="${MU_ZERO_REMOTE_URL:-git@github.com:Linmin1209/Mu-Zero.git}"
fi

if ! git remote | grep -qx "$REMOTE"; then
  git remote add "$REMOTE" "$REMOTE_URL"
else
  git remote set-url "$REMOTE" "$REMOTE_URL"
fi

REMOTE_URL="$(git remote get-url "$REMOTE")"
DISPLAY_URL="$REMOTE_URL"
if [[ "$DISPLAY_URL" == *"x-access-token:"*"@"* ]]; then
  DISPLAY_URL="https://x-access-token:***@github.com/Linmin1209/Mu-Zero.git"
elif [[ "$DISPLAY_URL" == https://*@* ]]; then
  DISPLAY_URL="https://***@github.com/Linmin1209/Mu-Zero.git"
fi
echo "[i] Remote: $REMOTE -> $DISPLAY_URL"
echo "[i] Branch: $BRANCH"
git log -1 --oneline
echo "[i] LFS upload: $([[ "$PUSH_LFS" == "1" ]] && echo forced || echo skip-by-default, auto-on-GH008)"

if [[ "$REMOTE_URL" == git@github.com:* ]]; then
  echo "[i] Testing GitHub SSH (timeout 15s) ..."
  set +e
  ssh -o BatchMode=yes -o ConnectTimeout=15 -T git@github.com 2>&1 | sed 's/^/[ssh] /'
  ssh_status=${PIPESTATUS[0]}
  set -e
  if [[ "$ssh_status" -eq 255 ]]; then
    echo "[x] SSH to github.com failed. Run: ssh -T git@github.com"
    exit "$ssh_status"
  fi
fi

PUSH_ARGS=(--progress --verbose)
if [[ "$FORCE" == "1" ]]; then
  PUSH_ARGS+=(--force)
fi

GIT_HTTP_CFG=(
  -c http.postBuffer=524288000
  -c http.lowSpeedLimit=0
  -c http.lowSpeedTime=999999
  -c http.version=HTTP/1.1
  -c lfs.locksverify=false
)
GIT_LFS_CFG=(-c lfs.locksverify=false)

push_git() {
  local log
  log="$(mktemp)"
  set +e
  if [[ "$REMOTE_URL" == https://* ]]; then
    git "${GIT_HTTP_CFG[@]}" "${GIT_LFS_CFG[@]}" push "$REMOTE" "$BRANCH" "${PUSH_ARGS[@]}" 2>&1 | tee "$log"
  else
    git "${GIT_LFS_CFG[@]}" push "$REMOTE" "$BRANCH" "${PUSH_ARGS[@]}" 2>&1 | tee "$log"
  fi
  local status=${PIPESTATUS[0]}
  set -e
  if [[ "$status" -ne 0 ]] && grep -qE 'GH008|unknown Git LFS objects' "$log"; then
    rm -f "$log"
    return 42
  fi
  rm -f "$log"
  return "$status"
}

push_lfs() {
  echo "[i] Uploading missing Git LFS objects to $REMOTE ..."
  git "${GIT_LFS_CFG[@]}" lfs fetch origin --all
  local lfs_try=1
  local lfs_max="${MU_ZERO_LFS_RETRIES:-5}"
  while [[ "$lfs_try" -le "$lfs_max" ]]; do
    echo "[i] LFS upload attempt $lfs_try/$lfs_max ..."
    set +e
    git "${GIT_LFS_CFG[@]}" lfs push "$REMOTE" --all "$BRANCH"
    local status=$?
    set -e
    [[ "$status" -eq 0 ]] && return 0
    if [[ "$lfs_try" -ge "$lfs_max" ]]; then
      echo "[x] git lfs push failed. One-time fix for last wheel (~387MB):"
      echo "  git -c lfs.locksverify=false lfs push --object-id $REMOTE \\"
      echo "    7127cf58a7642d7350527a57daf14b8d1a8301ccd2805eaea1897f5e85535f30"
      echo "  Or strip LFS once: bash scripts/prepare_mu_zero_no_lfs.sh"
      return "$status"
    fi
    sleep $(( lfs_try * 20 ))
    lfs_try=$(( lfs_try + 1 ))
  done
}

export GIT_PROGRESS_DELAY=0
export GIT_LFS_SKIP_PUSH="${GIT_LFS_SKIP_PUSH:-1}"
START_TS=$(date +%s)

if [[ "$PUSH_LFS" == "1" ]]; then
  push_lfs
fi

attempt=1
while true; do
  echo "[i] git push attempt $attempt/$MAX_RETRIES ..."
  set +e
  push_git
  push_status=$?
  set -e

  if [[ "$push_status" -eq 0 ]]; then
    break
  fi

  if [[ "$push_status" -eq 42 && "$PUSH_LFS" != "1" ]]; then
    echo "[i] GH008: remote missing LFS blobs; uploading once then retrying git push ..."
    set +e
    push_lfs
    lfs_status=$?
    set -e
    if [[ "$lfs_status" -eq 0 ]]; then
      set +e
      push_git
      push_status=$?
      set -e
      [[ "$push_status" -eq 0 ]] && break
    else
      push_status=$lfs_status
    fi
  fi

  if [[ "$attempt" -ge "$MAX_RETRIES" ]]; then
    echo "[x] Push failed (exit $push_status)."
    exit "$push_status"
  fi
  sleep $(( attempt * 15 ))
  attempt=$(( attempt + 1 ))
done

ELAPSED=$(( $(date +%s) - START_TS ))
echo "[ok] Push finished in ${ELAPSED}s -> https://github.com/Linmin1209/Mu-Zero"
