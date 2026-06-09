#!/usr/bin/env bash
# Push Isaac-GR00T (Mu-Zero fork) to https://github.com/Linmin1209/Mu-Zero.git
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

REMOTE="${MU_ZERO_REMOTE:-mu-zero}"
BRANCH="${MU_ZERO_BRANCH:-main}"
FORCE="${MU_ZERO_FORCE:-1}"
MAX_RETRIES="${MU_ZERO_MAX_RETRIES:-3}"

# Prefer SSH for large pushes (HTTPS often hits GnuTLS timeout on slow links).
if [[ "${MU_ZERO_USE_HTTPS:-0}" == "1" ]]; then
  REMOTE_URL="${MU_ZERO_REMOTE_URL:-https://github.com/Linmin1209/Mu-Zero.git}"
else
  REMOTE_URL="${MU_ZERO_REMOTE_URL:-git@github.com:Linmin1209/Mu-Zero.git}"
fi

if ! git remote | grep -qx "$REMOTE"; then
  git remote add "$REMOTE" "$REMOTE_URL"
else
  CURRENT_URL="$(git remote get-url "$REMOTE")"
  if [[ "$CURRENT_URL" == https://github.com/* && "${MU_ZERO_USE_HTTPS:-0}" != "1" ]]; then
    echo "[i] Switching remote from HTTPS to SSH (set MU_ZERO_USE_HTTPS=1 to keep HTTPS)."
    git remote set-url "$REMOTE" "$REMOTE_URL"
  elif [[ -n "${MU_ZERO_REMOTE_URL:-}" && "$CURRENT_URL" != "$REMOTE_URL" ]]; then
    git remote set-url "$REMOTE" "$REMOTE_URL"
  elif [[ "$CURRENT_URL" != "$REMOTE_URL" && "${MU_ZERO_USE_HTTPS:-0}" != "1" ]]; then
    git remote set-url "$REMOTE" "$REMOTE_URL"
  fi
fi

echo "[i] Remote: $REMOTE -> $(git remote get-url "$REMOTE")"
echo "[i] Branch: $BRANCH"
git log -1 --oneline

COMMIT_COUNT="$(git rev-list --count "$BRANCH")"
read -r PACK_COUNT PACK_SIZE _ <<<"$(git count-objects -vH | awk '
  /in-pack/ {c=$2}
  /size-pack/ {s=$2}
  END {print c, s, ""}
')"
LFS_COUNT="$(git lfs ls-files 2>/dev/null | wc -l | tr -d ' ')"
echo "[i] Commits on $BRANCH: $COMMIT_COUNT"
echo "[i] Pack objects: ${PACK_COUNT:-0}, pack size: ${PACK_SIZE:-unknown}"
echo "[i] LFS-tracked files in HEAD: ${LFS_COUNT:-0}"
echo "[i] GitHub rejects GH008 if commits reference LFS but blobs are missing."
echo "[i] Tip: first push may take several minutes; progress prints below."

REMOTE_URL="$(git remote get-url "$REMOTE")"
if [[ "$REMOTE_URL" == git@github.com:* ]]; then
  echo "[i] Testing GitHub SSH (timeout 15s) ..."
  set +e
  ssh -o BatchMode=yes -o ConnectTimeout=15 -T git@github.com 2>&1 | sed 's/^/[ssh] /'
  ssh_status=${PIPESTATUS[0]}
  set -e
  if [[ "$ssh_status" -eq 255 ]]; then
    echo "[x] SSH to github.com failed."
    echo "    Fix: ssh -T git@github.com"
    echo "    Or HTTPS token: MU_ZERO_USE_HTTPS=1 MU_ZERO_REMOTE_URL=https://<TOKEN>@github.com/Linmin1209/Mu-Zero.git $0"
    exit "$ssh_status"
  fi
fi

PUSH_ARGS=(--progress --verbose)
if [[ "$FORCE" == "1" ]]; then
  echo "[i] Force-pushing (Mu-Zero placeholder README will be replaced) ..."
  PUSH_ARGS+=(--force)
else
  echo "[i] Pushing ..."
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
  if [[ "$REMOTE_URL" == https://* ]]; then
    git "${GIT_HTTP_CFG[@]}" "${GIT_LFS_CFG[@]}" push "$REMOTE" "$BRANCH" "${PUSH_ARGS[@]}"
  else
    git "${GIT_LFS_CFG[@]}" push "$REMOTE" "$BRANCH" "${PUSH_ARGS[@]}"
  fi
}

push_lfs() {
  echo "[i] Ensuring local LFS cache is complete (fetch from origin) ..."
  git "${GIT_LFS_CFG[@]}" lfs fetch origin --all
  echo "[i] Uploading Git LFS objects to $REMOTE ..."
  local lfs_try=1
  local lfs_max="${MU_ZERO_LFS_RETRIES:-5}"
  while [[ "$lfs_try" -le "$lfs_max" ]]; do
    echo "[i] LFS upload attempt $lfs_try/$lfs_max ..."
    set +e
    git "${GIT_LFS_CFG[@]}" lfs push "$REMOTE" --all "$BRANCH"
    local status=$?
    set -e
    if [[ "$status" -eq 0 ]]; then
      return 0
    fi
    if [[ "$lfs_try" -ge "$lfs_max" ]]; then
      echo "[x] git lfs push failed after $lfs_max attempts."
      echo "    Last object is often scripts/deployment/dgpu/wheels/flash_attn-*.whl (~387MB)."
      echo "    Retry only that blob:"
      echo "      git -c lfs.locksverify=false lfs push --object-id $REMOTE 7127cf58a7642d7350527a57daf14b8d1a8301ccd2805eaea1897f5e85535f30"
      echo "    Or strip LFS once: bash scripts/prepare_mu_zero_no_lfs.sh"
      return "$status"
    fi
    local wait=$(( lfs_try * 20 ))
    echo "[!] LFS push failed; retrying in ${wait}s (already-uploaded objects are skipped) ..."
    sleep "$wait"
    lfs_try=$(( lfs_try + 1 ))
  done
}

export GIT_PROGRESS_DELAY=0
unset GIT_LFS_SKIP_PUSH

START_TS=$(date +%s)
attempt=1
while true; do
  echo "[i] Push attempt $attempt/$MAX_RETRIES ..."
  set +e
  if [[ "${LFS_COUNT:-0}" -gt 0 ]]; then
    push_lfs
    lfs_status=$?
    if [[ "$lfs_status" -ne 0 ]]; then
      echo "[x] git lfs push failed (exit $lfs_status)."
      echo "    Option A: retry on SSH (recommended)."
      echo "    Option B: remove LFS dependency once:"
      echo "      bash scripts/prepare_mu_zero_no_lfs.sh && bash $0"
      push_status=$lfs_status
    else
      push_git
      push_status=$?
    fi
  else
    push_git
    push_status=$?
  fi
  set -e
  if [[ "$push_status" -eq 0 ]]; then
    break
  fi
  if [[ "$attempt" -ge "$MAX_RETRIES" ]]; then
    echo "[x] Push failed after $MAX_RETRIES attempts (exit $push_status)."
    exit "$push_status"
  fi
  sleep_secs=$(( attempt * 15 ))
  echo "[!] Push failed; retrying in ${sleep_secs}s ..."
  sleep "$sleep_secs"
  attempt=$(( attempt + 1 ))
done

ELAPSED=$(( $(date +%s) - START_TS ))
echo "[ok] Push finished in ${ELAPSED}s -> https://github.com/Linmin1209/Mu-Zero"
