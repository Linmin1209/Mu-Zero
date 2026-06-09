#!/usr/bin/env bash
# Push Isaac-GR00T (Mu-Zero fork) to https://github.com/Linmin1209/Mu-Zero.git
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

REMOTE="${MU_ZERO_REMOTE:-mu-zero}"
REMOTE_URL="${MU_ZERO_REMOTE_URL:-git@github.com:Linmin1209/Mu-Zero.git}"
BRANCH="${MU_ZERO_BRANCH:-main}"
FORCE="${MU_ZERO_FORCE:-1}"

if ! git remote | grep -qx "$REMOTE"; then
  git remote add "$REMOTE" "$REMOTE_URL"
else
  git remote set-url "$REMOTE" "$REMOTE_URL"
fi

echo "[i] Remote: $REMOTE -> $REMOTE_URL"
echo "[i] Branch: $BRANCH"
git log -1 --oneline

if [[ "$FORCE" == "1" ]]; then
  echo "[i] Force-pushing (Mu-Zero placeholder README will be replaced)."
  git push "$REMOTE" "$BRANCH" --force
else
  git push "$REMOTE" "$BRANCH"
fi

echo "[ok] https://github.com/Linmin1209/Mu-Zero"
