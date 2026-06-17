#!/usr/bin/env bash
# Download Vicuna-7B + PointNet++ backbone; patch LEO config paths.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
LEO_REPO="${LEO_REPO:-$PROJECT_ROOT/../embodied-generalist}"
WEIGHTS_DIR="${WEIGHTS_DIR:-$LEO_REPO/weights}"
VICUNA_DIR="${VICUNA_DIR:-$WEIGHTS_DIR/vicuna-7b}"
CKPT_DIR="${CKPT_DIR:-$LEO_REPO/checkpoints}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

if [[ ! -d "$LEO_REPO" ]]; then
  echo "[x] LEO repo not found: $LEO_REPO"
  exit 1
fi

mkdir -p "$WEIGHTS_DIR" "$CKPT_DIR"
export HF_ENDPOINT HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-1800}"

if command -v hf >/dev/null 2>&1; then
  HF_DL=(hf download)
elif command -v huggingface-cli >/dev/null 2>&1; then
  HF_DL=(huggingface-cli download)
else
  echo "[x] Install huggingface_hub"
  exit 1
fi

echo "[i] LEO_REPO=$LEO_REPO"
echo "[i] VICUNA_DIR=$VICUNA_DIR"
echo "[i] CKPT_DIR=$CKPT_DIR"

# PointNet++ pretrained backbone (~5 MB)
PNPP="$CKPT_DIR/pointnetpp_vil3dref.pth"
if [[ ! -f "$PNPP" ]]; then
  echo "[i] Downloading pointnetpp_vil3dref.pth ..."
  "${HF_DL[@]}" huangjy-pku/LEO_data pointnetpp_vil3dref.pth \
    --repo-type dataset --local-dir "$CKPT_DIR"
else
  echo "[i] Already have $PNPP"
fi

# Vicuna-7B (~13 GB) — model repo
if [[ ! -f "$VICUNA_DIR/config.json" ]]; then
  echo "[i] Downloading Vicuna-7B (this may take a while) ..."
  mkdir -p "$VICUNA_DIR"
  "${HF_DL[@]}" huangjy-pku/vicuna-7b --local-dir "$VICUNA_DIR"
else
  echo "[i] Vicuna-7B already at $VICUNA_DIR"
fi

# Patch LEO yaml configs (replace TBD)
VICUNA_CFG="$LEO_REPO/configs/llm/vicuna7b.yaml"
PNPP_CFG="$LEO_REPO/configs/vision3d/backbone/pointnetpp.yaml"
DEFAULT_CFG="$LEO_REPO/configs/default.yaml"

python3 - <<PY
from pathlib import Path
import yaml

vicuna_dir = Path("$VICUNA_DIR")
pnpp = Path("$PNPP")
leo_repo = Path("$LEO_REPO")

vpath = leo_repo / "configs/llm/vicuna7b.yaml"
v = yaml.safe_load(vpath.read_text())
v["cfg_path"] = str(vicuna_dir)
vpath.write_text(yaml.dump(v, default_flow_style=False, sort_keys=False))

ppath = leo_repo / "configs/vision3d/backbone/pointnetpp.yaml"
p = yaml.safe_load(ppath.read_text())
p["path"] = str(pnpp)
ppath.write_text(yaml.dump(p, default_flow_style=False, sort_keys=False))

dpath = leo_repo / "configs/default.yaml"
d = yaml.safe_load(dpath.read_text())
if d.get("base_dir") in (None, "", "TBD"):
    d["base_dir"] = str(leo_repo / "output")
dpath.write_text(yaml.dump(d, default_flow_style=False, sort_keys=False))

print(f"[i] Patched {vpath}")
print(f"[i] Patched {ppath}")
print(f"[i] Patched {dpath}")
PY

echo "[i] Dependencies ready."
ls -lh "$PNPP" "$VICUNA_DIR/config.json" 2>/dev/null || true
