#!/usr/bin/env bash
set -euxo pipefail

# RoboCasa365 sim venv for GR00T eval (MuJoCo + robocasa365 gym envs).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_REPO="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
RLDX_REPO="${RLDX_REPO:-/HOME/sysu_xdliang/sysu_xdliang_1/HDD_POOL/linmin/RLDX-1}"
ROBOCASA365_REPO="${ROBOCASA365_REPO:-$PROJECT_REPO/external_dependencies/robocasa365}"
UV_ENV="$SCRIPT_DIR/robocasa365_uv"

if [[ ! -d "$ROBOCASA365_REPO" ]]; then
  SRC="$RLDX_REPO/external_dependencies/robocasa365"
  if [[ -d "$SRC" ]]; then
    mkdir -p "$(dirname "$ROBOCASA365_REPO")"
    ln -sfn "$SRC" "$ROBOCASA365_REPO"
    echo "Linked $ROBOCASA365_REPO -> $SRC"
  else
    echo "[x] robocasa365 not found. Clone/submodule init or set ROBOCASA365_REPO."
    exit 1
  fi
fi

rm -rf "$UV_ENV"
mkdir -p "$UV_ENV"
uv venv "$UV_ENV/.venv" --python 3.10
# shellcheck disable=SC1091
source "$UV_ENV/.venv/bin/activate"

RLDX_MUJOCO_GL="${RLDX_MUJOCO_GL:-egl}"
export MUJOCO_GL="$RLDX_MUJOCO_GL"
if [[ "$MUJOCO_GL" == "osmesa" ]]; then
  export PYOPENGL_PLATFORM=osmesa
else
  export PYOPENGL_PLATFORM=egl
fi

uv pip install setuptools wheel
uv pip install torch==2.5.1 torchvision==0.20.1
INSTALL_FLASH_ATTN=${INSTALL_FLASH_ATTN:-0}
if [[ "$(uname -s)" == "Linux" && "$INSTALL_FLASH_ATTN" == "1" ]]; then
  uv pip install --no-build-isolation flash-attn==2.7.4.post1 || true
fi

uv pip install "git+https://github.com/ARISE-Initiative/robosuite.git@master"
uv pip install -e "$ROBOCASA365_REPO" --config-settings editable_mode=compat

# Import gr00t rollout deps without resolving full training stack.
uv pip install --editable "$PROJECT_REPO" --no-deps
uv pip install \
  tyro==0.9.17 pandas==2.2.3 pydantic pyzmq==27.0.1 \
  msgpack==1.1.0 msgpack-numpy==0.4.8 einops==0.8.1 dm-tree==0.1.8 \
  tianshou==0.5.1 tqdm termcolor omegaconf==2.3.0 \
  'huggingface-hub>=0.34.0,<1.0'

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_ASSETS_REPO="${HF_ASSETS_REPO:-twilighted/Robocasa365-Assets}"
bash "$SCRIPT_DIR/download_kitchen_assets_hf_mirror.sh" --snapshot --types all

python - <<'PY'
import os
import shutil
import robosuite

base = robosuite.__path__[0]
priv = os.path.join(base, "macros_private.py")
if not os.path.isfile(priv):
    shutil.copyfile(os.path.join(base, "macros.py"), priv)
    print("Created", priv)
PY

python - <<'PY'
import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import gymnasium as gym
import robocasa  # noqa: F401
import robosuite

print("Imports OK:", robosuite.__version__)
env = gym.make("robocasa/NavigateKitchen", split="pretrain", seed=0)
env.reset()
env.close()
print("Env OK")
PY

echo "RoboCasa365 sim ready: $UV_ENV/.venv/bin/python"
