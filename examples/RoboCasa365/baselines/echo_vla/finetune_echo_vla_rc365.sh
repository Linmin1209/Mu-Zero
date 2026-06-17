#!/usr/bin/env bash
# Launch Echo VLA PI0.5 fine-tune with RC365-oriented Hydra overrides.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
ECHO_VLA_REPO="${ECHO_VLA_REPO:-/HOME/sysu_xdliang/sysu_xdliang_1/HDD_POOL/linmin/UR-manipulation-modelscope/Echo_VLA}"
DATASET_PATH="${DATASET_PATH:-/HOME/sysu_xdliang/sysu_xdliang_1/HDD_POOL/linmin/datasets/robocasa365-datasets}"
CONFIG="${CONFIG:-$SCRIPT_DIR/echo_vla_rc365_pi05.yaml}"
OUTPUT_TAG="${OUTPUT_TAG:-echo_vla_rc365_pi05}"
NUM_GPUS="${NUM_GPUS:-1}"

if [[ ! -d "$ECHO_VLA_REPO" ]]; then
  echo "[x] ECHO_VLA_REPO not found: $ECHO_VLA_REPO"
  exit 1
fi

export DATASET_PATH
export PYTHONPATH="$ECHO_VLA_REPO${PYTHONPATH:+:$PYTHONPATH}"

cd "$ECHO_VLA_REPO"
echo "[i] Echo VLA fine-tune"
echo "[i] DATASET_PATH=$DATASET_PATH"
echo "[i] tag=$OUTPUT_TAG"

# Use Echo native configs + CLI overrides (config yaml in Isaac-GR00T is reference only)
OVERRIDES=(
  "tag=$OUTPUT_TAG"
  "dataset_path=$DATASET_PATH"
  "skip_simulation_after_train=true"
  "wandb.project=robocasa365_echo_vla"
)

if [[ "$NUM_GPUS" -gt 1 ]]; then
  torchrun --nproc_per_node="$NUM_GPUS" fsdp_run.py \
    --config-name robocasa_config_pi05 \
    "${OVERRIDES[@]}"
else
  python run.py \
    --config-name robocasa_config_pi05 \
    "${OVERRIDES[@]}"
fi

echo "[i] Checkpoints under $ECHO_VLA_REPO/train_runs/ or Hydra output dir"
