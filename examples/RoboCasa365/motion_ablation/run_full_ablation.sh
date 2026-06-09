#!/usr/bin/env bash
# End-to-end: train baseline + motion (RLDX), then sim-eval both.
#
# Prerequisites:
#   - RLDX-1-PT at BASE_MODEL_PATH (HF: RLWRLD/RLDX-1-PT)
#   - GPU for training + eval
#
set -euo pipefail

export BASE_MODEL_PATH="${BASE_MODEL_PATH:-RLWRLD/RLDX-1-PT}"
export DATASET_PATH="${DATASET_PATH:-/HOME/sysu_xdliang/sysu_xdliang_1/HDD_POOL/linmin/datasets/robocasa365-datasets/pretrain/atomic/PickPlaceToasterToCounter/20250819/lerobot}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RLDX_ABL="/HOME/sysu_xdliang/sysu_xdliang_1/HDD_POOL/linmin/RLDX-1/run_scripts/train/ablations/finetune_pickplace_motion_ablation.sh"

for variant in baseline motion; do
  echo "========== TRAIN $variant =========="
  bash "$RLDX_ABL" "$variant"
done

for variant in baseline motion; do
  echo "========== EVAL $variant =========="
  bash "$SCRIPT_DIR/run_rldx_ablation_eval.sh" "$variant"
done

echo "[i] Compare summaries:"
echo "  baseline: .../pickplace_baseline_30k_b*/eval/*/summary_shard0of1.csv"
echo "  motion:   .../pickplace_motion_30k_b*/eval/*/summary_shard0of1.csv"
