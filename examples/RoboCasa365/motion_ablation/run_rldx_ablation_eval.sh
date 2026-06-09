#!/usr/bin/env bash
# Sim eval for RLDX motion ablation checkpoints (baseline vs +motion).
#
# Usage:
#   bash run_rldx_ablation_eval.sh baseline
#   bash run_rldx_ablation_eval.sh motion
#
set -euo pipefail

VARIANT="${1:-}"
if [[ "$VARIANT" != "baseline" && "$VARIANT" != "motion" ]]; then
  echo "[x] Usage: $0 {baseline|motion}"
  exit 1
fi

RLDX_ROOT="/HOME/sysu_xdliang/sysu_xdliang_1/HDD_POOL/linmin/RLDX-1"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}"
MAX_STEPS="${MAX_STEPS:-30000}"
CKPT_ROOT="$RLDX_ROOT/output/motion_ablation/pickplace_${VARIANT}_30k_b${GLOBAL_BATCH_SIZE}"

if [[ -d "$CKPT_ROOT/checkpoint-${MAX_STEPS}" ]]; then
  MODEL_PATH="$CKPT_ROOT/checkpoint-${MAX_STEPS}"
else
  MODEL_PATH="$(ls -d "$CKPT_ROOT"/checkpoint-* 2>/dev/null | sort -V | tail -1)"
fi

if [[ -z "${MODEL_PATH:-}" || ! -d "$MODEL_PATH" ]]; then
  echo "[x] Checkpoint not found under $CKPT_ROOT"
  echo "[x] Train first: bash $RLDX_ROOT/run_scripts/train/ablations/finetune_pickplace_motion_ablation.sh $VARIANT"
  exit 1
fi

echo "[i] Evaluating RLDX ablation variant=$VARIANT model=$MODEL_PATH"

bash "$RLDX_ROOT/run_scripts/eval/robocasa_365/eval_robocasa365.sh" \
  --model-path "$MODEL_PATH" \
  --task-set atomic_seen \
  --split pretrain \
  --tasks PickPlaceToasterToCounter \
  --n-episodes "${N_EPISODES:-50}" \
  --n-envs "${N_ENVS:-5}" \
  --n-action-steps "${N_ACTION_STEPS:-40}" \
  --output-root "$CKPT_ROOT/eval"
