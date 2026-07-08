#!/usr/bin/env bash
# MoT joint finetune: shared DiT inpaint + GR00T VISOR (arm-only, base decoupled)
# Design: examples/RoboCasa365/JOINT_DUAL_BRANCH_DESIGN.md
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
# shellcheck source=examples/RoboCasa365/env_defaults.sh
source "$SCRIPT_DIR/env_defaults.sh"

ROBOCASA365_SPLIT="${ROBOCASA365_SPLIT:-pretrain}"
ROBOCASA365_CATEGORY="${ROBOCASA365_CATEGORY:-atomic}"
ROBOCASA365_TASKS="${ROBOCASA365_TASKS:-PickPlaceToasterOvenToCounter}"
MODALITY_CONFIG="${MODALITY_CONFIG:-$SCRIPT_DIR/robocasa365_config_4frame.py}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/output/rc365_PickPlace_joint_dual_branch_30k}"
LOG_FILE="${LOG_FILE:-$OUTPUT_DIR/train.log}"

NUM_GPUS="${NUM_GPUS:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
MAX_STEPS=30000
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-8}"
SAVE_STEPS="${SAVE_STEPS:-5000}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-10}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-0}"
SHARD_SIZE="${SHARD_SIZE:-256}"

USE_MOTION="${USE_MOTION:-1}"
MOTION_INSERT_LAYER="${MOTION_INSERT_LAYER:-9}"
TUNE_MOTION="${TUNE_MOTION:-1}"
MOTION_USE_GATING="${MOTION_USE_GATING:-1}"
MOTION_GATE_INIT_BIAS="${MOTION_GATE_INIT_BIAS:-1.5}"

# --- MoT joint (shared DiT) ---
DECOUPLE_BASE_ARM="${DECOUPLE_BASE_ARM:-1}"
MOT_INPAINT_TOKENS="${MOT_INPAINT_TOKENS:-4}"
JOINT_ALPHA_P1=1.0
JOINT_BETA_P1=1.0
JOINT_ALPHA_P2=0.2
JOINT_BETA_P2=2.0
JOINT_PHASE1_RATIO=0.2
JOINT_TRAIN_MODE="${JOINT_TRAIN_MODE:-simultaneous}"
JOINT_FLUX_MODEL="${JOINT_FLUX_MODEL:-/HOME/sysu_xdliang/sysu_xdliang_1/HDD_POOL/linmin/models/FLUX.1-Fill-dev}"

cd "$PROJECT_ROOT"
mkdir -p "$OUTPUT_DIR"

LOCK_FILE="$OUTPUT_DIR/.train.lock"
exec 200>"$LOCK_FILE"
if ! flock -n 200; then
  echo "[ERROR] Another joint training instance holds $LOCK_FILE — stop it first."
  exit 1
fi

exec > >(tee -a "$LOG_FILE") 2>&1

echo "[i] MoT joint: shared DiT inpaint + VISOR (mode=${JOINT_TRAIN_MODE}, decouple_base=${DECOUPLE_BASE_ARM})"
echo "[i] alpha/beta phase1: ${JOINT_ALPHA_P1}/${JOINT_BETA_P1} | phase2: ${JOINT_ALPHA_P2}/${JOINT_BETA_P2}"
echo "[i] phase1_ratio=${JOINT_PHASE1_RATIO} max_steps=${MAX_STEPS} inpaint_tokens=${MOT_INPAINT_TOKENS}"
echo "[i] global_batch=${GLOBAL_BATCH_SIZE} motion=${USE_MOTION} gate_bias=${MOTION_GATE_INIT_BIAS} workers=${DATALOADER_NUM_WORKERS} shard_size=${SHARD_SIZE}"
echo "[i] debug GR00T-only: JOINT_TRAIN_MODE=gr00t_only bash $0"
echo "[i] tactile weight reduced (0.01); visual fusion emphasized (phase1 visual scale=1.0)"
echo "[i] output: $OUTPUT_DIR"

EXTRA=(--robocasa365-tasks "$ROBOCASA365_TASKS")
VISOR_ARGS=(
  --use-visor
  --no-use-component-factored-head
  --visor-gate-mode dual_split
  --visor-use-visual-supervision
  --visor-use-readout-fed-gates
  --visor-visual-gt-level flow
  --visor-visual-dim 2
  --visor-tactile-align-mode hold_last
  --visor-loss-weight-tactile 0.01
  --visor-loss-weight-visual 0.15
  --visor-aux-delay-steps 6000
)
MOTION_ARGS=()
if [[ "$USE_MOTION" == "1" ]]; then
  MOTION_ARGS+=(--use-motion --motion-insert-layer "$MOTION_INSERT_LAYER")
  if [[ "$TUNE_MOTION" == "1" ]]; then
    MOTION_ARGS+=(--tune-motion)
  else
    MOTION_ARGS+=(--no-tune-motion)
  fi
  append_motion_gate_cli_args MOTION_ARGS
fi
JOINT_ARGS=(
  --joint-train-mode "$JOINT_TRAIN_MODE"
  --joint-flux-model-path "$JOINT_FLUX_MODEL"
  --joint-alpha-phase1 "$JOINT_ALPHA_P1"
  --joint-beta-phase1 "$JOINT_BETA_P1"
  --joint-alpha-phase2 "$JOINT_ALPHA_P2"
  --joint-beta-phase2 "$JOINT_BETA_P2"
  --joint-phase1-ratio "$JOINT_PHASE1_RATIO"
  --joint-visor-visual-weight-phase1 1.0
  --joint-visor-visual-weight-phase2 0.15
  --joint-visor-tactile-weight-phase1 0.01
  --joint-visor-tactile-weight-phase2 0.02
  --mot-inpaint-tokens "$MOT_INPAINT_TOKENS"
)
if [[ "$DECOUPLE_BASE_ARM" == "1" ]]; then
  JOINT_ARGS+=(--decouple-base-arm)
else
  JOINT_ARGS+=(--no-decouple-base-arm)
fi

CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" .venv/bin/python -u gr00t/experiment/launch_joint_finetune.py \
  --base-model-path "$GR00T_BASE_MODEL" \
  --robocasa365-root "$ROBOCASA365_ROOT" \
  --robocasa365-split "$ROBOCASA365_SPLIT" \
  --robocasa365-category "$ROBOCASA365_CATEGORY" \
  "${EXTRA[@]}" \
  --embodiment-tag ROBOCASA_PANDA_OMRON \
  --modality-config-path "$MODALITY_CONFIG" \
  --num-gpus 1 \
  --output-dir "$OUTPUT_DIR" \
  --max-steps "$MAX_STEPS" \
  --global-batch-size "$GLOBAL_BATCH_SIZE" \
  --save-steps "$SAVE_STEPS" \
  --save-total-limit "$SAVE_TOTAL_LIMIT" \
  --dataloader-num-workers "$DATALOADER_NUM_WORKERS" \
  --shard-size "$SHARD_SIZE" \
  "${VISOR_ARGS[@]}" \
  "${MOTION_ARGS[@]}" \
  "${JOINT_ARGS[@]}"
