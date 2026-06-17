#!/usr/bin/env bash
# Fine-tune ONE LEO LoRA on all RoboCasa365 target50 tasks (multi-task SFT).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
LEO_REPO="${LEO_REPO:-$PROJECT_ROOT/../embodied-generalist}"
MANIFEST="${MANIFEST:-$SCRIPT_DIR/data/manifest_target50_pretrain.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/output/leo_rc365_target50_lora}"
LEO_BASE_CKPT="${LEO_BASE_CKPT:-align}"
LEO_LORA_R="${LEO_LORA_R:-16}"
LEO_LORA_ALPHA="${LEO_LORA_ALPHA:-32}"
MAX_STEPS="${MAX_STEPS:-30000}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
NUM_GPUS="${NUM_GPUS:-4}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

if [[ ! -f "$MANIFEST" ]]; then
  echo "[i] Manifest missing; running conversion..."
  bash "$SCRIPT_DIR/convert_robocasa365_data.sh"
fi

if [[ ! -d "$LEO_REPO" ]]; then
  echo "[x] LEO repo not found: $LEO_REPO"
  echo "    Run: bash examples/RoboCasa365/baselines/leo/setup_leo.sh"
  exit 1
fi

BRIDGE="$SCRIPT_DIR/leo_rc365_trainer.py"
if [[ ! -f "$BRIDGE" ]]; then
  echo "[x] Trainer bridge not found: $BRIDGE"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
LOG="$OUTPUT_DIR/train.log"

echo "[i] LEO multi-task LoRA (target50, one checkpoint)"
echo "[i] LEO_REPO=$LEO_REPO"
echo "[i] MANIFEST=$MANIFEST"
echo "[i] OUTPUT_DIR=$OUTPUT_DIR"
echo "[i] LoRA r=$LEO_LORA_R alpha=$LEO_LORA_ALPHA steps=$MAX_STEPS batch=$GLOBAL_BATCH_SIZE"

# Prefer conda leo env if active
PYTHON="${PYTHON:-python}"
if command -v conda >/dev/null 2>&1 && conda info --envs 2>/dev/null | grep -q "leo"; then
  PYTHON="conda run -n leo python"
fi

cd "$PROJECT_ROOT"
export PYTHONPATH="$LEO_REPO:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# shellcheck disable=SC2086
$PYTHON -u "$BRIDGE" \
  --leo-repo "$LEO_REPO" \
  --manifest "$MANIFEST" \
  --config "$SCRIPT_DIR/leo_rc365_target50.yaml" \
  --pretrained-ckpt "$LEO_BASE_CKPT" \
  --lora-r "$LEO_LORA_R" \
  --lora-alpha "$LEO_LORA_ALPHA" \
  --max-steps "$MAX_STEPS" \
  --global-batch-size "$GLOBAL_BATCH_SIZE" \
  --num-gpus "$NUM_GPUS" \
  --output-dir "$OUTPUT_DIR" \
  2>&1 | tee "$LOG"

echo "[i] Training log: $LOG"
