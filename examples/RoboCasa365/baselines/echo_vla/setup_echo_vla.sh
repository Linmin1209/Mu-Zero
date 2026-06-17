#!/usr/bin/env bash
# One-time Echo VLA setup pointers (external repo stays separate).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
ECHO_VLA_REPO="${ECHO_VLA_REPO:-/HOME/sysu_xdliang/sysu_xdliang_1/HDD_POOL/linmin/UR-manipulation-modelscope/Echo_VLA}"

if [[ ! -d "$ECHO_VLA_REPO" ]]; then
  echo "[x] Clone Echo_VLA first, e.g.:"
  echo "    git clone <UR-manipulation-modelscope> $ECHO_VLA_REPO"
  exit 1
fi

cat <<EOF

=== Echo VLA × RoboCasa365 setup ===

External repo: $ECHO_VLA_REPO

1. Echo env (conda recommended):
   cd "$ECHO_VLA_REPO"
   pip install -r requirements.txt
   bash custom_robocasa/install.sh    # RoboCasa sim + PandaOmron

2. PI0.5 base (if fine-tuning):
   ln -sfn checkpoint/pi05_base "$ECHO_VLA_REPO/pi05_base"  # or set pretrained_checkpoint

3. RoboCasa365 eval harness (this repo):
   bash examples/RoboCasa365/setup_eval.sh

4. Fine-tune toward RC365:
   export DATASET_PATH=/path/to/robocasa365-datasets
   bash examples/RoboCasa365/baselines/echo_vla/finetune_echo_vla_rc365.sh

5. Eval target50:
   export MODEL_PATH=/path/to/echo_checkpoint
   bash examples/RoboCasa365/baselines/echo_vla/run_echo_vla_baseline.sh

See: examples/RoboCasa365/baselines/echo_vla/README.md

EOF
