#!/usr/bin/env bash
# Clone LEO and prepare conda env (run once on a networked node).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
LEO_REPO="${LEO_REPO:-$PROJECT_ROOT/../embodied-generalist}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

if [[ ! -d "$LEO_REPO/.git" ]]; then
  echo "[i] Cloning embodied-generalist -> $LEO_REPO"
  git clone https://github.com/embodied-generalist/embodied-generalist.git "$LEO_REPO"
else
  echo "[i] LEO repo exists: $LEO_REPO"
fi

cat <<EOF

=== LEO setup (run in conda env 'leo') ===

  conda create -n leo python=3.9 -y
  conda activate leo
  conda install pytorch==1.12.1 torchvision==0.13.1 cudatoolkit=11.3 -c pytorch -y
  pip install -r "$LEO_REPO/requirements.txt"
  pip install peft==0.5.0 --no-deps "huggingface_hub>=0.26"

  # PointNet++ (required)
  cd "$LEO_REPO/model/pointnetpp" && python setup.py install && cd -

  # Download LEO align/sft checkpoints
  bash "$SCRIPT_DIR/download_leo_weights.sh"

  # Vicuna-7B + PointNet++ backbone + patch configs
  bash "$SCRIPT_DIR/download_leo_deps.sh"

  # PointNet++ CUDA extension (required for LeoAgent import)
  bash "$SCRIPT_DIR/install_leo_pointnetpp.sh"

Then:
  bash examples/RoboCasa365/baselines/leo/convert_robocasa365_data.sh
  bash examples/RoboCasa365/baselines/leo/finetune_leo_target50_lora.sh

EOF
