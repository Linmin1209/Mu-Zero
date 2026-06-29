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

  # Python 3.10 recommended for torch 2.x + H100 (sm_90)
  conda create -n leo python=3.10 -y
  conda activate leo

  # H100 / A100: torch 2.1+ cu121 (includes sm_90). Legacy V100-only nodes can use cu118.
  pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121

  # Or run the upgrade script on an existing leo env:
  #   bash "$SCRIPT_DIR/upgrade_leo_torch_h100.sh"

  pip install -r "$LEO_REPO/requirements.txt"
  pip install peft==0.5.0 --no-deps "huggingface_hub>=0.26"
  pip install pyarrow

  # PointNet++ (required)
  cd "$LEO_REPO/model/pointnetpp" && python setup.py install && cd -

  # Download LEO align/sft checkpoints
  bash "$SCRIPT_DIR/download_leo_weights.sh"

  # Vicuna-7B + PointNet++ backbone + patch configs
  bash "$SCRIPT_DIR/download_leo_deps.sh"

  # ConvNeXt/CLIP 2D backbone (local: HDD_POOL/linmin/models)
  bash "$SCRIPT_DIR/download_leo_vision2d.sh"

  # PointNet++ CUDA extension (required for LeoAgent import)
  bash "$SCRIPT_DIR/install_leo_pointnetpp.sh"

Then:
  bash examples/RoboCasa365/baselines/leo/convert_robocasa365_data.sh
  bash examples/RoboCasa365/baselines/leo/finetune_leo_target50_lora.sh

EOF
