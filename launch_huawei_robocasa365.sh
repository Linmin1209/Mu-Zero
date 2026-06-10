#!/usr/bin/env bash
# ==============================================================================
# Mu-Zero / Isaac-GR00T — Huawei Cloud (ModelArts/CCE) RoboCasa365 training
#
# Fixes vs legacy launch_huawei.sh:
#   - Pins torch==2.7.1 + torchvision==0.22.1 + transformers==4.57.3 (cu128)
#     (avoids torch 2.12 / transformers 5.x -> torchvision::nms crash)
#   - Uses gr00t/experiment/launch_finetune.py (official N1.7 finetune entry)
#   - RoboCasa365 dataset layout: {root}/{pretrain|target}/{atomic|composite}/{Task}/{date}/lerobot/
#   - Fails fast when PyTorch CUDA is unavailable (no fake 8-GPU launch)
#
# Usage (ModelArts console):
#   bash launch_huawei_robocasa365.sh
#
# Single task + MOSS:
#   ROBOCASA365_TASKS=NavigateKitchen USE_MOTION=1 bash launch_huawei_robocasa365.sh
#
# Native DiT + per-component MLP decoders (loads pretrained AlternateVLDiT):
#   ROBOCASA365_TASKS=PickPlaceToasterToCounter USE_COMPONENT_FACTORED=1 bash launch_huawei_robocasa365.sh
#
# Legacy adaptive MSAT head (decoder trains from scratch — not recommended):
#   USE_ADAPTIVE_COMPONENT=1 bash launch_huawei_robocasa365.sh
#
# Skip downloads (code/models/data already on /cache):
#   DOWNLOAD_CODE=0 DOWNLOAD_MODELS=0 DOWNLOAD_DATA=0 bash launch_huawei_robocasa365.sh
#
# OBS credentials: export before running (do NOT commit secrets):
#   export S3_ENDPOINT ACCESS_KEY_ID SECRET_ACCESS_KEY OBS_BUCKET OBS_PREFIX
#
# OBS code layout (upload your Mu-Zero repo to OBS):
#   ${OBS_BUCKET}/${OBS_PREFIX}/Mu-Zero/gr00t/
#   ${OBS_BUCKET}/${OBS_PREFIX}/Mu-Zero/examples/
#   ${OBS_BUCKET}/${OBS_PREFIX}/Mu-Zero/scripts/
#   ${OBS_BUCKET}/${OBS_PREFIX}/Mu-Zero/external_dependencies/
#   ${OBS_BUCKET}/${OBS_PREFIX}/Mu-Zero/pyproject.toml
#   ${OBS_BUCKET}/${OBS_PREFIX}/Mu-Zero/launch_huawei_robocasa365.sh
# Override repo folder name: OBS_REPO_NAME=Mu-Zero
# ==============================================================================
set -euo pipefail

# ---------- conda ----------
__conda_setup="$('/root/miniconda3/bin/conda' 'shell.bash' 'hook' 2>/dev/null)" || true
if [[ -n "${__conda_setup:-}" ]]; then
  eval "$__conda_setup"
elif [[ -f /root/miniconda3/etc/profile.d/conda.sh ]]; then
  # shellcheck source=/dev/null
  . /root/miniconda3/etc/profile.d/conda.sh
else
  export PATH="/root/miniconda3/bin:$PATH"
fi
unset __conda_setup

CONDA_ENV="${CONDA_ENV:-qwen3}"
conda activate "$CONDA_ENV"
PYTHON="$(command -v python)"
echo "[i] Python: $($PYTHON --version 2>&1) @ $PYTHON"

# ==============================================================================
# Paths & OBS (override via env)
# ==============================================================================
WORK_DIR="${WORK_DIR:-/cache/robocasa-gr00t}"
OBS_REPO_NAME="${OBS_REPO_NAME:-Mu-Zero}"
REPO_ROOT="${REPO_ROOT:-$WORK_DIR/$OBS_REPO_NAME}"
MODELS_DIR="${MODELS_DIR:-$WORK_DIR/Models}"
ROBOCASA365_ROOT="${ROBOCASA365_ROOT:-$WORK_DIR/Datasets_365/robocasa365-datasets}"

export S3_ENDPOINT="${S3_ENDPOINT:-}"
export S3_USE_HTTPS="${S3_USE_HTTPS:-0}"
export ACCESS_KEY_ID="${ACCESS_KEY_ID:-}"
export SECRET_ACCESS_KEY="${SECRET_ACCESS_KEY:-}"
OBS_BUCKET="${OBS_BUCKET:-obs://yw-2030-gy}"
OBS_PREFIX="${OBS_PREFIX:-LM}"
OBS_SRC="${OBS_BUCKET}/${OBS_PREFIX}"
OBS_CODE_BASE="${OBS_CODE_BASE:-${OBS_SRC}/${OBS_REPO_NAME}}"
OBS_UPLOAD_DEST="${OBS_UPLOAD_DEST:-${OBS_CODE_BASE}/outputs}"
ROBOCASA365_OBS_BASE="${ROBOCASA365_OBS_BASE:-${OBS_BUCKET}/data/opensource/robocasa365/pretrain}"

DOWNLOAD_CODE="${DOWNLOAD_CODE:-1}"
DOWNLOAD_MODELS="${DOWNLOAD_MODELS:-1}"
DOWNLOAD_DATA="${DOWNLOAD_DATA:-1}"

GR00T_MODEL_DIR="${GR00T_MODEL_DIR:-GR00T-N1.7-3B}"
COSMOS_MODEL_DIR="${COSMOS_MODEL_DIR:-Cosmos-Reason2-2B}"
export GR00T_MODELS_ROOT="$MODELS_DIR"
export GR00T_BASE_MODEL="${GR00T_BASE_MODEL:-$MODELS_DIR/$GR00T_MODEL_DIR}"
export GR00T_COSMOS_MODEL_PATH="${GR00T_COSMOS_MODEL_PATH:-$MODELS_DIR/$COSMOS_MODEL_DIR}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export GROOT_PATCH_MISTRAL=1
export GROOT_HF_LOCAL_FIRST=1
export NO_ALBUMENTATIONS_UPDATE=1
export PYTHONUNBUFFERED=1

# Training
ROBOCASA365_SPLIT="${ROBOCASA365_SPLIT:-pretrain}"
ROBOCASA365_CATEGORY="${ROBOCASA365_CATEGORY:-atomic}"
ROBOCASA365_TASKS="${ROBOCASA365_TASKS:-}"
MAX_STEPS="${MAX_STEPS:-30000}"
SAVE_STEPS="${SAVE_STEPS:-5000}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-128}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-8}"
MODALITY_CONFIG="${MODALITY_CONFIG:-$REPO_ROOT/examples/RoboCasa365/robocasa365_config_4frame.py}"

USE_MOTION="${USE_MOTION:-1}"
MOTION_INSERT_LAYER="${MOTION_INSERT_LAYER:-9}"
TUNE_MOTION="${TUNE_MOTION:-1}"
USE_ADAPTIVE_COMPONENT="${USE_ADAPTIVE_COMPONENT:-0}"
USE_COMPONENT_FACTORED="${USE_COMPONENT_FACTORED:-0}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-1}"

if [[ "$USE_ADAPTIVE_COMPONENT" == "1" && "$USE_COMPONENT_FACTORED" == "1" ]]; then
  echo "[ERROR] USE_ADAPTIVE_COMPONENT and USE_COMPONENT_FACTORED are mutually exclusive."
  exit 1
fi

HEAD_SUFFIX=""
if [[ "$USE_COMPONENT_FACTORED" == "1" ]]; then
  HEAD_SUFFIX="_component_factored"
elif [[ "$USE_ADAPTIVE_COMPONENT" == "1" ]]; then
  HEAD_SUFFIX="_adaptive_component"
fi
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/output/rc365_huawei_${ROBOCASA365_CATEGORY}_${MAX_STEPS}${HEAD_SUFFIX}}"

# ==============================================================================
# Helpers
# ==============================================================================
mox_download() {
  local obs_path="$1" local_path="$2" desc="${3:-$local_path}"
  if [[ -d "$local_path" ]] && [[ -n "$(ls -A "$local_path" 2>/dev/null)" ]]; then
    echo "  [SKIP] $desc"
    return 0
  fi
  echo "  [DOWN] $desc"
  mkdir -p "$local_path"
  "$PYTHON" -c "
import moxing as mox
mox.file.copy_parallel('$obs_path', '$local_path')
" || return 1
}

mox_download_file() {
  local obs_path="$1" local_path="$2"
  [[ -f "$local_path" ]] && return 0
  mkdir -p "$(dirname "$local_path")"
  "$PYTHON" -c "
import moxing as mox
mox.file.copy('$obs_path', '$local_path')
" || true
}

migrate_flat_panda_datasets() {
  # Legacy OBS layout: Datasets_365/PandaOmron.{Task} -> standard robocasa365 tree
  local legacy_root="${1:-$(dirname "$ROBOCASA365_ROOT")}"
  shopt -s nullglob
  local moved=0
  for d in "$legacy_root"/PandaOmron.*; do
    [[ -d "$d/meta" ]] || continue
    local task="${d##*/PandaOmron.}"
    local dest="$ROBOCASA365_ROOT/pretrain/atomic/$task/migrated/lerobot"
    if [[ ! -f "$dest/meta/info.json" ]]; then
      echo "  [migrate] $d -> $dest"
      mkdir -p "$(dirname "$dest")"
      cp -a "$d" "$dest"
      moved=$((moved + 1))
    fi
  done
  [[ "$moved" -gt 0 ]] && echo "  [migrate] $moved legacy dataset(s) copied into $ROBOCASA365_ROOT"
}

install_pytorch_cu128() {
  echo "[i] Installing pinned PyTorch stack (cu128, matches Isaac-GR00T pyproject.toml) ..."
  "$PYTHON" -m pip install --upgrade \
    "torch==2.7.1" "torchvision==0.22.1" "torchaudio==2.7.1" \
    --index-url https://download.pytorch.org/whl/cu128
  "$PYTHON" -m pip install --upgrade \
    "transformers==4.57.3" "peft==0.17.1" "accelerate==1.7.0" "diffusers==0.35.1" \
    "numpy==1.26.4" "scipy==1.15.3"
}

verify_imports() {
  "$PYTHON" <<'PY'
import torch, transformers, peft, torchvision
print(f"  torch {torch.__version__}, torchvision {torchvision.__version__}")
print(f"  transformers {transformers.__version__}, peft {peft.__version__}")
if not torch.cuda.is_available():
    raise SystemExit(f"CUDA unavailable: {torch.cuda.device_count()} devices")
n = torch.cuda.device_count()
print(f"  CUDA OK: {n} GPU(s), torch.cuda={torch.version.cuda}")
PY
}

detect_num_gpus() {
  local n
  n="$("$PYTHON" -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 0)"
  if [[ "$n" -gt 0 ]]; then
    echo "$n"
    return
  fi
  echo 0
}

# ==============================================================================
# Step 1: moxing
# ==============================================================================
echo ""
echo "============ Step 1: OBS / moxing ============"
"$PYTHON" -c "import moxing" 2>/dev/null || "$PYTHON" -m pip install -q moxing-framework

# ==============================================================================
# Step 2: code
# ==============================================================================
if [[ "$DOWNLOAD_CODE" == "1" ]]; then
  echo ""
  echo "============ Step 2: Download code ($OBS_CODE_BASE) ============"
  mkdir -p "$WORK_DIR"
  mox_download "${OBS_CODE_BASE}/gr00t" "$REPO_ROOT/gr00t" "gr00t/"
  mox_download "${OBS_CODE_BASE}/examples" "$REPO_ROOT/examples" "examples/"
  mox_download "${OBS_CODE_BASE}/scripts" "$REPO_ROOT/scripts" "scripts/"
  mox_download "${OBS_CODE_BASE}/external_dependencies" "$REPO_ROOT/external_dependencies" "external_dependencies/"
  mox_download_file "${OBS_CODE_BASE}/pyproject.toml" "$REPO_ROOT/pyproject.toml"
  mox_download_file "${OBS_CODE_BASE}/launch_huawei_robocasa365.sh" "$REPO_ROOT/launch_huawei_robocasa365.sh" || true
  mox_download_file "${OBS_CODE_BASE}/MU_ZERO.md" "$REPO_ROOT/MU_ZERO.md" || true
else
  echo "[SKIP] code download"
fi

# ==============================================================================
# Step 3: models
# ==============================================================================
if [[ "$DOWNLOAD_MODELS" == "1" ]]; then
  echo ""
  echo "============ Step 3: Download models ============"
  mkdir -p "$MODELS_DIR"
  mox_download "${OBS_SRC}/Models/${GR00T_MODEL_DIR}" "$GR00T_BASE_MODEL" "$GR00T_MODEL_DIR"
  mox_download "${OBS_SRC}/Models/${COSMOS_MODEL_DIR}" "$GR00T_COSMOS_MODEL_PATH" "$COSMOS_MODEL_DIR"
else
  echo "[SKIP] model download"
fi

# ==============================================================================
# Step 4: datasets (standard robocasa365 layout)
# ==============================================================================
if [[ "$DOWNLOAD_DATA" == "1" ]]; then
  echo ""
  echo "============ Step 4: Download RoboCasa365 ============"
  mkdir -p "$ROBOCASA365_ROOT"

  CATEGORIES=("$ROBOCASA365_CATEGORY")
  [[ "$ROBOCASA365_CATEGORY" == "all" ]] && CATEGORIES=(atomic composite)

  TASK_FILTER=()
  if [[ -n "$ROBOCASA365_TASKS" ]]; then
    IFS=',' read -ra TASK_FILTER <<< "$ROBOCASA365_TASKS"
  fi

  for cat in "${CATEGORIES[@]}"; do
    mapfile -t ALL_TASKS < <("$PYTHON" -c "
import moxing as mox
for d in sorted(mox.file.list_directory('${ROBOCASA365_OBS_BASE}/${ROBOCASA365_SPLIT}/${cat}')):
    print(d)
" 2>/dev/null || true)

    for task in "${ALL_TASKS[@]:-}"; do
      [[ -z "$task" ]] && continue
      if [[ ${#TASK_FILTER[@]} -gt 0 ]]; then
        skip=1
        for t in "${TASK_FILTER[@]}"; do
          [[ "$t" == "$task" ]] && skip=0
        done
        [[ "$skip" == "1" ]] && continue
      fi

      local_lerobot="$ROBOCASA365_ROOT/${ROBOCASA365_SPLIT}/${cat}/${task}"
      if find "$local_lerobot" -path '*/lerobot/meta/info.json' -print -quit 2>/dev/null | grep -q .; then
        echo "  [SKIP] ${ROBOCASA365_SPLIT}/${cat}/${task}"
        continue
      fi

      obs_lerobot="$("$PYTHON" -c "
import moxing as mox, sys
base = '${ROBOCASA365_OBS_BASE}/${ROBOCASA365_SPLIT}/${cat}/${task}'
try:
    dates = sorted(mox.file.list_directory(base))
except Exception:
    sys.exit(1)
for date in dates:
    p = f'{base}/{date}/lerobot'
    try:
        inner = mox.file.list_directory(p)
    except Exception:
        continue
    if 'lerobot' in inner:
        print(f'{p}/lerobot')
    else:
        print(p)
    sys.exit(0)
sys.exit(1)
" 2>/dev/null || true)"

      if [[ -z "$obs_lerobot" ]]; then
        echo "  [WARN] no lerobot path for ${cat}/${task}"
        continue
      fi

      # OBS: .../{date}/lerobot[/lerobot] -> local: .../{date}/lerobot
      date_dir="$(basename "$(dirname "$obs_lerobot")")"
      [[ "$date_dir" == "lerobot" ]] && date_dir="$(basename "$(dirname "$(dirname "$obs_lerobot")")")"
      dest="$ROBOCASA365_ROOT/${ROBOCASA365_SPLIT}/${cat}/${task}/${date_dir}/lerobot"
      echo "  [DOWN] ${cat}/${task} -> $dest"
      mkdir -p "$(dirname "$dest")"
      "$PYTHON" -c "
import moxing as mox
mox.file.copy_parallel('$obs_lerobot', '$dest')
" || echo "  [FAIL] ${cat}/${task}"
    done
  done

  migrate_flat_panda_datasets "$(dirname "$ROBOCASA365_ROOT")"
else
  echo "[SKIP] data download"
  migrate_flat_panda_datasets "$(dirname "$ROBOCASA365_ROOT")"
fi

# ==============================================================================
# Step 5: dependencies (pinned — do NOT pip install -U transformers/peft blindly)
# ==============================================================================
echo ""
echo "============ Step 5: Install dependencies ============"
cd "$REPO_ROOT"
install_pytorch_cu128

if [[ ! -f "$WORK_DIR/.gr00t_pip_done" ]]; then
  echo "[i] Installing Mu-Zero/gr00t package (no-deps) + runtime deps ..."
  "$PYTHON" -m pip install -e . --no-deps
  "$PYTHON" -m pip install \
    "albumentations==1.4.18" "av==16.1.0" "tyro==0.9.17" "pandas==2.2.3" \
    "einops==0.8.1" "omegaconf==2.3.0" "msgpack==1.1.0" "msgpack-numpy==0.4.8" \
    "datasets==3.6.0" "wandb==0.23.0" "pyzmq==27.0.1" "gitpython==3.1.46" \
    "jsonlines==4.0.0" "gymnasium==1.2.2" "matplotlib==3.10.1" "termcolor==3.2.0" \
    "opencv-python-headless>=4.5,<4.13" "lmdb==1.7.5" "dm-tree" "click==8.1.8"
  touch "$WORK_DIR/.gr00t_pip_done"
else
  echo "[SKIP] gr00t pip bundle (remove $WORK_DIR/.gr00t_pip_done to reinstall)"
fi

echo "[i] Verifying imports & CUDA ..."
verify_imports

NUM_GPUS="${NUM_GPUS:-$(detect_num_gpus)}"
if [[ "$NUM_GPUS" -le 0 ]]; then
  echo "[ERROR] PyTorch sees 0 GPUs. Fix torch/cu128 install before training."
  nvidia-smi || true
  exit 1
fi
echo "[i] Using NUM_GPUS=$NUM_GPUS"

# ==============================================================================
# Step 6: training
# ==============================================================================
echo ""
echo "============ Step 6: Launch finetune ============"
mkdir -p "$OUTPUT_DIR"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $((NUM_GPUS - 1)))}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29500}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-bond0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export UPLOAD_OBS_BASE="${OBS_UPLOAD_DEST}/$(date '+%Y%m%d_%H%M%S')"

echo "[i] dataset root: $ROBOCASA365_ROOT"
echo "[i] base model:   $GR00T_BASE_MODEL"
echo "[i] output:       $OUTPUT_DIR"
echo "[i] use_motion=$USE_MOTION use_component_factored=$USE_COMPONENT_FACTORED use_adaptive_component=$USE_ADAPTIVE_COMPONENT"
echo "[i] OBS upload:   $UPLOAD_OBS_BASE"

EXTRA=(--robocasa365-root "$ROBOCASA365_ROOT" --robocasa365-split "$ROBOCASA365_SPLIT" --robocasa365-category "$ROBOCASA365_CATEGORY")
[[ -n "$ROBOCASA365_TASKS" ]] && EXTRA+=(--robocasa365-tasks "$ROBOCASA365_TASKS")

MOTION_ARGS=()
[[ "$USE_MOTION" == "1" ]] && MOTION_ARGS+=(--use-motion --motion-insert-layer "$MOTION_INSERT_LAYER")
[[ "$TUNE_MOTION" == "1" ]] && MOTION_ARGS+=(--tune-motion)

ADAPTIVE_ARGS=()
[[ "$USE_ADAPTIVE_COMPONENT" == "1" ]] && ADAPTIVE_ARGS+=(--use-adaptive-component-head)

FACTORED_ARGS=()
[[ "$USE_COMPONENT_FACTORED" == "1" ]] && FACTORED_ARGS+=(--use-component-factored-head)

GC_ARGS=()
[[ "$GRADIENT_CHECKPOINTING" == "1" ]] && GC_ARGS+=(--gradient-checkpointing)

COMMON=(
  --base-model-path "$GR00T_BASE_MODEL"
  "${EXTRA[@]}"
  --embodiment-tag ROBOCASA_PANDA_OMRON
  --modality-config-path "$MODALITY_CONFIG"
  --output-dir "$OUTPUT_DIR"
  --max-steps "$MAX_STEPS"
  --global-batch-size "$GLOBAL_BATCH_SIZE"
  --save-steps "$SAVE_STEPS"
  --save-total-limit 10
  --dataloader-num-workers "$DATALOADER_NUM_WORKERS"
  "${MOTION_ARGS[@]}"
  "${ADAPTIVE_ARGS[@]}"
  "${FACTORED_ARGS[@]}"
  "${GC_ARGS[@]}"
)

if [[ "$NUM_GPUS" -eq 1 ]]; then
  exec "$PYTHON" -u gr00t/experiment/launch_finetune.py \
    --num-gpus 1 \
    "${COMMON[@]}"
else
  exec "$PYTHON" -u -m torch.distributed.run \
    --nproc_per_node="$NUM_GPUS" \
    --master_port="$MASTER_PORT" \
    gr00t/experiment/launch_finetune.py \
    --num-gpus "$NUM_GPUS" \
    "${COMMON[@]}"
fi
