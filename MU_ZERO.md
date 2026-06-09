# Mu-Zero

Mobile manipulation VLA research codebase built on [NVIDIA Isaac GR00T N1.7](https://github.com/NVIDIA/Isaac-GR00T).

This repository extends GR00T with RoboCasa365 finetuning/evaluation, STSS/MOSS motion tokens, and an optional component-level adaptive action head.

## Highlights

- **RoboCasa365 pipeline**: dataset discovery, finetune launchers, sim eval (`examples/RoboCasa365/`)
- **STSS / MOSS**: short-term spatiotemporal motion tokens injected into the Qwen3-VL backbone (`gr00t/model/modules/motion.py`, `qwen3_motion.py`)
- **Adaptive component action head**: per-component tokenization + MSAT-style decoder (`gr00t/model/modules/component_action/`, `--use-adaptive-component-head`)
- **Local-first model loading**: offline HF/Cosmos paths via `gr00t/experiment/local_models.py`

## Quick start

### Environment

Follow upstream GR00T installation in `README.md`, then:

```bash
cd Isaac-GR00T
uv sync
source .venv/bin/activate
```

Set model paths (example):

```bash
export GR00T_MODELS_ROOT=/path/to/models
export GR00T_BASE_MODEL=$GR00T_MODELS_ROOT/GR00T-N1.7-3B
export GR00T_COSMOS_MODEL_PATH=$GR00T_MODELS_ROOT/Cosmos-Reason2-2B
export ROBOCASA365_ROOT=/path/to/robocasa365-datasets
```

### Finetune (RoboCasa365)

```bash
# Baseline + MOSS (4-frame video)
bash examples/RoboCasa365/finetune_pickplace_motion_30k.sh

# Adaptive component head + MOSS
bash examples/RoboCasa365/finetune_pickplace_adaptive_component_30k.sh
bash examples/RoboCasa365/finetune_navigate_kitchen_adaptive_component_30k.sh
```

### Sim evaluation

```bash
bash examples/RoboCasa365/setup_eval.sh
GR00T_CKPT=/path/to/checkpoint-30000 \
  bash examples/RoboCasa365/run_eval_local.sh --tasks NavigateKitchen
```

See `examples/RoboCasa365/README.md` for task sets, flags, and two-process eval layout.

## Layout

| Path | Description |
|------|-------------|
| `gr00t/experiment/robocasa365_datasets.py` | RoboCasa365 LeRobot dataset factory |
| `gr00t/eval/sim/robocasa365/` | Sim env setup + asset download |
| `gr00t/model/modules/component_action/` | Adaptive embodiment action head |
| `gr00t/model/modules/motion.py` | MOSS motion encoder |
| `examples/RoboCasa365/` | Configs, finetune/eval scripts |
| `external_dependencies/robocasa365/` | RoboCasa365 sim package (editable install) |

## Not included in git

Large artifacts are excluded via `.gitignore`:

- Training outputs (`output/`)
- Virtualenv (`.venv/`)
- Pretrained weights (`/models`, download separately)
- Datasets (`ROBOCASA365_ROOT`, not vendored)
- `external_dependencies/robosuite_official/` (installed from GitHub during sim setup)

## Upstream

Based on Isaac GR00T N1.7 Early Access. See `README.md` and NVIDIA license terms.
