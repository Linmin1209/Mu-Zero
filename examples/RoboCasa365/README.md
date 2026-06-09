# RoboCasa365 finetuning (GR00T N1.7)

Finetune on local [RoboCasa365](https://www.modelscope.cn/datasets/nv-community/robocasa365-datasets) LeRobot exports under `robocasa365-datasets/`.

**Prerequisites**

- Extracted datasets: `.../<split>/<atomic|composite>/<Task>/<date>/lerobot/`
- Base model: `GR00T-N1.7-3B`
- Cosmos tokenizer (local): `Cosmos-Reason2-2B` via `GR00T_COSMOS_MODEL_PATH`

## Quick start

```bash
cd Isaac-GR00T

# Single pretrain atomic task
ROBOCASA365_SPLIT=pretrain ROBOCASA365_CATEGORY=atomic \
  ROBOCASA365_TASKS=CloseElectricKettleLid \
  OUTPUT_DIR=/tmp/rc365_one \
  bash examples/RoboCasa365/finetune_robocasa365.sh

# All extracted pretrain atomic tasks (multi-task)
ROBOCASA365_SPLIT=pretrain ROBOCASA365_CATEGORY=atomic \
  OUTPUT_DIR=/tmp/rc365_pretrain_atomic \
  bash examples/RoboCasa365/finetune_robocasa365.sh
```

## `launch_finetune.py` flags

| Flag | Description |
|------|-------------|
| `--robocasa365-root` | Root of downloaded dataset tree |
| `--robocasa365-split` | `pretrain`, `target`, or `all` |
| `--robocasa365-category` | `atomic`, `composite`, or `all` |
| `--robocasa365-tasks` | Comma-separated task folder names (optional) |
| `--embodiment-tag` | Use `ROBOCASA_PANDA_OMRON` |
| `--modality-config-path` | `examples/RoboCasa365/robocasa365_config.py` |

Example (direct Python):

```bash
CUDA_VISIBLE_DEVICES=0 uv run python gr00t/experiment/launch_finetune.py \
  --base-model-path /HOME/sysu_xdliang/sysu_xdliang_1/HDD_POOL/linmin/models/GR00T-N1.7-3B \
  --robocasa365-root /HOME/sysu_xdliang/sysu_xdliang_1/HDD_POOL/linmin/datasets/robocasa365-datasets \
  --robocasa365-split pretrain \
  --robocasa365-category atomic \
  --robocasa365-tasks WashLettuce,CloseElectricKettleLid \
  --embodiment-tag ROBOCASA_PANDA_OMRON \
  --modality-config-path examples/RoboCasa365/robocasa365_config.py \
  --output-dir /tmp/rc365_multi \
  --num-gpus 1 \
  --max-steps 5000 \
  --global-batch-size 32
```

Multi-task mixing uses equal weight per dataset (see `DatasetFactory`).

## Sim evaluation (RoboCasa365 benchmark)

Same two-process layout as RLDX-1: **GR00T policy server** (main `.venv`, GPU) + **MuJoCo rollout** (`robocasa365_uv`).

### One-time setup

```bash
cd Isaac-GR00T
bash examples/RoboCasa365/setup_eval.sh
```

Reuses `robocasa365` from `RLDX-1/external_dependencies/robocasa365` if not vendored locally.  
If you already built `RLDX-1/rldx/eval/sim/robocasa365/robocasa365_uv`, eval will auto-fallback to that Python.

### Quick eval (single task)

```bash
GR00T_CKPT=/path/to/checkpoint-5000 \
TASK_SET=atomic_seen \
SPLIT=pretrain \
N_EPISODES=10 \
N_ENVS=1 \
bash examples/RoboCasa365/run_eval_local.sh \
  --tasks NavigateKitchen
```

Or pass `--model-path` directly:

```bash
bash examples/RoboCasa365/eval_robocasa365.sh \
  --model-path /HOME/sysu_xdliang/sysu_xdliang_1/HDD_POOL/linmin/RLDX-1/output/checkpoint-10 \
  --task-set atomic_seen \
  --split pretrain \
  --tasks NavigateKitchen \
  --n-episodes 10 \
  --n-envs 1 \
  --n-action-steps 40
```

Use the **checkpoint directory** (`checkpoint-XXXX/`), not the training output root.

### Task sets

Defined in `examples/RoboCasa365/task_sets.yaml`:

| `--task-set` | Description |
|--------------|-------------|
| `atomic_seen` | 18 atomic pretrain tasks |
| `composite_seen` | 16 composite seen |
| `composite_unseen` | 16 composite OOD |
| `target50` | All 50 benchmark tasks |

`--split pretrain|target` maps to `gym.make(..., split=...)`.

### Output layout

```
output/robocasa365_eval/
  checkpoint-5000_atomic_seen_pretrain_exp20260529_123456/
    summary_shard0of1.csv
    NavigateKitchen/
      eval.log
      videos/
        robocasa_NavigateKitchen_env00-episode_0-success.mp4
```

### Manual two-terminal (debug)

```bash
# Terminal 1 — policy server
cd Isaac-GR00T
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
.venv/bin/python gr00t/eval/run_gr00t_server.py \
  --model-path /path/to/checkpoint-5000 \
  --embodiment-tag ROBOCASA_PANDA_OMRON \
  --use-sim-policy-wrapper

# Terminal 2 — sim rollout (set PYTHONPATH like eval_robocasa365.sh)
PY365=gr00t/eval/sim/robocasa365/robocasa365_uv/.venv/bin/python
SIM_SITE=$($PY365 -c "import site; print(site.getsitepackages()[0])")
export PYTHONPATH="$SIM_SITE:$PWD/.venv/lib/python3.10/site-packages"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl

$PY365 gr00t/eval/rollout_policy.py \
  --policy-client-host 127.0.0.1 \
  --policy-client-port 5555 \
  --env-name robocasa/NavigateKitchen \
  --robocasa-split pretrain \
  --n-episodes 10 \
  --n-envs 1 \
  --n-action-steps 40 \
  --max-episode-steps 300 \
  --video-dir /tmp/rc365_nav_videos
```

### Notes

- Default `--n-action-steps` is **40** (matches `robocasa365_config.py` action horizon).
- Sim obs keys (`robot0_agentview_*`, `state.base_position`, …) match the finetune modality config; use `--use-sim-policy-wrapper` on the server.
- Finetune with `examples/RoboCasa365/robocasa365_config.py` so the saved processor matches sim I/O.
