# Echo VLA baseline on RoboCasa365

Upstream: [Echo_VLA](https://github.com/Linmin1209/UR-manipulation-modelscope) (`UR-manipulation-modelscope/Echo_VLA`) — mobile-manipulation VLA (PI0.5 / DDPM) on Panda-Omron + RoboCasa.

## Integration strategy

Echo VLA was built on **classic RoboCasa** (`PnPCounterToStove`, 10D actions, HDF5/3D demos).  
RoboCasa365 uses **new task names**, **LeRobot v2.1** demos, and **12D Panda-Omron** actions.

This baseline bridges the two stacks in three layers:

```text
┌─────────────────────────────────────────────────────────────┐
│  Isaac-GR00T RoboCasa365 harness (target50, 50×50 eval)    │
│  eval_echo_vla_robocasa365.py → summary_shard0of1.csv       │
└───────────────────────────┬─────────────────────────────────┘
                            │ PolicyClient (ZMQ)
┌───────────────────────────▼─────────────────────────────────┐
│  run_echo_vla_server.py  (GR00T PolicyServer)                │
│  EchoVlaGr00tPolicy: obs RC365 → Echo → action 12D          │
└───────────────────────────┬─────────────────────────────────┘
                            │ hydra checkpoint
┌───────────────────────────▼─────────────────────────────────┐
│  Echo_VLA repo (PI05_Agent / DDPM_Agent)                    │
│  train: finetune_echo_vla_rc365.sh → Echo Hydra configs     │
└─────────────────────────────────────────────────────────────┘
```

### Phase 1 (this PR) — harness + bridge scaffold

| Component | Purpose |
|-----------|---------|
| `task_mapping.yaml` | RC365 task → Echo classic env name |
| `task_compatibility.yaml` | Which target50 tasks Echo can eval today |
| `action_adapter.py` | 10D Echo ↔ 12D RC365 Panda-Omron |
| `echo_vla_policy.py` | `BasePolicy` wrapper around Echo agent |
| `run_echo_vla_server.py` | ZMQ server for `rollout_policy` |
| `eval_echo_vla_robocasa365.py` | 50-task matrix, same CSV as GR00T/LEO |

### Phase 2 — data + training on RC365 LeRobot

| Step | Echo-side work |
|------|----------------|
| Dataset | LeRobot → Echo `RobocasaDataset` adapter (3× RGB + proprio + language) |
| Config | `echo_vla_rc365_pi05.yaml` overrides `dataset_path`, `action_dim: 12` |
| Train | `finetune_echo_vla_rc365.sh` → `torchrun fsdp_run.py` or `run.py` |

### Phase 3 — full target50 parity

- Extend `task_mapping.yaml` for all 50 tasks (or train multi-task on RC365 manifest)
- Optional 3D point-cloud path (`robocasa_config_pi05_3d.yaml` pattern)
- Memory-bank / slot variants for long-horizon composite tasks

## Action space bridge

| Echo VLA (10D) | RoboCasa365 (12D) |
|----------------|-------------------|
| 7 arm (OSC delta pos/rot + gripper) | `gripper_close` (1) + `end_effector_position` (3) + `end_effector_rotation` (3) |
| 3 base (x, y, yaw) | `base_motion` (3) + `control_mode` (1) |

See `action_adapter.py` for the default layout (tune per checkpoint scaler).

## Pipeline

```bash
# 1. Point at local Echo_VLA clone
export ECHO_VLA_REPO=/path/to/UR-manipulation-modelscope/Echo_VLA
bash examples/RoboCasa365/baselines/echo_vla/setup_echo_vla.sh

# 2. Fine-tune (Echo Hydra — uses Echo repo, RC365-oriented config)
export DATASET_PATH=/path/to/robocasa365-datasets   # or robocasa_3d HDF5
bash examples/RoboCasa365/baselines/echo_vla/finetune_echo_vla_rc365.sh

# 3. Eval on RoboCasa365 sim (target50 × 50 ep)
export MODEL_PATH=/path/to/echo_checkpoint_dir
bash examples/RoboCasa365/baselines/echo_vla/run_echo_vla_baseline.sh
```

## Env vars

```bash
export ECHO_VLA_REPO=/path/to/Echo_VLA
export MODEL_PATH=/path/to/checkpoint   # eval_dir with best_val_model*.pth
export ECHO_VLA_CONFIG=robocasa_config_pi05.yaml
export TASK_SET=target50
export SPLIT=pretrain
export N_EPISODES=50
export SERVER_PORT=5560
```

## Outputs

| Path | Content |
|------|---------|
| `output/echo_vla_rc365_pi05/` | Echo training runs (via Hydra) |
| `output/robocasa365_eval_echo_vla/` | `summary_shard0of1.csv` (50 rows) |

## References

- Echo VLA: PI0.5 + optional 3D point cloud, RoboCasa parallel sim
- RoboCasa365 leaderboard: 50-task multi-task eval on pretrain kitchens
