# LEO baseline on RoboCasa365 (single multi-task LoRA, 50-task eval)

Upstream: [LEO (ICML 2024)](https://embodied-generalist.github.io/) / [embodied-generalist](https://github.com/embodied-generalist/embodied-generalist).

## Strategy (aligned with RoboCasa365 leaderboard)

For **multi-task LoRA**, you train on all **locally available** pretrain demos among the 50 tasks.  
`composite_unseen` (16 tasks) may be missing locally — they are still **evaluated in sim** (zero-shot generalization) if the checkpoint exists.

| Split | Train data | Sim eval |
|-------|------------|----------|
| atomic_seen (18) | pretrain demos | pretrain kitchens |
| composite_seen (16) | pretrain demos | pretrain kitchens |
| composite_unseen (16) | optional / zero-shot | pretrain kitchens |

This mirrors the [RoboCasa365 leaderboard](https://robocasa.ai/leaderboard.html) multi-task setting: one policy, 50-task matrix.

## Why one LoRA (not 50 separate)

- Fair comparison to GR00T multi-task / single-checkpoint eval
- Matches LEO official recipe (LoRA on Vicuna-7B after `align.pth`)
- Feasible on 4×A100 80G with mixed 2D + proprio + action head

## Pipeline

```text
1. bash baselines/leo/setup_leo.sh
2. bash baselines/leo/convert_robocasa365_data.sh      # manifest for 50 tasks
3. bash baselines/leo/finetune_leo_target50_lora.sh    # one LoRA ckpt
4. bash baselines/leo/run_leo_baseline.sh              # 50×50 sim eval
```

## LEO ↔ RoboCasa365 bridge (v1)

LEO has no native RoboCasa sim hook. This baseline uses:

- **Input**: 3× RGB (agentview L/R + eye-in-hand) + language + proprio (5D state)
- **3D**: optional depth/pointcloud (v2); v1 trains **2D branch + proprio** only
- **Output**: 12D Panda-Omron action (gripper, EEF delta, base, control_mode) via **action MLP head** on LLM hidden state

Weights: start from LEO `sft_noact.pth` or `align.pth` + new action head.

## Outputs

| Path | Content |
|------|---------|
| `output/leo_rc365_target50_lora/` | LoRA adapter + action head |
| `output/robocasa365_eval_leo/` | `summary_shard0of1.csv` (50 rows) |

## Env vars

```bash
export LEO_REPO=/path/to/embodied-generalist
export ROBOCASA365_ROOT=/path/to/robocasa365-datasets
export LEO_BASE_CKPT=align   # align | sft_noact | path/to/ckpt
export LEO_LORA_R=16
export LEO_LORA_ALPHA=32
export MAX_STEPS=30000
export GLOBAL_BATCH_SIZE=32
```

## References

- Huang et al., *An Embodied Generalist Agent in 3D World*, ICML 2024
- RoboCasa365 benchmark: 50-task multi-task eval on pretrain kitchens
