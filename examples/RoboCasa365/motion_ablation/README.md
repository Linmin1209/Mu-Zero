# STSS / MOSS Motion 模块验证方案

在 Vision Encoder（Qwen3-VL layer 9）插入 **STSS → MotionModule（MOSS）** 后是否有提升。

## 背景

| 项目 | Vision 骨干 | STSS/MOSS |
|------|-------------|-----------|
| **Isaac-GR00T N1.7** | Cosmos-Reason2-2B（`Qwen3VLForConditionalGeneration`） | **无** |
| **RLDX-1** | 定制 `modeling_qwen3_vl.py` | **有**（`rldx/model/modules/backbone/motion.py`） |

本地 RC365 相关 checkpoint 状态：

- GR00T finetune：`output/rc365_PickPlaceToasterToCounter_30k_b128/checkpoint-30000`（无 motion）
- RLDX RC365 finetune：`RLDX-1/checkpoints/robocasa365`（`use_video=true`，**无** `motion_block` 权重）

因此：**不能**用现有 checkpoint 直接做 with/without 对比，需要 **同数据、同步数、两路训练**。

## 推荐验证路径（两阶段）

### 阶段 1：RLDX 对照实验（最快，模块已集成）

在同一任务 **PickPlaceToasterToCounter / pretrain** 上训练两版：

| 实验 | 关键 flag | 说明 |
|------|-----------|------|
| **baseline** | `--video-length 4` | 多帧输入，**不**启用 STSS |
| **+motion** | 上式 + `--use-motion --motion-insert-layer 9 --new-param-warmup-steps 2000` | Vision block 9 后 residual 注入 MOSS |

脚本（已生成）：

```bash
# 1) 训练对照（RLDX-1 仓库）
export BASE_MODEL_PATH=/path/to/RLDX-1-PT   # 或 RLWRLD/RLDX-1-PT
bash /HOME/sysu_xdliang/sysu_xdliang_1/HDD_POOL/linmin/RLDX-1/run_scripts/train/ablations/finetune_pickplace_motion_ablation.sh baseline
bash /HOME/sysu_xdliang/sysu_xdliang_1/HDD_POOL/linmin/RLDX-1/run_scripts/train/ablations/finetune_pickplace_motion_ablation.sh motion

# 2) 分别 sim eval（RLDX eval 脚本）
bash examples/RoboCasa365/motion_ablation/run_rldx_ablation_eval.sh baseline
bash examples/RoboCasa365/motion_ablation/run_rldx_ablation_eval.sh motion

# 或一键 train + eval
bash examples/RoboCasa365/motion_ablation/run_full_ablation.sh
```

输出目录：

- `RLDX-1/output/motion_ablation/pickplace_baseline_30k_b64/`
- `RLDX-1/output/motion_ablation/pickplace_motion_30k_b64/`

**已有 GR00T baseline（无 motion）** 可作为架构参考，但不与 RLDX ablation 直接对比（不同代码路径）：

- `output/rc365_PickPlaceToasterToCounter_30k_b128/checkpoint-30000`（eval success_rate 见 `eval/.../summary_shard0of1.csv`）

评测（RLDX checkpoint 用 RLDX eval；GR00T checkpoint 用 GR00T eval）：

```bash
# RLDX ablation
bash examples/RoboCasa365/motion_ablation/run_rldx_ablation_eval.sh motion

# GR00T 已有 ckpt
bash examples/RoboCasa365/eval_robocasa365.sh \
  --model-path output/rc365_PickPlaceToasterToCounter_30k_b128/checkpoint-30000 \
  --task-set atomic_seen --split pretrain --tasks PickPlaceToasterToCounter \
  --n-episodes 50 --n-envs 5 --n-action-steps 40
```

**成功标准**：同 eval 配置下 `success_rate`（+motion）> baseline，且训练 loss 收敛正常。

### 阶段 2：移植到 GR00T（目标基座）— **已实现**

GR00T 现已支持通过训练 flag 启用 STSS/MOSS：

| 实验 | 脚本 / flag |
|------|-------------|
| **4-frame baseline** | `examples/RoboCasa365/finetune_pickplace_toaster_30k.sh` |
| **4-frame + motion** | `examples/RoboCasa365/finetune_pickplace_motion_30k.sh` 或 `launch_finetune.py --use-motion --motion-insert-layer 9` |

```bash
cd /HOME/sysu_xdliang/sysu_xdliang_1/HDD_POOL/linmin/Isaac-GR00T

# baseline（已有 4-frame ckpt）
bash examples/RoboCasa365/finetune_pickplace_toaster_30k.sh

# +motion ablation
bash examples/RoboCasa365/finetune_pickplace_motion_30k.sh

# eval 对照（同一 eval 脚本）
bash examples/RoboCasa365/eval_robocasa365.sh \
  --model-path output/rc365_PickPlaceToasterToCounter_30k_b64_4frame_motion/checkpoint-30000 \
  --task-set atomic_seen --split pretrain --tasks PickPlaceToasterToCounter \
  --n-episodes 50 --n-envs 5 --n-action-steps 40
```

实现位置：

- STSS/MOSS：`gr00t/model/modules/motion.py`
- Vision 插入：`gr00t/model/modules/qwen3_motion.py`（layer 9 后 `_apply_moss`）
- Backbone：`gr00t/model/modules/qwen3_backbone.py`（`install_motion_module` + `num_frames`/`num_views`）
- Config：`gr00t/configs/model/gr00t_n1d7.py`、`gr00t/configs/finetune_config.py`（`--use-motion`）

**注意**：MOSS 需要多帧 video（`robocasa365_config_4frame.py` 的 `delta_indices=[-6,-4,-2,0]`）；单帧无效。

---

### 阶段 2（旧说明）：移植要点（参考 RLDX）

1. 拷贝 `RLDX-1/rldx/model/modules/backbone/motion.py`
2. 在 Vision forward 第 9 层后调用 `_apply_moss()`（见 `modeling_qwen3_vl.py` ~942 行）
3. 训练侧：`video-length ≥ 2`（建议 4），modality `delta_indices` 覆盖多帧
4. 新增 config：`use_motion`, `motion_insert_layer`, `new_param_warmup_steps`
5. 在 GR00T 上重复阶段 1 的 baseline / +motion 对照

## RLDX 核心代码位置

- STSS/MOSS：`RLDX-1/rldx/model/modules/backbone/motion.py`
- 插入点：`RLDX-1/rldx/model/modules/backbone/modeling_qwen3_vl.py`（`MotionModule` + `_apply_moss`）
- 训练开关：`--use-motion` → `rldx/experiment/features/motion.py`

## 注意

- MOSS 需要 **时序视频**（`--video-length ≥ 2`）；单帧 `delta_indices=[0]` 不足以发挥 STSS。
- 评测时不要固定 `--server-port 5555`（易被 RLDX server 占用）；使用 `eval_robocasa365.sh` 自动选端口。
- Mid-train 权重 `RLDX-1-MT-*` 含 motion，但与 RC365 human 数据分布不同，**不能**替代 RC365 上的 controlled ablation。
