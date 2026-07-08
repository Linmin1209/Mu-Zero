# 双分支流匹配联合训练设计（FLUX LoRA + GR00T VISOR）

> 状态：**v0.2 MoT 融合架构已实现**（`use_mot_joint_expert=True` 默认开启）  
> 目标：对齐论文 \(L_{\text{total}} = \alpha L_{\text{img}} + \beta L_{\text{act}}\) 两阶段漏斗，同时保留现有 VISOR v4.2b 能力。

### v0.2 MoT 融合（VT-WAM 风格，推荐）

```
DiT token 序列: [state | action(H) | IHT(K) | anchor_inpaint(N) | future_inpaint(N)]
                         └─ action 去噪 ─┘      └─ FLUX VAE latent flow matching ─┘

- 同一 DiT、同一 backward（`joint_train_mode=simultaneous`）
- 冻结 FLUX VAE；可训练 `inpaint_expert` 投影 + velocity head + DiT/VISOR
- Asymmetric MoT SA mask（`build_mot_inpaint_sa_mask`）：
  - action 可读 anchor，不可读 future（推理安全）
  - inpaint expert 与 action/IHT 键隔离
- `L_total = L_act + L_visor_aux + α(s)·L_inpaint`（`JointMotModel` 调度 α）
- 底盘解耦：`decouple_base_arm=True` 时屏蔽 `base_motion` flow loss、nav 视觉辅助与 base gate

模块：`gr00t/model/modules/mot/*`，`visor_mot_joint_action_head.py`

---

## 1. 架构总览

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     JointTrainingStep (同一 batch, 同一 step t)          │
├──────────────────────────────┬──────────────────────────────────────────┤
│  Image Branch (FLUX Fill)    │  Action Branch (GR00T N1.7 + VISOR)       │
│  t_img ~ U(0,1) 独立采样      │  t_act ~ Beta(·) 独立采样                  │
│  L_img = MSE(v_pred, v_gt)   │  L_act = MSE(vel_pred, action - noise)    │
│  可训练: FLUX LoRA           │  可训练: projector/DiT/VISOR/MOSS          │
│  冻结: VAE, text encoders    │  冻结: VLM backbone (默认)                 │
└──────────────────────────────┴──────────────────────────────────────────┘
                                    │
                    L_total = α(s)·L_img + β(s)·L_act + L_visor_aux
                    α, β 由 DualBranchSchedule 按 global_step 切换
```

**与当前 v4.2b 的差异**

| 项 | 当前 | 联合训练 |
|----|------|----------|
| FLUX | 独立 `train_flux_fill_lora_rc365.py` | 同一 `training_step` 内前向 |
| 时间步 | 仅 `t_act` | `t_img`、`t_act` 独立 |
| 视觉监督 | 在线 Farneback 光流 → DiT readout | 阶段一主训 FLUX；阶段二 FLUX 弱正则 + 可选 `flux_fill_flow` GT |
| KV 注入 | 无 | **v0.2**：FLUX hidden → Action cross-attn（本版先不做） |

---

## 2. 损失定义

### 2.1 图像分支 \(L_{\text{img}}\)

沿用 `train_flux_fill_lora_rc365.py` 的 Flow Match MSE（对 packed latent velocity）：

- 输入：`(I_t, mask, I_{t+k}, prompt)`，来自与 GR00T 相同的 `(episode, step)`
- `t_img`：每样本独立 `u ~ density_sampling`，映射到 scheduler timesteps
- 仅 **FLUX transformer LoRA** 反传

### 2.2 动作分支 \(L_{\text{act}}\)

沿用 `VisorFlatActionHead` 的 `flow_loss`（velocity MSE，mask 加权）：

- `t_act`：`Gr00tN1d7ActionHead.sample_time()`（Beta），与 `t_img` **解耦**
- VISOR gate 仍用 `refine_active(t_act)`（`t_act >= tau_split`）

### 2.3 VISOR 辅助项 \(L_{\text{visor\_aux}}\)（保留，权重随阶段调节）

```
L_visor_aux = w_tac(s) · L_tactile + w_vis(s) · L_visual
```

- 阶段一：`w_vis` 高（对齐视觉 readout 与光流/FLUX 衍生 GT）
- 阶段二：`w_tac` 正常，`w_vis` 降低（防止抢动作梯度）

### 2.4 总损失

```
L_total = α(s) · L_img + β(s) · L_act + L_visor_aux
```

---

## 3. 两阶段漏斗（默认 30k steps）

| 阶段 | step 范围 | α (`joint_alpha`) | β (`joint_beta`) | 意图 |
|------|-----------|-------------------|------------------|------|
| **一：视觉对齐** | `[0, 0.2·S)` | **1.0** | **0.1** | FLUX LoRA 学刚体/透视/布局；动作头旁听 |
| **二：肌肉记忆** | `[0.2·S, S)` | **0.2** | **2.0** | 动作主导；FLUX 弱正则防遗忘 |

`S = max_steps`（默认 30000）。可通过 `joint_phase1_ratio=0.2` 调整分界。

**与旧 VISOR warmup 的关系**

- `coupling_scale` / `aux_scale`（gate 与 aux  ramp）**保留**，但 tactile/visual aux 在阶段一不应压过 `L_img`
- 建议：`visor_aux_delay_steps` 与阶段边界对齐（6000），或改用 `DualBranchSchedule.visor_aux_scale(step)`

---

## 4. 数据与 Batch 对齐

### 4.1 共享锚点

`ShardedSingleStepDataset.get_datapoint(episode, step)` 已具备：

- `video.*`（历史 4 帧）
- `video_future_manip` @ deltas `[0,5,10,15,20,25,30,35]`
- `action`, `tactile`, `language`

### 4.2 FLUX 字段从同一 step 派生

| FLUX 字段 | GR00T 来源 |
|-----------|------------|
| `I_t` | `video_future_manip` delta=0 或 `video` 最后一帧 `robot0_eye_in_hand` |
| `I_{t+k}` | `video_future_manip` delta=`joint_flux_future_delta`（默认 5） |
| `mask` | `mask_mode=keep_reference`（训练）/ 可配置 |
| `prompt` | `language` 任务描述 |

实现：`gr00t/data/joint_batch.py::build_flux_batch_from_vla_step(metadata)`

### 4.3 时间步独立采样（每 step）

```python
t_act = action_head.sample_time(B, device, dtype)      # (B,) continuous
t_img = sample_flux_timesteps(B, scheduler, mode=...)   # (B,) discrete indices
```

---

## 5. 模块与文件布局

```
gr00t/configs/joint_finetune_config.py      # 联合训练超参（扩展 FinetuneConfig）
gr00t/experiment/joint_train/
  __init__.py
  dual_branch_schedule.py                   # α, β, w_vis, w_tac 调度
  flux_branch.py                            # FLUX 前向 + L_img（从 lora 脚本抽取）
  joint_loss.py                             # 组装 L_total
  joint_trainer.py                          # Gr00tTrainer 子类 / 自定义 loop（TODO）
  joint_model.py                            # 包装 Gr00tN1d7 + FluxBranch（TODO 前向）
gr00t/data/joint_batch.py                   # VLA step → FLUX tensor（TODO）
examples/RoboCasa365/
  finetune_pickplace_joint_dual_branch_30k.sh
  JOINT_DUAL_BRANCH_DESIGN.md               # 本文档
```

### 5.1 训练器策略（推荐 v0.1）

**单进程单卡顺序反传**（实现简单，适合 A800 80G 试探）：

1. `loss_act = model.action_head(...)` → `β · L_act`
2. `loss_img = flux_branch(...)` → `α · L_img`
3. `loss = loss_act + loss_img + L_visor_aux`
4. 一次 `backward()`；优化器 param groups：
   - `flux_lora`: lr_flux
   - `gr00t`: lr（现有 finetune lr）

**OOM 时**：`joint_train_mode=alternate` — 奇偶 step 只训一支（α/β 在当步视为 1）。

### 5.2 需改动的现有文件（实现阶段）

| 文件 | 改动 |
|------|------|
| `visor_flat_action_head.py` | 接受可选外部 `t_act`；`flow_loss` 乘 `joint_beta` 或在 trainer 层乘 |
| `launch_finetune.py` | `--joint-dual-branch` 分支 → `joint_trainer.run()` |
| `trainer.py` | 记录 `L_img`, `alpha`, `beta` 到 log |
| `train_flux_fill_lora_rc365.py` | 核心 step 迁入 `flux_branch.py`（原脚本变 thin wrapper） |

---

## 6. 默认超参（`JointFinetuneConfig`）

```yaml
# 双分支主开关
use_joint_dual_branch: true
joint_flux_model_path: .../FLUX.1-Fill-dev
joint_flux_lora_rank: 8
joint_flux_future_delta: 5          # 与 FLUX 单训一致；VISOR 仍用 8 waypoints
joint_flux_resolution: 256
joint_flux_mask_mode: keep_reference

# 漏斗
joint_phase1_ratio: 0.2
joint_alpha_phase1: 1.0
joint_beta_phase1: 0.1
joint_alpha_phase2: 0.2
joint_beta_phase2: 2.0

# 优化
joint_lr_flux: 1.0e-4
joint_train_mode: simultaneous      # simultaneous | alternate
joint_flux_weight_decay: 0.01

# VISOR aux 与阶段对齐
joint_visor_aux_delay_steps: 6000   # = 0.2 * 30k
joint_visor_visual_weight_phase1: 0.3
joint_visor_visual_weight_phase2: 0.03
```

---

## 7. 实现路线图

| 阶段 | 内容 | 产出 |
|------|------|------|
| **P0（本文）** | 设计 + config + schedule + loss 组装接口 | 可 review 的骨架 |
| **P1** | `joint_batch.py` + `flux_branch.py` 单测 | 固定 batch 上 `L_img` 有限 |
| **P2** | `joint_trainer.py` 接 `ShardedMixtureDataset` | 联合 step 跑通 |
| **P3** | `visor_flat_action_head` 外部 `t_act`；log `visual_loss` | 指标可对标论文 |
| **P4（论文完整）** | FLUX KV → DiT cross-attn；`t_img` 特征注入 IHT | 纠偏能力验证 |

---

## 8. 与当前在跑 30k 的关系

- **不要**中断 `rc365_PickPlaceToasterOvenToCounter_30k_visor_v42_flow`（纯动作+VISOR baseline）
- 联合训练用新 output dir：`output/rc365_PickPlace_joint_dual_branch_30k`
- 可先 **P1**：冻结 GR00T，只验证联合 dataloader + `L_img`；再开全联合

---

## 9. 启动命令（骨架脚本，P2 后可用）

```bash
source /app/bin/proxy.sh
cd Isaac-GR00T
CUDA_VISIBLE_DEVICES=0 bash examples/RoboCasa365/finetune_pickplace_joint_dual_branch_30k.sh
```
