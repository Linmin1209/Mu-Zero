# VT Closed-Loop Policy：基于 GR00T N1.7 的视觉-触觉闭环改造

> **状态**：设计 + MVP 骨架 + `VTClosedLoopActionHead`（**不用 VISOR**，**保留 MOSS**）  
> **原则**：不重写 GR00T N1.7 backbone / MOSS；**替换** VISOR 触觉 action decoder。

---

## 0. 架构决策（2026-07 更新）

| 组件 | 本方案 | 说明 |
|------|--------|------|
| **VLM + MOSS** | ✅ **保留** | `use_motion=True`, `motion_use_gating=True`, `motion_block` + `MotionFusionGate` |
| **VISOR** | ❌ **不用** | 不用 `VisorFlat/Factored/MotJointActionHead`、IHT、split gates、visor aux loss |
| **VT 触觉栈** | ✅ **替代 VISOR** | `TactileEncoder` → `ContactGate` → `TactileLateDenoisingRefiner` |
| **动作解码** | ✅ **StructuredActionDiT** | base/arm/gripper 分组 + asymmetric SA mask |
| **MoT + FLUX joint** | ⚪ 可选后续 | 与 VISOR-MoT **独立**；VT 路径默认 `use_joint_dual_branch=False` |

互斥规则（config 强制）：

```text
use_vt_closed_loop=True  →  use_visor=False, use_joint_dual_branch=False
use_motion=True          →  MOSS 仍在 vision layer 9 注入（与是否 VT 无关）
```

---

## 1. 与旧方案对比

```
【旧】GR00T + MOSS + VISOR
  Backbone(MOSS) → VisorFlatActionHead
                     ├── IHT tactile tokens 进 DiT
                     ├── flow-late split gates (arm/base/hand)
                     └── tactile/visual aux losses

【新】GR00T + MOSS + VT Closed-Loop
  Backbone(MOSS) → VTClosedLoopActionHead
                     ├── IntentManifoldAdapter
                     ├── TactileEncoder（非 VISOR IHT）
                     ├── StructuredActionDiT（分组 velocity）
                     ├── TactileLateDenoisingRefiner（g_contact · Δa）
                     ├── Monitor / Recovery（可选）
                     └── 无 visor_gate_mode / visor_loss_*
```

---

## 2. 目标数据流

```
MultiModal Batch (images, language, state, tactile, actions)
        ↓
GR00T N1.7 Backbone + MOSS (4-frame video, layer 9)
        ↓
AlternateVLDiT [state | action_tokens]  (+ base/arm SA mask)
        ↓
VTClosedLoopPolicy.forward_stages()
  Intent → TactileEncoder → ContactGate
  → StructuredActionDiT → TactileLateRefiner → Monitor → Recovery
        ↓
flow_loss (velocity MSE, grouped mask)
```

训练辅助（可选，不进默认推理）：

```
FutureHead + offline FLUX teacher
AVTAG (training-only)
```

---

## 3. 代码布局

```
gr00t/model/modules/vt_closed_loop/
  vt_closed_loop_action_head.py   ← 替代 Visor*ActionHead
  closed_loop_policy.py
  tactile_encoder.py
  structured_action_dit.py
  ...

gr00t/configs/vt_closed_loop_config.py
  use_motion=True (default)
  use_visor=False (forced when VT on)

gr00t/model/gr00t_n1d7/gr00t_n1d7.py
  use_vt_closed_loop → VTClosedLoopActionHead (优先于 use_visor)
```

---

## 4. RoboCasa365 动作分组（11 维）

| Group | Indices |
|-------|---------|
| gripper | `[0, 1)` |
| arm | `[1, 7)` |
| base | `[7, 11)` |

`decouple_base_arm=True`：base 组不参与 flow loss / refiner 强修正。

---

## 5. 训练阶段

| Stage | 训练 | 冻结 |
|-------|------|------|
| 1 | Intent + StructuredActionDiT | VLM, refiner |
| 2 | TactileEncoder + ContactGate + Refiner | VLM |
| 3 | FutureHead + FLUX cache | FLUX 权重 |
| 4 | AVTAG | — |
| 5 | Monitor + Recovery | — |
| 6 | Rollout future schedule | — |

**MOSS**：Stage 1 起默认与 action head 同训（`tune_motion=True`）；VLM 仍冻结。

---

## 6. 实现原则

1. **不重写** Qwen3-VL backbone；**保留 MOSS**。
2. **禁用 VISOR** 与 VT 同开；不 import `VisorModule` 于 VT head。
3. 触觉 **不** 进 VLM；只进 VT tactile 栈。
4. FLUX **仅训练**；推理 `use_flux=false`。
5. Recovery 为 **residual**；partial-chunk 闭环推理。
6. MoT joint（VisorMotJoint）为 **legacy 路径**，VT 方案不依赖。

---

## 7. 启动示例（计划）

```bash
# VT + MOSS，无 VISOR
USE_MOTION=1 TUNE_MOTION=1 MOTION_USE_GATING=1 \
  .venv/bin/python gr00t/experiment/launch_finetune.py \
  --use-vt-closed-loop --no-use-visor \
  --use-motion --tune-motion --motion-use-gating \
  ...
```

---

## 8. 参考（只读，VT 不接入）

- 旧 VISOR：`visor.py`, `visor_flat_action_head.py`（**VT 替代**）
- MOSS：`qwen3_motion.py`, `motion_gate_init_bias`
- 语义分析：`examples/RoboCasa365/analysis/task_semantics/`
