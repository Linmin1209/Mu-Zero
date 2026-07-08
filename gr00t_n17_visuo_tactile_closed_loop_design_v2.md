# 基于 GR00T N1.7 的视觉-触觉闭环移动操作模型改造方案 v2

> 目标：在 **GR00T N1.7 backbone** 基础上，改造 action decoder，使其适配真实 mobile manipulation 场景中的 **移动-操作解耦、触觉接触闭环、错误检测与恢复、未来视觉辅助监督**。
>
> 本版重点修正上一版文档中的张力：
>
> - 不再同时使用「4 组 intent tokens」和「扁平 `num_intent_tokens: 16`」两套表达。
> - 不再使用 `phase_classes=8` 这种容易被误解为人工阶段分类的接口。
> - 明确区分：
>   - **route_probs**：低容量软路由变量，用于专家调度和 contact gate 的先验。
>   - **intent tokens**：高容量连续动作意图流形，用于表达真实 mobile manipulation 的复杂动作意图。
> - `route_probs` 默认 **不需要人工阶段标签**，也不默认使用 cross-entropy 阶段监督。
> - 语义阶段名称只用于 debug / 可视化 / post-hoc interpretation，不是训练标签。

---

## 1. 设计原则

### 1.1 不重写 GR00T N1.7 Backbone

GR00T N1.7 继续作为视觉语言与动作先验主干：

```text
多视角视觉 + 语言 + 机器人状态
        ↓
GR00T N1.7 Backbone
        ↓
h_vlm / action-level hidden states
```

改造重点放在 backbone 之后：

```text
Intent Manifold Adapter
Structured Action DiT
Tactile Late-Denoising Refiner
Execution Monitor
Recovery Expert
```

### 1.2 不把动作意图压缩成少数阶段类别

真实 mobile manipulation 的动作意图远不止：

```text
navigation / pre_grasp / contact / transport / place / recovery
```

这些只能作为粗粒度控制模式，不能作为完整动作意图。

因此本方案采用：

```text
route_probs:
  低容量软路由变量，用于专家门控和接触先验

hierarchical intent tokens:
  高容量连续 latent tokens，用于表达具体动作意图
```

### 1.3 触觉不污染 VLM 主干，只在动作解码和闭环监控中强参与

触觉主要用于：

```text
接触建立
抓取稳定
滑移检测
力控修正
stop-base
错误恢复
```

不建议将触觉早期强行注入 GR00T VLM backbone。

### 1.4 FLUX 只作为训练时未来视觉补全 teacher

FLUX 不参与真实推理，不作为 action decoder 的必需输入。

```text
FLUX:
  training-only teacher

Future Head:
  推理时可选使用的轻量未来特征预测分支
```

### 1.5 闭环能力需要显式建模

只用成功数据做行为克隆不能自然学会错误恢复。

必须加入：

```text
Execution Monitor
Recovery Expert
扰动数据 / 失败状态 / 恢复轨迹
```

---

## 2. 总体架构

```text
MultiModal Batch
  ├── images / videos
  ├── language
  ├── robot_state
  └── tactile_state
        ↓
GR00T N1.7 Backbone
        ↓
Hierarchical Intent Manifold Adapter
  ├── global_intent_tokens
  ├── motion_intent_tokens
  ├── contact_intent_tokens
  ├── recovery_intent_tokens
  └── route_probs
        ↓
Coarse Structured Action DiT
        ↓
a_mid
        ↓
Contact-Gated Tactile Late-Denoising Refiner
        ↓
a_refined
        ↓
Execution Monitor
        ↓
Recovery Expert
        ↓
Safety / Projection Layer
        ↓
Executable Action Chunk
        ↓
Robot Execution
        ↓
Visual + Tactile + State Feedback
        ↺ closed-loop receding-horizon inference
```

训练时辅助分支：

```text
h_vlm + intent tokens
        ↓
Visuo-Tactile Future Head
  ├── future visual tokens
  ├── dynamic / object / contact masks
  ├── contact affordance
  ├── future tactile latent
  ├── slip risk
  └── grasp stability
        ↑
        │ training-only distillation
        │
FLUX Inpainting Teacher
```

---

## 3. Batch 数据结构

```python
from dataclasses import dataclass
from typing import Optional
import torch


@dataclass
class MultiModalRobotBatch:
    images: torch.Tensor
    # [B, T, V, C, H, W]

    language_input: dict
    # tokenizer output or raw text depending on existing GR00T pipeline

    robot_state: torch.Tensor
    # [B, T, state_dim]

    tactile: dict
    # force_torque: Optional[Tensor] [B, T, F]
    # pressure_map: Optional[Tensor] [B, T, C, Ht, Wt]
    # gripper_current: Optional[Tensor] [B, T, 1]
    # slip_signal: Optional[Tensor] [B, T, 1]
    # contact_flag: Optional[Tensor] [B, T, 1]

    actions: torch.Tensor
    # [B, H, action_dim]

    action_groups: dict
    # indices of base / arm / gripper / posture / extra dimensions

    noisy_action: Optional[torch.Tensor] = None
    # [B, H, action_dim], for flow/diffusion training

    diffusion_timestep: Optional[torch.Tensor] = None
    # [B] or [B, 1]

    future_images: Optional[torch.Tensor] = None
    # [B, T_future, V, C, H, W]

    future_masks: Optional[dict] = None
    # dynamic_mask / object_mask / contact_mask

    labels: Optional[dict] = None
    # contact_gate_target
    # slip_risk
    # grasp_stability
    # progress_score
    # error_type
    # recovery_action
```

---

## 4. Action Space 分组

动作必须显式分组，不建议一个 head 预测全部动作维度。

```yaml
action_space:
  action_dim: 32

  groups:
    base:
      indices: [0, 1, 2]
      type: "base_velocity_or_delta_pose"

    arm:
      indices: [3, 4, 5, 6, 7, 8]
      type: "relative_eef_delta"

    gripper:
      indices: [9]
      type: "open_close_or_force"

    posture:
      indices: [10, 11, 12, 13]
      type: "posture_balance_residual"

    extra:
      indices: [14, 31]
      type: "robot_specific"
```

每个动作组对应不同专家和不同 loss 权重。

---

## 5. 模块 1：Hierarchical Intent Manifold Adapter

### 5.1 作用

该模块不是阶段分类器，而是 **高容量连续动作意图建模器**。

它输出四组 intent tokens：

```text
global_intent_tokens:
  任务级目标、场景语义、子目标、导航目标。

motion_intent_tokens:
  局部运动意图，例如接近、绕障、对齐、伸手、抬起、搬运、放置。

contact_intent_tokens:
  接触相关意图，例如接触建立、夹持稳定、滑移修正、释放时机。

recovery_intent_tokens:
  错误恢复意图，例如重新对齐、重新抓取、后退避碰、停止底盘。
```

同时输出：

```text
route_probs:
  learned latent control modes，用于专家软路由和 contact gate 先验。
```

### 5.2 重要澄清

```text
route_probs 不是完整动作意图。
route modes 不需要人工标签。
route modes 不等价于 navigation/pre_grasp/contact 这种离散阶段。
语义阶段名称只用于可视化和调试。
完整动作意图容量来自多组连续 intent tokens。
```

### 5.3 接口

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class HierarchicalIntentManifoldAdapter(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        state_dim: int,
        num_route_modes: int = 16,
        num_global_tokens: int = 8,
        num_motion_tokens: int = 32,
        num_contact_tokens: int = 8,
        num_recovery_tokens: int = 8,
        num_heads: int = 8,
    ):
        super().__init__()

        self.global_queries = nn.Parameter(
            torch.randn(1, num_global_tokens, hidden_dim) * 0.02
        )
        self.motion_queries = nn.Parameter(
            torch.randn(1, num_motion_tokens, hidden_dim) * 0.02
        )
        self.contact_queries = nn.Parameter(
            torch.randn(1, num_contact_tokens, hidden_dim) * 0.02
        )
        self.recovery_queries = nn.Parameter(
            torch.randn(1, num_recovery_tokens, hidden_dim) * 0.02
        )

        self.state_proj = nn.Sequential(
            nn.LayerNorm(state_dim),
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.global_attn = nn.MultiheadAttention(
            hidden_dim, num_heads=num_heads, batch_first=True
        )
        self.motion_attn = nn.MultiheadAttention(
            hidden_dim, num_heads=num_heads, batch_first=True
        )
        self.contact_attn = nn.MultiheadAttention(
            hidden_dim, num_heads=num_heads, batch_first=True
        )
        self.recovery_attn = nn.MultiheadAttention(
            hidden_dim, num_heads=num_heads, batch_first=True
        )

        self.router_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_route_modes),
        )

        self.intent_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        h_vlm: torch.Tensor,
        robot_state: torch.Tensor,
        tactile_tokens: torch.Tensor | None = None,
        contact_gate: torch.Tensor | None = None,
    ) -> dict:
        """
        Args:
            h_vlm: [B, N, D]
            robot_state: [B, T, state_dim]
            tactile_tokens: optional [B, Nt, D]
            contact_gate: optional [B, 1] or [B, 1, 1]

        Returns:
            {
              "global_intent_tokens": [B, Kg, D],
              "motion_intent_tokens": [B, Km, D],
              "contact_intent_tokens": [B, Kc, D],
              "recovery_intent_tokens": [B, Kr, D],
              "all_intent_tokens": [B, Kg+Km+Kc+Kr, D],
              "route_logits": [B, R],
              "route_probs": [B, R],
              "attn": {...}
            }
        """
        B = h_vlm.shape[0]

        state_summary = robot_state[:, -1]
        state_token = self.state_proj(state_summary).unsqueeze(1)

        base_context = torch.cat([h_vlm, state_token], dim=1)

        global_q = self.global_queries.expand(B, -1, -1)
        motion_q = self.motion_queries.expand(B, -1, -1)

        global_tokens, global_attn = self.global_attn(
            global_q, base_context, base_context
        )
        motion_tokens, motion_attn = self.motion_attn(
            motion_q, base_context, base_context
        )

        contact_context = base_context
        if tactile_tokens is not None:
            contact_context = torch.cat([base_context, tactile_tokens], dim=1)

        contact_q = self.contact_queries.expand(B, -1, -1)
        contact_tokens, contact_attn = self.contact_attn(
            contact_q, contact_context, contact_context
        )

        if contact_gate is not None:
            if contact_gate.dim() == 2:
                contact_gate = contact_gate[:, None, :]
            contact_tokens = contact_gate * contact_tokens

        recovery_q = self.recovery_queries.expand(B, -1, -1)
        recovery_tokens, recovery_attn = self.recovery_attn(
            recovery_q, contact_context, contact_context
        )

        all_tokens = torch.cat(
            [
                global_tokens,
                motion_tokens,
                contact_tokens,
                recovery_tokens,
            ],
            dim=1,
        )
        all_tokens = self.intent_norm(all_tokens)

        pooled = all_tokens.mean(dim=1)
        route_logits = self.router_head(pooled)
        route_probs = F.softmax(route_logits, dim=-1)

        return {
            "global_intent_tokens": global_tokens,
            "motion_intent_tokens": motion_tokens,
            "contact_intent_tokens": contact_tokens,
            "recovery_intent_tokens": recovery_tokens,
            "all_intent_tokens": all_tokens,
            "route_logits": route_logits,
            "route_probs": route_probs,
            "attn": {
                "global": global_attn,
                "motion": motion_attn,
                "contact": contact_attn,
                "recovery": recovery_attn,
            },
        }
```

### 5.4 Router 训练方式

默认不使用人工阶段标签。

Route modes 通过以下信号学习：

```text
下游 action loss
contact consistency
temporal smoothness
batch balance
optional high-confidence pseudo labels
```

#### Router loss

```python
def compute_router_losses(
    route_logits,
    route_probs,
    pseudo_targets=None,
    confidence=None,
    prev_route_probs=None,
    contact_gate=None,
    contact_route_indices=(2, 3),
    ignore_index=-100,
):
    losses = {}

    if pseudo_targets is not None:
        ce = F.cross_entropy(
            route_logits,
            pseudo_targets,
            ignore_index=ignore_index,
            reduction="none",
        )
        valid = pseudo_targets != ignore_index
        if confidence is not None:
            ce = ce * confidence
        losses["router_pseudo_ce"] = (
            ce[valid].mean() if valid.any() else route_logits.sum() * 0.0
        )

    if prev_route_probs is not None:
        losses["router_temporal_smooth"] = F.mse_loss(
            route_probs,
            prev_route_probs.detach(),
        )

    mean_prob = route_probs.mean(dim=0)
    uniform = torch.ones_like(mean_prob) / mean_prob.numel()
    losses["router_balance"] = F.kl_div(
        (mean_prob + 1e-6).log(),
        uniform,
        reduction="batchmean",
    )

    if contact_gate is not None:
        contact_prob = route_probs[:, list(contact_route_indices)].sum(
            dim=-1, keepdim=True
        )
        losses["router_contact_consistency"] = F.binary_cross_entropy(
            contact_prob.clamp(1e-5, 1 - 1e-5),
            contact_gate.detach().clamp(0, 1),
        )

    return losses
```

建议权重：

```yaml
router_loss:
  pseudo_ce: 0.0          # 默认关闭，避免退化成阶段分类器
  temporal_smooth: 0.01
  balance: 0.005
  contact_consistency: 0.05
```

---

## 6. 模块 2：Tactile Encoder

### 6.1 作用

编码触觉历史，得到：

```text
tactile_tokens
tactile_summary
contact_logits
slip_logits
force_summary
```

由于当前仿真验证中视频和触觉同频，本方案不强调高频触觉，而强调：

```text
触觉在接触阶段参与 late denoising 和 error recovery。
```

### 6.2 接口

```python
class TactileEncoder(nn.Module):
    def __init__(
        self,
        tactile_dim: int,
        hidden_dim: int,
        use_pressure_map: bool = False,
        history_len: int = 16,
    ):
        super().__init__()
        ...

    def forward(self, tactile: dict) -> dict:
        """
        Returns:
            {
              "tactile_tokens": [B, Nt, D],
              "tactile_summary": [B, D],
              "contact_logits": [B, 1],
              "slip_logits": [B, 1],
              "force_summary": [B, D]
            }
        """
```

---

## 7. 模块 3：Contact Gate

### 7.1 作用

计算接触门控：

```text
g_contact ∈ [0, 1]
```

用于控制 tactile refiner 和 contact intent tokens 的激活强度。

### 7.2 接口

```python
class ContactGate(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        ...

    def forward(
        self,
        tactile_summary: torch.Tensor,
        robot_state: torch.Tensor,
        route_probs: torch.Tensor,
        sim_contact_flag: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Returns:
            g_contact: [B, 1] or [B, H, 1]
        """
```

### 7.3 说明

这里使用 `route_probs`，而不是 `phase_probs`。

```text
route_probs:
  learned latent control modes
  not manually labeled phase probabilities
```

---

## 8. 模块 4：Coarse Structured Action DiT

### 8.1 作用

生成中间动作：

```text
a_mid
```

主要依赖：

```text
h_vlm
global_intent_tokens
motion_intent_tokens
robot_state
optional stop-grad future tokens
```

### 8.2 专家结构

```text
Shared Self-Attention
        ↓
Group-specific FFN / Heads
        ├── Base Expert
        ├── Arm Expert
        ├── Gripper Expert
        ├── Posture Expert
        └── Coupling Expert
```

### 8.3 不同专家读取不同 intent tokens

```text
Base Expert:
  global_intent_tokens + motion_intent_tokens + route_probs

Arm Expert:
  motion_intent_tokens + contact_intent_tokens

Gripper Expert:
  contact_intent_tokens + tactile summary

Posture Expert:
  global_intent_tokens + contact_intent_tokens + robot_state

Coupling Expert:
  all_intent_tokens + robot_state
```

### 8.4 接口

```python
class StructuredActionDiT(nn.Module):
    def forward(
        self,
        noisy_action: torch.Tensor,
        timestep: torch.Tensor,
        h_vlm: torch.Tensor,
        global_intent_tokens: torch.Tensor,
        motion_intent_tokens: torch.Tensor,
        contact_intent_tokens: torch.Tensor,
        recovery_intent_tokens: torch.Tensor,
        route_probs: torch.Tensor,
        robot_state: torch.Tensor,
        future_tokens: torch.Tensor | None = None,
    ) -> dict:
        """
        Returns:
            {
              "a_mid": [B, H, action_dim],
              "group_outputs": {...},
              "attn_maps": optional
            }
        """
```

---

## 9. 模块 5：Contact-Gated Tactile Late-Denoising Refiner

### 9.1 作用

在 denoising 后半段引入触觉，对 `a_mid` 做接触残差修正。

```text
a_refined = a_mid + g_contact · Δa_tactile
```

这里借鉴 T-Rex 的 late tactile refinement 思想，但不依赖触觉与视觉的频率差异。

### 9.2 接口

```python
class TactileLateDenoisingRefiner(nn.Module):
    def forward(
        self,
        a_mid: torch.Tensor,
        timestep: torch.Tensor,
        tactile_tokens: torch.Tensor,
        h_vlm_cache: torch.Tensor,
        global_intent_tokens: torch.Tensor,
        motion_intent_tokens: torch.Tensor,
        contact_intent_tokens: torch.Tensor,
        recovery_intent_tokens: torch.Tensor,
        route_probs: torch.Tensor,
        robot_state: torch.Tensor,
        contact_gate: torch.Tensor,
    ) -> dict:
        """
        Returns:
            {
              "delta_tactile": [B, H, action_dim],
              "a_refined": [B, H, action_dim],
              "attn_maps": optional
            }
        """
```

### 9.3 触觉修正权重

```yaml
tactile_refiner:
  group_scales:
    base_navigation: 0.1
    base_stop: 0.5
    arm: 1.0
    gripper: 1.5
    posture: 0.5
    recovery: 1.0
```

---

## 10. 模块 6：VT-WAM 风格 Contact-Gated AVTAG Loss

### 10.1 作用

训练时引导 action tokens 在接触阶段关注触觉 tokens，防止强视觉 backbone 让模型忽略触觉。

### 10.2 公式

```text
L_AVTAG = g_contact · max(0, p_vis - p_tac + margin)
```

其中：

```text
p_vis:
  action query 对 visual tokens 的注意力总量

p_tac:
  action query 对 tactile tokens 的注意力总量
```

### 10.3 分组约束

不要对所有 action token 一视同仁。

```yaml
avtag:
  enabled: true
  margin: 0.05
  weight: 0.02
  group_weights:
    arm: 1.0
    gripper: 1.5
    recovery: 1.5
    posture: 0.5
    base_stop: 0.5
    base_navigation: 0.1
```

---

## 11. 模块 7：Future Head + FLUX Training-Only Teacher

### 11.1 作用

Future Head 预测未来视觉-触觉相关表征。

FLUX 只作为 masked future visual supervision teacher。

### 11.2 Future Head 输出

```text
future_tokens
future_visual_latent
dynamic_mask_logits
object_mask_logits
contact_mask_logits
contact_affordance_logits
future_tactile_latent
slip_logits
grasp_stability_logits
```

### 11.3 FLUX 原则

```text
FLUX 不参与推理。
FLUX 不进入闭环。
FLUX 不作为 action decoder 的必需输入。
FLUX 只对 dynamic/object/contact mask 区域提供训练监督。
```

---

## 12. 模块 8：Execution Monitor

### 12.1 作用

判断当前执行是否正常。

输出：

```text
progress_score
error_logits
recovery_gate
```

### 12.2 错误类型

```python
ERROR_TYPES = [
    "none",
    "no_contact_when_expected",
    "unexpected_contact",
    "slip_detected",
    "grasp_unstable",
    "object_lost",
    "force_too_large",
    "base_misaligned",
    "collision_risk",
    "posture_unstable",
    "progress_stalled",
]
```

---

## 13. 模块 9：Recovery Expert

### 13.1 作用

当 Execution Monitor 判断异常时，输出恢复 residual action。

```text
a_final = a_refined + recovery_gate · Δa_recovery
```

### 13.2 接口

```python
class RecoveryExpert(nn.Module):
    def forward(
        self,
        error_logits: torch.Tensor,
        recovery_gate: torch.Tensor,
        h_vlm: torch.Tensor,
        global_intent_tokens: torch.Tensor,
        motion_intent_tokens: torch.Tensor,
        contact_intent_tokens: torch.Tensor,
        recovery_intent_tokens: torch.Tensor,
        tactile_tokens: torch.Tensor,
        robot_state: torch.Tensor,
        a_refined: torch.Tensor,
    ) -> dict:
        """
        Returns:
            {
              "delta_recovery": [B, H, action_dim],
              "a_recovered": [B, H, action_dim]
            }
        """
```

---

## 14. 总模型封装

```python
class GR00TN17VisuoTactileClosedLoopPolicy(nn.Module):
    def __init__(self, gr00t_backbone, cfg):
        super().__init__()

        self.backbone = gr00t_backbone

        self.tactile_encoder = TactileEncoder(...)
        self.contact_gate = ContactGate(...)
        self.intent_adapter = HierarchicalIntentManifoldAdapter(...)

        self.future_head = VisuoTactileFutureHead(...)

        self.coarse_action_dit = StructuredActionDiT(...)
        self.tactile_refiner = TactileLateDenoisingRefiner(...)

        self.execution_monitor = ExecutionMonitor(...)
        self.recovery_expert = RecoveryExpert(...)

        self.safety_projector = SafetyProjector(...)

    def forward(self, batch, mode="train"):
        tactile_out = self.tactile_encoder(batch.tactile)

        backbone_out = self.backbone(
            images=batch.images,
            language=batch.language_input,
            robot_state=batch.robot_state,
        )
        h_vlm = backbone_out["hidden_states"]

        # First pass: get route prior without contact-gated contact tokens.
        intent_out_pre = self.intent_adapter(
            h_vlm=h_vlm,
            robot_state=batch.robot_state,
            tactile_tokens=tactile_out["tactile_tokens"],
            contact_gate=None,
        )

        contact_gate = self.contact_gate(
            tactile_summary=tactile_out["tactile_summary"],
            robot_state=batch.robot_state,
            route_probs=intent_out_pre["route_probs"],
            sim_contact_flag=batch.tactile.get("contact_flag", None),
        )

        # Second pass: activate contact intent tokens with contact gate.
        intent_out = self.intent_adapter(
            h_vlm=h_vlm,
            robot_state=batch.robot_state,
            tactile_tokens=tactile_out["tactile_tokens"],
            contact_gate=contact_gate,
        )

        future_out = None
        future_tokens = None
        if self.training or self.cfg.use_future_tokens_at_inference:
            future_out = self.future_head(
                h_vlm=h_vlm,
                intent_tokens=intent_out["all_intent_tokens"],
                robot_state=batch.robot_state,
            )
            if self.cfg.action_decoder.use_future_tokens:
                future_tokens = future_out["future_tokens"].detach()

        coarse_out = self.coarse_action_dit(
            noisy_action=batch.noisy_action,
            timestep=batch.diffusion_timestep,
            h_vlm=h_vlm,
            global_intent_tokens=intent_out["global_intent_tokens"],
            motion_intent_tokens=intent_out["motion_intent_tokens"],
            contact_intent_tokens=intent_out["contact_intent_tokens"],
            recovery_intent_tokens=intent_out["recovery_intent_tokens"],
            route_probs=intent_out["route_probs"],
            robot_state=batch.robot_state,
            future_tokens=future_tokens,
        )

        tactile_refine_out = self.tactile_refiner(
            a_mid=coarse_out["a_mid"],
            timestep=batch.diffusion_timestep,
            tactile_tokens=tactile_out["tactile_tokens"],
            h_vlm_cache=h_vlm.detach(),
            global_intent_tokens=intent_out["global_intent_tokens"],
            motion_intent_tokens=intent_out["motion_intent_tokens"],
            contact_intent_tokens=intent_out["contact_intent_tokens"],
            recovery_intent_tokens=intent_out["recovery_intent_tokens"],
            route_probs=intent_out["route_probs"],
            robot_state=batch.robot_state,
            contact_gate=contact_gate,
        )

        monitor_out = self.execution_monitor(
            h_vlm=h_vlm,
            tactile_tokens=tactile_out["tactile_tokens"],
            robot_state=batch.robot_state,
            predicted_future_tokens=future_tokens,
            executed_action=getattr(batch, "executed_action", None),
        )

        recovery_out = self.recovery_expert(
            error_logits=monitor_out["error_logits"],
            recovery_gate=monitor_out["recovery_gate"],
            h_vlm=h_vlm,
            global_intent_tokens=intent_out["global_intent_tokens"],
            motion_intent_tokens=intent_out["motion_intent_tokens"],
            contact_intent_tokens=intent_out["contact_intent_tokens"],
            recovery_intent_tokens=intent_out["recovery_intent_tokens"],
            tactile_tokens=tactile_out["tactile_tokens"],
            robot_state=batch.robot_state,
            a_refined=tactile_refine_out["a_refined"],
        )

        action = self.safety_projector(
            recovery_out["a_recovered"],
            robot_state=batch.robot_state,
        )

        return {
            "action": action,
            "h_vlm": h_vlm,
            "tactile": tactile_out,
            "intent": intent_out,
            "future": future_out,
            "coarse": coarse_out,
            "tactile_refine": tactile_refine_out,
            "monitor": monitor_out,
            "recovery": recovery_out,
            "contact_gate": contact_gate,
        }
```

---

## 15. 总损失

```python
loss = (
    cfg.loss.action * loss_action
    + cfg.loss.future * loss_future
    + cfg.loss.flux * loss_flux
    + cfg.loss.mask * loss_mask
    + cfg.loss.contact * loss_contact
    + cfg.loss.tactile * loss_tactile
    + cfg.loss.avtag * loss_avtag
    + cfg.loss.router * loss_router
    + cfg.loss.intent_diversity * loss_intent_diversity
    + cfg.loss.monitor * loss_monitor
    + cfg.loss.recovery * loss_recovery
    + cfg.loss.smooth * loss_smooth
    + cfg.loss.safe * loss_safe
)
```

### 15.1 Intent diversity loss

防止高容量 intent tokens 塌缩成重复表达。

```python
def intent_token_diversity_loss(tokens: torch.Tensor) -> torch.Tensor:
    """
    tokens: [B, K, D]
    """
    tokens = F.normalize(tokens, dim=-1)
    sim = torch.matmul(tokens, tokens.transpose(-1, -2))

    K = tokens.shape[1]
    eye = torch.eye(K, device=tokens.device).unsqueeze(0)
    off_diag = sim * (1.0 - eye)

    return off_diag.pow(2).mean()
```

---

## 16. 训练阶段

### Stage 1：最小行为克隆恢复

训练：

```text
Hierarchical Intent Manifold Adapter
Coarse Structured Action DiT
```

关闭：

```text
router pseudo CE
AVTAG
FLUX
Recovery Expert
```

### Stage 2：加入触觉编码和 Contact Gate

训练：

```text
Tactile Encoder
Contact Gate
Tactile Late-Denoising Refiner
```

### Stage 3：加入 Future Head + FLUX 监督

训练：

```text
Future Head
mask heads
contact affordance
future tactile latent
```

FLUX：

```text
使用离线缓存 target
只训练，不推理
```

### Stage 4：加入 Contact-Gated AVTAG

训练：

```text
AVTAG loss
```

只对：

```text
arm
gripper
recovery
posture
base_stop
```

施加强触觉注意力引导。

### Stage 5：加入扰动数据和 Recovery

训练：

```text
Execution Monitor
Recovery Expert
```

数据增强：

```text
base misalignment
gripper miss
slip
unexpected contact
force too large
object lost
target occlusion
collision risk
```

### Stage 6：Rollout-aware 微调

逐步从 GT future tokens 切换到 predicted future tokens。

```yaml
rollout_schedule:
  early:
    gt_future_ratio: 0.8
    pred_future_ratio: 0.2
  mid:
    gt_future_ratio: 0.5
    pred_future_ratio: 0.5
  late:
    gt_future_ratio: 0.2
    pred_future_ratio: 0.8
```

---

## 17. 推理流程

```text
1. 读取视觉、语言、状态、触觉
2. GR00T N1.7 提取 h_vlm
3. Tactile Encoder 编码触觉
4. Intent Adapter 生成多组 intent tokens 和 route_probs
5. Contact Gate 判断触觉修正强度
6. Structured Action DiT 输出 a_mid
7. Tactile Refiner 输出 a_refined
8. Monitor 判断是否异常
9. Recovery Expert 产生恢复 residual
10. Safety Projector 输出可执行 action chunk
11. 只执行前 k 步
12. 重新观测并进入下一轮闭环
```

```yaml
inference:
  action_horizon: 16
  execute_steps: 2-4
  use_flux: false
  use_future_tokens: true
  use_recovery: true
  use_safety_projector: true
```

---

## 18. YAML 配置模板

本节与 §5 保持一致：使用层次化 intent tokens，不再使用扁平 `num_intent_tokens` 或 `phase_classes`。

```yaml
model:
  name: "gr00t_n17_visuo_tactile_closed_loop_v2"
  backbone: "gr00t_n1_7"
  freeze_backbone: true
  tune_vlm_lora: false
  hidden_dim: 1024

intent_adapter:
  enabled: true
  type: "hierarchical_intent_manifold"
  num_route_modes: 16

  num_global_tokens: 8
  num_motion_tokens: 32
  num_contact_tokens: 8
  num_recovery_tokens: 8

  route_supervision:
    use_manual_phase_labels: false
    use_pseudo_labels: false
    use_contact_consistency: true
    use_temporal_smoothness: true
    use_batch_balance: true

  route_loss:
    pseudo_ce: 0.0
    contact_consistency: 0.05
    temporal_smooth: 0.01
    batch_balance: 0.005

  intent_regularization:
    diversity: 0.01
    temporal_smooth: 0.01

tactile_encoder:
  enabled: true
  history_len: 16
  use_force_torque: true
  use_pressure_map: false
  use_gripper_current: true
  use_slip_signal: true
  hidden_dim: 1024

contact_gate:
  enabled: true
  use_sim_contact_flag: true
  soft_gate: true
  input_route_probs: true

action_decoder:
  type: "structured_dit"
  num_layers: 12
  num_heads: 16
  use_future_tokens: false
  detach_future_tokens: true

  action_groups:
    base:
      weight: 1.0
    arm:
      weight: 1.0
    gripper:
      weight: 2.0
    posture:
      weight: 0.5

tactile_refiner:
  enabled: true
  type: "contact_gated_late_denoising"
  num_layers: 4
  num_heads: 8
  contact_residual_scale: 1.0

  group_scales:
    base_navigation: 0.1
    base_stop: 0.5
    arm: 1.0
    gripper: 1.5
    posture: 0.5
    recovery: 1.0

future_head:
  enabled: true
  use_flux_teacher: true
  flux_training_only: true
  predict_visual_future: true
  predict_contact_mask: true
  predict_tactile_future: true
  predict_slip: true
  predict_grasp_stability: true

avtag:
  enabled: true
  margin: 0.05
  weight: 0.02
  group_weights:
    arm: 1.0
    gripper: 1.5
    recovery: 1.5
    posture: 0.5
    base_stop: 0.5
    base_navigation: 0.1

monitor:
  enabled: true
  error_types:
    - none
    - no_contact_when_expected
    - unexpected_contact
    - slip_detected
    - grasp_unstable
    - object_lost
    - force_too_large
    - base_misaligned
    - collision_risk
    - posture_unstable
    - progress_stalled

recovery:
  enabled: true
  residual_scale: 1.0

loss:
  action: 1.0
  future: 0.1
  flux: 0.05
  mask: 0.1
  contact: 0.2
  tactile: 0.1
  avtag: 0.02
  router: 0.05
  intent_diversity: 0.01
  monitor: 0.2
  recovery: 0.5
  smooth: 0.01
  safe: 0.01

training:
  stage: 1
  batch_size: 128
  lr: 3.0e-5
  weight_decay: 1.0e-5
  bf16: true

inference:
  action_horizon: 16
  execute_steps: 4
  use_flux: false
  use_future_tokens: true
  use_recovery: true
  use_safety_projector: true
```

---

## 19. 必做消融实验

```text
A0: GR00T N1.7 baseline
A1: + Hierarchical Intent Manifold Adapter
A2: + Structured Action DiT
A3: + Tactile Encoder early fusion
A4: + Tactile Late-Denoising Refiner
A5: + Contact Gate
A6: + Contact-Gated AVTAG
A7: + Future Head
A8: + Future Head + FLUX teacher
A9: + Execution Monitor
A10: + Recovery Expert
A11: Full model
```

指标：

```text
success rate
progress score
contact success rate
grasp stability
slip count
force overshoot
recovery success rate
base-stop accuracy
action smoothness
collision rate
posture stability
tactile attention ratio during contact
route entropy
intent token diversity
```

---

## 20. Cursor 执行优先级

```text
1. 扩展 batch schema，加入 tactile、action_groups、future labels、error labels
2. 实现 TactileEncoder
3. 实现 HierarchicalIntentManifoldAdapter
4. 实现 ContactGate
5. 实现 StructuredActionDiT 的 group heads
6. 实现 TactileLateDenoisingRefiner
7. 实现 grouped action loss
8. 实现 router regularization，不启用 manual CE
9. 实现 intent diversity loss
10. 实现 AVTAG loss
11. 实现 FutureHead，先不接 FLUX
12. 接入离线 FLUX target loss
13. 实现 ExecutionMonitor
14. 实现 RecoveryExpert
15. 实现 closed-loop inference wrapper
16. 添加 configs 和 ablation flags
```

---

## 21. Cursor 实现约束

```text
1. 不要重写 GR00T N1.7 backbone。
2. 不要使用 phase_classes=8 作为主接口。
3. 不要要求人工阶段标签。
4. 不要把 route_probs 当作完整动作意图。
5. 使用 hierarchical intent tokens 表达高容量连续动作意图。
6. FLUX 只允许在训练 loss 中使用，禁止推理依赖 FLUX。
7. 触觉不要污染 VLM backbone，只进入 tactile encoder、contact gate、late refiner 和 monitor。
8. AVTAG 是 training-only loss，不改变推理计算图。
9. action 必须按 base / arm / gripper / posture 分组处理。
10. Recovery Expert 输出 residual action，不直接替代主策略。
11. 推理时执行 partial chunk，然后重新观测，形成闭环。
```

---

## 22. 最小可行版本 MVP

第一版先实现：

```text
GR00T N1.7
+ TactileEncoder
+ HierarchicalIntentManifoldAdapter
+ ContactGate
+ StructuredActionDiT
+ TactileLateDenoisingRefiner
```

暂时关闭：

```text
FLUX
Future Head
AVTAG
Execution Monitor
Recovery Expert
```

第二版再加入：

```text
AVTAG
Future Head + FLUX teacher
Execution Monitor
Recovery Expert
```

---

## 23. 本版文档相对上一版的关键修正

```text
1. 删除扁平 num_intent_tokens: 16 的 MVP 表述。
2. 删除 phase_classes: 8 的主接口表述。
3. 将 phase_probs 改为 route_probs。
4. 明确 route_probs 是 learned latent control modes，不是人工阶段分类。
5. YAML 与 §5 层次化流形保持一致。
6. ContactGate 使用 route_probs 作为软先验，而不是 phase_probs。
7. Router loss 默认不启用 pseudo CE，避免退化成阶段分类器。
8. 增加 intent diversity regularization，防止高容量 token 塌缩。
9. 明确 route names 只用于 post-hoc interpretation。
10. 明确完整动作意图容量来自 global/motion/contact/recovery 四组连续 tokens。
```
