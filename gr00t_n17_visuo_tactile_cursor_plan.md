# 基于 GR00T N1.7 的视觉-触觉闭环移动操作模型：Cursor 改造方案

## 0. 改造目标

在现有 **GR00T N1.7 backbone** 基础上，增加一个面向移动操作任务的结构化动作解码框架，使模型支持：

1. **多模态输入**：视觉、语言、机器人状态、触觉/力觉。
2. **动作意图流形**：将高层语义映射成连续技能阶段 tokens。
3. **结构化 Action DiT**：将动作拆成 `base / arm / gripper / posture` 等子空间分别解码。
4. **触觉后段去噪**：在 denoising 后半段引入触觉信息，做接触残差修正。
5. **VT-WAM 式接触门控注意力引导**：在接触阶段强制 action tokens 更关注触觉。
6. **Future Head + FLUX 训练监督**：仅训练时使用 FLUX 作为局部未来视觉补全 teacher。
7. **Execution Monitor + Recovery Expert**：让模型具备错误检测和恢复能力。
8. **闭环推理**：执行 partial action chunk 后重新观测，根据视觉/触觉/状态反馈滚动预测或触发恢复。

---

## 1. 总体架构

```text
MultiModal Batch
  ├── images / videos
  ├── language
  ├── robot_state
  └── tactile_state
        ↓
GR00T N1.7 Backbone
        ↓
Intent Manifold Adapter
        ↓
Coarse Structured Action DiT
        ↓
Tactile Late-Denoising Refiner
        ↓
Execution Monitor + Recovery Expert
        ↓
Safety / Projection Layer
        ↓
Executable Action Chunk
```

训练时额外使用：

```text
Future Head
  ├── future visual feature
  ├── dynamic / object / contact mask
  ├── tactile future latent
  ├── slip risk
  └── grasp stability

FLUX Inpainting Teacher
  └── only used for masked future visual supervision
```

核心原则：

- 不重写 GR00T N1.7 backbone。
- 优先在 action decoder 外围加 adapter、refiner、monitor。
- FLUX 只用于训练监督，不参与真实推理。
- 触觉不直接污染 VLM backbone，只进入 tactile encoder、contact gate、late refiner 和 monitor。
- Action 必须按子空间分组处理。
- Recovery Expert 输出 residual action，不直接替代主策略。
- 推理时执行 partial chunk，再重新观测，形成闭环。

---

## 2. 推荐新增文件结构

如果 repo 结构允许，请新增以下文件。若实际项目目录不同，请按现有结构放到对应 `models/`、`modules/`、`losses/`、`trainers/` 下。

```text
models/
  gr00t_vt_closed_loop_policy.py

models/modules/
  intent_manifold_adapter.py
  tactile_encoder.py
  contact_gate.py
  structured_action_dit.py
  tactile_late_denoising_refiner.py
  future_head.py
  execution_monitor.py
  recovery_expert.py
  safety_projector.py

losses/
  action_losses.py
  future_losses.py
  avtag_loss.py
  monitor_recovery_losses.py

data/
  multimodal_batch.py
  tactile_transforms.py
  error_augmentation.py

train/
  train_stage1_intent_action.py
  train_stage2_tactile_refiner.py
  train_stage3_future_flux.py
  train_stage4_avtag.py
  train_stage5_recovery.py

configs/
  gr00t_n17_vt_closed_loop.yaml
```

---

## 3. Batch 数据结构

扩展 dataloader 输出，支持触觉、动作分组和错误恢复标签。

```python
from dataclasses import dataclass
from typing import Optional, Dict, Any
import torch

@dataclass
class MultiModalRobotBatch:
    images: torch.Tensor
    # [B, T, V, C, H, W]
    # V = number of camera views

    language_input: Dict[str, Any]
    # tokenizer output or raw text depending on existing GR00T pipeline

    robot_state: torch.Tensor
    # [B, T, state_dim]
    # includes joint, eef pose, gripper qpos, base pose, imu, etc.

    tactile: Dict[str, torch.Tensor]
    # force_torque: [B, T, F]
    # pressure_map: optional [B, T, C, Ht, Wt]
    # gripper_current: optional [B, T, 1]
    # slip_signal: optional [B, T, 1]
    # contact_flag: optional [B, T, 1]

    actions: torch.Tensor
    # [B, H, action_dim]

    action_groups: Dict[str, torch.Tensor]
    # group indices:
    # base_idx, arm_idx, gripper_idx, posture_idx

    future_images: Optional[torch.Tensor] = None
    # [B, T_future, V, C, H, W]

    future_masks: Optional[Dict[str, torch.Tensor]] = None
    # dynamic_mask, object_mask, contact_mask

    labels: Optional[Dict[str, torch.Tensor]] = None
    # contact_phase
    # slip_risk
    # grasp_stability
    # progress_score
    # error_type
    # recovery_action
```

---

## 4. 动作空间分组

不要再用单一 action head 直接预测所有动作维度。请在配置里显式定义 action group。

```yaml
action_space:
  action_dim: 32

  groups:
    base:
      indices: [0, 1, 2]
      type: "velocity_or_delta_pose"

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

如果实际 `action_dim` 不同，请 Cursor 根据现有 data config 自动映射。

---

#模块 1：Hierarchical Intent Manifold Adapter
作用

该模块不是阶段分类器，而是高容量连续动作意图建模器。

它将 GR00T N1.7 的 VLM / action token 表征、机器人状态、可选触觉信息映射为多尺度 intent tokens：

global_intent_tokens:
  表达任务级目标、物体语义、子目标和导航目标。

motion_intent_tokens:
  表达局部运动意图，例如靠近、绕障、对齐、伸手、抬起、搬运、放置。

contact_intent_tokens:
  表达接触相关意图，例如接触建立、夹持稳定、滑移修正、释放时机。

recovery_intent_tokens:
  表达错误恢复意图，例如重新对齐、重新抓取、后退避碰、停止底盘。

同时输出 route_probs 作为低容量软路由变量，用于调节 base / arm / gripper / tactile / recovery experts 的权重。

重要原则
1. route_probs 不是完整动作意图。
2. 不需要人工阶段标签。
3. intent tokens 是高容量连续 latent space。
4. route modes 是 learned latent control modes。
5. 语义名称只用于可视化，不用于强监督。

---

## 6. 模块 2：Tactile Encoder

### 作用

将触觉历史编码成 tokens。由于当前模拟验证中视频和触觉是同频的，不强调“触觉高频”，而强调：

```text
触觉用于接触阶段的局部动作修正。
```

### 输入

```text
force_torque history
pressure map / tactile image
gripper current
gripper qpos
joint torque
slip signal
contact flag
```

### 接口

```python
class TactileEncoder(nn.Module):
    def __init__(
        self,
        tactile_dim: int,
        hidden_dim: int,
        history_len: int = 16,
        use_pressure_map: bool = False,
    ):
        super().__init__()
        self.history_len = history_len
        self.temporal = nn.Sequential(
            nn.Conv1d(tactile_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.contact_head = nn.Linear(hidden_dim, 1)
        self.slip_head = nn.Linear(hidden_dim, 1)

    def forward(self, tactile: dict) -> dict:
        """
        Required tactile keys:
          - force_torque: [B, T, F]
        Optional keys:
          - gripper_current, slip_signal, contact_flag, etc.

        Returns:
            {
              "tactile_tokens": [B, Nt, D],
              "tactile_summary": [B, D],
              "contact_logits": [B, 1],
              "slip_logits": [B, 1],
              "force_summary": [B, D]
            }
        """
        x = tactile["force_torque"][:, -self.history_len:]

        extra = []
        for key in ["gripper_current", "slip_signal"]:
            if key in tactile and tactile[key] is not None:
                extra.append(tactile[key][:, -self.history_len:])
        if extra:
            x = torch.cat([x] + extra, dim=-1)

        # [B, T, C] -> [B, C, T]
        x = x.transpose(1, 2)
        x = self.temporal(x).transpose(1, 2)
        tokens = self.transformer(x)
        summary = tokens.mean(dim=1)

        return {
            "tactile_tokens": tokens,
            "tactile_summary": summary,
            "contact_logits": self.contact_head(summary),
            "slip_logits": self.slip_head(summary),
            "force_summary": summary,
        }
```

---

## 7. 模块 3：Contact Gate

### 作用

计算接触门控 `g_contact ∈ [0, 1]`，决定触觉 refiner 是否强介入动作修正。

### 接口

```python
class ContactGate(nn.Module):
    def __init__(self, hidden_dim: int, phase_dim: int, state_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim + phase_dim + state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        tactile_summary: torch.Tensor,
        robot_state_summary: torch.Tensor,
        phase_probs: torch.Tensor,
        sim_contact_flag: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Returns:
            g_contact: [B, 1]
        """
        x = torch.cat([tactile_summary, robot_state_summary, phase_probs], dim=-1)
        g = torch.sigmoid(self.net(x))

        if sim_contact_flag is not None:
            # Use simulation contact as soft teacher, not a hard replacement.
            sim_g = sim_contact_flag.float().view(g.shape)
            g = 0.5 * g + 0.5 * sim_g

        return g
```

### 使用原则

```text
contact / grasp / transport / release / recovery 阶段 → gate 更高
navigation / pre_grasp 阶段 → gate 更低
```

---

## 8. 模块 4：Coarse Structured Action DiT

### 作用

第一阶段动作解码器，主要依赖视觉、语言、状态和 intent，生成中间动作 `a_mid`。

负责：

```text
base 粗运动
arm 粗轨迹
gripper 初始开合
posture 粗平衡残差
```

### 结构

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

### 接口

```python
class StructuredActionDiT(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        action_dim: int,
        action_groups: dict,
        num_layers: int = 12,
        num_heads: int = 16,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.action_groups = action_groups

        self.action_proj = nn.Linear(action_dim, hidden_dim)
        self.cond_proj = nn.Linear(hidden_dim, hidden_dim)

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            batch_first=True,
        )
        self.shared_transformer = nn.TransformerEncoder(layer, num_layers=num_layers)

        self.group_heads = nn.ModuleDict({
            name: nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, len(group["indices"])),
            )
            for name, group in action_groups.items()
        })

    def forward(
        self,
        noisy_action: torch.Tensor,
        timestep: torch.Tensor,
        h_vlm: torch.Tensor,
        intent_tokens: torch.Tensor,
        robot_state_tokens: torch.Tensor,
        future_tokens: torch.Tensor | None = None,
    ) -> dict:
        """
        Args:
            noisy_action: [B, H, action_dim]
            timestep: [B] or [B, 1]
            h_vlm: [B, N, D]
            intent_tokens: [B, K, D]
            robot_state_tokens: [B, Ns, D]
            future_tokens: optional [B, Nf, D], stop-grad recommended

        Returns:
            {
              "a_mid": [B, H, action_dim],
              "group_outputs": {...},
              "attn_maps": optional
            }
        """
        action_tokens = self.action_proj(noisy_action)
        cond = [h_vlm, intent_tokens, robot_state_tokens, action_tokens]

        if future_tokens is not None:
            cond.insert(-1, future_tokens.detach())

        x = torch.cat(cond, dim=1)
        x = self.shared_transformer(x)

        # Use the last H tokens as action tokens.
        h_action = x[:, -noisy_action.shape[1]:]

        a_mid = torch.zeros_like(noisy_action)
        group_outputs = {}
        for name, group in self.action_groups.items():
            idx = group["indices"]
            out = self.group_heads[name](h_action)
            a_mid[..., idx] = out
            group_outputs[name] = out

        return {
            "a_mid": a_mid,
            "group_outputs": group_outputs,
            "attn_maps": None,
        }
```

---

## 9. 模块 5：Tactile Late-Denoising Refiner

### 作用

第二阶段动作解码器。借鉴 T-Rex 的后段触觉修正思想，但由于当前视频和触觉同频，因此命名为：

```text
Synchronous Tactile Late-Denoising Refiner
```

它的目标不是利用高频触觉，而是让触觉在 denoising 后半段参与接触残差修正。

### 公式

```text
a_refined = a_mid + g_contact · Δa_tactile
```

### 接口

```python
class TactileLateDenoisingRefiner(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        action_dim: int,
        action_groups: dict,
        group_scales: dict,
        num_layers: int = 4,
        num_heads: int = 8,
    ):
        super().__init__()
        self.action_groups = action_groups
        self.group_scales = group_scales
        self.action_proj = nn.Linear(action_dim, hidden_dim)

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            batch_first=True,
        )
        self.refiner = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.delta_head = nn.Linear(hidden_dim, action_dim)

    def forward(
        self,
        a_mid: torch.Tensor,
        timestep: torch.Tensor,
        tactile_tokens: torch.Tensor,
        h_vlm_cache: torch.Tensor,
        intent_tokens: torch.Tensor,
        robot_state_tokens: torch.Tensor,
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
        action_tokens = self.action_proj(a_mid)
        x = torch.cat([
            h_vlm_cache,
            intent_tokens,
            robot_state_tokens,
            tactile_tokens,
            action_tokens,
        ], dim=1)

        x = self.refiner(x)
        h_action = x[:, -a_mid.shape[1]:]
        delta = self.delta_head(h_action)

        # Apply group-wise residual scaling.
        for name, group in self.action_groups.items():
            idx = group["indices"]
            scale = self.group_scales.get(name, 1.0)
            delta[..., idx] *= scale

        g = contact_gate.view(contact_gate.shape[0], 1, 1)
        a_refined = a_mid + g * delta

        return {
            "delta_tactile": delta,
            "a_refined": a_refined,
            "attn_maps": None,
        }
```

### 触觉主要修正的动作维度

```text
arm residual
gripper force / open-close
compliance / stiffness
stop-base signal
posture residual
recovery residual
```

建议不要让 tactile refiner 强行修改所有动作。

---

## 10. 模块 6：Future Head + FLUX 训练监督

### 作用

仅训练时使用。Future Head 预测未来视觉和触觉相关表征，FLUX 只作为局部图像补全 teacher。

### 接口

```python
class VisuoTactileFutureHead(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.future_query = nn.Parameter(torch.randn(1, 16, hidden_dim))
        self.cross_attn = nn.MultiheadAttention(hidden_dim, 8, batch_first=True)
        self.contact_affordance_head = nn.Linear(hidden_dim, 1)
        self.future_tactile_head = nn.Linear(hidden_dim, hidden_dim)
        self.slip_head = nn.Linear(hidden_dim, 1)
        self.grasp_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        h_vlm: torch.Tensor,
        intent_tokens: torch.Tensor,
        robot_state_tokens: torch.Tensor,
    ) -> dict:
        """
        Returns:
            {
              "future_tokens": [B, Nf, D],
              "contact_affordance_logits": [B, Nf, 1],
              "future_tactile_latent": [B, D],
              "slip_logits": [B, 1],
              "grasp_stability_logits": [B, 1]
            }
        """
        b = h_vlm.shape[0]
        q = self.future_query.expand(b, -1, -1)
        context = torch.cat([h_vlm, intent_tokens, robot_state_tokens], dim=1)
        future_tokens, _ = self.cross_attn(q, context, context)
        pooled = future_tokens.mean(dim=1)

        return {
            "future_tokens": future_tokens,
            "contact_affordance_logits": self.contact_affordance_head(future_tokens),
            "future_tactile_latent": self.future_tactile_head(pooled),
            "slip_logits": self.slip_head(pooled),
            "grasp_stability_logits": self.grasp_head(pooled),
        }
```

### FLUX 使用原则

- 不在线跑 FLUX。
- 不让推理依赖 FLUX。
- 训练数据预处理阶段可以离线生成：

```text
flux_refined_future_patch
flux_refined_future_feature
```

训练 loss 只在 masked region 上计算：

```python
def masked_feature_distillation(pred, target, mask):
    return ((pred - target) ** 2 * mask).sum() / (mask.sum() + 1e-6)
```

---

## 11. 模块 7：VT-WAM 风格 AVTAG Loss

### 作用

训练时引导 action tokens 在接触阶段关注触觉 tokens，避免强视觉 backbone 让模型忽略触觉。

### 核心公式

```text
L_AVTAG = g_contact · max(0, p_vis - p_tac + margin)
```

其中：

```text
p_vis = action query 对 visual tokens 的注意力总量
p_tac = action query 对 tactile tokens 的注意力总量
```

### 接口

```python
def compute_avtag_loss(
    attn_maps: dict,
    token_groups: dict,
    contact_gate: torch.Tensor,
    action_group_weights: dict,
    margin: float = 0.05,
) -> torch.Tensor:
    """
    Use only during training.

    attn_maps should contain attention from action queries to:
      - visual tokens
      - tactile tokens

    Apply stronger loss to:
      - gripper tokens
      - arm residual tokens
      - recovery tokens

    Apply weaker or no loss to:
      - base navigation tokens
      - high-level intent tokens
    """
    p_vis = attn_maps["action_to_visual"]
    p_tac = attn_maps["action_to_tactile"]

    # Expected shapes can be normalized to [B, action_group]
    loss = 0.0
    total_weight = 0.0

    for group_name, weight in action_group_weights.items():
        if group_name not in token_groups:
            continue
        idx = token_groups[group_name]
        group_p_vis = p_vis[:, idx].mean(dim=-1)
        group_p_tac = p_tac[:, idx].mean(dim=-1)
        group_loss = torch.relu(group_p_vis - group_p_tac + margin)
        group_loss = (contact_gate.view(-1) * group_loss).mean()
        loss = loss + weight * group_loss
        total_weight += weight

    return loss / max(total_weight, 1e-6)
```

### 分组权重建议

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

## 12. 模块 8：Execution Monitor

### 作用

判断当前执行是否正常，输出任务进度、错误类型和 recovery gate。

### 接口

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

class ExecutionMonitor(nn.Module):
    def __init__(self, hidden_dim: int, num_error_types: int):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.progress_head = nn.Linear(hidden_dim, 1)
        self.error_head = nn.Linear(hidden_dim, num_error_types)
        self.recovery_gate_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        h_vlm_summary: torch.Tensor,
        tactile_summary: torch.Tensor,
        robot_state_summary: torch.Tensor,
    ) -> dict:
        """
        Returns:
            {
              "progress_score": [B, 1],
              "error_logits": [B, num_error_types],
              "recovery_gate": [B, 1]
            }
        """
        x = torch.cat([h_vlm_summary, tactile_summary, robot_state_summary], dim=-1)
        h = self.fuse(x)
        return {
            "progress_score": torch.sigmoid(self.progress_head(h)),
            "error_logits": self.error_head(h),
            "recovery_gate": torch.sigmoid(self.recovery_gate_head(h)),
        }
```

---

## 13. 模块 9：Recovery Expert

### 作用

当 Execution Monitor 发现异常时，输出恢复 residual action。

### 接口

```python
class RecoveryExpert(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        action_dim: int,
        num_error_types: int,
    ):
        super().__init__()
        self.error_embed = nn.Linear(num_error_types, hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(
        self,
        error_logits: torch.Tensor,
        recovery_gate: torch.Tensor,
        h_vlm_summary: torch.Tensor,
        intent_summary: torch.Tensor,
        tactile_summary: torch.Tensor,
        a_refined: torch.Tensor,
    ) -> dict:
        """
        Returns:
            {
              "delta_recovery": [B, H, action_dim],
              "a_recovered": [B, H, action_dim]
            }
        """
        error_prob = error_logits.softmax(dim=-1)
        error_h = self.error_embed(error_prob)
        context = torch.cat([h_vlm_summary, intent_summary, tactile_summary, error_h], dim=-1)
        delta_one = self.net(context)

        delta = delta_one[:, None, :].expand_as(a_refined)
        gate = recovery_gate[:, None, :]
        a_recovered = a_refined + gate * delta

        return {
            "delta_recovery": delta,
            "a_recovered": a_recovered,
        }
```

---

## 14. 模型总封装类

请新增总模型类：

```python
class GR00TN17VisuoTactileClosedLoopPolicy(nn.Module):
    def __init__(self, gr00t_backbone, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = gr00t_backbone

        self.tactile_encoder = TactileEncoder(...)
        self.intent_adapter = IntentManifoldAdapter(...)
        self.contact_gate = ContactGate(...)

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
        h_vlm_summary = h_vlm.mean(dim=1)

        # Project robot state to tokens / summary according to existing codebase.
        robot_state_tokens = self.encode_robot_state_as_tokens(batch.robot_state)
        robot_state_summary = robot_state_tokens.mean(dim=1)

        intent_out = self.intent_adapter(
            h_vlm=h_vlm,
            robot_state=batch.robot_state,
            tactile_tokens=tactile_out["tactile_tokens"],
        )
        intent_summary = intent_out["intent_tokens"].mean(dim=1)

        g_contact = self.contact_gate(
            tactile_summary=tactile_out["tactile_summary"],
            robot_state_summary=robot_state_summary,
            phase_probs=intent_out["phase_probs"],
            sim_contact_flag=batch.tactile.get("contact_flag", None),
        )

        future_out = None
        future_tokens = None
        if self.training or self.cfg.inference.use_future_tokens:
            future_out = self.future_head(
                h_vlm=h_vlm,
                intent_tokens=intent_out["intent_tokens"],
                robot_state_tokens=robot_state_tokens,
            )
            if self.cfg.action_decoder.use_future_tokens:
                future_tokens = future_out["future_tokens"].detach()

        coarse_out = self.coarse_action_dit(
            noisy_action=batch.noisy_action,
            timestep=batch.diffusion_timestep,
            h_vlm=h_vlm,
            intent_tokens=intent_out["intent_tokens"],
            robot_state_tokens=robot_state_tokens,
            future_tokens=future_tokens,
        )

        tactile_refine_out = self.tactile_refiner(
            a_mid=coarse_out["a_mid"],
            timestep=batch.diffusion_timestep,
            tactile_tokens=tactile_out["tactile_tokens"],
            h_vlm_cache=h_vlm.detach() if self.cfg.detach_vlm_for_refiner else h_vlm,
            intent_tokens=intent_out["intent_tokens"],
            robot_state_tokens=robot_state_tokens,
            contact_gate=g_contact,
        )

        monitor_out = self.execution_monitor(
            h_vlm_summary=h_vlm_summary,
            tactile_summary=tactile_out["tactile_summary"],
            robot_state_summary=robot_state_summary,
        )

        recovery_out = self.recovery_expert(
            error_logits=monitor_out["error_logits"],
            recovery_gate=monitor_out["recovery_gate"],
            h_vlm_summary=h_vlm_summary,
            intent_summary=intent_summary,
            tactile_summary=tactile_out["tactile_summary"],
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
            "contact_gate": g_contact,
        }
```

> 注意：`encode_robot_state_as_tokens` 和 `SafetyProjector` 需要根据现有 GR00T / robot control 代码实现。前期可以先用 MLP placeholder 跑通训练。

---

## 15. 总损失

训练 loss 汇总：

```python
loss = (
    loss_action
    + cfg.loss.future * loss_future
    + cfg.loss.flux * loss_flux
    + cfg.loss.mask * loss_mask
    + cfg.loss.contact * loss_contact
    + cfg.loss.tactile * loss_tactile
    + cfg.loss.avtag * loss_avtag
    + cfg.loss.monitor * loss_monitor
    + cfg.loss.recovery * loss_recovery
    + cfg.loss.smooth * loss_smooth
    + cfg.loss.safe * loss_safe
)
```

建议封装成：

```python
class ClosedLoopPolicyLoss(nn.Module):
    def forward(self, outputs, batch):
        loss_dict = {}

        loss_dict["action"] = compute_grouped_action_loss(
            outputs["action"],
            batch.actions,
            batch.action_groups,
        )

        if outputs["future"] is not None:
            loss_dict["future"] = compute_future_loss(outputs["future"], batch)
            loss_dict["flux"] = compute_flux_loss(outputs["future"], batch)
            loss_dict["mask"] = compute_mask_loss(outputs["future"], batch)

        loss_dict["contact"] = compute_contact_loss(outputs["contact_gate"], batch)
        loss_dict["tactile"] = compute_tactile_aux_loss(outputs["tactile"], batch)
        loss_dict["monitor"] = compute_monitor_loss(outputs["monitor"], batch)
        loss_dict["recovery"] = compute_recovery_loss(outputs["recovery"], batch)
        loss_dict["smooth"] = compute_smooth_loss(outputs["action"])

        if self.cfg.loss.avtag > 0:
            loss_dict["avtag"] = compute_avtag_loss(...)

        total = sum(self.cfg.loss[k] * v for k, v in loss_dict.items() if k in self.cfg.loss)
        return total, loss_dict
```

---

## 16. 训练阶段

### Stage 1：冻结 / 轻调 GR00T N1.7

训练：

```text
Intent Manifold Adapter
Structured Action DiT
```

冻结：

```text
GR00T VLM backbone
Future Head
Tactile Refiner
Recovery Expert
```

目标：先稳定恢复原始行为克隆能力。

---

### Stage 2：加入触觉编码器和接触门控

训练：

```text
Tactile Encoder
Contact Gate
Gripper / Contact Head
Tactile Late-Denoising Refiner
```

目标：验证触觉是否改善接触成功率、抓取稳定性、滑移检测。

---

### Stage 3：加入 Future Head + FLUX 监督

训练：

```text
Future Head
mask head
contact affordance head
future tactile latent head
```

FLUX：

```text
只用缓存的离线 target / feature
不参与推理
```

---

### Stage 4：加入 AVTAG

训练：

```text
contact-gated tactile attention guidance
```

只对这些 action token 强加：

```text
arm
gripper
recovery
posture
base_stop
```

不要强加给 base navigation 和 high-level intent。

---

### Stage 5：加入扰动数据和恢复训练

需要数据增强：

```text
object pose perturbation
base misalignment
gripper miss
slip
unexpected contact
force too large
target occlusion
object lost
```

训练：

```text
Execution Monitor
Recovery Expert
```

---

### Stage 6：Rollout-aware 微调

逐步从 GT future feature 切到 predicted future feature：

```yaml
rollout_schedule:
  step_0:
    gt_future_ratio: 0.8
    pred_future_ratio: 0.2

  step_mid:
    gt_future_ratio: 0.5
    pred_future_ratio: 0.5

  step_final:
    gt_future_ratio: 0.2
    pred_future_ratio: 0.8
```

---

## 17. 推理流程

真实推理时不要一次执行完整 chunk。

```python
while not done:
    obs = env.get_observation()
    batch = build_batch(obs)

    outputs = policy(batch, mode="inference")
    action_chunk = outputs["action"]

    # Execute only first k steps.
    for i in range(exec_steps):
        env.step(action_chunk[:, i])

    feedback = env.get_observation()

    # The next iteration uses updated visual + tactile + state feedback.
    # If recovery_gate is high, model should enter recovery phase.
    if outputs["monitor"]["recovery_gate"].max() > recovery_threshold:
        continue
```

推荐配置：

```yaml
inference:
  action_horizon: 16
  execute_steps: 4
  use_flux: false
  use_future_tokens: true
  use_recovery: true
  use_safety_projector: true
```

---

## 18. YAML 配置模板

```yaml
model:
  name: "gr00t_n17_visuo_tactile_closed_loop"
  backbone: "gr00t_n1_7"
  freeze_backbone: true
  tune_vlm_lora: false
  hidden_dim: 1024

intent_adapter:
  enabled: true
  num_intent_tokens: 16
  phase_classes:
    - navigation
    - pre_grasp
    - contact
    - transport
    - place
    - release
    - recovery
    - idle

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
  type: "late_denoising"
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
  use_recovery: true
  use_safety_projector: true
```

---

## 19. 必做消融实验

请预留 config flags，方便做以下 ablation：

```text
A0: GR00T N1.7 baseline
A1: + Intent Manifold Adapter
A2: + Structured Action DiT
A3: + Early tactile fusion
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
```

---

## 20. Cursor 执行优先级

建议按以下顺序实现：

```text
1. 扩展 batch，加入 tactile 和 action group mapping
2. 实现 TactileEncoder
3. 实现 IntentManifoldAdapter
4. 实现 StructuredActionDiT 的 group heads
5. 实现 ContactGate
6. 实现 TactileLateDenoisingRefiner
7. 实现 action loss 和 grouped loss
8. 实现 AVTAG loss
9. 实现 FutureHead，先不接 FLUX
10. 接入离线 FLUX target loss
11. 实现 ExecutionMonitor
12. 实现 RecoveryExpert
13. 实现 closed-loop inference wrapper
14. 加 configs 和 ablation flags
```

---

## 21. 最小可行版本 MVP

先做：

```text
GR00T N1.7
+ TactileEncoder
+ IntentManifoldAdapter
+ StructuredActionDiT
+ ContactGate
+ TactileLateDenoisingRefiner
```

跑通以后再加：

```text
AVTAG
Future Head + FLUX
Execution Monitor
Recovery Expert
```

---

## 22. Cursor 需要遵守的实现原则

```text
1. 不要重写 GR00T N1.7 backbone。
2. 优先在 action decoder 外围加 adapter / refiner / monitor。
3. FLUX 只允许在训练 loss 中使用，禁止推理依赖 FLUX。
4. 触觉不要直接污染 VLM backbone，只进入 tactile encoder、contact gate、late refiner 和 monitor。
5. AVTAG 是 training-only loss，不改变推理计算图。
6. action 必须按 base / arm / gripper / posture 分组处理。
7. Recovery Expert 输出 residual action，不直接替代主策略。
8. 推理时执行 partial chunk，然后重新观测，形成闭环。
```
