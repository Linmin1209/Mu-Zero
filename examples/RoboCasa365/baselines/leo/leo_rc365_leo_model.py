#!/usr/bin/env python3
"""LEO LeoAgent + LoRA + continuous action head for RoboCasa365."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf


def resolve_pretrained_ckpt(leo_repo: Path, pretrained_ckpt: str) -> Path:
    ckpt = Path(pretrained_ckpt)
    if ckpt.is_file():
        return ckpt
    name = pretrained_ckpt.strip()
    if name in {"align", "sft_noact"}:
        path = leo_repo / "checkpoints" / f"{name}.pth"
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"LEO pretrained checkpoint not found: {pretrained_ckpt!r} "
        f"(looked under {leo_repo / 'checkpoints'})"
    )


def build_leo_cfg(
    leo_repo: Path,
    *,
    lora_r: int,
    lora_alpha: int,
    clip_txt_guidance: bool = False,
) -> Any:
    cfg_dir = leo_repo / "configs"
    cfg = OmegaConf.load(cfg_dir / "default.yaml")
    cfg = OmegaConf.merge(cfg, OmegaConf.load(cfg_dir / "data/default.yaml"))
    cfg = OmegaConf.merge(cfg, OmegaConf.load(cfg_dir / "task/tuning_noact.yaml"))
    cfg.llm = OmegaConf.load(cfg_dir / "llm/vicuna7b.yaml")
    cfg.vision2d = OmegaConf.load(cfg_dir / "vision2d/convnext.yaml")
    cfg.vision3d = OmegaConf.load(cfg_dir / "vision3d/ose3d_pointnetpp.yaml")
    cfg.vision3d.backbone = OmegaConf.load(cfg_dir / "vision3d/backbone/pointnetpp.yaml")
    cfg.llm.lora.flag = True
    cfg.llm.lora.rank = int(lora_r)
    cfg.llm.lora.alpha = int(lora_alpha)
    cfg.clip_txt_guidance.flag = bool(clip_txt_guidance)
    return cfg


class LeoRc365ActionHead(nn.Module):
    def __init__(self, hidden_dim: int, state_dim: int, action_dim: int, mlp_hidden: int):
        super().__init__()
        self.state_mlp = nn.Sequential(
            nn.Linear(state_dim, mlp_hidden // 4),
            nn.ReLU(),
            nn.Linear(mlp_hidden // 4, mlp_hidden // 4),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + mlp_hidden // 4, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, action_dim),
        )

    def forward(self, hidden: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        st = self.state_mlp(state)
        return self.head(torch.cat([hidden, st], dim=-1))


class LeoRc365ActionModel(nn.Module):
    """Wrap LeoAgent encoders + Vicuna (LoRA) and regress 12D Panda-Omron actions."""

    def __init__(
        self,
        leo_agent: nn.Module,
        *,
        state_dim: int = 16,
        action_dim: int = 12,
        action_hidden: int = 1024,
    ):
        super().__init__()
        self.leo_agent = leo_agent
        hidden_dim = int(leo_agent.llm_model.config.hidden_size)
        self.action_head = LeoRc365ActionHead(hidden_dim, state_dim, action_dim, action_hidden)

    @property
    def device(self) -> torch.device:
        return getattr(self.leo_agent, "_rc365_llm_input_device", self.leo_agent.device)

    def _vision_device(self) -> torch.device:
        """PointNet++ / ConvNeXt must run on CUDA (not CPU-offloaded LLM device)."""
        if torch.cuda.is_available():
            leo = self.leo_agent
            for module_name in ("pcd_encoder", "img_encoder"):
                module = getattr(leo, module_name, None)
                if module is not None:
                    try:
                        return next(module.parameters()).device
                    except StopIteration:
                        pass
            return torch.device("cuda:0")
        return self.device

    _BATCH_TENSOR_KEYS = (
        "obj_fts",
        "obj_locs",
        "obj_masks",
        "anchor_locs",
        "anchor_orientation",
        "img_fts",
        "img_masks",
        "state",
        "action",
        "obj_tokens",
    )

    def _move_batch_tensors(self, data_dict: dict[str, Any], device: torch.device) -> dict[str, Any]:
        batch = dict(data_dict)
        for key in self._BATCH_TENSOR_KEYS:
            if key in batch and torch.is_tensor(batch[key]):
                batch[key] = batch[key].to(device, non_blocking=True)
        return batch

    def _encode_multiview_images(self, img_fts: torch.Tensor, img_masks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """img_fts: (B, V, 3, H, W) -> tokens (B, V*T, D), masks (B, V*T)."""
        leo = self.leo_agent
        b, v, _c, _h, _w = img_fts.shape
        tokens = []
        for view_idx in range(v):
            with torch.no_grad():
                view_feats = leo.img_encoder(img_fts[:, view_idx])
            tokens.append(leo.img_proj(view_feats))
        img_tokens = torch.cat(tokens, dim=1)
        b = img_fts.shape[0]
        if img_masks.ndim == 1:
            view_mask = img_masks.reshape(b, 1)
        else:
            view_mask = img_masks.reshape(b, -1)
        if view_mask.shape[1] == 1:
            token_masks = view_mask.repeat(1, img_tokens.shape[1])
        else:
            token_masks = view_mask
        if token_masks.shape[1] != img_tokens.shape[1]:
            token_masks = torch.ones(
                b, img_tokens.shape[1], device=img_tokens.device, dtype=view_mask.dtype
            )
        return img_tokens, token_masks

    def encode_hidden(self, data_dict: dict[str, Any]) -> torch.Tensor:
        from model.utils import maybe_autocast

        leo = self.leo_agent
        llm_device = self.device
        vision_device = self._vision_device()
        bs = len(data_dict["prompt_after_obj"])

        batch = self._move_batch_tensors(data_dict, vision_device)
        if "obj_tokens" not in batch:
            from leo_rc365_sanitize import apply_leo_numeric_patches, sanitize_leo_batch_3d

            apply_leo_numeric_patches()
            batch = sanitize_leo_batch_3d(batch)
            with torch.cuda.amp.autocast(enabled=False):
                batch = leo.pcd_encoder(batch)
            if not torch.isfinite(batch.get("obj_tokens", torch.tensor(0.0))).all():
                raise RuntimeError("pcd_encoder produced non-finite obj_tokens")
        batch["obj_tokens"] = leo.pcd_proj(batch["obj_tokens"])

        img_tokens, _ = self._encode_multiview_images(batch["img_fts"], batch["img_masks"])
        batch["img_tokens"] = img_tokens

        inputs_embeds, attention_mask = leo.build_right_justified_sequence(batch)
        inputs_embeds = inputs_embeds.to(llm_device, non_blocking=True)
        attention_mask = attention_mask.to(llm_device, non_blocking=True)

        autocast_dtype = "fp16"
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            autocast_dtype = "bf16"
        with maybe_autocast(leo, dtype=autocast_dtype):
            outputs = leo.llm_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                use_cache=False,
                output_hidden_states=True,
            )

        hidden = getattr(outputs, "last_hidden_state", None)
        if hidden is None:
            hidden = outputs.hidden_states[-1]
        lengths = attention_mask.sum(dim=1).long().clamp(min=1) - 1
        pooled = hidden[torch.arange(bs, device=llm_device), lengths]
        return pooled.float()

    def forward(self, data_dict: dict[str, Any]) -> dict[str, Any]:
        hidden = self.encode_hidden(data_dict)
        state = data_dict["state"].to(hidden.device).float()
        target = data_dict["action"].to(hidden.device).float()
        pred = self.action_head(hidden, state.to(self.action_head.head[0].weight.device))
        loss = F.mse_loss(pred, target)
        return {"loss": loss, "pred_action": pred.detach()}

    def move_trainable_modules_to_cuda(self) -> None:
        if not torch.cuda.is_available():
            return
        dev = torch.device("cuda:0")
        leo = self.leo_agent
        for module_name in ("img_encoder", "img_proj", "pcd_encoder", "pcd_proj"):
            getattr(leo, module_name).to(dev)
        self.action_head.to(dev)
        if not getattr(leo, "_rc365_offload", False):
            leo.llm_model.to(dev)
            if hasattr(leo, "_rc365_llm_input_device"):
                leo._rc365_llm_input_device = leo.llm_model.get_input_embeddings().weight.device
        if leo.clip_txt_guidance:
            leo.clip_model.to(dev)
            leo.clip_proj.to(dev)

    def load_pretrained_leo(self, ckpt_path: Path, strict: bool = False) -> tuple[list[str], list[str]]:
        state = torch.load(ckpt_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        missing, unexpected = self.leo_agent.load_state_dict(state, strict=strict)
        return list(missing), list(unexpected)

    def learnable_state_dict(self) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for name, param in self.named_parameters():
            if param.requires_grad:
                out[name] = param.detach().cpu()
        return out
