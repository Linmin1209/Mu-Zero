#!/usr/bin/env python3
"""LeoAgent builder with LLM CPU offload for 16GB GPUs."""

from __future__ import annotations

import os
from pathlib import Path

import torch
from accelerate.logging import get_logger
from peft import LoraConfig, get_peft_model
from transformers import LlamaForCausalLM, LlamaTokenizer

from model.build import build_module
from model.leo_agent import LeoAgent
from model.utils import disabled_train

logger = get_logger(__name__)


class LeoAgentRc365(LeoAgent):
    """LeoAgent with device routed to LLM embedding weights (offload-safe)."""

    @property
    def device(self):
        override = getattr(self, "_rc365_llm_input_device", None)
        if override is not None:
            return override
        return super().device

    def __init__(self, cfg):
        super().__init__(cfg)
        self._rc365_llm_input_device = self.llm_model.get_input_embeddings().weight.device


def _gpu_total_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.get_device_properties(0).total_memory / (1024**3)


def _should_offload_llm(force_offload: bool | None) -> bool:
    if force_offload is not None:
        return force_offload
    return _gpu_total_gb() < 24.0


def _init_offload_llm(agent: LeoAgentRc365, cfg) -> None:
    torch.nn.Module.__init__(agent)

    agent.llm_tokenizer = LlamaTokenizer.from_pretrained(
        cfg.llm.cfg_path, truncation_side=cfg.llm.truncation_side
    )
    agent.llm_tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    offload_dir = Path(os.environ.get("LEO_OFFLOAD_DIR", "/tmp/leo_llm_offload"))
    offload_dir.mkdir(parents=True, exist_ok=True)
    gpu_cap = max(10, int(_gpu_total_gb() * 0.72))
    max_memory = {0: f"{gpu_cap}GiB", "cpu": "64GiB"}
    logger.info(
        f"Loading {cfg.llm.name} with CPU offload (GPU cap {gpu_cap}GiB, folder={offload_dir})"
    )
    agent.llm_model = LlamaForCausalLM.from_pretrained(
        cfg.llm.cfg_path,
        torch_dtype=torch.float16,
        device_map="auto",
        max_memory=max_memory,
        offload_folder=str(offload_dir),
    )
    agent.llm_model.resize_token_embeddings(len(agent.llm_tokenizer))
    logger.info(f"Build {cfg.llm.name} from {cfg.llm.cfg_path} (offloaded)")

    for param in agent.llm_model.parameters():
        param.requires_grad = False
    agent.llm_model.eval()
    agent.llm_model.train = disabled_train
    logger.info("Freeze LLM")

    agent.img_encoder = build_module(cfg.vision2d)
    agent.img_proj = torch.nn.Linear(
        agent.img_encoder.out_channels, agent.llm_model.config.hidden_size
    )

    agent.pcd_encoder = build_module(cfg.vision3d)
    agent.pcd_proj = torch.nn.Linear(cfg.vision3d.hidden_dim, agent.llm_model.config.hidden_size)

    if cfg.llm.lora.flag:
        logger.info(f"Apply LoRA with configs: {cfg.llm.lora}")
        lora_config = LoraConfig(
            r=cfg.llm.lora.rank,
            lora_alpha=cfg.llm.lora.alpha,
            target_modules=cfg.llm.lora.target_modules,
            lora_dropout=cfg.llm.lora.dropout,
            bias="none",
            modules_to_save=[],
        )
        agent.llm_model = get_peft_model(agent.llm_model, peft_config=lora_config)
        if hasattr(agent.llm_model, "enable_input_require_grads"):
            agent.llm_model.enable_input_require_grads()

    agent._rc365_offload = True

    agent.max_context_len = cfg.llm.max_context_len
    agent.max_out_len = cfg.llm.max_out_len

    agent.clip_txt_guidance = cfg.clip_txt_guidance.flag
    if agent.clip_txt_guidance:
        import clip

        logger.info("Add CLIP semantics guidance")
        agent.clip_model = clip.load("RN50")[0]
        for param in agent.clip_model.parameters():
            param.requires_grad = False
        agent.clip_model.eval()
        agent.clip_model.train = disabled_train
        agent.clip_proj = torch.nn.Linear(
            cfg.clip_txt_guidance.clip_out_dim, agent.llm_model.config.hidden_size
        )


def build_leo_agent_for_training(cfg, offload_llm: bool | None = None) -> LeoAgentRc365:
    """Build LeoAgent; offload LLM weights to CPU on GPUs with <24GB VRAM."""
    use_offload = _should_offload_llm(offload_llm)
    if not use_offload:
        return LeoAgentRc365(cfg)

    agent = LeoAgentRc365.__new__(LeoAgentRc365)
    _init_offload_llm(agent, cfg)
    agent._rc365_llm_input_device = agent.llm_model.get_input_embeddings().weight.device
    return agent
