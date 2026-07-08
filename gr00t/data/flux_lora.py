# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for FLUX.1-Fill-dev LoRA fine-tuning."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

DEFAULT_FLUX_LORA_TARGET_MODULES = [
    "attn.to_k",
    "attn.to_q",
    "attn.to_v",
    "attn.to_out.0",
    "attn.add_k_proj",
    "attn.add_q_proj",
    "attn.add_v_proj",
    "attn.to_add_out",
    "ff.net.0.proj",
    "ff.net.2",
    "ff_context.net.0.proj",
    "ff_context.net.2",
]

DEFAULT_LORA_WEIGHT_NAME = "pytorch_lora_weights.safetensors"


def snap_dim(value: int, multiple: int = 16) -> int:
    return max(multiple, (value // multiple) * multiple)


def mask_image_from_mode(size: tuple[int, int], mode: str) -> Image.Image:
    """Build a single-channel mask for Fill training/inference."""
    width, height = size
    if mode == "full_inpaint":
        return Image.new("L", (width, height), 255)
    if mode == "keep_reference":
        return Image.new("L", (width, height), 0)
    raise ValueError(f"Unknown mask mode: {mode!r}")


def prepare_mask_and_masked_image(
    image: Image.Image,
    mask: Image.Image,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert PIL image/mask to tensors used by FLUX Fill LoRA training."""
    image_arr = np.array(image.convert("RGB"))
    image_tensor = torch.from_numpy(image_arr[None].transpose(0, 3, 1, 2)).to(dtype=torch.float32)
    image_tensor = image_tensor / 127.5 - 1.0

    mask_arr = np.array(mask.convert("L"), dtype=np.float32) / 255.0
    mask_tensor = torch.from_numpy(mask_arr[None, None])
    mask_tensor = (mask_tensor >= 0.5).to(dtype=torch.float32)

    masked_image = image_tensor * (mask_tensor < 0.5)
    return mask_tensor, masked_image


def tokenize_prompt(tokenizer, prompt: str | list[str], max_sequence_length: int) -> torch.Tensor:
    if isinstance(prompt, str):
        prompt = [prompt]
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        return_length=False,
        return_overflowing_tokens=False,
        return_tensors="pt",
    )
    return text_inputs.input_ids


def _encode_prompt_with_t5(
    text_encoder,
    tokenizer,
    *,
    max_sequence_length: int = 512,
    prompt: str | list[str] | None = None,
    num_images_per_prompt: int = 1,
    device=None,
    text_input_ids: torch.Tensor | None = None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
    elif text_input_ids is None:
        raise ValueError("text_input_ids must be provided when tokenizer is None")

    prompt_embeds = text_encoder(text_input_ids.to(device))[0]
    dtype = text_encoder.dtype
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
    _, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
    return prompt_embeds


def _encode_prompt_with_clip(
    text_encoder,
    tokenizer,
    *,
    prompt: str | list[str],
    device=None,
    text_input_ids: torch.Tensor | None = None,
    num_images_per_prompt: int = 1,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_overflowing_tokens=False,
            return_length=False,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
    elif text_input_ids is None:
        raise ValueError("text_input_ids must be provided when tokenizer is None")

    prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=False)
    prompt_embeds = prompt_embeds.pooler_output
    prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, -1)
    return prompt_embeds


def encode_prompt(
    text_encoders,
    tokenizers,
    prompt: str | list[str],
    *,
    max_sequence_length: int,
    device=None,
    num_images_per_prompt: int = 1,
    text_input_ids_list: list[torch.Tensor] | None = None,
):
    dtype = text_encoders[0].dtype
    pooled_prompt_embeds = _encode_prompt_with_clip(
        text_encoder=text_encoders[0],
        tokenizer=tokenizers[0],
        prompt=prompt,
        device=device if device is not None else text_encoders[0].device,
        num_images_per_prompt=num_images_per_prompt,
        text_input_ids=text_input_ids_list[0] if text_input_ids_list else None,
    )
    prompt_embeds = _encode_prompt_with_t5(
        text_encoder=text_encoders[1],
        tokenizer=tokenizers[1],
        max_sequence_length=max_sequence_length,
        prompt=prompt,
        num_images_per_prompt=num_images_per_prompt,
        device=device if device is not None else text_encoders[1].device,
        text_input_ids=text_input_ids_list[1] if text_input_ids_list else None,
    )
    text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(device=device, dtype=dtype)
    return prompt_embeds, pooled_prompt_embeds, text_ids
