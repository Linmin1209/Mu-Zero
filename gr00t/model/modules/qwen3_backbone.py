# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

import torch
import torch.nn as nn
from transformers.feature_extraction_utils import BatchFeature

from gr00t.model.modules.qwen3_motion import (
    MotionConfig,
    install_motion_module,
    set_motion_trainable,
)


logger = logging.getLogger(__name__)


try:
    from transformers import Qwen3VLForConditionalGeneration

    _QWEN3VL_AVAILABLE = True
except ImportError:
    _QWEN3VL_AVAILABLE = False


class Qwen3Backbone(torch.nn.Module):
    def __init__(
        self,
        model_name: str = "nvidia/Cosmos-Reason2-2B",
        tune_llm: bool = False,
        tune_visual: bool = False,
        select_layer: int = -1,
        reproject_vision: bool = True,
        use_flash_attention: bool = False,
        projector_dim: int = -1,
        load_bf16: bool = False,
        tune_top_llm_layers: int = 0,
        trainable_params_fp32: bool = False,
        transformers_loading_kwargs: dict = {},
        motion_config: MotionConfig | None = None,
        tune_motion: bool = True,
    ):
        """
        Qwen3Backbone is to generate n_queries to represent the future action hidden states.
        Args:
            model_name: nvidia/Cosmos-Reason2-2B
            tune_llm: whether to tune the LLM model (default: False)
            tune_visual: whether to tune the visual model (default: False)
            motion_config: optional STSS/MOSS motion module config
            tune_motion: train motion_block while keeping the rest of vision frozen
        """
        if not _QWEN3VL_AVAILABLE:
            raise ImportError(
                "Qwen3VLForConditionalGeneration is not available. "
                "Please upgrade transformers to a version that supports Qwen3-VL: "
                "pip install transformers>=4.57.0"
            )

        super().__init__()

        # Add attention kwargs
        extra_kwargs = {}
        if use_flash_attention:
            try:
                import flash_attn  # noqa: F401

                extra_kwargs["attn_implementation"] = "flash_attention_2"
            except ImportError:
                logger.warning(
                    "flash_attn is not installed. Falling back to sdpa attention. "
                    "Install flash-attn for better performance: pip install flash-attn"
                )
                extra_kwargs["attn_implementation"] = "sdpa"
        if load_bf16:
            extra_kwargs["torch_dtype"] = torch.bfloat16

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            **extra_kwargs,
            **transformers_loading_kwargs,
        ).eval()

        # needed since we don't use these layers. Also saves compute
        while len(self.model.language_model.layers) > select_layer:
            self.model.language_model.layers.pop(-1)

        self.motion_config = motion_config
        self.tune_motion = tune_motion
        if motion_config is not None and motion_config.use_motion:
            install_motion_module(self.model.visual, motion_config)
            self._convert_motion_bn_to_float()

        self.select_layer = select_layer
        self.set_trainable_parameters(tune_llm, tune_visual, tune_top_llm_layers)
        if load_bf16 and trainable_params_fp32:
            # cast trainable parameters to fp32
            for n, p in self.named_parameters():
                if p.requires_grad:
                    p.data = p.data.to(torch.float32)
                    logger.debug(f"Casting trainable parameter {n} to fp32")

    def _convert_motion_bn_to_float(self) -> None:
        motion_block = getattr(self.model.visual, "motion_block", None)
        if motion_block is None:
            return
        for module in motion_block.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)):
                module.float()

    def set_trainable_parameters(self, tune_llm: bool, tune_visual: bool, tune_top_llm_layers: int):
        self.tune_llm = tune_llm
        self.tune_visual = tune_visual
        for p in self.parameters():
            p.requires_grad = True
        if not tune_llm:
            self.model.language_model.requires_grad_(False)
        if not tune_visual:
            self.model.visual.requires_grad_(False)

        if self.motion_config is not None and self.motion_config.use_motion:
            set_motion_trainable(self.model.visual, tune_visual, self.tune_motion)
            self.model.visual._gr00t_tune_motion_only = (
                self.tune_motion and not self.tune_visual
            )

        if tune_top_llm_layers > 0:
            for layer in self.model.language_model.layers[-tune_top_llm_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True

        logger.debug(f"Tune backbone llm: {self.tune_llm}")
        logger.debug(f"Tune backbone visual: {self.tune_visual}")
        if self.motion_config is not None and self.motion_config.use_motion:
            logger.debug(f"Tune backbone motion: {self.tune_motion}")
        # Check if any parameters are still trainable. If not, log a warning.
        for name, p in self.named_parameters():
            if p.requires_grad:
                logger.debug(f"Backbone trainable parameter: {name}")
        if not any(p.requires_grad for p in self.parameters()):
            logger.warning("No backbone trainable parameters found.")

    def set_frozen_modules_to_eval_mode(self):
        """
        Huggingface will call model.train() at each training_step. To ensure
        the expected behaviors for modules like dropout, batchnorm, etc., we
        need to call model.eval() for the frozen modules.
        """
        if self.training:
            if self.model.language_model and not self.tune_llm:
                self.model.language_model.eval()
            if self.model.visual and not self.tune_visual:
                self.model.visual.eval()
                if (
                    self.motion_config is not None
                    and self.motion_config.use_motion
                    and self.tune_motion
                ):
                    if hasattr(self.model.visual, "motion_block") and self.model.visual.motion_block:
                        self.model.visual.motion_block.train()

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)

    def _set_motion_batch_meta(self, vl_input: dict) -> None:
        num_frames = vl_input.get("num_frames")
        num_views = vl_input.get("num_views")
        if num_frames is None or num_views is None:
            return
        nf = int(num_frames[0]) if torch.is_tensor(num_frames) else int(num_frames)
        nv = int(num_views[0]) if torch.is_tensor(num_views) else int(num_views)
        self.model.visual._gr00t_num_frames = nf
        self.model.visual._gr00t_num_views = nv

    def forward(self, vl_input: BatchFeature) -> BatchFeature:
        self.set_frozen_modules_to_eval_mode()
        vl_dict = dict(vl_input)
        self._set_motion_batch_meta(vl_dict)
        keys_to_use = ["input_ids", "attention_mask", "pixel_values", "image_grid_thw"]
        vl_input = {k: vl_dict[k] for k in keys_to_use if k in vl_dict}
        outputs = self.model(**vl_input, output_hidden_states=True)
        outputs = outputs.hidden_states[-1]
        image_mask = vl_input["input_ids"] == self.model.config.image_token_id
        attention_mask = vl_input["attention_mask"] == 1
        return BatchFeature(
            data={
                "backbone_features": outputs,
                "backbone_attention_mask": attention_mask,
                "image_mask": image_mask,
            }
        )  # [B, T2, hidden_size]
