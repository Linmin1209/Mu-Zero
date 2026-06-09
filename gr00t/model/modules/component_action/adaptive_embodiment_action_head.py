# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adaptive component-level flow-matching action head."""

from __future__ import annotations

import logging
from typing import Any

import torch
from torch import nn
from torch.distributions import Beta
import torch.nn.functional as F
from transformers.feature_extraction_utils import BatchFeature

from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
from gr00t.model.modules.component_action.component_schema import (
    CANONICAL_COMPONENTS,
    ComponentSchemaConfig,
    DEFAULT_COMPONENT_DIMS,
    DEFAULT_COMPONENT_LOSS_WEIGHTS,
)
from gr00t.model.modules.component_action.msat_joint import ComponentActionMSAT
from gr00t.model.modules.component_action.packing import (
    pack_action_stream,
    unpack_action_predictions_batched,
)
from gr00t.model.modules.dit import SelfAttentionTransformer
from gr00t.model.modules.embodiment_conditioned_mlp import CategorySpecificMLP


logger = logging.getLogger(__name__)


class AdaptiveEmbodimentActionHead(nn.Module):
    """Component-level variable-length flow matching action head with MSAT."""

    supports_gradient_checkpointing = True

    def __init__(self, config: Gr00tN1d7Config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.input_embedding_dim = config.input_embedding_dim
        self.action_horizon = config.action_horizon
        self.num_inference_timesteps = config.num_inference_timesteps

        schema_cfg = getattr(config, "component_schema", None) or {}
        override_dims = getattr(config, "component_projector_dims", None) or {}
        if override_dims:
            component_dims = dict(override_dims)
        else:
            component_dims = dict(DEFAULT_COMPONENT_DIMS)
        loss_weights = dict(DEFAULT_COMPONENT_LOSS_WEIGHTS)
        loss_weights.update(getattr(config, "component_loss_weights", {}) or {})
        self.schema = ComponentSchemaConfig(
            component_dims=component_dims,
            loss_weights=loss_weights,
        )

        msat_cfg = dict(getattr(config, "component_msat_cfg", {}) or {})
        self.msat = ComponentActionMSAT(
            sa_dim=self.input_embedding_dim,
            vl_dim=config.backbone_embedding_dim,
            num_attention_heads=msat_cfg.get("num_attention_heads", 24),
            attention_head_dim=msat_cfg.get("attention_head_dim", 64),
            depth_multi_stream=msat_cfg.get("depth_multi_stream", 4),
            depth_single_stream=msat_cfg.get("depth_single_stream", 8),
            dropout=msat_cfg.get("dropout", config.attn_dropout),
            output_dim=self.hidden_size,
            max_seq_len=config.max_seq_len,
        )
        self.msat_decode = nn.Linear(self.hidden_size, self.input_embedding_dim)

        projector_dims = {
            comp: self.schema.component_dims[comp]
            for comp in CANONICAL_COMPONENTS
            if comp in self.schema.component_dims
        }
        self.component_projectors = nn.ModuleDict(
            {comp: nn.Linear(dim, self.input_embedding_dim) for comp, dim in projector_dims.items()}
        )
        self.component_inverse_projectors = nn.ModuleDict(
            {comp: nn.Linear(self.input_embedding_dim, dim) for comp, dim in projector_dims.items()}
        )
        self.component_type_embed = nn.Embedding(len(CANONICAL_COMPONENTS), self.input_embedding_dim)

        self.state_encoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=config.max_state_dim * config.state_history_length,
            hidden_dim=self.hidden_size,
            output_dim=self.input_embedding_dim,
        )

        self.vlln = (
            nn.LayerNorm(config.backbone_embedding_dim) if config.use_vlln else nn.Identity()
        )
        vl_self_attention_cfg = getattr(config, "vl_self_attention_cfg", None)
        if vl_self_attention_cfg and vl_self_attention_cfg.get("num_layers", 0) > 0:
            self.vl_self_attention = SelfAttentionTransformer(**vl_self_attention_cfg)
        else:
            self.vl_self_attention = nn.Identity()

        self.state_dropout_prob = config.state_dropout_prob
        self.beta_dist = Beta(config.noise_beta_alpha, config.noise_beta_beta)
        self.num_timestep_buckets = config.num_timestep_buckets
        self.set_trainable_parameters(
            config.tune_projector, config.tune_diffusion_model, config.tune_vlln
        )

    def set_trainable_parameters(
        self, tune_projector: bool, tune_diffusion_model: bool, tune_vlln: bool
    ):
        self.tune_projector = tune_projector
        self.tune_diffusion_model = tune_diffusion_model
        self.tune_vlln = tune_vlln
        for p in self.parameters():
            p.requires_grad = True
        if not tune_projector:
            self.state_encoder.requires_grad_(False)
            self.component_projectors.requires_grad_(False)
            self.component_inverse_projectors.requires_grad_(False)
            self.component_type_embed.requires_grad_(False)
            self.msat_decode.requires_grad_(False)
        if not tune_diffusion_model:
            self.msat.requires_grad_(False)
        if not tune_vlln:
            self.vlln.requires_grad_(False)
            self.vl_self_attention.requires_grad_(False)

    def set_frozen_modules_to_eval_mode(self):
        if not self.training:
            return
        if not self.tune_projector:
            self.state_encoder.eval()
            self.component_projectors.eval()
            self.component_inverse_projectors.eval()
            self.component_type_embed.eval()
            self.msat_decode.eval()
        if not self.tune_diffusion_model:
            self.msat.eval()
        if not self.tune_vlln:
            self.vlln.eval()
            self.vl_self_attention.eval()

    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        return (1 - sample) * self.config.noise_s

    def process_backbone_output(self, backbone_output: BatchFeature) -> BatchFeature:
        backbone_features = backbone_output["backbone_features"]
        backbone_features = self.vlln(backbone_features)
        backbone_features = self.vl_self_attention(backbone_features)
        backbone_output["backbone_features"] = backbone_features
        return backbone_output

    def _encode_state(self, action_input: BatchFeature, embodiment_id: torch.Tensor):
        state = action_input.state
        assert state.shape[1] == self.config.state_history_length
        state = state.view(state.shape[0], 1, -1)
        state_features = self.state_encoder(state, embodiment_id)
        if self.training and self.state_dropout_prob > 0:
            do_dropout = (
                torch.rand(state_features.shape[0], device=state_features.device)
                < self.state_dropout_prob
            )
            do_dropout = do_dropout[:, None, None].to(dtype=state_features.dtype)
            state_features = state_features * (1 - do_dropout)
        return state_features

    def _component_dict_from_input(self, action_input: BatchFeature) -> dict[str, torch.Tensor]:
        if hasattr(action_input, "component_actions"):
            return action_input.component_actions
        return action_input["component_actions"]

    def _active_lists_from_input(self, action_input: BatchFeature) -> list[list[str]]:
        mask = action_input.active_component_mask
        active_lists: list[list[str]] = []
        for b in range(mask.shape[0]):
            active_lists.append(
                [CANONICAL_COMPONENTS[i] for i, v in enumerate(mask[b]) if v > 0.5]
            )
        return active_lists

    def _flow_match_components(
        self,
        component_actions: dict[str, torch.Tensor],
        device,
        dtype,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        batch_size = next(iter(component_actions.values())).shape[0]
        t = self.sample_time(batch_size, device, dtype)
        noisy: dict[str, torch.Tensor] = {}
        velocity: dict[str, torch.Tensor] = {}
        for comp, actions in component_actions.items():
            noise = torch.randn_like(actions)
            t_b = t[:, None, None]
            noisy[comp] = (1 - t_b) * noise + t_b * actions
            velocity[comp] = actions - noise
        t_discretized = (t * self.num_timestep_buckets).long()
        return noisy, velocity, t, t_discretized

    def _run_msat(
        self,
        *,
        component_tokens: dict[str, torch.Tensor],
        active_lists: list[list[str]],
        state_features: torch.Tensor,
        t_discretized: torch.Tensor,
        vl_embeds: torch.Tensor,
        vl_attn_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        tau_token = self.msat.encode_tau_token(t_discretized)
        packed = pack_action_stream(
            state_features=state_features,
            tau_token=tau_token,
            component_tokens=component_tokens,
            active_components=active_lists,
            component_projectors=self.component_projectors,
            component_type_embed=self.component_type_embed,
        )
        sa_out = self.msat(
            sa_tokens=packed.tokens,
            vl_tokens=vl_embeds,
            timesteps=t_discretized,
            sa_horizon_ids=packed.horizon_ids,
            sa_attention_mask=packed.attention_mask,
            vl_attention_mask=vl_attn_mask,
        )
        sa_out = self.msat_decode(sa_out)
        self._last_packed = packed
        return sa_out

    def forward(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
        self.set_frozen_modules_to_eval_mode()
        backbone_output = self.process_backbone_output(backbone_output)
        vl_embeds = backbone_output.backbone_features
        device = vl_embeds.device
        embodiment_id = action_input.embodiment_id
        state_features = self._encode_state(action_input, embodiment_id)

        component_actions = self._component_dict_from_input(action_input)
        active_lists = self._active_lists_from_input(action_input)
        noisy, velocity, _t, t_discretized = self._flow_match_components(
            component_actions, device, vl_embeds.dtype
        )

        sa_out = self._run_msat(
            component_tokens=noisy,
            active_lists=active_lists,
            state_features=state_features,
            t_discretized=t_discretized,
            vl_embeds=vl_embeds,
            vl_attn_mask=getattr(backbone_output, "backbone_attention_mask", None),
        )
        pred_velocity = unpack_action_predictions_batched(
            sa_out, self._last_packed, self.component_inverse_projectors
        )

        total_loss = torch.tensor(0.0, device=device, dtype=vl_embeds.dtype)
        n_terms = 0
        for comp, target in velocity.items():
            if comp not in pred_velocity:
                continue
            weight = self.schema.loss_weights.get(comp, 1.0)
            total_loss = total_loss + weight * F.mse_loss(pred_velocity[comp], target)
            n_terms += 1
        if n_terms == 0:
            raise RuntimeError("No component losses computed — check active_component_mask.")
        loss = total_loss / n_terms

        return {
            "loss": loss,
            "backbone_features": vl_embeds,
            "state_features": state_features,
        }

    def _encode_features(
        self, backbone_output: BatchFeature, action_input: BatchFeature
    ) -> BatchFeature:
        backbone_output = self.process_backbone_output(backbone_output)
        vl_embeds = backbone_output.backbone_features
        state_features = self._encode_state(action_input, action_input.embodiment_id)
        return BatchFeature(data={"backbone_features": vl_embeds, "state_features": state_features})

    @torch.no_grad()
    def get_action_with_features(
        self,
        backbone_features: torch.Tensor,
        state_features: torch.Tensor,
        embodiment_id: torch.Tensor,
        backbone_output: BatchFeature,
        action_input: BatchFeature,
        options: dict[str, Any] | None = None,
    ) -> BatchFeature:
        vl_embeds = backbone_features
        batch_size = vl_embeds.shape[0]
        device = vl_embeds.device
        dtype = vl_embeds.dtype

        active_lists = self._active_lists_from_input(action_input)
        component_noise: dict[str, torch.Tensor] = {}
        for comp_list in active_lists:
            for comp in comp_list:
                if comp in component_noise:
                    continue
                dim = self.schema.component_dims[comp]
                component_noise[comp] = torch.randn(
                    batch_size, self.action_horizon, dim, device=device, dtype=dtype
                )

        dt = 1.0 / self.num_inference_timesteps
        for step in range(self.num_inference_timesteps):
            t_cont = step / float(self.num_inference_timesteps)
            t_discretized = torch.full(
                (batch_size,), int(t_cont * self.num_timestep_buckets), device=device, dtype=torch.long
            )
            sa_out = self._run_msat(
                component_tokens=component_noise,
                active_lists=active_lists,
                state_features=state_features,
                t_discretized=t_discretized,
                vl_embeds=vl_embeds,
                vl_attn_mask=getattr(backbone_output, "backbone_attention_mask", None),
            )
            pred_velocity = unpack_action_predictions_batched(
                sa_out, self._last_packed, self.component_inverse_projectors
            )
            for comp, vel in pred_velocity.items():
                component_noise[comp] = component_noise[comp] + dt * vel

        return BatchFeature(
            data={
                "component_action_pred": component_noise,
                "backbone_features": vl_embeds,
                "state_features": state_features,
            }
        )

    @torch.no_grad()
    def get_action(
        self,
        backbone_output: BatchFeature,
        action_input: BatchFeature,
        options: dict[str, Any] | None = None,
    ) -> BatchFeature:
        features = self._encode_features(backbone_output, action_input)
        return self.get_action_with_features(
            backbone_features=features.backbone_features,
            state_features=features.state_features,
            embodiment_id=action_input.embodiment_id,
            backbone_output=backbone_output,
            action_input=action_input,
            options=options,
        )

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype
