# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DexJoCo bimanual + 4-finger + palm tactile per hand (20D stacked vector)."""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)

from dexjoco_tactile_schema import BIMANUAL_TACTILE_KEYS, tactile_num_contact, tactile_num_force

ACTION_HORIZON = 16

dexjoco_bimanual_visor_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["base", "wrist_left", "wrist_right"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=["right_tcp", "left_tcp", "right_hand", "left_hand"],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(ACTION_HORIZON)),
        modality_keys=["right_eef", "right_hand", "left_eef", "left_hand"],
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
    "tactile": ModalityConfig(
        delta_indices=[0],
        modality_keys=list(BIMANUAL_TACTILE_KEYS),
    ),
    "tactile_future": ModalityConfig(
        delta_indices=list(range(ACTION_HORIZON)),
        modality_keys=list(BIMANUAL_TACTILE_KEYS),
    ),
}

DEXJOCo_VISOR_TACTILE_NUM_FORCE = tactile_num_force(dual_arm=True)
DEXJOCo_VISOR_TACTILE_NUM_CONTACT = tactile_num_contact(dual_arm=True)

register_modality_config(
    dexjoco_bimanual_visor_config,
    embodiment_tag=EmbodimentTag.DEXJOCo_BIMANUAL,
)
