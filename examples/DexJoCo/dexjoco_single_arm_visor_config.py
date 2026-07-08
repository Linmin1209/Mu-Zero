# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DexJoCo single-arm + 4-finger + palm Allegro tactile (10D stacked vector)."""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)

from dexjoco_tactile_schema import SINGLE_ARM_TACTILE_KEYS, tactile_num_contact, tactile_num_force

ACTION_HORIZON = 16

dexjoco_single_arm_visor_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["base", "wrist"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=["tcp_pose", "hand"],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(ACTION_HORIZON)),
        modality_keys=["eef_rotvec", "hand"],
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
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
    "tactile": ModalityConfig(
        delta_indices=[0],
        modality_keys=list(SINGLE_ARM_TACTILE_KEYS),
    ),
    "tactile_future": ModalityConfig(
        delta_indices=list(range(ACTION_HORIZON)),
        modality_keys=list(SINGLE_ARM_TACTILE_KEYS),
    ),
}

# VISOR model dims — pass via finetune / launch_finetune:
#   visor_tactile_num_force=5, visor_tactile_num_contact=5
#   (ff/mf/rf/thumb/palm + 5 contact flags)
#   visor_arm_action_slice=(0,6), visor_hand_action_slice=(6,22)
#   visor_arm_action_dim=6, visor_hand_action_dim=16
DEXJOCo_VISOR_TACTILE_NUM_FORCE = tactile_num_force(dual_arm=False)
DEXJOCo_VISOR_TACTILE_NUM_CONTACT = tactile_num_contact(dual_arm=False)

register_modality_config(
    dexjoco_single_arm_visor_config,
    embodiment_tag=EmbodimentTag.DEXJOCo_SINGLE_ARM,
)
