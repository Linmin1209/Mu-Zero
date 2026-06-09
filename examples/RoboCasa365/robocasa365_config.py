# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Modality config for RoboCasa365 human teleop (PandaOmron) LeRobot v2.1 datasets.

Keys match meta/modality.json shipped with nv-community/robocasa365-datasets.
"""

from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)

# Match GR00T N1.7 action horizon (see config.json action_horizon=40).
ACTION_HORIZON = 40

robocasa365_panda_omron_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "robot0_agentview_left",
            "robot0_agentview_right",
            "robot0_eye_in_hand",
        ],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "base_position",
            "base_rotation",
            "end_effector_position_relative",
            "end_effector_rotation_relative",
            "gripper_qpos",
        ],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(ACTION_HORIZON)),
        modality_keys=[
            "gripper_close",
            "end_effector_position",
            "end_effector_rotation",
            "base_motion",
            "control_mode",
        ],
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
}

# Override pre-registered sim keys so finetune uses human-dataset field names.
MODALITY_CONFIGS[EmbodimentTag.ROBOCASA_PANDA_OMRON.value] = robocasa365_panda_omron_config

# Canonical component dims for AdaptiveEmbodimentActionHead (pos3+rot3, gripper, base3).
ROBOCASA365_COMPONENT_PROJECTOR_DIMS = {
    "right_arm": 6,
    "right_hand": 1,
    "base": 4,
}
