# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
RoboCasa365 modality config with 4-frame video history: delta_indices [-6, -4, -2, 0].

Tactile uses the same delta_indices so at decision step t you load
tactile at t-6, t-4, t-2, t (aligned with video). tactile_future loads
t..t+39 for WWM future tactile supervision. State / language at [0].
Action horizon remains 40 future steps.
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

ACTION_HORIZON = 40
VIDEO_DELTA_INDICES = [-6, -4, -2, 0]
# Tactile history aligned with 4-frame video (same relative times as images).
TACTILE_HISTORY_DELTA_INDICES = VIDEO_DELTA_INDICES
# Future tactile chunk (t..t+39) for WWM auxiliary supervision.
TACTILE_FUTURE_DELTA_INDICES = list(range(ACTION_HORIZON))

robocasa365_panda_omron_config = {
    "video": ModalityConfig(
        delta_indices=VIDEO_DELTA_INDICES,
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
    "tactile": ModalityConfig(
        delta_indices=TACTILE_HISTORY_DELTA_INDICES,
        modality_keys=["left", "right", "contact"],
    ),
    "tactile_future": ModalityConfig(
        delta_indices=TACTILE_FUTURE_DELTA_INDICES,
        modality_keys=["left", "right", "contact"],
    ),
}

# Override pre-registered sim keys so finetune uses human-dataset field names.
MODALITY_CONFIGS[EmbodimentTag.ROBOCASA_PANDA_OMRON.value] = robocasa365_panda_omron_config

ROBOCASA365_COMPONENT_PROJECTOR_DIMS = {
    "right_arm": 6,
    "right_hand": 1,
    "base": 4,
}

ROBOCASA365_ACTION_KEY_DIMS = {
    "gripper_close": 1,
    "end_effector_position": 3,
    "end_effector_rotation": 3,
    "base_motion": 4,
    "control_mode": 1,
}
