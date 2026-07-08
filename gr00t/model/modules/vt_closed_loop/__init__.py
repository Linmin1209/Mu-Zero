# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""VT closed-loop policy modules (skeleton — wrap GR00T N1.7 action decoder)."""

from gr00t.model.modules.vt_closed_loop.action_groups import (
    ERROR_TYPES,
    INTENT_PHASES,
    ActionGroupSpec,
    AVTAGGroupWeights,
    get_action_groups,
)
from gr00t.model.modules.vt_closed_loop.batch_types import MultiModalRobotBatch

__all__ = [
    "ActionGroupSpec",
    "AVTAGGroupWeights",
    "ERROR_TYPES",
    "INTENT_PHASES",
    "MultiModalRobotBatch",
    "get_action_groups",
]


def __getattr__(name: str):
    """Lazy imports to avoid config ↔ policy circular dependency."""
    if name == "GR00TN17VisuoTactileClosedLoopPolicy":
        from gr00t.model.modules.vt_closed_loop.closed_loop_policy import (
            GR00TN17VisuoTactileClosedLoopPolicy,
        )
        return GR00TN17VisuoTactileClosedLoopPolicy
    if name == "ClosedLoopPolicyLoss":
        from gr00t.model.modules.vt_closed_loop.losses import ClosedLoopPolicyLoss
        return ClosedLoopPolicyLoss
    if name == "compute_avtag_loss":
        from gr00t.model.modules.vt_closed_loop.losses import compute_avtag_loss
        return compute_avtag_loss
    lazy = {
        "ContactGate": "contact_gate",
        "ExecutionMonitor": "execution_monitor",
        "IntentManifoldAdapter": "intent_manifold_adapter",
        "RecoveryExpert": "recovery_expert",
        "SafetyProjector": "safety_projector",
        "StructuredActionDiT": "structured_action_dit",
        "TactileEncoder": "tactile_encoder",
        "TactileLateDenoisingRefiner": "tactile_late_denoising_refiner",
        "VisuoTactileFutureHead": "future_head",
    }
    if name in lazy:
        import importlib
        mod = importlib.import_module(f"gr00t.model.modules.vt_closed_loop.{lazy[name]}")
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
