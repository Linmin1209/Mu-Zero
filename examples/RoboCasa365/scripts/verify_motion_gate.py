#!/usr/bin/env python3
"""Smoke test MOSS task-modality gating. Run with project venv:

  /HOME/sysu_xdliang/sysu_xdliang_1/HDD_POOL/linmin/Isaac-GR00T/.venv/bin/python \\
    examples/RoboCasa365/scripts/verify_motion_gate.py
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

print("[i] verify_motion_gate starting...", flush=True)

REPO = Path(__file__).resolve().parents[3]
VENV_PYTHON = REPO / ".venv/bin/python"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

MODULES_DIR = REPO / "gr00t/model/modules"


def _torch():
    import torch

    return torch


def _ensure_pkg(name: str) -> None:
    if name not in sys.modules:
        sys.modules[name] = types.ModuleType(name)


def _load_module_file(module_name: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {module_name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_qwen3_motion():
    _ensure_pkg("gr00t")
    _ensure_pkg("gr00t.model")
    _ensure_pkg("gr00t.model.modules")
    if "gr00t.model.modules.motion" not in sys.modules:
        _load_module_file("gr00t.model.modules.motion", MODULES_DIR / "motion.py")
    if "gr00t.model.modules.qwen3_motion" not in sys.modules:
        _load_module_file(
            "gr00t.model.modules.qwen3_motion",
            MODULES_DIR / "qwen3_motion.py",
        )
    return sys.modules["gr00t.model.modules.qwen3_motion"]


def test_imports_fast() -> None:
    qm = load_qwen3_motion()
    assert hasattr(qm, "MotionFusionGate")
    assert hasattr(qm, "ensure_motion_gate")
    assert qm.is_motion_missing_key("model.visual.motion_gate.net.0.weight")
    assert any("motion_gate" in p for p in qm.motion_state_dict_prefixes())
    print("[ok] qwen3_motion load (fast path)")


def test_motion_fusion_gate() -> None:
    torch = _torch()
    qm = load_qwen3_motion()
    MotionFusionGate = qm.MotionFusionGate

    B, D = 4, 128
    gate = MotionFusionGate(D, hidden_dim=32, init_bias=0.0, g_min=0.0, g_max=0.8)
    text = torch.randn(B, D)
    vision = torch.randn(B, D)
    temporal = torch.randn(B, D)
    out = gate(text, vision, temporal)
    assert out.shape == (B, 1)
    assert (out >= 0).all() and (out <= 0.8).all()
    assert abs(out.mean().item() - 0.4) < 0.15
    print("[ok] MotionFusionGate forward")


def test_apply_motion_gate_mock() -> None:
    torch = _torch()
    qm = load_qwen3_motion()
    MotionFusionGate = qm.MotionFusionGate
    _apply_motion_gate = qm._apply_motion_gate
    _pool_hidden_for_gate = qm._pool_hidden_for_gate

    B, T, V, P, D = 2, 4, 3, 16, 64
    hidden_5d = torch.randn(B, T, V, P, D)
    moss_delta = torch.ones(B, T, V, P, D)
    visual = SimpleNamespace(
        motion_gate=MotionFusionGate(D, 32),
        _gr00t_motion_text_context=torch.randn(B, D),
    )
    gated = _apply_motion_gate(visual, moss_delta.clone(), hidden_5d, B)
    assert gated.shape == moss_delta.shape
    vision_ctx, temporal_ctx = _pool_hidden_for_gate(hidden_5d)
    scale = visual.motion_gate(
        visual._gr00t_motion_text_context, vision_ctx, temporal_ctx
    ).view(B, 1, 1, 1, 1)
    assert torch.allclose(gated, moss_delta * scale)
    print("[ok] _apply_motion_gate scales moss_delta")


def test_ensure_motion_gate() -> None:
    qm = load_qwen3_motion()
    MotionConfig = qm.MotionConfig
    ensure_motion_gate = qm.ensure_motion_gate

    visual = SimpleNamespace(config=SimpleNamespace(hidden_size=96), motion_gate=None)
    cfg = MotionConfig(use_motion=True, motion_use_gating=True, motion_gate_hidden=32)
    ensure_motion_gate(visual, cfg)
    assert visual.motion_gate is not None
    ensure_motion_gate(visual, cfg)
    print("[ok] ensure_motion_gate installs once")


def test_pool_motion_gate_text_context() -> None:
    torch = _torch()
    from gr00t.model.modules.qwen3_motion import pool_motion_gate_text_context

    B, T, D = 2, 8, 16
    embeds = torch.randn(B, T, D)
    input_ids = torch.tensor([[1, 2, 99, 99, 3, 4, 99, 5], [1, 2, 3, 99, 4, 5, 6, 7]])
    attention_mask = torch.ones(B, T)
    image_token_id = 99
    ctx = pool_motion_gate_text_context(embeds, attention_mask, input_ids, image_token_id)
    assert ctx.shape == (B, D)
    print("[ok] pool_motion_gate_text_context")


def test_motion_config_from_model() -> None:
    qm = load_qwen3_motion()
    motion_cfg = qm.motion_config_from_model_config(
        SimpleNamespace(
            use_motion=True,
            motion_use_gating=True,
            motion_gate_hidden=128,
        )
    )
    assert motion_cfg.motion_use_gating is True
    assert motion_cfg.motion_gate_hidden == 128
    print("[ok] motion_config_from_model_config")


def test_setup_import() -> None:
    for key in list(sys.modules.keys()):
        if key == "gr00t" or key.startswith("gr00t."):
            del sys.modules[key]
    from gr00t.model.gr00t_n1d7.setup import Gr00tN1d7Pipeline  # noqa: F401
    from gr00t.model.modules.qwen3_motion import ensure_motion_gate

    assert ensure_motion_gate is not None
    print("[ok] setup + canonical qwen3_motion import")


def main() -> None:
    if VENV_PYTHON.exists():
        print(f"[i] venv python: {VENV_PYTHON}", flush=True)
    print("[i] loading torch (first import may take a few minutes on NFS)...", flush=True)
    _torch()
    print("[ok] torch ready", flush=True)

    for fn in [
        test_imports_fast,
        test_motion_fusion_gate,
        test_apply_motion_gate_mock,
        test_ensure_motion_gate,
        test_pool_motion_gate_text_context,
        test_motion_config_from_model,
        test_setup_import,
    ]:
        fn()
    print("[pass] motion gate smoke: all checks ok")


if __name__ == "__main__":
    main()
