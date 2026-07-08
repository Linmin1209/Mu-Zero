# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backward-compatible alias for native 4-frame config (no tactile / no FLUX)."""

from pathlib import Path

_src = Path(__file__).resolve().parent / "robocasa365_config_4frame.py"
exec(compile(_src.read_text(), str(_src), "exec"))
