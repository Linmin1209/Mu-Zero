#!/usr/bin/env bash
# One-time RoboCasa365 sim setup for GR00T eval.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "$SCRIPT_DIR/../../gr00t/eval/sim/robocasa365/setup_RoboCasa365.sh"
