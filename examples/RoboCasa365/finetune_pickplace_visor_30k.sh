#!/usr/bin/env bash
# Finetune GR00T N1.7 on RoboCasa365 PickPlaceToasterToCounter (30k steps)
# T-Rex-style sensor VISOR (tri-path VQ + flow-late). Alias for finetune_pickplace_visor_trex_30k.sh.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/finetune_pickplace_visor_trex_30k.sh" "$@"
