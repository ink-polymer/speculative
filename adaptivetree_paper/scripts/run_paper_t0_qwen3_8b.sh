#!/usr/bin/env bash
set -euo pipefail
ADAPTIVE_8B_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ADAPTIVE_8B_ROOT"
export PYTHONPATH="$ADAPTIVE_8B_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
# No arguments only prints the plan. Use an explicit 'all' to launch experiments.
"${PAPER_PYTHON:-python}" -m dflash_specblock.paper.qwen3_8b "$@"
