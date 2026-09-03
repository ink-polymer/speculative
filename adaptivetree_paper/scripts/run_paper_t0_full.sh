#!/usr/bin/env bash
set -euo pipefail
PAPER_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PAPER_ROOT"
export PYTHONPATH="$PAPER_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
# "full" is the complete OFFICIAL matrix, not full dataset splits.
# Limits and seed=0 sampling exactly follow the pinned DDTree run_benchmark.sh.
if [[ "$#" -eq 0 ]]; then
  set -- all
fi
"${PAPER_PYTHON:-python}" -m dflash_specblock.paper "$@"
