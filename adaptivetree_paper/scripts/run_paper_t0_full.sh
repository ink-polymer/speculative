#!/usr/bin/env bash
set -euo pipefail
PAPER_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PAPER_ROOT"
export PYTHONPATH="$PAPER_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
# Default is the complete protocol. No max-samples / 2K truncation flags.
if [[ "$#" -eq 0 ]]; then
  set -- all
fi
"${PAPER_PYTHON:-python}" -m dflash_specblock.paper "$@"
