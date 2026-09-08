#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
GBV_FORMAL_PYTHON="${GBV_FORMAL_PYTHON:-python}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
exec "$GBV_FORMAL_PYTHON" -m gbv_experiments.terminal_formal "$@"
