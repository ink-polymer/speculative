#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
DIFFUSION_PYTHON="${DIFFUSION_PYTHON:-.artifacts/gbv-test-venv/bin/python}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
if [[ $# -eq 0 ]]; then
  for tag in t03 t06 t10; do
    "$DIFFUSION_PYTHON" -m gbv_experiments.terminal_formal plan --study "configs/diffusion_tree_${tag}.json"
  done
else
  exec "$DIFFUSION_PYTHON" -m gbv_experiments.terminal_formal --study configs/diffusion_tree_t10.json "$@"
fi
