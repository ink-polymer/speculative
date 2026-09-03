#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python}"
if [[ -n "${CONFIG:-}" ]]; then
  echo "This entry uses SUITE for two-model scheduling. For a single config, use scripts/gbv_paper.py --help." >&2
  exit 2
fi
SUITE="${SUITE:-configs/gbv_paper_suite.json}"
DATA_DIR="${DATA_DIR:-datasets/gbv_paper_ddtree_counts}"
OUTPUT="${OUTPUT:-outputs/gbv_paper_two_models}"
DEVICE="${DEVICE:-cuda:0}"
CODE_BACKEND="${CODE_BACKEND:-docker}"
PHASE="${1:-gbv-first}"
if (( $# )); then shift; fi
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

mkdir -p "$OUTPUT"
if [[ "$PHASE" == "plan" ]]; then
  "$PYTHON_BIN" scripts/gbv_paper.py plan-suite --suite "$SUITE" --phase complete --output "$OUTPUT/plan_complete.json" "$@"
  exit 0
fi
case "$PHASE" in gbv-first|main|complete) ;; *) echo "Usage: $0 [gbv-first|main|complete|plan] [--model-ids qwen3_4b qwen3_8b]" >&2; exit 2 ;; esac
"$PYTHON_BIN" -m pytest tests/gbv_paper --junitxml="$OUTPUT/unit_tests.xml"
"$PYTHON_BIN" scripts/gbv_paper.py run-suite --suite "$SUITE" --phase "$PHASE" --data-dir "$DATA_DIR" --device "$DEVICE" --code-backend "$CODE_BACKEND" --output "$OUTPUT" "$@"
