#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="${CONFIG:-configs/gbv_paper_full.json}"
DATA_DIR="${DATA_DIR:-datasets/gbv_paper_full}"
OUTPUT="${OUTPUT:-outputs/gbv_paper_full}"
DEVICE="${DEVICE:-cuda:0}"
CODE_BACKEND="${CODE_BACKEND:-docker}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

mkdir -p "$OUTPUT"
"$PYTHON_BIN" -m pytest tests/gbv_paper --junitxml="$OUTPUT/unit_tests.xml"
"$PYTHON_BIN" scripts/gbv_paper.py prepare --config "$CONFIG" --data-dir "$DATA_DIR"
"$PYTHON_BIN" scripts/gbv_paper.py audit --config "$CONFIG" --data-dir "$DATA_DIR" --output "$OUTPUT/data_audit.json"
"$PYTHON_BIN" scripts/gbv_paper.py validate-gold --config "$CONFIG" --data-dir "$DATA_DIR" --output "$OUTPUT/gold_audit.json" --code-backend "$CODE_BACKEND"
"$PYTHON_BIN" scripts/gbv_paper.py plan --config "$CONFIG" --output "$OUTPUT/plan.json"
"$PYTHON_BIN" scripts/gbv_paper.py check-model --config "$CONFIG" --output "$OUTPUT/gpu_preflight.json" --device "$DEVICE" --code-backend "$CODE_BACKEND"
"$PYTHON_BIN" scripts/gbv_paper.py run --config "$CONFIG" --data-dir "$DATA_DIR" --device "$DEVICE" --output "$OUTPUT"
"$PYTHON_BIN" scripts/gbv_paper.py score --run-dir "$OUTPUT" --data-dir "$DATA_DIR" --code-backend "$CODE_BACKEND"
"$PYTHON_BIN" scripts/gbv_paper.py report --run-dir "$OUTPUT" --output "$OUTPUT/report"
