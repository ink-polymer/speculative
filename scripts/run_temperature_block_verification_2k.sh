#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${PROJECT_ROOT}"

PROMPTS="${PROMPTS:-datasets/processed/dflash_official/prompts_benchmark_2k.jsonl}"
OUTPUT="${OUTPUT:-outputs/temperature_1_block_verification_2k.jsonl}"
SUMMARY="${SUMMARY:-outputs/temperature_1_block_verification_2k_summary.json}"
MODEL="${MODEL:-models/Qwen3-4B}"
DRAFT="${DRAFT:-models/Qwen3-4B-DFlash-b16}"
TEMPERATURE="${TEMPERATURE:-1.0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
TREE_BUDGET="${TREE_BUDGET:-60}"
MAX_SAMPLES="${MAX_SAMPLES:-2000}"
SEED="${SEED:-42}"
GBV_PATHS="${GBV_PATHS:-3}"

python scripts/benchmark_official_ddtree.py \
  --prompts "${PROMPTS}" \
  --output "${OUTPUT}" \
  --model "${MODEL}" \
  --draft "${DRAFT}" \
  --temperature "${TEMPERATURE}" \
  --tree-budget "${TREE_BUDGET}" \
  --gbv-paths "${GBV_PATHS}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --max-samples "${MAX_SAMPLES}" \
  --seed "${SEED}" \
  --resume

python scripts/summarize_temperature_comparison.py \
  --input "${OUTPUT}" \
  --output "${SUMMARY}"
