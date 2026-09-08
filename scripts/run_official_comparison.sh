#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PROMPTS="${PROMPTS:-${PROJECT_ROOT}/datasets/processed/specblock_official/prompts_benchmark_tree15.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/official_references}"
MAX_SAMPLES="${MAX_SAMPLES:-200}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
DEVICE="${DEVICE:-cuda:0}"
DFLASH_PYTHON="${DFLASH_PYTHON:-${PROJECT_ROOT}/.venvs/dflash-official/bin/python}"
DDTREE_PYTHON="${DDTREE_PYTHON:-${PROJECT_ROOT}/.venvs/ddtree-official/bin/python}"

mkdir -p "${OUTPUT_DIR}"

"${DFLASH_PYTHON}" "${PROJECT_ROOT}/scripts/benchmark_official_dflash.py" \
  --prompts "${PROMPTS}" \
  --output "${OUTPUT_DIR}/dflash_official.jsonl" \
  --device "${DEVICE}" \
  --max-samples "${MAX_SAMPLES}" \
  --max-new-tokens "${MAX_NEW_TOKENS}"

"${DDTREE_PYTHON}" "${PROJECT_ROOT}/scripts/benchmark_official_ddtree.py" \
  --prompts "${PROMPTS}" \
  --output "${OUTPUT_DIR}/ddtree_official.jsonl" \
  --device "${DEVICE}" \
  --tree-budget 60 \
  --max-samples "${MAX_SAMPLES}" \
  --max-new-tokens "${MAX_NEW_TOKENS}"
