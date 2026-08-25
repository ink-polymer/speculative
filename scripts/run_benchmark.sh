#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

python -m dflash_specblock.benchmark \
  --config configs/qwen3_4b_cuda.json \
  --prompts examples/prompts.jsonl \
  --output outputs/qwen3_4b_cuda.jsonl \
  --max-new-tokens 128
