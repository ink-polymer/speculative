#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

prompts="datasets/processed/dflash_official/prompts_benchmark_2k.jsonl"
out="outputs/remote_4090/dflash_official_2k_2048"
log="logs/official_2k_2048.log"
mkdir -p "$out" logs

echo "run_started=$(date -Iseconds)" | tee -a "$log"
.venvs/bench/bin/python scripts/benchmark_official_ddtree.py \
  --prompts "$prompts" \
  --output "$out/qwen3_4b_dflash_ddtree_bf16.jsonl" \
  --model models/Qwen3-4B \
  --draft models/Qwen3-4B-DFlash-b16 \
  --device cuda:0 \
  --tree-budget 60 \
  --max-samples 2000 \
  --max-new-tokens 2048 \
  --resume 2>&1 | tee -a "$log"

.venvs/bench/bin/python -m dflash_specblock.benchmark \
  --config configs/qwen3_4b_cuda_2k_retrained.json \
  --prompts "$prompts" \
  --output "$out/qwen3_4b_own_bf16.jsonl" \
  --max-prompts 2000 \
  --max-new-tokens 2048 \
  --device cuda:0 2>&1 | tee -a "$log"

echo "run_finished=$(date -Iseconds)" | tee -a "$log"
