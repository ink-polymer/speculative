#!/usr/bin/env bash
set -euo pipefail

# CUDA_VISIBLE_DEVICES 决定进程可见的 NVIDIA GPU；进程内首张卡仍为 cuda:0。
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

python -m dflash_specblock.cli \
  --config configs/qwen3_4b_cuda.json \
  --prompt "请解释为什么推测解码能够无损加速大语言模型推理。" \
  --max-new-tokens 128
