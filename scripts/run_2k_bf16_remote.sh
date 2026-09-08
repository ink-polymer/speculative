#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${PROJECT_ROOT}"

export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PROMPTS="datasets/processed/specblock_official/prompts_benchmark.jsonl"
TRAIN_DATA="datasets/generated/rank_train_tree15_disjoint_2k.jsonl"
CONFIG="configs/qwen3_4b_cuda_2k_retrained.json"
CHECKPOINT="checkpoints/rank_head_tree15_retrained_2k.pt"
OUTPUT_DIR="outputs/remote_4090/2k_bf16"
LOG="logs/bench_2k_bf16_clean.log"

mkdir -p "${OUTPUT_DIR}" logs checkpoints
exec > >(tee -a "${LOG}") 2>&1

echo "run_started=$(date -Iseconds)"
nvidia-smi
sha256sum "${PROMPTS}" "${TRAIN_DATA}" "${CONFIG}" > "${OUTPUT_DIR}/input_sha256.txt"

# The target and official DFlash draft stay frozen. The architecture-specific
# rank head is randomly initialized by train_rank_head and trained from scratch.
# A completed checkpoint is reused when resuming an interrupted benchmark.
if [[ ! -f "${CHECKPOINT}" ]]; then
  .venvs/bench/bin/python -m dflash_specblock.train_rank_head \
    --config "${CONFIG}" \
    --train-data "${TRAIN_DATA}" \
    --output "${CHECKPOINT}" \
    --epochs 3 \
    --learning-rate 2e-4 \
    --device cuda:0
else
  echo "reusing_rank_checkpoint=${CHECKPOINT}"
fi
sha256sum "${CHECKPOINT}" >> "${OUTPUT_DIR}/input_sha256.txt"

.venvs/bench/bin/python scripts/benchmark_official_ddtree.py \
  --prompts "${PROMPTS}" \
  --output "${OUTPUT_DIR}/qwen3_4b_official_dflash_ddtree_bf16.jsonl" \
  --model models/Qwen3-4B \
  --draft models/Qwen3-4B-DFlash-b16 \
  --device cuda:0 \
  --tree-budget 60 \
  --max-samples 2000 \
  --max-new-tokens 128 \
  --resume

.venvs/bench/bin/python -m dflash_specblock.benchmark \
  --config "${CONFIG}" \
  --prompts "${PROMPTS}" \
  --output "${OUTPUT_DIR}/qwen3_4b_own_bf16.jsonl" \
  --max-prompts 2000 \
  --max-new-tokens 128 \
  --device cuda:0

.venvs/bench/bin/python scripts/summarize_remote_benchmarks.py \
  --input-dir "${OUTPUT_DIR}" \
  --prompts "${PROMPTS}" \
  --json-output "${OUTPUT_DIR}/summary_bf16.json" \
  --markdown-output "${OUTPUT_DIR}/summary_bf16.md"

{
  echo "run_finished=$(date -Iseconds)"
  .venvs/bench/bin/python --version
  .venvs/bench/bin/python -c 'import torch, transformers; print("torch=" + torch.__version__); print("transformers=" + transformers.__version__); print("cuda=" + str(torch.version.cuda))'
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
} > "${OUTPUT_DIR}/environment_manifest.txt"
echo "run_finished=$(date -Iseconds)"
