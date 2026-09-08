#!/usr/bin/env bash
###############################################################################
# DDTree 拓扑 benchmark（本工程实现，NVIDIA CUDA）
#
# 与 scripts/run_benchmark.sh 的唯一差别是 tree_mode=ddtree：DFlash drafter、
# ancestor-only 树验证、KV 压缩和 CUDA 后端完全共用同一套代码路径，因此本脚本与
# SpecBlock benchmark 的结果可以直接对照。
#
# DDTree 不需要 rank head，所以无需先跑 rank 数据生成与训练；prompt 文件默认复用
# 已固定的 200 条 benchmark JSONL，保证与 scripts/run_official_comparison.sh 的
# 官方 DDTree 对照口径一致。
#
# 可选环境变量:
#   CONFIG          默认 configs/qwen3_4b_cuda_ddtree.json
#                   （configs/qwen3_4b_cuda_ddtree_reserve_chain.json 为消融配置）
#   PROMPTS         默认 datasets/processed/specblock_official/prompts_benchmark_tree15.jsonl
#   OUTPUT          默认 outputs/benchmark_ddtree.jsonl
#   MAX_PROMPTS     默认 0（全部）
#   MAX_NEW_TOKENS  默认 128
###############################################################################
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${PROJECT_ROOT}"

CONFIG="${CONFIG:-configs/qwen3_4b_cuda_ddtree.json}"
PROMPTS="${PROMPTS:-datasets/processed/specblock_official/prompts_benchmark_tree15.jsonl}"
OUTPUT="${OUTPUT:-outputs/benchmark_ddtree.jsonl}"
MAX_PROMPTS="${MAX_PROMPTS:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"

if [[ ! -f "${CONFIG}" ]]; then
  echo "配置不存在: ${CONFIG}" >&2
  exit 1
fi
if [[ ! -f "${PROMPTS}" ]]; then
  echo "prompt 文件不存在: ${PROMPTS}" >&2
  echo "先运行 python -m dflash_specblock.dataset_pipeline 生成 benchmark 数据。" >&2
  exit 1
fi

mkdir -p "$(dirname "${OUTPUT}")" logs

echo "配置:      ${CONFIG}"
echo "prompts:   ${PROMPTS}"
echo "输出:      ${OUTPUT}"

python scripts/check_cuda_env.py

python -m dflash_specblock.benchmark \
  --config "${CONFIG}" \
  --prompts "${PROMPTS}" \
  --output "${OUTPUT}" \
  --max-prompts "${MAX_PROMPTS}" \
  --max-new-tokens "${MAX_NEW_TOKENS}"
