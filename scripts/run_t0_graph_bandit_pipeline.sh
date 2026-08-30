#!/usr/bin/env bash
set -euo pipefail

export PATH=/root/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

actions="ddtree:128,calibrated:128:0.9:0,calibrated:128:1.15:0,calibrated:128:1:0.15"
checkpoint="checkpoints/topology_bandit_t0_graph_v1.json"
mkdir -p outputs/topology_bandit_t0_graph_v1 logs

python scripts/benchmark_topology_bandit.py \
  --prompts datasets/processed/dflash_official/prompts_policy_train_4domain_disjoint_192.jsonl \
  --output outputs/topology_bandit_t0_graph_v1/train192x128.jsonl \
  --checkpoint "${checkpoint}" \
  --draft checkpoints/dflash_draft_t0_disjoint_v1 \
  --fixed-draft models/Qwen3-4B-DFlash-b16 \
  --temperature 0 \
  --actions "${actions}" \
  --initial-action ddtree:128 \
  --bandit-exploration-scale 0.2 \
  --bandit-warmup-episodes 12 \
  --bandit-train \
  --skip-comparators \
  --max-new-tokens 128 \
  --enable-cpp-compact \
  --cuda-graph-policy \
  --cuda-graph-max-cache-len 2048 \
  > logs/topology_bandit_t0_graph_train.log 2>&1

python scripts/benchmark_topology_bandit.py \
  --prompts datasets/processed/dflash_official/prompts_policy_validation_4domain_disjoint_48.jsonl \
  --output outputs/topology_bandit_t0_graph_v1/validation48x512.jsonl \
  --checkpoint "${checkpoint}" \
  --draft checkpoints/dflash_draft_t0_disjoint_v1 \
  --fixed-draft models/Qwen3-4B-DFlash-b16 \
  --temperature 0 \
  --actions "${actions}" \
  --initial-action ddtree:128 \
  --bandit-exploration-scale 0.2 \
  --bandit-warmup-episodes 12 \
  --bandit-eval \
  --max-new-tokens 512 \
  --enable-cpp-compact \
  --cuda-graph-policy \
  --cuda-graph-max-cache-len 2048 \
  > logs/topology_bandit_t0_graph_validation.log 2>&1

python scripts/summarize_topology_bandit.py \
  --input outputs/topology_bandit_t0_graph_v1/validation48x512.jsonl \
  --output outputs/topology_bandit_t0_graph_v1/validation48x512_summary.json \
  > logs/topology_bandit_t0_graph_validation_summary.log 2>&1
