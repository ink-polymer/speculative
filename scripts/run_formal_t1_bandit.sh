#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
export PATH=/root/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

tmux kill-session -t rl_t1_formal2k 2>/dev/null || true
tmux new-session -d -s rl_t1_formal2k \
  "cd \"${PROJECT_ROOT}\" && python scripts/benchmark_topology_bandit.py \
    --prompts datasets/processed/dflash_official/prompts_benchmark_2k.jsonl \
    --output outputs/topology_bandit_t1_v1/formal2k_x128.jsonl \
    --checkpoint checkpoints/topology_bandit_t1_v1.json \
    --temperature 1 \
    --actions gbv:2,gbv:3,gbv:5 \
    --initial-action gbv:2 \
    --bandit-exploration-scale 0.2 \
    --bandit-warmup-episodes 16 \
    --bandit-eval \
    --max-new-tokens 128 \
    --enable-cpp-compact \
    --resume \
    > logs/rl_t1_formal2k.log 2>&1 && \
   python scripts/summarize_topology_bandit.py \
    --input outputs/topology_bandit_t1_v1/formal2k_x128.jsonl \
    --output outputs/topology_bandit_t1_v1/formal2k_x128_summary.json \
    >> logs/rl_t1_formal2k.log 2>&1"

sleep 3
tmux ls
wc -l outputs/topology_bandit_t1_v1/formal2k_x128.jsonl
tail -n 4 logs/rl_t1_formal2k.log
