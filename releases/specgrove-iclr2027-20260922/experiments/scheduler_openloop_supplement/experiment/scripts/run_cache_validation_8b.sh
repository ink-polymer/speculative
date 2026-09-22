#!/usr/bin/env bash
set -euo pipefail
study=/root/dp-cache-reuse-20260917
results=/root/dp-cache-results-20260917
runtime=/root/autodl-tmp/envs/speculative/bin/python
export PYTHONPATH="$study/src:$study/scripts"
export HF_HOME=/root/autodl-tmp/hf-cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
cd "$study"
# Wait behind the already inspected original 8B queue, not alongside it.
# The command check prevents a reused PID from becoming a new wait target.
while ps -p 18951 -o args= | grep -q 'run_dp_paper_memory_safe_shard.sh qwen3_8b'; do
  sleep 30
done
"$runtime" scripts/benchmark_draft_cache_reuse.py --model qwen3_8b \
  --output "$results/smoke/qwen3_8b" --prefixes 8192 --concurrencies 8 \
  --datasets '' --seeds 17 --repeats 1 \
  --data-dir /root/autodl-tmp/data/sampling-t1-formal-16c0e91 \
  --calibration /root/autodl-tmp/outputs/audited-global-tree-matrix-20260916/qwen3_8b/calibration.json
"$runtime" scripts/benchmark_draft_cache_reuse.py --model qwen3_8b \
  --output "$results/matched/qwen3_8b" \
  --data-dir /root/autodl-tmp/data/sampling-t1-formal-16c0e91 \
  --calibration /root/autodl-tmp/outputs/audited-global-tree-matrix-20260916/qwen3_8b/calibration.json
