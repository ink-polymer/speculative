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
resume_original() {
  # Resume the unchanged old-version quality study, not the cache candidate.
  screen -dmS dp-memory-safe-4b-resume-20260917 bash -c \
    'bash /root/dp-paper-study-20260916/scripts/run_dp_paper_memory_safe_shard.sh qwen3_4b >> /root/dp-paper-results-memory-safe-20260917/queue-4b.log 2>&1'
}
trap resume_original EXIT
"$runtime" scripts/benchmark_draft_cache_reuse.py --model qwen3_4b \
  --output "$results/smoke/qwen3_4b" --prefixes 8192 --concurrencies 8 \
  --datasets '' --seeds 17 --repeats 1 \
  --data-dir /root/autodl-tmp/data/sampling-t1-formal-16c0e91 \
  --calibration /root/autodl-tmp/outputs/audited-global-tree-matrix-20260916/qwen3_4b/calibration.json
"$runtime" scripts/benchmark_draft_cache_reuse.py --model qwen3_4b \
  --output "$results/matched/qwen3_4b" \
  --data-dir /root/autodl-tmp/data/sampling-t1-formal-16c0e91 \
  --calibration /root/autodl-tmp/outputs/audited-global-tree-matrix-20260916/qwen3_4b/calibration.json
