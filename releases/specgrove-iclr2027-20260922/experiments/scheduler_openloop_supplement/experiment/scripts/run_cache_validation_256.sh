#!/usr/bin/env bash
set -euo pipefail
model=${1:?Use qwen3_4b or qwen3_8b}
case "$model" in
  qwen3_4b|qwen3_8b) ;;
  *) exit 2 ;;
esac
study=/root/dp-cache-reuse-20260917
results=/root/dp-cache-results-short256-20260917
runtime=/root/autodl-tmp/envs/speculative/bin/python
export PYTHONPATH="$study/src:$study/scripts"
export HF_HOME=/root/autodl-tmp/hf-cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
cd "$study"
"$runtime" scripts/benchmark_draft_cache_reuse.py --model "$model" \
  --output "$results/matched/$model" --prefixes 256 --concurrencies 1,4,8 \
  --datasets '' --seeds 17,29 --repeats 3 --max-new-tokens 128 \
  --data-dir /root/autodl-tmp/data/sampling-t1-formal-16c0e91 \
  --calibration "/root/autodl-tmp/outputs/audited-global-tree-matrix-20260916/$model/calibration.json"
# Resume only natural-prompt quality generation. Do not restart the old
# context matrix, whose contract still contains the cancelled 8192 cases.
export PYTHONPATH=/root/dp-paper-study-20260916/src:/root/dp-paper-study-20260916/scripts
cd /root/dp-paper-study-20260916
"$runtime" scripts/run_dp_paper_memory_safe.py --model "$model" --phase quality_math \
  --output "/root/dp-paper-results-memory-safe-20260917/quality_math/$model" \
  --reuse-directory "/root/dp-paper-results-single-wave-20260917/quality_math/$model" \
  --data-dir /root/autodl-tmp/data/sampling-t1-formal-16c0e91 \
  --calibration "/root/autodl-tmp/outputs/audited-global-tree-matrix-20260916/$model/calibration.json"
