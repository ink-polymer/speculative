#!/usr/bin/env bash
set -euo pipefail
model=${1:?Use qwen3_4b or qwen3_8b}
case "$model" in
  qwen3_4b|qwen3_8b) ;;
  *) printf 'Unknown model shard\n' >&2; exit 2 ;;
esac
study=/root/dp-paper-study-20260916
results=/root/dp-paper-results-memory-safe-20260917
previous=/root/dp-paper-results-single-wave-20260917
runtime=/root/autodl-tmp/envs/speculative/bin/python
export PYTHONPATH="$study/src:$study/scripts"
export HF_HOME=/root/autodl-tmp/hf-cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
cd "$study"
for phase in context quality_math; do
  "$runtime" scripts/run_dp_paper_memory_safe.py --model "$model" --phase "$phase" \
    --output "$results/$phase/$model" --reuse-directory "$previous/$phase/$model" \
    --data-dir /root/autodl-tmp/data/sampling-t1-formal-16c0e91 \
    --calibration "/root/autodl-tmp/outputs/audited-global-tree-matrix-20260916/$model/calibration.json"
done
"$runtime" scripts/summarize_dp_paper_single_wave.py --directory "$results"
