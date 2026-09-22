#!/usr/bin/env bash
set -euo pipefail
study=/root/dp-paper-study-20260916
results=/root/dp-paper-results-20260916
runtime=/root/autodl-tmp/envs/speculative/bin/python
export PYTHONPATH="$study/src:$study/scripts"
export HF_HOME=/root/autodl-tmp/hf-cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
cd "$study"
mkdir -p "$results"
# Pilot both models first so the second model's ETA does not wait for full4B.
for phase in pilot main budget ablation temperature0 long_output context quality_math; do
  for model in qwen3_4b qwen3_8b; do
    "$runtime" scripts/run_dp_paper_study.py --model "$model" --phase "$phase" \
      --output "$results/$phase/$model" \
      --data-dir /root/autodl-tmp/data/sampling-t1-formal-16c0e91 \
      --calibration "/root/autodl-tmp/outputs/audited-global-tree-matrix-20260916/$model/calibration.json"
  done
done
"$runtime" scripts/summarize_dp_paper_study.py --directory "$results"
printf 'Registered executable batch-study queue complete. Native-serving and official-related-baseline stages are separate prerequisites.\n'
