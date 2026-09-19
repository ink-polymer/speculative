#!/usr/bin/env bash
set -euo pipefail
model=${1:?Use qwen3_4b or qwen3_8b}
case "$model" in
  qwen3_4b|qwen3_8b) ;;
  *) exit 2 ;;
esac
study=/root/dp-cache-reuse-20260917
results=/root/dp-main-natural-cache-20260917
runtime=/root/autodl-tmp/envs/speculative/bin/python
export PYTHONPATH="$study/src:$study/scripts"
export HF_HOME=/root/autodl-tmp/hf-cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
cd "$study"
resume_quality() {
  screen -dmS "dp-quality-after-natural-$model" bash -c \
    "cd /root/dp-paper-study-20260916 && PYTHONPATH=/root/dp-paper-study-20260916/src:/root/dp-paper-study-20260916/scripts HF_HOME=/root/autodl-tmp/hf-cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 $runtime scripts/run_dp_paper_memory_safe.py --model $model --phase quality_math --output /root/dp-paper-results-memory-safe-20260917/quality_math/$model --reuse-directory /root/dp-paper-results-single-wave-20260917/quality_math/$model --data-dir /root/autodl-tmp/data/sampling-t1-formal-16c0e91 --calibration /root/autodl-tmp/outputs/audited-global-tree-matrix-20260916/$model/calibration.json >> /root/dp-paper-results-memory-safe-20260917/queue-after-input256-$model.log 2>&1"
}
trap resume_quality EXIT
"$runtime" -m pytest -q tests/gbv_paper
"$runtime" scripts/benchmark_main_natural.py --model "$model" \
  --output "$results/$model" --datasets gsm8k,math500,humaneval,mbpp_sanitized \
  --concurrencies 1,4,8 --requests 32 --seeds 17,29,43 --repeats 3 --max-new-tokens 256 \
  --data-dir /root/autodl-tmp/data/sampling-t1-formal-16c0e91 \
  --calibration "/root/autodl-tmp/outputs/audited-global-tree-matrix-20260916/$model/calibration.json"
