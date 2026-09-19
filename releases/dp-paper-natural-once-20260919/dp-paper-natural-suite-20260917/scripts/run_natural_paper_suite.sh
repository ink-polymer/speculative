#!/usr/bin/env bash
set -uo pipefail
model=${1:?model required}
case "$model" in qwen3_4b|qwen3_8b) ;; *) exit 2;; esac
study=/root/dp-paper-natural-suite-20260917
results=/root/dp-paper-natural-results-20260917
runtime=/root/autodl-tmp/envs/speculative/bin/python
data=/root/autodl-tmp/data/sampling-t1-formal-16c0e91
calibration="/root/autodl-tmp/outputs/audited-global-tree-matrix-20260916/$model/calibration.json"
export PYTHONPATH="$study/src:$study/scripts"
export HF_HOME=/root/autodl-tmp/hf-cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$study" || exit 2
mkdir -p "$results/$model"
"$runtime" -m pytest -q tests/gbv_paper || exit 3
"$runtime" scripts/report_natural_paper.py --results "$results" --plan-only || exit 3
for smoke_phase in native_c1 ablation correctness; do
  "$runtime" scripts/run_natural_paper_phase.py --model "$model" --phase "$smoke_phase" --smoke \
    --output "$results/$model/nonformal-smoke-$smoke_phase" --data-dir "$data" --calibration "$calibration" \
    >> "$results/$model/smoke.log" 2>&1 || exit 4
done
for phase in main128 quality correctness native_c1 budget ablation long_output heterogeneous; do
  "$runtime" scripts/run_natural_paper_phase.py --model "$model" --phase "$phase" \
    --output "$results/$model/$phase" --data-dir "$data" --calibration "$calibration" \
    >> "$results/$model/phase-$phase.log" 2>&1
  status=$?
  printf '%s phase=%s exit=%s\n' "$(date -u +%FT%TZ)" "$phase" "$status"
  if [ "$status" -ne 0 ]; then
    printf 'phase=%s exit=%s\n' "$phase" "$status" >> "$results/$model/FAILED_PHASES.txt"
  fi
  if [ "$phase" = quality ]; then
    "$runtime" scripts/grade_natural_paper_quality.py --input "$results/$model/quality" --data-dir "$data" \
      >> "$results/$model/scoring.log" 2>&1
    status=$?
    if [ "$status" -ne 0 ]; then printf 'phase=quality_scoring exit=%s\n' "$status" >> "$results/$model/FAILED_PHASES.txt"; fi
  fi
  "$runtime" scripts/report_natural_paper.py --results "$results" >> "$results/$model/report.log" 2>&1
done
printf '%s queue-finished; inspect FAILED_PHASES.txt and per-phase complete.json before claiming completion\n' "$(date -u +%FT%TZ)"
