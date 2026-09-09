#!/usr/bin/env bash
#SBATCH --job-name=adaptive-cost
#SBATCH --account=naiss2026-3-658-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=300G
#SBATCH --time=3-00:00:00
#SBATCH --output=/nobackup/proj/disk/naiss2026-3-658/personal/jiage91/fuyile/logs/adaptive-cost-%j.out
#SBATCH --error=/nobackup/proj/disk/naiss2026-3-658/personal/jiage91/fuyile/logs/adaptive-cost-%j.err
set -euo pipefail

source /nobackup/proj/disk/naiss2026-3-658/personal/jiage91/fuyile/scripts/common_env.sh
test -f "$FUYILE_ROOT/logs/adaptive-record-validated.txt"
fuyile_load_gpu_toolchain
: "${ADAPTIVE_MODEL_INDEX:?Set ADAPTIVE_MODEL_INDEX to 0 (Qwen3-4B) or 1 (Qwen3-8B)}"
: "${ADAPTIVE_SOURCE:?Set ADAPTIVE_SOURCE to the immutable source worktree}"
ADAPTIVE_SMOKE_COUNT="${ADAPTIVE_SMOKE_COUNT:-2}"
case "$ADAPTIVE_MODEL_INDEX" in
  0) MODEL_TAG=4b ;;
  1) MODEL_TAG=8b ;;
  *) echo "ADAPTIVE_MODEL_INDEX must be 0 or 1" >&2; exit 2 ;;
esac
if [[ ! "$ADAPTIVE_SMOKE_COUNT" =~ ^[0-9]+$ ]]; then
  echo "ADAPTIVE_SMOKE_COUNT must be a nonnegative integer" >&2
  exit 2
fi

RUN_KIND=diagnostic
if (( ADAPTIVE_SMOKE_COUNT > 0 )); then
  RUN_KIND=diagnostic-smoke
fi
RUN_DIR="$FUYILE_ROOT/runs/adaptive-cost-${MODEL_TAG}-${RUN_KIND}-${SLURM_JOB_ID}"
cd "$ADAPTIVE_SOURCE/adaptivetree_paper"
export PATH="$FUYILE_ROOT/envs/adaptive/bin:$PATH"
export PAPER_PYTHON="$FUYILE_ROOT/envs/adaptive/bin/python"
export PYTHONPATH="$PWD/src"
trap 'rc=$?; printf "%s model=%s run_dir=%s exit_code=%s\n" "$(date -Is)" "$MODEL_TAG" "$RUN_DIR" "$rc" > "$FUYILE_ROOT/logs/adaptive-cost-${SLURM_JOB_ID}.status"' EXIT

arguments=(
  all
  --experimental-cost-attribution
  --model-index "$ADAPTIVE_MODEL_INDEX"
  --nproc-per-node 1
  --greedy-audit-policy record-bf16-mismatches
  --data-dir "$FUYILE_ROOT/data/adaptive"
  --run-dir "$RUN_DIR"
)
if (( ADAPTIVE_SMOKE_COUNT > 0 )); then
  arguments+=(--smoke-count "$ADAPTIVE_SMOKE_COUNT")
fi
bash scripts/run_paper_t0_full.sh "${arguments[@]}"
