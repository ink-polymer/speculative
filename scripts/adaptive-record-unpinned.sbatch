#!/usr/bin/env bash
#SBATCH --job-name=fuyile-adaptive-record
#SBATCH --account=naiss2026-3-658-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=300G
#SBATCH --time=3-00:00:00
#SBATCH --output=/nobackup/proj/disk/naiss2026-3-658/personal/jiage91/fuyile/logs/adaptive-record-%j.out
#SBATCH --error=/nobackup/proj/disk/naiss2026-3-658/personal/jiage91/fuyile/logs/adaptive-record-%j.err
set -euo pipefail
source /nobackup/proj/disk/naiss2026-3-658/personal/jiage91/fuyile/scripts/common_env.sh
test -f "$FUYILE_ROOT/logs/adaptive-record-validated.txt"
fuyile_load_gpu_toolchain
: "${ADAPTIVE_MODEL_INDEX:?Set ADAPTIVE_MODEL_INDEX to 0 (Qwen3-4B) or 1 (Qwen3-8B)}"
ADAPTIVE_SOURCE="${ADAPTIVE_SOURCE:-$FUYILE_ROOT/src/adaptive}"
case "$ADAPTIVE_MODEL_INDEX" in
  0) MODEL_TAG=4b ;;
  1) MODEL_TAG=8b ;;
  *) echo "ADAPTIVE_MODEL_INDEX must be 0 or 1" >&2; exit 2 ;;
esac
ADAPTIVE_RUN_TAG="${ADAPTIVE_RUN_TAG:-${MODEL_TAG}-unpinned}"
if [[ ! "$ADAPTIVE_RUN_TAG" =~ ^[a-z0-9][a-z0-9._-]*$ ]]; then
  echo "ADAPTIVE_RUN_TAG must be a safe relative name" >&2
  exit 2
fi
cd "$ADAPTIVE_SOURCE/adaptivetree_paper"
export PATH="$FUYILE_ROOT/envs/adaptive/bin:$PATH"
export PAPER_PYTHON="$FUYILE_ROOT/envs/adaptive/bin/python"
export PYTHONPATH="$PWD/src"
trap 'rc=$?; printf "%s model=%s exit_code=%s\n" "$(date -Is)" "$MODEL_TAG" "$rc" > "$FUYILE_ROOT/logs/adaptive-record-${SLURM_JOB_ID}.status"' EXIT
bash scripts/run_paper_t0_full.sh all \
  --model-index "$ADAPTIVE_MODEL_INDEX" \
  --nproc-per-node 1 \
  --greedy-audit-policy record-bf16-mismatches \
  --data-dir "$FUYILE_ROOT/data/adaptive" \
  --run-dir "$FUYILE_ROOT/runs/adaptive-record-${ADAPTIVE_RUN_TAG}"
