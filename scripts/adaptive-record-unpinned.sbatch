#!/usr/bin/env bash
#SBATCH --job-name=fuyile-adaptive8b-record
#SBATCH --account=naiss2026-3-658-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=3-00:00:00
#SBATCH --output=/nobackup/proj/disk/naiss2026-3-658/personal/jiage91/fuyile/logs/adaptive-record-%j.out
#SBATCH --error=/nobackup/proj/disk/naiss2026-3-658/personal/jiage91/fuyile/logs/adaptive-record-%j.err
set -euo pipefail
source /nobackup/proj/disk/naiss2026-3-658/personal/jiage91/fuyile/scripts/common_env.sh
test -f "$FUYILE_ROOT/logs/adaptive-record-validated.txt"
fuyile_load_gpu_toolchain
cd "$FUYILE_ROOT/src/adaptive/adaptivetree_paper"
export PATH="$FUYILE_ROOT/envs/adaptive/bin:$PATH"
export PAPER_PYTHON="$FUYILE_ROOT/envs/adaptive/bin/python"
export PYTHONPATH="$PWD/src"
trap 'rc=$?; printf "%s exit_code=%s\n" "$(date -Is)" "$rc" > "$FUYILE_ROOT/logs/adaptive-record-${SLURM_JOB_ID}.status"' EXIT
bash scripts/run_paper_t0_qwen3_8b.sh all --nproc-per-node 1 --greedy-audit-policy record-bf16-mismatches --data-dir "$FUYILE_ROOT/data/adaptive" --run-dir "$FUYILE_ROOT/runs/adaptive-record-8b-unpinned"
