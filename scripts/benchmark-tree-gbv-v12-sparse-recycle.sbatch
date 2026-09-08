#!/usr/bin/env bash
#SBATCH --job-name=tree-gbv-v12-sparse
#SBATCH --account=naiss2026-3-658-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=00:30:00
#SBATCH --output=/nobackup/proj/disk/naiss2026-3-658/personal/jiage91/fuyile/logs/tree-gbv-v12-sparse-%j.out
#SBATCH --error=/nobackup/proj/disk/naiss2026-3-658/personal/jiage91/fuyile/logs/tree-gbv-v12-sparse-%j.err
set -euo pipefail

source /nobackup/proj/disk/naiss2026-3-658/personal/jiage91/fuyile/scripts/common_env.sh
cd "$FUYILE_ROOT/src/gbv-optimized"
export PYTHONPATH="$PWD/src"
PYTHON="$FUYILE_ROOT/envs/gbv/bin/python"
trap 'rc=$?; printf "%s exit_code=%s\n" "$(date -Is)" "$rc" > "$FUYILE_ROOT/logs/tree-gbv-v12-sparse-${SLURM_JOB_ID}.status"' EXIT
"$PYTHON" -m pytest tests/gbv_paper/test_probability_law.py -q
"$PYTHON" -m py_compile "$FUYILE_ROOT/scripts/benchmark-tree-gbv-v12-sparse-recycle.py"
"$PYTHON" "$FUYILE_ROOT/scripts/benchmark-tree-gbv-v12-sparse-recycle.py"
