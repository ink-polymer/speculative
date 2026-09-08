#!/usr/bin/env bash
#SBATCH --job-name=tree-gbv-v7-lazy
#SBATCH --account=naiss2026-3-658-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=00:20:00
#SBATCH --output=/nobackup/proj/disk/naiss2026-3-658/personal/jiage91/fuyile/logs/tree-gbv-v7-lazy-%j.out
#SBATCH --error=/nobackup/proj/disk/naiss2026-3-658/personal/jiage91/fuyile/logs/tree-gbv-v7-lazy-%j.err
set -euo pipefail
source /nobackup/proj/disk/naiss2026-3-658/personal/jiage91/fuyile/scripts/common_env.sh
cd "$FUYILE_ROOT/src/gbv-optimized"
export PYTHONPATH="$PWD/src"
PYTHON="$FUYILE_ROOT/envs/gbv/bin/python"
trap 'rc=$?; printf "%s exit_code=%s\n" "$(date -Is)" "$rc" > "$FUYILE_ROOT/logs/tree-gbv-v7-lazy-${SLURM_JOB_ID}.status"' EXIT
date -Is
hostname
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
"$PYTHON" -m py_compile "$FUYILE_ROOT/scripts/benchmark-tree-gbv-v7-lazy.py"
"$PYTHON" "$FUYILE_ROOT/scripts/benchmark-tree-gbv-v7-lazy.py"
