#!/bin/bash
#SBATCH --job-name=gbv-v21-bench
#SBATCH --account=naiss2026-3-658-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=00:30:00
#SBATCH --output=/nobackup/proj/disk/naiss2026-3-658/personal/jiage91/fuyile/logs/gbv-v21-bench-%j.out
#SBATCH --error=/nobackup/proj/disk/naiss2026-3-658/personal/jiage91/fuyile/logs/gbv-v21-bench-%j.err
set -euo pipefail
source /nobackup/proj/disk/naiss2026-3-658/personal/jiage91/fuyile/scripts/common_env.sh
cd "$FUYILE_ROOT/src/gbv-optimized"
export PYTHONPATH="$PWD/src"
PYTHON="$FUYILE_ROOT/envs/adaptive/bin/python"
"$PYTHON" scripts/benchmark-tree-gbv-v21-ratio-transport.py
