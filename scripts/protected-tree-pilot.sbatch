#!/usr/bin/env bash
#SBATCH --job-name=fuyile-protected-pilot
#SBATCH --account=naiss2026-3-658-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=01:00:00
#SBATCH --output=/nobackup/proj/disk/naiss2026-3-658/personal/jiage91/fuyile/logs/protected-pilot-%j.out
#SBATCH --error=/nobackup/proj/disk/naiss2026-3-658/personal/jiage91/fuyile/logs/protected-pilot-%j.err
set -euo pipefail
source /nobackup/proj/disk/naiss2026-3-658/personal/jiage91/fuyile/scripts/common_env.sh
fuyile_load_gpu_toolchain
: "${PROTECTED_PILOT_SOURCE:?Set isolated source snapshot}"
cd "$PROTECTED_PILOT_SOURCE"
export PYTHONPATH="$PWD/src"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHON_BIN="$FUYILE_ROOT/envs/gbv/bin/python"
export PATH="$FUYILE_ROOT/envs/gbv/bin:$PATH"
PROTECTED_PILOT_OUTPUT="$FUYILE_ROOT/runs/protected-pilot-${SLURM_JOB_ID}"
mkdir -p "$PROTECTED_PILOT_OUTPUT"
trap 'rc=$?; printf "job=%s exit=%s\n" "$SLURM_JOB_ID" "$rc"' EXIT
srun --ntasks=1 "$PYTHON_BIN" -m pytest tests/gbv_paper -q --tb=short \
    --junitxml="$PROTECTED_PILOT_OUTPUT/unit_tests.xml"
srun --ntasks=1 "$PYTHON_BIN" -m gbv_experiments.protected_pilot \
    --output "$PROTECTED_PILOT_OUTPUT" --gpu-family GH200 --tokens 96
