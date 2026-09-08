#!/usr/bin/env bash
#SBATCH --job-name=fuyile-protected-fusion
#SBATCH --account=naiss2026-3-658-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=00:20:00
#SBATCH --output=/nobackup/proj/disk/naiss2026-3-658/personal/jiage91/fuyile/logs/protected-fusion-%j.out
#SBATCH --error=/nobackup/proj/disk/naiss2026-3-658/personal/jiage91/fuyile/logs/protected-fusion-%j.err
set -euo pipefail
source /nobackup/proj/disk/naiss2026-3-658/personal/jiage91/fuyile/scripts/common_env.sh
fuyile_load_gpu_toolchain
: "${PROTECTED_PILOT_SOURCE:?Set isolated source snapshot}"
: "${PROTECTED_CAPTURE_RUN:?Set completed capture run}"
cd "$PROTECTED_PILOT_SOURCE"
export PYTHONPATH="$PWD/src"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PATH="$FUYILE_ROOT/envs/gbv/bin:$PATH"
PROTECTED_FUSION_OUTPUT="$FUYILE_ROOT/runs/protected-fusion-${SLURM_JOB_ID}"
export TORCHINDUCTOR_CACHE_DIR="$TMPDIR/inductor"
export TRITON_CACHE_DIR="$TMPDIR/triton"
mkdir -p "$PROTECTED_FUSION_OUTPUT"
trap 'rc=$?; printf "job=%s exit=%s\n" "$SLURM_JOB_ID" "$rc"' EXIT
srun --ntasks=1 "$FUYILE_ROOT/envs/gbv/bin/python" -m gbv_experiments.protected_compile_probe \
    --capture-run "$PROTECTED_CAPTURE_RUN" --output "$PROTECTED_FUSION_OUTPUT"
