#!/usr/bin/env bash
#SBATCH --job-name=fuyile-atom-tree-pilot
#SBATCH --account=naiss2026-3-658-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=01:00:00
#SBATCH --output=/nobackup/proj/disk/naiss2026-3-658/personal/jiage91/fuyile/logs/atom-tree-pilot-%j.out
#SBATCH --error=/nobackup/proj/disk/naiss2026-3-658/personal/jiage91/fuyile/logs/atom-tree-pilot-%j.err
set -euo pipefail
source /nobackup/proj/disk/naiss2026-3-658/personal/jiage91/fuyile/scripts/common_env.sh
fuyile_load_gpu_toolchain
: "${ATOM_PILOT_SOURCE:?Set the isolated new source snapshot, not an active experiment directory}"
cd "$ATOM_PILOT_SOURCE"
export PYTHONPATH="$PWD/src"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PATH="$FUYILE_ROOT/envs/gbv/bin:$PATH"
ATOM_PILOT_PYTHON="$FUYILE_ROOT/envs/gbv/bin/python"
ATOM_PILOT_OUTPUT="$FUYILE_ROOT/runs/atom-tree-pilot-${SLURM_JOB_ID}"
mkdir -p "$ATOM_PILOT_OUTPUT"
trap 'rc=$?; printf "job=%s exit=%s\n" "$SLURM_JOB_ID" "$rc"' EXIT
srun --ntasks=1 "$ATOM_PILOT_PYTHON" -m pytest tests/gbv_paper -q --tb=short \
  --junitxml="$ATOM_PILOT_OUTPUT/unit_tests.xml"
# Arrhenius allocates GH200; the pilot records it explicitly. Formal H200
# registration is not modified, and a pilot never launches the full matrix.
srun --ntasks=1 "$ATOM_PILOT_PYTHON" -m gbv_experiments.protected_pilot \
  --study configs/atom_tree_formal.json --output "$ATOM_PILOT_OUTPUT" \
  --gpu-family GH200 --tokens 96
