#!/usr/bin/env bash
#SBATCH --job-name=fuyile-diffusion-tree-pilot
#SBATCH --account=naiss2026-3-658-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=02:00:00
#SBATCH --output=/nobackup/proj/disk/naiss2026-3-658/personal/jiage91/fuyile/logs/diffusion-tree-pilot-%j.out
#SBATCH --error=/nobackup/proj/disk/naiss2026-3-658/personal/jiage91/fuyile/logs/diffusion-tree-pilot-%j.err
set -euo pipefail
source /nobackup/proj/disk/naiss2026-3-658/personal/jiage91/fuyile/scripts/common_env.sh
fuyile_load_gpu_toolchain
: "${DIFFUSION_PILOT_SOURCE:?Set the isolated source snapshot path}"
cd "$DIFFUSION_PILOT_SOURCE"
export PYTHONPATH="$PWD/src"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PATH="$FUYILE_ROOT/envs/gbv/bin:$PATH"
DIFFUSION_PYTHON="$FUYILE_ROOT/envs/gbv/bin/python"
DIFFUSION_PILOT_FAMILY="${DIFFUSION_PILOT_FAMILY:-diffusion_tree}"
DIFFUSION_PILOT_MODEL="${DIFFUSION_PILOT_MODEL:-qwen3_8b}"
case "$DIFFUSION_PILOT_FAMILY" in
  diffusion_tree|diffusion_scaffold) ;;
  *) echo "Pilot family must be diffusion_tree or diffusion_scaffold" >&2; exit 2 ;;
esac
case "$DIFFUSION_PILOT_MODEL" in
  qwen3_8b) STUDY_INFIX="" ;;
  qwen3_4b)
    test "$DIFFUSION_PILOT_FAMILY" = diffusion_scaffold
    STUDY_INFIX="_qwen3_4b"
    ;;
  *) echo "Pilot model must be qwen3_4b or qwen3_8b" >&2; exit 2 ;;
esac
DIFFUSION_OUTPUT="$FUYILE_ROOT/runs/${DIFFUSION_PILOT_FAMILY}-${DIFFUSION_PILOT_MODEL}-pilot-${SLURM_JOB_ID}"
mkdir -p "$DIFFUSION_OUTPUT"
srun --ntasks=1 "$DIFFUSION_PYTHON" -m pytest tests/gbv_paper -q --tb=short \
  --junitxml="$DIFFUSION_OUTPUT/unit_tests.xml"
for tag in t03 t06 t10; do
  srun --ntasks=1 "$DIFFUSION_PYTHON" -m gbv_experiments.protected_pilot \
    --study "configs/${DIFFUSION_PILOT_FAMILY}${STUDY_INFIX}_${tag}.json" \
    --output "$DIFFUSION_OUTPUT/$tag" \
    --gpu-family GH200 --tokens 96
done
