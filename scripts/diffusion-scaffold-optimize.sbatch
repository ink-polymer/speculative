#!/usr/bin/env bash
#SBATCH --job-name=fuyile-diffusion-opt
#SBATCH --account=naiss2026-3-658-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=/nobackup/proj/disk/naiss2026-3-658/personal/jiage91/fuyile/logs/diffusion-scaffold-optimize-%j.out
#SBATCH --error=/nobackup/proj/disk/naiss2026-3-658/personal/jiage91/fuyile/logs/diffusion-scaffold-optimize-%j.err
set -euo pipefail
source /nobackup/proj/disk/naiss2026-3-658/personal/jiage91/fuyile/scripts/common_env.sh
fuyile_load_gpu_toolchain
: "${DIFFUSION_OPT_SOURCE:?Set the isolated source snapshot path}"
cd "$DIFFUSION_OPT_SOURCE"
export PYTHONPATH="$PWD/src"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PATH="$FUYILE_ROOT/envs/gbv/bin:$PATH"
DIFFUSION_OPT_PYTHON="$FUYILE_ROOT/envs/gbv/bin/python"
DIFFUSION_OPT_MODEL="${DIFFUSION_OPT_MODEL:-qwen3_8b}"
case "$DIFFUSION_OPT_MODEL" in
  qwen3_4b) DIFFUSION_OPT_CONFIG="configs/gbv_paper_ddtree_counts.json" ;;
  qwen3_8b) DIFFUSION_OPT_CONFIG="configs/gbv_paper_qwen3_8b.json" ;;
  *) echo "Optimization model must be qwen3_4b or qwen3_8b" >&2; exit 2 ;;
esac
DIFFUSION_OPT_OUTPUT="$FUYILE_ROOT/runs/diffusion-scaffold-opt-${DIFFUSION_OPT_MODEL}-${SLURM_JOB_ID}"
mkdir -p "$DIFFUSION_OPT_OUTPUT"
srun --ntasks=1 "$DIFFUSION_OPT_PYTHON" -m pytest \
  tests/gbv_paper/test_diffusion_optimization.py \
  tests/gbv_paper/test_diffusion_scaffold.py \
  tests/gbv_paper/test_diffusion_preflight.py -q --tb=short \
  --junitxml="$DIFFUSION_OPT_OUTPUT/unit_tests.xml"
srun --ntasks=1 "$DIFFUSION_OPT_PYTHON" scripts/optimize_diffusion_scaffold.py \
  --config "$DIFFUSION_OPT_CONFIG" \
  --output "$DIFFUSION_OPT_OUTPUT/scan" \
  --tokens "${DIFFUSION_OPT_TOKENS:-96}" \
  --repeats "${DIFFUSION_OPT_REPEATS:-2}"
