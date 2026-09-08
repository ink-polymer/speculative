#!/usr/bin/env bash
#SBATCH --job-name=fuyile-diffusion-tree-formal
#SBATCH --account=naiss2026-3-658-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=12:00:00
#SBATCH --output=/nobackup/proj/disk/naiss2026-3-658/personal/jiage91/fuyile/logs/diffusion-tree-formal-%j.out
#SBATCH --error=/nobackup/proj/disk/naiss2026-3-658/personal/jiage91/fuyile/logs/diffusion-tree-formal-%j.err
set -euo pipefail
source /nobackup/proj/disk/naiss2026-3-658/personal/jiage91/fuyile/scripts/common_env.sh
fuyile_load_gpu_toolchain
: "${DIFFUSION_FORMAL_SOURCE:?Set the isolated, frozen source snapshot path}"
: "${DIFFUSION_FORMAL_STUDY:?Set the temperature-specific registered study config}"
: "${DIFFUSION_FORMAL_OUTPUT:?Set that study's registered output directory}"
: "${DIFFUSION_FORMAL_DATA:?Set the existing frozen dataset directory}"
cd "$DIFFUSION_FORMAL_SOURCE"
export DIFFUSION_PYTHON="$FUYILE_ROOT/envs/gbv/bin/python"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PATH="$FUYILE_ROOT/envs/gbv/bin:$PATH"
DIFFUSION_STAGE="${1:-run}"
DIFFUSION_REPEAT="${2:-0}"
if [[ "$DIFFUSION_STAGE" != diagnostics && "$DIFFUSION_STAGE" != run ]]; then
  echo "GPU stages are diagnostics and run; freeze/test/score/report run separately." >&2
  exit 2
fi
# The registered GPU-family gate rejects a mismatched allocation. Do not edit
# an existing registration to turn GH200 pilot timings into H200 evidence.
srun --ntasks=1 bash scripts/run_diffusion_tree.sh "$DIFFUSION_STAGE" \
  --study "$DIFFUSION_FORMAL_STUDY" --data-dir "$DIFFUSION_FORMAL_DATA" \
  --output "$DIFFUSION_FORMAL_OUTPUT" --repeat "$DIFFUSION_REPEAT" --device cuda:0
