#!/usr/bin/env bash
#SBATCH --job-name=tm-formal
#SBATCH --account=naiss2026-3-658-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=12:00:00
#SBATCH --output=/nobackup/proj/disk/naiss2026-3-658/personal/jiage91/fuyile/logs/tm-formal-%j.out
#SBATCH --error=/nobackup/proj/disk/naiss2026-3-658/personal/jiage91/fuyile/logs/tm-formal-%j.err
set -euo pipefail
source /nobackup/proj/disk/naiss2026-3-658/personal/jiage91/fuyile/scripts/common_env.sh
cd "$FUYILE_ROOT/src/gbv-optimized"
export GBV_FORMAL_PYTHON="$FUYILE_ROOT/envs/adaptive/bin/python"
: "${GBV_FORMAL_OUTPUT:?Set GBV_FORMAL_OUTPUT to the registered study directory}"
: "${GBV_FORMAL_DATA:?Set GBV_FORMAL_DATA to the existing frozen dataset directory}"
GBV_FORMAL_STAGE="${1:-run}"
GBV_FORMAL_REPEAT="${2:-0}"
if [[ "$GBV_FORMAL_STAGE" != diagnostics && "$GBV_FORMAL_STAGE" != run ]]; then
  echo "GPU batch stages are diagnostics and run. Run freeze/test/score/report separately." >&2
  exit 2
fi
trap 'rc=$?; printf "job=%s exit=%s\n" "$SLURM_JOB_ID" "$rc"' EXIT
srun --ntasks=1 bash scripts/run_terminal_mass_formal.sh "$GBV_FORMAL_STAGE" \
  --study "${GBV_FORMAL_STUDY:-configs/terminal_mass_formal.json}" \
  --data-dir "$GBV_FORMAL_DATA" --output "$GBV_FORMAL_OUTPUT" \
  --repeat "$GBV_FORMAL_REPEAT" --device cuda:0
