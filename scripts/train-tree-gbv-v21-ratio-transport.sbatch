#!/bin/bash
#SBATCH --job-name=gbv-v21-train
#SBATCH --account=naiss2026-3-658-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=01:00:00
#SBATCH --output=/nobackup/proj/disk/naiss2026-3-658/personal/jiage91/fuyile/logs/gbv-v21-train-%j.out
#SBATCH --error=/nobackup/proj/disk/naiss2026-3-658/personal/jiage91/fuyile/logs/gbv-v21-train-%j.err
set -euo pipefail
source /nobackup/proj/disk/naiss2026-3-658/personal/jiage91/fuyile/scripts/common_env.sh
cd "$FUYILE_ROOT/src/gbv-optimized"
export PYTHONPATH="$PWD/src"
PYTHON="$FUYILE_ROOT/envs/adaptive/bin/python"
"$PYTHON" -m pytest -q tests/gbv_paper/test_slot_mixer.py tests/gbv_paper/test_probability_law.py tests/gbv_paper/test_fairness.py
"$PYTHON" -m gbv_experiments.train_ratio_transport \
  --config configs/gbv_paper_full.json \
  --train-data datasets/generated/rank_train_tree15_disjoint_2k.jsonl \
  --output "$FUYILE_ROOT/checkpoints/tree_gbv_ratio_transport_v21.pt"
