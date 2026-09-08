#!/usr/bin/env bash
#SBATCH --job-name=tm-replay
#SBATCH --account=naiss2026-3-658-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=01:00:00
#SBATCH --output=/nobackup/proj/disk/naiss2026-3-658/personal/jiage91/fuyile/logs/tm-replay-%j.out
#SBATCH --error=/nobackup/proj/disk/naiss2026-3-658/personal/jiage91/fuyile/logs/tm-replay-%j.err
set -euo pipefail

source /nobackup/proj/disk/naiss2026-3-658/personal/jiage91/fuyile/scripts/common_env.sh
fuyile_load_gpu_toolchain
: "${TERMINAL_REPLAY_SOURCE:?Set the immutable source worktree path}"
TERMINAL_REPLAY_MODEL="${TERMINAL_REPLAY_MODEL:-qwen3_8b}"
case "$TERMINAL_REPLAY_MODEL" in
  qwen3_4b) TERMINAL_REPLAY_CONFIG="configs/tree_bv_qwen3_4b.json" ;;
  qwen3_8b) TERMINAL_REPLAY_CONFIG="configs/tree_bv_qwen3_8b.json" ;;
  *) echo "TERMINAL_REPLAY_MODEL must be qwen3_4b or qwen3_8b" >&2; exit 2 ;;
esac

cd "$TERMINAL_REPLAY_SOURCE"
export PYTHONPATH="$PWD/src"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PATH="$FUYILE_ROOT/envs/adaptive/bin:$PATH"
# Inline CUDA builds are not concurrency-safe when jobs share one cache path.
export TORCH_EXTENSIONS_DIR="$FUYILE_ROOT/cache/torch_extensions/$SLURM_JOB_ID"
TERMINAL_REPLAY_PYTHON="$FUYILE_ROOT/envs/gbv/bin/python"
TERMINAL_REPLAY_OUTPUT="$FUYILE_ROOT/logs/terminal-architecture-${TERMINAL_REPLAY_MODEL}-${SLURM_JOB_ID}.json"

trap 'rc=$?; printf "job=%s model=%s output=%s exit=%s\n" "$SLURM_JOB_ID" "$TERMINAL_REPLAY_MODEL" "$TERMINAL_REPLAY_OUTPUT" "$rc"' EXIT
srun --ntasks=1 "$TERMINAL_REPLAY_PYTHON" -m pytest \
  tests/gbv_paper/test_terminal_mass.py \
  tests/gbv_paper/test_probability_law.py \
  tests/gbv_paper/test_fairness.py -q --tb=short
srun --ntasks=1 "$TERMINAL_REPLAY_PYTHON" scripts/replay_terminal_architecture.py \
  --config "$TERMINAL_REPLAY_CONFIG" \
  --output "$TERMINAL_REPLAY_OUTPUT" \
  --device cuda:0
