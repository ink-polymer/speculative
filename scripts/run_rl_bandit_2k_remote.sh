#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

TRAIN_CONFIG="${TRAIN_CONFIG:-configs/qwen3_4b_cuda_ddtree_bandit.json}"
EVAL_CONFIG="${EVAL_CONFIG:-configs/qwen3_4b_cuda_ddtree_bandit_eval.json}"
TRAIN_PROMPTS="${TRAIN_PROMPTS:-datasets/generated/rank_train_tree15_disjoint_2k.jsonl}"
EVAL_PROMPTS="${EVAL_PROMPTS:-datasets/processed/dflash_official/prompts_benchmark_2k.jsonl}"
TRAIN_OUTPUT="${TRAIN_OUTPUT:-outputs/rl_bandit_2k/train_policy_566_128.jsonl}"
EVAL_OUTPUT="${EVAL_OUTPUT:-outputs/rl_bandit_2k/eval_2k_2048.jsonl}"
TRAIN_MAX_NEW_TOKENS="${TRAIN_MAX_NEW_TOKENS:-128}"
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-2048}"
STATUS_FILE="${STATUS_FILE:-outputs/rl_bandit_2k/run_status.txt}"

mkdir -p "$(dirname "${TRAIN_OUTPUT}")" "$(dirname "${EVAL_OUTPUT}")" checkpoints logs

finish() {
    rc=$?
    if [[ ${rc} -eq 0 ]]; then
        state="complete"
    else
        state="failed"
    fi
    printf 'state=%s\nexit_code=%s\nfinished_at=%s\n' \
        "${state}" "${rc}" "$(date --iso-8601=seconds)" > "${STATUS_FILE}"
}
trap finish EXIT

printf 'state=running\nstarted_at=%s\n' "$(date --iso-8601=seconds)" > "${STATUS_FILE}"

python -m dflash_specblock.benchmark \
    --config "${TRAIN_CONFIG}" \
    --prompts "${TRAIN_PROMPTS}" \
    --output "${TRAIN_OUTPUT}" \
    --max-prompts 0 \
    --max-new-tokens "${TRAIN_MAX_NEW_TOKENS}"

python -m dflash_specblock.benchmark \
    --config "${EVAL_CONFIG}" \
    --prompts "${EVAL_PROMPTS}" \
    --output "${EVAL_OUTPUT}" \
    --max-prompts 0 \
    --max-new-tokens "${EVAL_MAX_NEW_TOKENS}"

sha256sum checkpoints/ddtree_bandit_policy.json "${TRAIN_OUTPUT}" "${EVAL_OUTPUT}" \
    > outputs/rl_bandit_2k/SHA256SUMS
