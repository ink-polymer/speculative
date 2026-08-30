#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${PROJECT_ROOT}"

SOURCE_TRAIN_PROMPTS="${SOURCE_TRAIN_PROMPTS:-datasets/processed/specblock_official/prompts_rank_train_tree15.jsonl}"
TRAIN_PROMPTS="${TRAIN_PROMPTS:-datasets/processed/specblock_official/prompts_ppo_train_tree15_disjoint_2k.jsonl}"
EVAL_PROMPTS="${EVAL_PROMPTS:-datasets/processed/dflash_official/prompts_benchmark_2k.jsonl}"
TRAIN_OUTPUT="${TRAIN_OUTPUT:-outputs/temperature_1_ppo_ddtree/train_policy.jsonl}"
EVAL_OUTPUT="${EVAL_OUTPUT:-outputs/temperature_1_ppo_ddtree/eval_2k.jsonl}"
SUMMARY="${SUMMARY:-outputs/temperature_1_ppo_ddtree/eval_2k_summary.json}"
PPO_CHECKPOINT="${PPO_CHECKPOINT:-checkpoints/ddtree_ppo_temperature1.pt}"
MODEL="${MODEL:-models/Qwen3-4B}"
DRAFT="${DRAFT:-models/Qwen3-4B-DFlash-b16}"
DRAFT_ATTN_IMPLEMENTATION="${DRAFT_ATTN_IMPLEMENTATION:-flash_attention_2}"
TEMPERATURE="${TEMPERATURE:-1.0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
TREE_BUDGET="${TREE_BUDGET:-256}"
FIXED_TREE_BUDGET="${FIXED_TREE_BUDGET:-60}"
TREE_BUDGETS="${TREE_BUDGETS:-30,40,50,60,70,80,90,100,112,128,144,160,192,224,256}"
INITIAL_BUDGET="${INITIAL_BUDGET:-60}"
PPO_HIDDEN_SIZE="${PPO_HIDDEN_SIZE:-64}"
PPO_LEARNING_RATE="${PPO_LEARNING_RATE:-0.0003}"
PPO_GAMMA="${PPO_GAMMA:-0.99}"
PPO_GAE_LAMBDA="${PPO_GAE_LAMBDA:-0.95}"
PPO_CLIP_RANGE="${PPO_CLIP_RANGE:-0.2}"
PPO_VALUE_COEFFICIENT="${PPO_VALUE_COEFFICIENT:-0.5}"
PPO_ENTROPY_COEFFICIENT="${PPO_ENTROPY_COEFFICIENT:-0.01}"
PPO_ROLLOUT_STEPS="${PPO_ROLLOUT_STEPS:-256}"
PPO_UPDATE_EPOCHS="${PPO_UPDATE_EPOCHS:-4}"
PPO_MINIBATCH_SIZE="${PPO_MINIBATCH_SIZE:-64}"
PPO_MAX_GRAD_NORM="${PPO_MAX_GRAD_NORM:-0.5}"
PPO_TREE_BUILD_COST_WEIGHT="${PPO_TREE_BUILD_COST_WEIGHT:-2.0}"
PPO_CONTEXT_LENGTH_SCALE="${PPO_CONTEXT_LENGTH_SCALE:-4096}"
TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:-0}"
MAX_SAMPLES="${MAX_SAMPLES:-2000}"
SEED="${SEED:-42}"

mkdir -p "$(dirname "${TRAIN_OUTPUT}")" "$(dirname "${EVAL_OUTPUT}")" \
  "$(dirname "${PPO_CHECKPOINT}")"

python scripts/filter_rank_training_disjoint.py \
  --train "${SOURCE_TRAIN_PROMPTS}" \
  --eval "${EVAL_PROMPTS}" \
  --output "${TRAIN_PROMPTS}"

TRAIN_LIMIT_ARGS=()
if [[ "${TRAIN_MAX_SAMPLES}" -gt 0 ]]; then
  TRAIN_LIMIT_ARGS=(--max-samples "${TRAIN_MAX_SAMPLES}")
fi

python scripts/benchmark_official_ddtree.py \
  --prompts "${TRAIN_PROMPTS}" \
  --output "${TRAIN_OUTPUT}" \
  --model "${MODEL}" \
  --draft "${DRAFT}" \
  --draft-attn-implementation "${DRAFT_ATTN_IMPLEMENTATION}" \
  --temperature "${TEMPERATURE}" \
  --tree-budget "${TREE_BUDGET}" \
  --fixed-tree-budget "${FIXED_TREE_BUDGET}" \
  --budget-candidates "${TREE_BUDGETS}" \
  --initial-budget "${INITIAL_BUDGET}" \
  --ppo-hidden-size "${PPO_HIDDEN_SIZE}" \
  --ppo-learning-rate "${PPO_LEARNING_RATE}" \
  --ppo-gamma "${PPO_GAMMA}" \
  --ppo-gae-lambda "${PPO_GAE_LAMBDA}" \
  --ppo-clip-range "${PPO_CLIP_RANGE}" \
  --ppo-value-coefficient "${PPO_VALUE_COEFFICIENT}" \
  --ppo-entropy-coefficient "${PPO_ENTROPY_COEFFICIENT}" \
  --ppo-rollout-steps "${PPO_ROLLOUT_STEPS}" \
  --ppo-update-epochs "${PPO_UPDATE_EPOCHS}" \
  --ppo-minibatch-size "${PPO_MINIBATCH_SIZE}" \
  --ppo-max-grad-norm "${PPO_MAX_GRAD_NORM}" \
  --ppo-tree-build-cost-weight "${PPO_TREE_BUILD_COST_WEIGHT}" \
  --ppo-context-length-scale "${PPO_CONTEXT_LENGTH_SCALE}" \
  --ppo-checkpoint "${PPO_CHECKPOINT}" \
  --ppo-train \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  "${TRAIN_LIMIT_ARGS[@]}" \
  --seed "${SEED}" \
  --resume

python scripts/benchmark_official_ddtree.py \
  --prompts "${EVAL_PROMPTS}" \
  --output "${EVAL_OUTPUT}" \
  --model "${MODEL}" \
  --draft "${DRAFT}" \
  --draft-attn-implementation "${DRAFT_ATTN_IMPLEMENTATION}" \
  --temperature "${TEMPERATURE}" \
  --tree-budget "${TREE_BUDGET}" \
  --fixed-tree-budget "${FIXED_TREE_BUDGET}" \
  --budget-candidates "${TREE_BUDGETS}" \
  --initial-budget "${INITIAL_BUDGET}" \
  --ppo-hidden-size "${PPO_HIDDEN_SIZE}" \
  --ppo-learning-rate "${PPO_LEARNING_RATE}" \
  --ppo-gamma "${PPO_GAMMA}" \
  --ppo-gae-lambda "${PPO_GAE_LAMBDA}" \
  --ppo-clip-range "${PPO_CLIP_RANGE}" \
  --ppo-value-coefficient "${PPO_VALUE_COEFFICIENT}" \
  --ppo-entropy-coefficient "${PPO_ENTROPY_COEFFICIENT}" \
  --ppo-rollout-steps "${PPO_ROLLOUT_STEPS}" \
  --ppo-update-epochs "${PPO_UPDATE_EPOCHS}" \
  --ppo-minibatch-size "${PPO_MINIBATCH_SIZE}" \
  --ppo-max-grad-norm "${PPO_MAX_GRAD_NORM}" \
  --ppo-tree-build-cost-weight "${PPO_TREE_BUILD_COST_WEIGHT}" \
  --ppo-context-length-scale "${PPO_CONTEXT_LENGTH_SCALE}" \
  --ppo-checkpoint "${PPO_CHECKPOINT}" \
  --ppo-eval \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --max-samples "${MAX_SAMPLES}" \
  --seed "${SEED}" \
  --resume

python scripts/summarize_temperature_comparison.py \
  --input "${EVAL_OUTPUT}" \
  --output "${SUMMARY}"
