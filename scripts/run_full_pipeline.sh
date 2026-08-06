#!/usr/bin/env bash
set -euo pipefail

source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"

CONFIG="${CONFIG:-configs/qwen3_4b_a2.json}"
DATASET_DIR="${DATASET_DIR:-datasets/processed/specblock_official}"
PROMPTS_FILE="${PROMPTS_FILE:-${DATASET_DIR}/prompts_all.jsonl}"
RANK_TRAIN_DATA="${RANK_TRAIN_DATA:-datasets/generated/rank_train.jsonl}"
RANK_CHECKPOINT="${RANK_CHECKPOINT:-checkpoints/rank_head.pt}"
BENCHMARK_OUTPUT="${BENCHMARK_OUTPUT:-outputs/specblock_official.jsonl}"
OFFICIAL_EVAL_OUTPUT="${OFFICIAL_EVAL_OUTPUT:-outputs/official_eval}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
RANK_EPOCHS="${RANK_EPOCHS:-1}"
RANK_LR="${RANK_LR:-2e-4}"
MAX_SAMPLES_PER_DATASET="${MAX_SAMPLES_PER_DATASET:-0}"
MAX_PROMPTS="${MAX_PROMPTS:-0}"
RUN_OFFICIAL_EVAL="${RUN_OFFICIAL_EVAL:-1}"
TRANSLATION_DATASET="${TRANSLATION_DATASET:-wmt14}"
TRANSLATION_CONFIG="${TRANSLATION_CONFIG:-de-en}"
TRANSLATION_SPLIT="${TRANSLATION_SPLIT:-test}"
SOURCE_LANG="${SOURCE_LANG:-de}"
TARGET_LANG="${TARGET_LANG:-en}"

print_step() {
  printf '\n[%s/6] %s\n' "$1" "$2"
}

print_step 1 "下载模型"
python scripts/download_models.py --config "${CONFIG}"

print_step 2 "下载并转换 benchmark 数据集"
python -m dflash_specblock.dataset_pipeline \
  --output-dir "${DATASET_DIR}" \
  --max-samples-per-dataset "${MAX_SAMPLES_PER_DATASET}" \
  --translation-dataset "${TRANSLATION_DATASET}" \
  --translation-config "${TRANSLATION_CONFIG}" \
  --translation-split "${TRANSLATION_SPLIT}" \
  --source-lang "${SOURCE_LANG}" \
  --target-lang "${TARGET_LANG}"

print_step 3 "生成 rank 训练集"
python -m dflash_specblock.generate_rank_data \
  --config "${CONFIG}" \
  --prompts "${PROMPTS_FILE}" \
  --output "${RANK_TRAIN_DATA}" \
  --max-prompts "${MAX_PROMPTS}" \
  --max-new-tokens "${MAX_NEW_TOKENS}"

print_step 4 "训练 rank head"
python -m dflash_specblock.train_rank_head \
  --config "${CONFIG}" \
  --train-data "${RANK_TRAIN_DATA}" \
  --output "${RANK_CHECKPOINT}" \
  --epochs "${RANK_EPOCHS}" \
  --learning-rate "${RANK_LR}"

print_step 5 "运行 baseline benchmark"
python -m dflash_specblock.benchmark \
  --config "${CONFIG}" \
  --prompts "${PROMPTS_FILE}" \
  --output "${BENCHMARK_OUTPUT}" \
  --max-prompts "${MAX_PROMPTS}" \
  --max-new-tokens "${MAX_NEW_TOKENS}"

if [[ "${RUN_OFFICIAL_EVAL}" == "1" ]]; then
  print_step 6 "运行官方任务评测"
  python -m dflash_specblock.official_eval \
    --config "${CONFIG}" \
    --prompts "${PROMPTS_FILE}" \
    --output-dir "${OFFICIAL_EVAL_OUTPUT}" \
    --max-prompts "${MAX_PROMPTS}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --mode hybrid
fi
