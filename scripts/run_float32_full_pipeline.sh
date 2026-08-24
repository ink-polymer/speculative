#!/usr/bin/env bash
###############################################################################
# float32 从头训练 + 基准流水线 (新优化代码)
#
# 用 float32 消除 bfloat16 数值精度差异, 且 rank head 用新优化代码从头训练:
#   - train_rank_head: max_blocks=1 时跳过 continuation (训练/推理一致)
#   - rank_head: top20_values 预算复用, max_blocks 元数据校验
#   - tree: is_main_chain 主链保护, build 接受 budget
#   - engine: 自适应 tree_budget
#   - verification: 连续区间 narrow 快速路径
#
# 流程:
#   步骤 1: 生成 rank 训练集 (float32 target greedy)
#   步骤 2: 训练 rank head (float32, 新代码)
#   步骤 3: Vanilla DFlash benchmark (heuristic ranker, float32)
#   步骤 4: Tree15 DFlash-SpecBlock benchmark (learned ranker, float32)
#
# 预计耗时 (float32 比 bf16 慢 ~2x):
#   步骤 1:  ~1.5-2 小时 (450 条 × ~12-15s)
#   步骤 2:  ~15-20 分钟 (3 epochs)
#   步骤 3:  ~1 小时 (200 条, heuristic, 无 learned ranker 开销)
#   步骤 4:  ~1.5-2 小时 (200 条, learned ranker)
#   总计:    ~4-6 小时
###############################################################################
set -uo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_DATASETS_ENDPOINT="${HF_DATASETS_ENDPOINT:-https://hf-mirror.com}"

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${PROJECT_ROOT}"
mkdir -p logs outputs checkpoints datasets/generated

MAX_NEW_TOKENS=128
RANK_EPOCHS=3
RANK_LR=2e-4

VANILLA_CONFIG="configs/qwen3_4b_cuda_tree15_float32.json"
VANILLA_OUTPUT="outputs/benchmark_vanilla_dflash_float32.jsonl"

TREE15_CONFIG="configs/qwen3_4b_cuda_float32.json"
TREE15_OUTPUT="outputs/benchmark_tree15_float32.jsonl"

RANK_PROMPTS_FILE="datasets/processed/specblock_official/prompts_rank_train_tree15.jsonl"
RANK_TRAIN_DATA="datasets/generated/rank_train_tree15_float32.jsonl"
RANK_CHECKPOINT="checkpoints/rank_head_tree15_float32.pt"

BENCH_PROMPTS_FILE="datasets/processed/specblock_official/prompts_benchmark_tree15.jsonl"

ts() { date '+%Y-%m-%d %H:%M:%S'; }
step_banner() {
  printf '\n%s\n' "============================================================"
  printf '[%s] [步骤 %s] %s\n' "$(ts)" "$1" "$2"
  printf '============================================================\n'
}
log() { printf '[%s] %s\n' "$(ts)" "$*"; }
die() { printf '[%s] *** 错误 *** %s\n' "$(ts)" "$*" >&2; exit 1; }

STEP_START=0
step_timer_start() { STEP_START=$(date +%s); }
step_timer_elapsed() { echo $(( $(date +%s) - STEP_START )); }
fmt_duration() {
  local s=$1
  printf '%dh %dm %ds' $((s/3600)) $(((s%3600)/60)) $((s%60))
}

for f in "${RANK_PROMPTS_FILE}" "${BENCH_PROMPTS_FILE}" "${VANILLA_CONFIG}" "${TREE15_CONFIG}"; do
  [[ -f "${f}" ]] || die "缺少必要文件: ${f}"
done

RANK_PROMPT_COUNT=$(wc -l < "${RANK_PROMPTS_FILE}")
BENCH_PROMPT_COUNT=$(wc -l < "${BENCH_PROMPTS_FILE}")
OVERALL_START=$(date +%s)

log "============================================================"
log "  float32 从头训练 + 基准流水线 (新优化代码)"
log "  dtype: float32 (消除 bfloat16 数值精度差异)"
log "  rank 训练: ${RANK_PROMPT_COUNT} 条, ${RANK_EPOCHS} epochs, lr=${RANK_LR}"
log "  benchmark: ${BENCH_PROMPT_COUNT} 条, max_new_tokens=${MAX_NEW_TOKENS}"
log "  rank checkpoint: ${RANK_CHECKPOINT}"
log "============================================================"

# ── 步骤 1: 生成 rank 训练集 (float32 target greedy) ───────────────────────
step_banner 1 "生成 rank 训练集 (float32, ${RANK_PROMPT_COUNT} 条)"
if [[ -f "${RANK_TRAIN_DATA}" && "${#1}" == "force" ]]; then
  log "强制重生成"
fi
if [[ -f "${RANK_TRAIN_DATA}" && "$1" != "force" ]]; then
  log "rank 训练集已存在, 跳过 (用 force 参数强制重生成): ${RANK_TRAIN_DATA}"
else
  step_timer_start
  EST_SECS=$(( RANK_PROMPT_COUNT * 15 ))
  log "预计耗时: ~$(fmt_duration ${EST_SECS})"
  python -m dflash_specblock.generate_rank_data \
    --config "${TREE15_CONFIG}" \
    --prompts "${RANK_PROMPTS_FILE}" \
    --output "${RANK_TRAIN_DATA}" \
    --max-prompts 0 \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --device cuda:0 \
    || die "Rank 训练集生成失败"
  TRAIN_ROWS=$(wc -l < "${RANK_TRAIN_DATA}")
  log "Rank 训练集完成: ${TRAIN_ROWS} 条 (耗时: $(fmt_duration $(step_timer_elapsed)))"
fi

# ── 步骤 2: 训练 rank head (float32, 新代码) ──────────────────────────────
step_banner 2 "训练 rank head (float32, ${RANK_EPOCHS} epochs, lr=${RANK_LR})"
if [[ -f "${RANK_CHECKPOINT}" && "$1" != "force" ]]; then
  log "rank checkpoint 已存在, 跳过 (用 force 参数强制重训): ${RANK_CHECKPOINT}"
else
  step_timer_start
  python -m dflash_specblock.train_rank_head \
    --config "${TREE15_CONFIG}" \
    --train-data "${RANK_TRAIN_DATA}" \
    --output "${RANK_CHECKPOINT}" \
    --epochs "${RANK_EPOCHS}" \
    --learning-rate "${RANK_LR}" \
    --device cuda:0 \
    || die "Rank head 训练失败"
  log "Rank head 训练完成 (耗时: $(fmt_duration $(step_timer_elapsed)))"
fi

# ── 步骤 3: Vanilla DFlash benchmark (heuristic, float32) ──────────────────
step_banner 3 "Vanilla DFlash benchmark (heuristic ranker, float32)"
log "  config: ${VANILLA_CONFIG}"
log "  output: ${VANILLA_OUTPUT}"
step_timer_start
python -m dflash_specblock.benchmark_vanilla \
  --config "${VANILLA_CONFIG}" \
  --prompts "${BENCH_PROMPTS_FILE}" \
  --output "${VANILLA_OUTPUT}" \
  --max-prompts 0 \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --device cuda:0 \
  || die "Vanilla DFlash (float32) benchmark 运行失败"
log "Vanilla DFlash 完成 (耗时: $(fmt_duration $(step_timer_elapsed)))"

# ── 步骤 4: Tree15 DFlash-SpecBlock benchmark (learned, float32) ───────────
step_banner 4 "Tree15 DFlash-SpecBlock benchmark (learned ranker, float32)"
log "  config: ${TREE15_CONFIG}"
log "  output: ${TREE15_OUTPUT}"
log "  rank checkpoint: ${RANK_CHECKPOINT}"
step_timer_start
python -m dflash_specblock.benchmark \
  --config "${TREE15_CONFIG}" \
  --prompts "${BENCH_PROMPTS_FILE}" \
  --output "${TREE15_OUTPUT}" \
  --max-prompts 0 \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --device cuda:0 \
  || die "Tree15 DFlash-SpecBlock (float32) benchmark 运行失败"
log "Tree15 DFlash-SpecBlock 完成 (耗时: $(fmt_duration $(step_timer_elapsed)))"

# ── 汇总 ──────────────────────────────────────────────────────────────────
OVERALL_ELAPSED=$(( $(date +%s) - OVERALL_START ))
printf '\n============================================================\n'
log "  float32 从头训练 + 基准流水线全部完成!"
log "  总耗时: $(fmt_duration ${OVERALL_ELAPSED})"
log ""
log "  输出文件:"
log "    rank checkpoint: ${RANK_CHECKPOINT}"
log "    rank 训练数据:   ${RANK_TRAIN_DATA}"
log "    vanilla (float32): ${VANILLA_OUTPUT}"
log "    tree15  (float32): ${TREE15_OUTPUT}"
log ""
log "  对比 bf16 版本:"
log "    vanilla bf16: outputs/benchmark_vanilla_dflash.jsonl"
log "    tree15  bf16: outputs/benchmark_tree15.jsonl"
printf '============================================================\n'
