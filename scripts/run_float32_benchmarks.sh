#!/usr/bin/env bash
###############################################################################
# float32 精度基准：tree15 (DFlash-SpecBlock) + vanilla DFlash
#
# 用 float32 消除 bfloat16 数值精度差异，验证 greedy exact match 是否达到 100%。
# 两个 benchmark 使用完全相同的 200 条 prompts，与 bf16 版本一一对应。
#
# 预计耗时（float32 比 bf16 慢 ~2x）:
#   vanilla:  ~2-3 小时 (200 条 × ~30s)
#   tree15:   ~3-4 小时 (200 条 × ~40s)
#   总计:     ~5-7 小时
###############################################################################
set -uo pipefail

source /opt/home/developer/Ascend/ascend-toolkit/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_DATASETS_ENDPOINT="${HF_DATASETS_ENDPOINT:-https://hf-mirror.com}"

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${PROJECT_ROOT}"
mkdir -p logs outputs

PROMPTS_FILE="datasets/processed/specblock_official/prompts_benchmark_tree15.jsonl"
MAX_NEW_TOKENS=128

VANILLA_CONFIG="configs/qwen3_4b_a2_tree15_float32.json"
VANILLA_OUTPUT="outputs/benchmark_vanilla_dflash_float32.jsonl"

TREE15_CONFIG="configs/qwen3_4b_a2_float32.json"
TREE15_OUTPUT="outputs/benchmark_tree15_float32.jsonl"

ts() { date '+%Y-%m-%d %H:%M:%S'; }
log() { printf '[%s] %s\n' "$(ts)" "$*"; }
die() { printf '[%s] *** 错误 *** %s\n' "$(ts)" "$*" >&2; exit 1; }

if [[ ! -f "${PROMPTS_FILE}" ]]; then
  die "benchmark prompts 文件不存在: ${PROMPTS_FILE}"
fi

PROMPT_COUNT=$(wc -l < "${PROMPTS_FILE}")
OVERALL_START=$(date +%s)

log "============================================================"
log "  float32 精度基准 (tree15 + vanilla DFlash)"
log "  prompts: ${PROMPT_COUNT} 条 (与 bf16 版本同数据)"
log "  dtype: float32 (消除 bfloat16 数值精度差异)"
log "  max_new_tokens: ${MAX_NEW_TOKENS}"
log "============================================================"

# ── 1. Vanilla DFlash (official, float32) ────────────────────────────────
log "[1/2] 正版 DFlash (vanilla, float32)"
log "  config: ${VANILLA_CONFIG}"
log "  output: ${VANILLA_OUTPUT}"

VANILLA_START=$(date +%s)
python -m dflash_specblock.benchmark_vanilla \
  --config "${VANILLA_CONFIG}" \
  --prompts "${PROMPTS_FILE}" \
  --output "${VANILLA_OUTPUT}" \
  --max-prompts 0 \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --device npu:0 \
  || die "Vanilla DFlash (float32) benchmark 运行失败"

VANILLA_ELAPSED=$(( $(date +%s) - VANILLA_START ))
log "[1/2] Vanilla DFlash (float32) 完成, 耗时: $(( VANILLA_ELAPSED / 3600 ))h $(( (VANILLA_ELAPSED % 3600) / 60 ))m"

# ── 2. Tree15 DFlash-SpecBlock (float32) ──────────────────────────────────
log "[2/2] DFlash-SpecBlock tree15 (float32)"
log "  config: ${TREE15_CONFIG}"
log "  output: ${TREE15_OUTPUT}"

TREE15_START=$(date +%s)
python -m dflash_specblock.benchmark \
  --config "${TREE15_CONFIG}" \
  --prompts "${PROMPTS_FILE}" \
  --output "${TREE15_OUTPUT}" \
  --max-prompts 0 \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --device npu:0 \
  || die "Tree15 DFlash-SpecBlock (float32) benchmark 运行失败"

TREE15_ELAPSED=$(( $(date +%s) - TREE15_START ))
log "[2/2] Tree15 DFlash-SpecBlock (float32) 完成, 耗时: $(( TREE15_ELAPSED / 3600 ))h $(( (TREE15_ELAPSED % 3600) / 60 ))m"

# ── 汇总 ──────────────────────────────────────────────────────────────────
OVERALL_ELAPSED=$(( $(date +%s) - OVERALL_START ))
printf '\n============================================================\n'
log "  float32 基准全部完成!"
log "  总耗时: $(( OVERALL_ELAPSED / 3600 ))h $(( (OVERALL_ELAPSED % 3600) / 60 ))m"
log ""
log "  输出文件:"
log "    vanilla (float32):  ${VANILLA_OUTPUT}"
log "    tree15  (float32):  ${TREE15_OUTPUT}"
log ""
log "  对比 bf16 版本:"
log "    vanilla bf16:       outputs/benchmark_vanilla_dflash.jsonl"
log "    tree15  bf16:       outputs/benchmark_tree15.jsonl"
printf '============================================================\n'
