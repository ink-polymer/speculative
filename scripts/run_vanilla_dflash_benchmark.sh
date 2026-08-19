#!/usr/bin/env bash
###############################################################################
# 正版 DFlash（线性 speculative decoding）benchmark
#
# 在与 tree15 完全相同的 200 条 benchmark prompts 上运行 vanilla DFlash：
#   - 同一 target/draft 模型、同一 block_size=15、同一 max_new_tokens=128
#   - 唯一区别：线性 draft（top-1）+ 因果 verify（无 SpecBlock 树/兄弟/ancestor mask）
#   - 不需要 rank head checkpoint
#
# 输出格式与 benchmark_tree15.jsonl 完全一致，可直接对比加速比与平均接受长度。
###############################################################################
set -uo pipefail

source /opt/home/developer/Ascend/ascend-toolkit/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_DATASETS_ENDPOINT="${HF_DATASETS_ENDPOINT:-https://hf-mirror.com}"

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${PROJECT_ROOT}"
mkdir -p logs outputs

CONFIG="configs/qwen3_4b_a2_tree15.json"
PROMPTS_FILE="datasets/processed/specblock_official/prompts_benchmark_tree15.jsonl"
OUTPUT_FILE="outputs/benchmark_vanilla_dflash.jsonl"
MAX_NEW_TOKENS=128

ts() { date '+%Y-%m-%d %H:%M:%S'; }
log() { printf '[%s] %s\n' "$(ts)" "$*"; }
die() { printf '[%s] *** 错误 *** %s\n' "$(ts)" "$*" >&2; exit 1; }

if [[ ! -f "${PROMPTS_FILE}" ]]; then
  die "benchmark prompts 文件不存在: ${PROMPTS_FILE}"
fi

PROMPT_COUNT=$(wc -l < "${PROMPTS_FILE}")
EST_SECS=$(( PROMPT_COUNT * 22 ))
log "============================================================"
log "  正版 DFlash 线性 benchmark"
log "  prompts: ${PROMPT_COUNT} 条 (与 tree15 同数据)"
log "  config: ${CONFIG}"
log "  block_size=15, 线性 draft + 因果 verify (无树扩展)"
log "  预计耗时: ~$(( EST_SECS / 3600 ))h $(( (EST_SECS % 3600) / 60 ))m"
log "============================================================"

START=$(date +%s)

python -m dflash_specblock.benchmark_vanilla \
  --config "${CONFIG}" \
  --prompts "${PROMPTS_FILE}" \
  --output "${OUTPUT_FILE}" \
  --max-prompts 0 \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --device npu:0 \
  || die "Benchmark 运行失败"

ELAPSED=$(( $(date +%s) - START ))
printf '\n============================================================\n'
log "  正版 DFlash benchmark 完成!"
log "  耗时: $(( ELAPSED / 3600 ))h $(( (ELAPSED % 3600) / 60 ))m $(( ELAPSED % 60 ))s"
log "  输出: ${OUTPUT_FILE}"
log ""
log "  对比 tree15 结果:"
log "    tree15:          outputs/benchmark_tree15.jsonl"
log "    vanilla dflash:  ${OUTPUT_FILE}"
printf '============================================================\n'
