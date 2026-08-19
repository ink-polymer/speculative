#!/usr/bin/env bash
###############################################################################
# DFlash-SpecBlock 树模式训练流水线（block_size=15, ~3h 版）
#
# 配置: block_size=15 (官方一致), max_blocks=1, beam_width=4, tree_budget=60
# P0 修复: prune 保护 greedy 主链(is_main_chain)，主链不再被浅层兄弟挤掉
# 时间预估:
#   步骤 3 rank 训练集:  ~1.5 小时 (每数据集采样 100, 共 ~450 条 × ~12s)
#   步骤 4 rank 训练:    ~10 分钟 (3 epochs)
#   步骤 5 benchmark:    ~1.5 小时 (200 条 × ~27s, baseline+hybrid)
#   总计:                ~3 小时
###############################################################################
set -uo pipefail

source /opt/home/developer/Ascend/ascend-toolkit/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_DATASETS_ENDPOINT="${HF_DATASETS_ENDPOINT:-https://hf-mirror.com}"

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${PROJECT_ROOT}"
mkdir -p logs checkpoints outputs datasets/generated datasets/processed/specblock_official

MAX_TRAIN_PER_DATASET=100
MAX_BENCHMARK_PROMPTS=200
MAX_NEW_TOKENS=128
RANK_EPOCHS=3
RANK_LR=2e-4

CONFIG="configs/qwen3_4b_a2.json"
DATASET_DIR="datasets/processed/specblock_official"
PROMPTS_FILE="${DATASET_DIR}/prompts_all.jsonl"
RANK_TRAIN_DATA="datasets/generated/rank_train_tree15.jsonl"
RANK_CHECKPOINT="checkpoints/rank_head_tree15.pt"
BENCHMARK_OUTPUT="outputs/benchmark_tree15.jsonl"

CKPT_DIR=".pipeline_checkpoints"
mkdir -p "${CKPT_DIR}"

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

OVERALL_START=$(date +%s)
log "============================================================"
log "  DFlash-SpecBlock 树模式训练流水线 (block_size=15)"
log "  block_size=15, beam_width=4, max_blocks=1, tree_budget=60"
log "  P0: prune 保护 greedy 主链 (is_main_chain)"
log "  rank 采样: ${MAX_TRAIN_PER_DATASET}/数据集, benchmark: ${MAX_BENCHMARK_PROMPTS} 条"
log "============================================================"

# ── 步骤 3: 生成 rank 训练集 ─────────────────────────────────────────────
step_banner 3 "生成 rank 训练集 (每数据集上限 ${MAX_TRAIN_PER_DATASET})"
if [[ -f "${CKPT_DIR}/03_rank_data" ]]; then
  log "步骤3已完成，跳过"
else
  step_timer_start

  RANK_PROMPTS_FILE="${DATASET_DIR}/prompts_rank_train_tree15.jsonl"
  python3 -c "
import json, random, os
random.seed(42)
max_per = ${MAX_TRAIN_PER_DATASET}
dataset_dir = '${DATASET_DIR}'
out_path = '${RANK_PROMPTS_FILE}'
rows = []
for name in ['mt_bench', 'humaneval', 'math_500', 'alpaca', 'nq_open', 'translation']:
    fpath = os.path.join(dataset_dir, f'{name}.jsonl')
    if not os.path.exists(fpath):
        continue
    with open(fpath) as f:
        dataset_rows = [json.loads(l) for l in f if l.strip()]
    orig = len(dataset_rows)
    if max_per > 0 and len(dataset_rows) > max_per:
        dataset_rows = random.sample(dataset_rows, max_per)
    rows.extend(dataset_rows)
    print(f'  {name}: {len(dataset_rows)} / {orig}')
random.shuffle(rows)
with open(out_path, 'w') as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=False) + '\n')
print(f'分层采样总计: {len(rows)} 条 -> {out_path}')
"

  RANK_PROMPT_COUNT=$(wc -l < "${RANK_PROMPTS_FILE}")
  EST_TRAIN_SECS=$(( RANK_PROMPT_COUNT * 12 ))
  log "rank 训练集: ${RANK_PROMPT_COUNT} 条, 预计 ~$(fmt_duration ${EST_TRAIN_SECS})"

  python -m dflash_specblock.generate_rank_data \
    --config "${CONFIG}" \
    --prompts "${RANK_PROMPTS_FILE}" \
    --output "${RANK_TRAIN_DATA}" \
    --max-prompts 0 \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    || die "Rank 训练集生成失败"

  TRAIN_ROWS=$(wc -l < "${RANK_TRAIN_DATA}")
  log "Rank 训练集完成: ${TRAIN_ROWS} 条 (耗时: $(fmt_duration $(step_timer_elapsed)))"
  touch "${CKPT_DIR}/03_rank_data"
fi

# ── 步骤 4: 训练 rank head ───────────────────────────────────────────────
step_banner 4 "训练 rank head (${RANK_EPOCHS} epochs, lr=${RANK_LR})"
if [[ -f "${CKPT_DIR}/04_rank_head" ]]; then
  log "步骤4已完成，跳过"
else
  step_timer_start
  python -m dflash_specblock.train_rank_head \
    --config "${CONFIG}" \
    --train-data "${RANK_TRAIN_DATA}" \
    --output "${RANK_CHECKPOINT}" \
    --epochs "${RANK_EPOCHS}" \
    --learning-rate "${RANK_LR}" \
    || die "Rank head 训练失败"
  log "Rank head 训练完成 (耗时: $(fmt_duration $(step_timer_elapsed)))"
  touch "${CKPT_DIR}/04_rank_head"
fi

# 更新配置中的 rank_checkpoint 路径
log "更新配置 rank_checkpoint -> ${RANK_CHECKPOINT}"

# ── 生成 benchmark prompt 文件 ────────────────────────────────────────────
BENCH_PROMPTS_FILE="${DATASET_DIR}/prompts_benchmark_tree15.jsonl"
if [[ -f "${BENCH_PROMPTS_FILE}" ]]; then
  log "benchmark prompts 文件已存在，跳过生成"
else
  TOTAL_PROMPTS=${TOTAL_PROMPTS:-$(wc -l < "${PROMPTS_FILE}")}
  python3 -c "
import json, random
random.seed(123)
with open('${PROMPTS_FILE}') as f:
    rows = [l for l in f if l.strip()]
sampled = random.sample(rows, min(${MAX_BENCHMARK_PROMPTS}, len(rows)))
with open('${BENCH_PROMPTS_FILE}', 'w') as f:
    f.writelines(sampled)
print(f'benchmark prompts: {len(sampled)} 条')
"
fi
BENCH_PROMPT_COUNT=$(wc -l < "${BENCH_PROMPTS_FILE}")

# ── 步骤 5: Benchmark ────────────────────────────────────────────────────
step_banner 5 "运行 benchmark (${BENCH_PROMPT_COUNT} 条, baseline+hybrid, learned ranker)"
if [[ -f "${CKPT_DIR}/05_benchmark" ]]; then
  log "步骤5已完成，跳过"
else
  step_timer_start
  EST_BENCH_SECS=$(( BENCH_PROMPT_COUNT * 27 ))
  log "预计 benchmark 耗时: ~$(fmt_duration ${EST_BENCH_SECS})"

  python -m dflash_specblock.benchmark \
    --config "${CONFIG}" \
    --prompts "${BENCH_PROMPTS_FILE}" \
    --output "${BENCHMARK_OUTPUT}" \
    --max-prompts 0 \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --device npu:0 \
    || die "Benchmark 运行失败"
  log "Benchmark 完成 (耗时: $(fmt_duration $(step_timer_elapsed)))"
  touch "${CKPT_DIR}/05_benchmark"
fi

# ── 完成 ─────────────────────────────────────────────────────────────────
OVERALL_ELAPSED=$(( $(date +%s) - OVERALL_START ))
printf '\n============================================================\n'
log "  树模式训练流水线完成!"
log "  配置: block_size=15, beam_width=4, max_blocks=1, tree_budget=60"
log "  总耗时: $(fmt_duration ${OVERALL_ELAPSED})"
log ""
log "  输出文件:"
log "    benchmark:     ${BENCHMARK_OUTPUT}"
log "    rank ckpt:     ${RANK_CHECKPOINT}"
log "    训练数据:      ${RANK_TRAIN_DATA}"
printf '============================================================\n'
