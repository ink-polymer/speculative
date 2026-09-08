#!/usr/bin/env bash
###############################################################################
# DFlash-SpecBlock 全量过夜流水线（~2 天版）
#
# 时间预估需在目标 NVIDIA GPU 上重新标定；下列步骤规模仅供排程参考：
#   步骤 1 模型校验:     秒级（已下载则跳过）
#   步骤 2 全量下载:     ~5 分钟（mt_bench 80 + humaneval 164 + math 500
#                                  + alpaca 52K + nq 3610 + translation 3003
#                                  = ~59K 条）
#   步骤 3 rank 训练集:  ~7 小时（每数据集采样 500，共 2244 条 × ~12s）
#   步骤 4 rank 训练:    ~10 分钟（3 epochs）
#   步骤 5 benchmark:    ~20 小时（2000 条 × ~36s, baseline+hybrid 双轮）
#   步骤 6 官方评测:     ~17 小时（2000 条 × ~30s, hybrid 单轮）
#   总计:                ~45 小时
#
# 用法:
#   tmux new-session -d -s speculative \
#     'bash scripts/run_full_pipeline_overnight.sh 2>&1 \
#      | tee logs/overnight_$(date +%Y%m%d_%H%M%S).log'
###############################################################################
set -uo pipefail

# ── 环境 ──────────────────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_DATASETS_ENDPOINT="${HF_DATASETS_ENDPOINT:-https://hf-mirror.com}"

# ── 目录 ──────────────────────────────────────────────────────────────────
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${PROJECT_ROOT}"
mkdir -p logs checkpoints outputs datasets/generated datasets/processed/specblock_official

# ── 参数 ──────────────────────────────────────────────────────────────────
# 数据集下载：0=不限制，全量下载
MAX_SAMPLES_PER_DATASET=0
# rank 训练：每数据集最多采样 500 条（Alpaca 52K 太大，必须采样）
MAX_TRAIN_PER_DATASET=500
# benchmark / eval 使用的 prompt 上限（~59K 全量跑需要 500h，必须截断）
MAX_BENCHMARK_PROMPTS=2000
MAX_NEW_TOKENS=128
RANK_EPOCHS=3
RANK_LR=2e-4

# ── 路径 ──────────────────────────────────────────────────────────────────
CONFIG="configs/qwen3_4b_cuda.json"
DATASET_DIR="datasets/processed/specblock_official"
PROMPTS_FILE="${DATASET_DIR}/prompts_all.jsonl"
RANK_TRAIN_DATA="datasets/generated/rank_train_overnight.jsonl"
RANK_CHECKPOINT="checkpoints/rank_head_overnight.pt"
BENCHMARK_OUTPUT="outputs/benchmark_overnight.jsonl"
OFFICIAL_EVAL_OUTPUT="outputs/official_eval_overnight"

CKPT_DIR=".pipeline_checkpoints"
mkdir -p "${CKPT_DIR}"

# ── 日志工具 ──────────────────────────────────────────────────────────────
ts() { date '+%Y-%m-%d %H:%M:%S'; }
step_banner() {
  printf '\n%s\n' "============================================================"
  printf '[%s] [步骤 %s/6] %s\n' "$(ts)" "$1" "$2"
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
log "  DFlash-SpecBlock 全量过夜流水线（~2 天版）"
log "============================================================"
log "全量下载 (MAX_SAMPLES=0), rank 训练采样 ${MAX_TRAIN_PER_DATASET}/数据集"
log "benchmark/eval 上限: ${MAX_BENCHMARK_PROMPTS} 条"
log "MAX_NEW_TOKENS=${MAX_NEW_TOKENS}, RANK_EPOCHS=${RANK_EPOCHS}"
log "============================================================"

# ── 步骤 1: 模型校验 ──────────────────────────────────────────────────────
step_banner 1 "模型校验"
if [[ -f "${CKPT_DIR}/01_models" ]]; then
  log "步骤1已完成，跳过"
else
  step_timer_start
  if [[ -d "models/Qwen3-4B" ]] && [[ -d "models/Qwen3-4B-DFlash-b16" ]]; then
    log "模型已存在，跳过下载"
  else
    log "下载模型..."
    python scripts/download_models.py --config "${CONFIG}" || die "模型下载失败"
  fi
  log "模型校验完成 (耗时: $(fmt_duration $(step_timer_elapsed)))"
  touch "${CKPT_DIR}/01_models"
fi

# ── 步骤 2: 全量下载 benchmark 数据集 ─────────────────────────────────────
step_banner 2 "全量下载 benchmark 数据集（无上限）"
if [[ -f "${CKPT_DIR}/02_datasets" ]]; then
  log "步骤2已完成，跳过"
  TOTAL_PROMPTS=$(wc -l < "${PROMPTS_FILE}")
else
  step_timer_start
  python -m dflash_specblock.dataset_pipeline \
    --output-dir "${DATASET_DIR}" \
    --max-samples-per-dataset "${MAX_SAMPLES_PER_DATASET}" \
    --translation-dataset "wmt14" \
    --translation-config "de-en" \
    --translation-split "test" \
    --source-lang "de" \
    --target-lang "en" \
    || die "数据集下载失败"

  TOTAL_PROMPTS=$(wc -l < "${PROMPTS_FILE}")
  log "全量数据集下载完成: ${TOTAL_PROMPTS} 条 (耗时: $(fmt_duration $(step_timer_elapsed)))"
  if [[ -f "${DATASET_DIR}/manifest.json" ]]; then
    log "各数据集样本数:"
    python3 -c "
import json
with open('${DATASET_DIR}/manifest.json') as f:
    m = json.load(f)
for name, count in m.get('counts', {}).items():
    print(f'  {name}: {count}')
"
  fi
  touch "${CKPT_DIR}/02_datasets"
fi

# ── 步骤 3: 生成 rank 训练集（分层采样）─────────────────────────────────
step_banner 3 "生成 rank 训练集 (分层采样, 每数据集上限 ${MAX_TRAIN_PER_DATASET})"
if [[ -f "${CKPT_DIR}/03_rank_data" ]]; then
  log "步骤3已完成，跳过"
else
  step_timer_start

  RANK_PROMPTS_FILE="${DATASET_DIR}/prompts_rank_train.jsonl"
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

# ── 生成 benchmark/eval 用的截断 prompt 文件 ─────────────────────────────
BENCH_PROMPTS_FILE="${DATASET_DIR}/prompts_benchmark.jsonl"
if [[ -f "${BENCH_PROMPTS_FILE}" ]]; then
  log "benchmark prompts 文件已存在，跳过生成"
else
  TOTAL_PROMPTS=${TOTAL_PROMPTS:-$(wc -l < "${PROMPTS_FILE}")}
  if [[ "${MAX_BENCHMARK_PROMPTS}" -gt 0 ]] && [[ "${TOTAL_PROMPTS}" -gt "${MAX_BENCHMARK_PROMPTS}" ]]; then
    log "从全量 ${TOTAL_PROMPTS} 条中随机抽取 ${MAX_BENCHMARK_PROMPTS} 条用于 benchmark/eval"
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
  else
    cp "${PROMPTS_FILE}" "${BENCH_PROMPTS_FILE}"
  fi
fi
BENCH_PROMPT_COUNT=$(wc -l < "${BENCH_PROMPTS_FILE}")

# ── 步骤 5: Benchmark ────────────────────────────────────────────────────
step_banner 5 "运行 benchmark (${BENCH_PROMPT_COUNT} 条, baseline+hybrid)"
if [[ -f "${CKPT_DIR}/05_benchmark" ]]; then
  log "步骤5已完成，跳过"
else
  step_timer_start
  EST_BENCH_SECS=$(( BENCH_PROMPT_COUNT * 36 ))
  log "预计 benchmark 耗时: ~$(fmt_duration ${EST_BENCH_SECS})"

  python -m dflash_specblock.benchmark \
    --config "${CONFIG}" \
    --prompts "${BENCH_PROMPTS_FILE}" \
    --output "${BENCHMARK_OUTPUT}" \
    --max-prompts 0 \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    || die "Benchmark 运行失败"
  log "Benchmark 完成 (耗时: $(fmt_duration $(step_timer_elapsed)))"
  touch "${CKPT_DIR}/05_benchmark"
fi

# ── 步骤 6: 官方任务评测 ─────────────────────────────────────────────────
step_banner 6 "运行官方评测 (${BENCH_PROMPT_COUNT} 条, hybrid 模式)"
if [[ -f "${CKPT_DIR}/06_official_eval" ]]; then
  log "步骤6已完成，跳过"
else
  step_timer_start
  EST_EVAL_SECS=$(( BENCH_PROMPT_COUNT * 30 ))
  log "预计评测耗时: ~$(fmt_duration ${EST_EVAL_SECS})"

  python -m dflash_specblock.official_eval \
    --config "${CONFIG}" \
    --prompts "${BENCH_PROMPTS_FILE}" \
    --output-dir "${OFFICIAL_EVAL_OUTPUT}" \
    --max-prompts 0 \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --mode hybrid \
    || die "官方评测运行失败"
  log "官方评测完成 (耗时: $(fmt_duration $(step_timer_elapsed)))"
  touch "${CKPT_DIR}/06_official_eval"
fi

# ── 完成 ─────────────────────────────────────────────────────────────────
OVERALL_ELAPSED=$(( $(date +%s) - OVERALL_START ))
printf '\n============================================================\n'
log "  全量过夜流水线完成!"
log "  总耗时: $(fmt_duration ${OVERALL_ELAPSED})"
log ""
log "  输出文件:"
log "    benchmark:     ${BENCHMARK_OUTPUT}"
log "    官方评测:      ${OFFICIAL_EVAL_OUTPUT}/"
log "    rank ckpt:     ${RANK_CHECKPOINT}"
log "    训练数据:      ${RANK_TRAIN_DATA}"
printf '============================================================\n'
