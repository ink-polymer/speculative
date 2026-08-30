#!/usr/bin/env bash
set -euo pipefail

export PATH=/root/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
cd "$(dirname "$0")/.."

python scripts/finetune_dflash_draft.py \
  --config configs/qwen3_4b_cuda.json \
  --train-data datasets/generated/rank_choice_train_4domain_disjoint_192.jsonl \
  --output checkpoints/dflash_draft_t0_disjoint_v1 \
  --epochs 2 \
  --anchors-per-row 4 \
  --learning-rate 0.000002 \
  --gradient-accumulation 8 \
  > logs/draft_ft_t0_train.log 2>&1

mkdir -p outputs/draft_ft_t0_v1
python scripts/scan_lookup_ddtree.py \
  --prompts datasets/processed/dflash_official/prompts_policy_validation_4domain_disjoint_48.jsonl \
  --output outputs/draft_ft_t0_v1/screen8x512.jsonl \
  --draft checkpoints/dflash_draft_t0_disjoint_v1 \
  --fixed-draft models/Qwen3-4B-DFlash-b16 \
  --temperature 0 \
  --candidates "60:0:4:1,128:0:4:1,192:0:4:1,224:0:4:1,192:2:4:1,224:2:4:0.9" \
  --max-samples 8 \
  --max-new-tokens 512 \
  --enable-cpp-compact \
  > logs/draft_ft_t0_scan.log 2>&1
