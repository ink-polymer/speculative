#!/usr/bin/env bash
set -euo pipefail

source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"

python -m dflash_specblock.benchmark \
  --config configs/qwen3_4b_a2.json \
  --prompts examples/prompts.jsonl \
  --output outputs/qwen3_4b_a2.jsonl \
  --max-new-tokens 128

