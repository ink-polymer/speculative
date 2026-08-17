#!/usr/bin/env bash
set -euo pipefail

# Atlas A2 使用 ASCEND_RT_VISIBLE_DEVICES；不要替换为其他后端的可见卡变量。
source /opt/home/developer/Ascend/ascend-toolkit/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"

python -m dflash_specblock.cli \
  --config configs/qwen3_4b_a2.json \
  --prompt "请解释为什么推测解码能够无损加速大语言模型推理。" \
  --max-new-tokens 128

