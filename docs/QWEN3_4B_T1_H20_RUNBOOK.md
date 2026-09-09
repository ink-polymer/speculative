# Qwen3-4B / T=1 / H20 登记子矩阵运行与审计

本流程只完成 Qwen3-4B、T=1、H20 上的 4 方法登记子矩阵。它不代表 4B/8B、T=0/T=1 全部 65,280 次生成已完成，也不允许根据该子矩阵单独发布“树状块验证显著更快”的全局结论。

以下命令假定已进入最终 commit 的全新源码目录，且正式输出目录之前从未用于其他运行。

```bash
set -euo pipefail
export GBV_REPO=/root/autodl-tmp/speculative-treebv-FULL_COMMIT_SHA
export GBV_PY=/root/autodl-tmp/envs/speculative-py311/bin/python
export GBV_CONFIG="$GBV_REPO/configs/adaptive_block_qwen3_4b.json"
export GBV_SUITE="$GBV_REPO/configs/adaptive_tree_block_suite.json"
export GBV_DATA=/root/autodl-tmp/data/sampling-t1-formal-20260910-v1
export GBV_RUN=/root/autodl-tmp/outputs/qwen3-4b-t1-tree-block-h20-FULL_COMMIT_SHA
export GBV_EVIDENCE="$GBV_RUN/evidence"
export GBV_PREFLIGHT="$GBV_RUN/gpu_preflight.json"
export PYTHONPATH="$GBV_REPO/src"
export HF_HOME=/root/autodl-tmp/hf-home
export GBV_PROCESS_PYTHON=/opt/gbv-code-eval/bin/python
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=0
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
mkdir -p "$GBV_EVIDENCE"
```

在任何正式计时前，依次生成并保存这些证据：

```bash
"$GBV_PY" -c 'import sys; from pathlib import Path; from gbv_experiments.common import write_json; from gbv_experiments.integrated_suite import _run_distribution_law_audit; write_json(Path(sys.argv[1]), _run_distribution_law_audit())' \
  "$GBV_EVIDENCE/distribution_law_audit.json"

"$GBV_PY" -m gbv_experiments plan \
  --config "$GBV_CONFIG" \
  --output "$GBV_EVIDENCE/plan.json"

"$GBV_PY" -c 'import sys; from pathlib import Path; from gbv_experiments.common import write_json; from gbv_experiments.integrated_suite import _formal_matrix_audit; write_json(Path(sys.argv[1]), _formal_matrix_audit())' \
  "$GBV_EVIDENCE/formal_matrix_audit.json"

"$GBV_PY" -m gbv_experiments audit-integrated-suite \
  --suite "$GBV_SUITE" \
  --model-ids qwen3_4b \
  --output "$GBV_EVIDENCE/fairness_audit_qwen3_4b.json"

"$GBV_PY" -m gbv_experiments audit \
  --config "$GBV_CONFIG" \
  --data-dir "$GBV_DATA" \
  --output "$GBV_EVIDENCE/sampling_data_audit.json"

"$GBV_PY" -m gbv_experiments validate-gold \
  --config "$GBV_CONFIG" \
  --data-dir "$GBV_DATA" \
  --output "$GBV_EVIDENCE/sampling_gold_audit.json" \
  --code-backend process \
  --timeout 10

"$GBV_PY" "$GBV_REPO/scripts/capture_process_evaluator_self_test.py" \
  --output "$GBV_EVIDENCE/process_evaluator_self_test.json"

"$GBV_PY" -m gbv_experiments check-model \
  --config "$GBV_CONFIG" \
  --output "$GBV_PREFLIGHT" \
  --device cuda:0 \
  --code-backend process
```

`gpu_preflight.json` 是子矩阵的真实运行环境身份源：独立审计会重新检查其 `NVIDIA H20` 名称、`102085623808` 字节显存、Python/CUDA/PyTorch 版本、所有 T=1 相关包版本、固定 process 评分器路径与二进制 SHA-256、模型 revision/runtime 状态以及真实的同树验证路由。process evaluator 为独立 conda 环境，不伪造或要求不存在的 `pyvenv.cfg`。该 T=1 审计不要求 T=0 AdaptiveTree 才需要的 FlashAttention2 或官方 DDTree C++ cache compaction。

本次已准备数据的 `manifest.json` 文件 SHA-256 固定为 `3c46d4f25a3d4009ed9784c0b09aa2a870546ac2fed912204be6d8d3ce267faf`；post-run 审计会同时重读所有数据文件、source selection 和 prompt hash，而不只信任该顶层 SHA。

紧接正式计时前执行独占 GPU 门禁，然后在同一 `screen` 会话中启动全量运行：

```bash
"$GBV_PY" -c 'import sys,torch; from pathlib import Path; from gbv_experiments.common import write_json; from gbv_experiments.integrated_suite import _gpu_allocation_gate; write_json(Path(sys.argv[1]), _gpu_allocation_gate(torch.device("cuda:0")))' \
  "$GBV_EVIDENCE/gpu_allocation_before_timing.json"

"$GBV_PY" -m gbv_experiments run \
  --config "$GBV_CONFIG" \
  --data-dir "$GBV_DATA" \
  --output "$GBV_RUN" \
  --device cuda:0
```

不要添加 `--groups`、`--only-variants`、`--smoke` 或 `--profile`。断连后先确认原 `screen` 会话已停止；只有在旧进程确实不在运行时，才能以原命令续跑同一输出目录。

生成完成后，使用与 preflight 完全相同的评分后端和参数：

```bash
"$GBV_PY" -m gbv_experiments score \
  --run-dir "$GBV_RUN" \
  --data-dir "$GBV_DATA" \
  --code-backend process \
  --workers 4 \
  --timeout 10 \
  --lcb-timeout 6

"$GBV_PY" -m gbv_experiments report \
  --run-dir "$GBV_RUN" \
  --output "$GBV_RUN/report" \
  --bootstrap 10000
```

最后运行独立 post-run 审计：

```bash
"$GBV_PY" "$GBV_REPO/scripts/audit_t1_submatrix.py" \
  --run "$GBV_RUN" \
  --data "$GBV_DATA" \
  --preflight "$GBV_PREFLIGHT" \
  --audit-evidence "$GBV_EVIDENCE" \
  --output "$GBV_RUN/qwen3_4b_t1_h20_submatrix_audit.json"
```

成功时审计强制核对 9,792 条生成记录、10,752 个对话 turn、9,792 条评分（其中 8,832 条客观评分、960 条 MT-Bench `not_scored`）、32 行汇总、2,448 个完整方法顺序组、4 种方法和 16 行成对比较。输出只会将 `submatrix_complete` 置为 `true`；`formal_complete`、`whole_formal_matrix_complete`、`publication_claim_eligible` 和 `tree_block_superiority_claim_eligible` 仍强制为 `false`。
