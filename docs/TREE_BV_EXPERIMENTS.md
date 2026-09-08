# BRBV（tree_gbv_recycle）正式评测协议与运行脚本

本文件对应 `configs/tree_bv_suite.json` 的两模型（Qwen3-4B + Qwen3-8B）树分支回收实验。
协议沿用 `GBV` 的固定数据规模和抽样规则，新增对照项为：

- `tree_gbv_base`：树提议 + GBV 非回收基线（固定 `method=tree_gbv_full`）。
- `tree_bv_recycle`：树提议 + BRBV 回收（`method=tree_bv_recycle`）。
- 还保留 `target_t1`、`gbv`、`ddtree`。

## 版本锁定（与 2026-09-05 套件对齐）

- 评测集与种子：与 `configs/tree_bv_qwen3_4b.json`、`configs/tree_bv_qwen3_8b.json` 中声明一致（`gbv-paper ddtree-counts` 协议、seed 17/29/43）。
- 关键数值设置：`temperature=1.0`，`paths=3` 的 GBV；`ddtree` 固定 `tree_budget=45`；
  `tree_gbv_*` 固定 `paths=10`、`length=14`、`tree_budget=6`。
- 概率精度保持 FP64，模型与运行参数保持冻结。

当前 `tree_gbv_recycle` 在本地实现中对每次回收段使用 `sparse_lazy` 分段块验证（与 `tree_gbv_full_sparse_lazy` 保持一致的尾部一次性采样优化），并仅在树节点确认子树后复用树结构；不会改变数学分布。

## 三阶段调度

阶段与 `configs/tree_bv_suite.json` 的计划保持一致：

1. `recycle-first`：只跑 `tree_bv_recycle`（先快速验证可用性和回收路径）
2. `main`：补齐 `target_t1`、`gbv`、`ddtree`、`tree_gbv_base`
3. `complete`：补齐所有阶段并在完成后输出决策文件

在 `complete` 阶段，脚本会自动在输出目录下生成：

- `tree_bv_decision.json`：全局决策汇总
- `tree_bv_decision.md`：可读结果摘要

## 统计口径（严格定义）

只做统计闸门，不做“必胜”承诺。判定规则是：

- 对每个 `(dataset, source_id)` 聚类进行成对 bootstrap（同源同题的所有 seed 一起采样）；
- 计算 `tree_bv_recycle` 相对 `ddtree` 的速度倍率；
- 若 95% CI 下界 > 1.0，则该模型通过该指标；
- 所有模型都通过才整体通过。

决策文件同时会记录：

- 总体/各数据集的加速比与 CI；
- 回收段数量（`tree_bv_segments`）、回收命中率、每 round 复用纠正次数；
- 选择/回收阶段耗时（host 与 cuda-event）。

## 运行方式

推荐入口：

```bash
PYTHON_BIN=.venv-gbv/bin/python bash scripts/run_tree_bv_paper.sh plan        # 打印完整计划（写 plan_complete.json）
PYTHON_BIN=.venv-gbv/bin/python bash scripts/run_tree_bv_paper.sh recycle-first  # 阶段1
PYTHON_BIN=.venv-gbv/bin/python bash scripts/run_tree_bv_paper.sh main          # 阶段2
PYTHON_BIN=.venv-gbv/bin/python bash scripts/run_tree_bv_paper.sh complete      # 阶段3并输出决策
```

常用覆盖参数：

- `SUITE`（默认 `configs/tree_bv_suite.json`）
- `DATA_DIR`（默认 `datasets/gbv_paper_ddtree_counts`）
- `OUTPUT`（默认 `outputs/tree_bv_benchmark`）
- `DEVICE`（默认 `cuda:0`）
- `CODE_BACKEND`（默认 `docker`）

单模型续跑示例：

```bash
PYTHON_BIN=.venv-gbv/bin/python bash scripts/run_tree_bv_paper.sh main --model-ids qwen3_8b
```

## 解释边界

本框架只输出统计闸门与可复现记录；不承诺“比 DDTree 一定更快”。
硬件抖动、采样波动和机器状态可能使同一配置在重跑时在 CI 区间内波动。
