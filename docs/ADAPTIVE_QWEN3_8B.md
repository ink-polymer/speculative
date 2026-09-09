# Qwen3-8B AdaptiveTree 历史独立入口

> 本文档只保留 2026-09-03 旧版 Qwen3-8B 单模型入口的追溯信息，**不再定义当前正式实验**。
> 当前必须以 [正式实验矩阵](FORMAL_EXPERIMENT_MATRIX.md) 和
> [统一运行与审计说明](INTEGRATED_ADAPTIVE_TREE_BLOCK_EXPERIMENT.md) 为准。

## 新服务器正式口径

Qwen3-8B 仍是正式模型之一，但不再通过旧的 `run_paper_t0_qwen3_8b.sh` 独立入口定义实验。
当前统一入口同时覆盖 Qwen3-4B 和 Qwen3-8B，并把温度严格限制为 T=0 与 T=1：

- T=0 比较 Target、DFlash、DDTree 的 7 个固定预算，以及修正后的 canonical AdaptiveTree。
- canonical AdaptiveTree 使用 `B<=256`、budget-aware tree-build cost attribution，并关闭周期探索。
- AdaptiveTree 固定为 8 方法注册表：主方法、`adaptive_legacy` 历史对照及 6 项消融/控制；
  `adaptive_legacy_cost_attribution` 是只切换成本归因的严格单因素对照。
- T=1 比较 Target、DFlash、DDTree-B45 与同树 `tree_block_verification`
  （`L=15, B=45`），运行 3 个生成 seed。
- 30B 模型和其他温度延期；树状块只在新服务器的全新矩阵运行，不续接
  此前基于 `16c0e91` 启动过且不含树状块的 H20 任务。

查看当前计划应使用：

```bash
bash scripts/run_integrated_fresh_server.sh plan
```

## 保留的历史事实

旧独立入口使用以下固定权重配对；当前统一矩阵仍会独立校验其 revision，不允许混入可变本地权重：

| 角色 | 模型 | 固定 revision |
|---|---|---|
| Target | Qwen/Qwen3-8B | `b968826d9c46dd6066d109eabc6255188de91218` |
| Draft | z-lab/Qwen3-8B-DFlash-b16 | `9b41424b7109f9c5413454f481b09a82b85333f4` |

旧入口曾采用 T=0、最大 128 节点、预算 30/45/60/80/100/128、周期探索和四项消融，
并计划在十数据集上单独运行 8B。该控制器现在命名为 `adaptive_legacy`，只作为历史对照；
其预算上限、计时归因与探索策略不能再冠以 canonical AdaptiveTree 名称。旧结果目录也不能续接到
当前运行目录。

旧版与 GBV 数据口径的差异（MBPP 划分、MT-Bench 上下文来源、LiveCodeBench 格式及 seed）
仍具有追溯价值，但不用于定义当前矩阵。当前数据集、样本数、种子、评分和预热约束均由
正式矩阵及统一审计器冻结。

## 历史命令的状态

`scripts/run_paper_t0_qwen3_8b.sh` 仅为复查旧协议保留。不要用它启动或汇总本轮正式实验，
也不要把它生成的结果与统一入口结果合并。正式运行只使用
`scripts/run_integrated_fresh_server.sh`，并从新的数据目录和结果目录开始。
