# AdaptiveTree T=0 历史协议说明

> 本文档归档 2026-09-03 的旧 T=0 三模型方案，**不是当前正式实验规范或启动入口**。
> 当前唯一权威规范是 [正式实验矩阵](FORMAL_EXPERIMENT_MATRIX.md)；部署、doctor、审计和汇总
> 以 [统一运行说明](INTEGRATED_ADAPTIVE_TREE_BLOCK_EXPERIMENT.md) 为准。

## 当前正式 T=0 范围

当前只运行 Qwen3-4B 与 Qwen3-8B，不运行 30B。两个模型采用同一份冻结数据、顺序、
生成长度、后端和环境契约，从新的数据及结果目录完整重跑。

| 类别 | 当前正式方法 |
|---|---|
| 基线 | Target、DFlash |
| 固定树 | DDTree：7 个固定预算 |
| AdaptiveTree 主方法 | `adaptive`：`B<=256`，构树成本采用 budget-aware attribution，无周期探索 |
| 历史对照 | `adaptive_legacy`：保留旧 B<=128、旧计时归因和周期探索语义 |
| 消融/控制 | 6 项；其中 `adaptive_legacy_cost_attribution` 只改变成本归因，是严格单因素对照 |

AdaptiveTree 的完整注册表固定为 8 个方法（主方法 + 1 个历史对照 + 6 项消融/控制）。
方法键、角色、预算和控制器参数由配置与审计器共同冻结，不能在运行时用同名参数覆盖。

T=0 之外，新服务器还包含一个独立 T=1 矩阵：Target、DFlash、DDTree-B45 与同树
`tree_block_verification`（`L=15, B=45`），使用 3 个生成 seed。T=0/T=1 之外的温度及
30B 均延期；该新矩阵不能续接此前基于 `16c0e91` 启动过且不含树状块的 H20 任务。

正式计划只通过统一入口查看：

```bash
bash scripts/run_integrated_fresh_server.sh plan
```

## 公平性与审计边界

- 同一模型、数据集和温度内，各方法共享输入、顺序、输出长度与环境身份。
- T=0 的固定预算 DDTree 和 AdaptiveTree 使用相同 target/draft 权重与验证路径；差异只来自
  已注册的构树预算策略。
- `adaptive_legacy_cost_attribution` 除成本归因外必须与 canonical 配置相同，避免把多个变化
  混成一个消融结论。
- 汇总前重新验证数据、代码、权重、Python/CUDA/GPU UUID、包版本、结果覆盖与数值完整性；
  旧目录、部分结果或缺少契约字段的清单不得续用。
- T=0 的 greedy 一致性与 T=1 的采样分布正确性使用不同的验证门，不能相互替代。
- 树状块验证不改变本文的 T=0 表；它只进入新服务器独立 T=1 的调用数和表格。

数据集、样本数、调用数、T=1 seed、评分后端及报告字段的精确定义不在本文重复，避免形成
第二套会漂移的规范；请直接查阅 [正式实验矩阵](FORMAL_EXPERIMENT_MATRIX.md)。

## 旧协议为何归档

旧方案曾把“原版 adaptive”描述为最大 128 节点、六候选预算、每 64 次周期探索，并只列四项
消融；还计划同时运行 Qwen3-Coder-30B-A3B-Instruct。该描述现在仅对应
`adaptive_legacy` 历史对照，不能代表修正后的 canonical AdaptiveTree。

旧 `run_paper_t0_full.sh`、`run_paper_t0_qwen3_8b.sh`、历史数据目录和历史结果目录仍可用于
追溯旧实验，但不得用于启动、续跑或汇总当前正式矩阵。旧证明说明也只覆盖其明确写出的
B<=128 理想化方法边界；参见 [历史证明核校记录](ADAPTIVE_PAPER_PROOF_CHECK.md)。

## 尚不能提前声称的结论

代码审计、单元测试和数学证明不等于新服务器上的完整 GPU 结果。正式运行结束并通过汇总审计
之前，不能声称 corrected AdaptiveTree 更快、任务质量更高、所有 BF16 输出逐 token 一致，
也不能把历史截图或部分运行结果作为本轮结论。
