# Adaptive DDTree：Qwen3-8B 独立版本

这是原版非 RL、T=0 AdaptiveTree 的 8B 入口；完整矩阵原本已包含 8B，现在提供独立启动脚本和结果目录，不必同时运行 4B、30B。方法、证明、官方数据与四项消融不变，不训练模型或策略。

## 模型与固定版本

| 角色 | 模型 | 固定 revision |
|---|---|---|
| Target | Qwen/Qwen3-8B | b968826d9c46dd6066d109eabc6255188de91218 |
| Draft | z-lab/Qwen3-8B-DFlash-b16 | 9b41424b7109f9c5413454f481b09a82b85333f4 |

与已上传的 [GBV 8B 配置](https://github.com/ink-polymer/speculative/blob/63635871b7176bfee5583be5d3fd01239066ea4f/configs/gbv_paper_qwen3_8b.json) 使用相同模型版本，但不复用 GBV 的温度、概率精度或验证规则。Target/Draft 均冻结；正式权重精度 BF16。

本地通过真实配置的 meta tensor 构造检查：hidden size 4096、词表 151936，Target 36 层、Draft 5 层，Target 特征层为 1/9/17/25/33，融合输入维度为 5×4096。Draft block size 16，仍是 1 个 anchor 加 15 个未来位置。此检查不下载权重，不等于真实 GPU 推理验收。

旧清单若记录了不同的 8B revision，会明确报错，必须使用新数据/结果目录；不能静默改权重并续用旧结果。

## 实验范围

- 复用官方十数据集、seed=0 抽样；1,072 题/对话、每方法 1,152 次回答。
- SDPA：官方 AR、DFlash、7 个 DDTree 预算、Adaptive、4 项消融；FA2：官方 AR、DFlash。Draft 始终 FA2。
- Adaptive 候选预算仍为 30/45/60/80/100/128。四项消融仍为去接受校准、去延迟项、去周期探索、预热后冻结校准。
- T=0，每回答最多 2,048 新 token。8B 完整子矩阵为 20 个进程组、18,432 次生成调用，另加预热。
- 默认八进程与官方相同；单卡需显式指定 --nproc-per-node 1，并记录硬件偏离。报告明确标为单模型子矩阵，不称为三模型完整复现。
- 使用官方 TPOT 口径及逐 token 对照；失败保存诊断并停止，不过滤失败题。没有新的实测加速比。

## 独立运行

先按 [环境说明](PAPER_T0_EXPERIMENTS.md) 安装 CUDA PyTorch、FA2 及编译依赖。下列命令从项目根目录运行：

```bash
# 不传参数时只展示计划，不下载、不运行 GPU。
bash scripts/run_paper_t0_qwen3_8b.sh
bash scripts/run_paper_t0_qwen3_8b.sh plan

# 单卡服务器检查及数据准备。
CUDA_VISIBLE_DEVICES=0 bash scripts/run_paper_t0_qwen3_8b.sh doctor --nproc-per-node 1
bash scripts/run_paper_t0_qwen3_8b.sh prepare

# 单独输出的 smoke；不能用于论文。
CUDA_VISIBLE_DEVICES=0 bash scripts/run_paper_t0_qwen3_8b.sh all \
  --nproc-per-node 1 --dataset gsm8k --smoke-count 2 \
  --run-dir outputs/adaptive_qwen3_8b_smoke

# 单卡、完整 8B 子矩阵。核验 smoke 后再运行。
CUDA_VISIBLE_DEVICES=0 bash scripts/run_paper_t0_qwen3_8b.sh all --nproc-per-node 1

# 单独汇总时保持相同进程数、数据目录及实验范围。
bash scripts/run_paper_t0_qwen3_8b.sh summarize --nproc-per-node 1
```

默认输出 outputs/adaptive_ddtree_official_t0_qwen3_8b；共享数据目录仍为 datasets/ddtree_official_t0。支持自定义 --run-dir 和 --data-dir；8B 入口不允许改 --model-index。模型加载和生成仍调用同一份官方执行器，没有另写一套验证算法。

## 与 GitHub GBV 数据的区别

已核对 GBV 分支 codex/gbv-paper-full-seven-datasets 的提交 63635871b7176bfee5583be5d3fd01239066ea4f。其 [8B 配置](https://github.com/ink-polymer/speculative/blob/63635871b7176bfee5583be5d3fd01239066ea4f/configs/gbv_paper_qwen3_8b.json) 使用 GSM8K 128、MATH-500 128、AIME25 30、HumanEval 164、MBPP 128、LiveCodeBench 128、MT-Bench 80，共 786 题/对话、866 次回答；选题 seed=0，生成种子 17/29/43。

当前 Adaptive 仍遵循用户此前确认的完整 DDTree 官方 T=0 数据协议，另外包含 AIME24 30、SWE-bench 128、Alpaca 128。不因“参考 GBV”而自动替换已经确认的数据协议；同名数据集或同样条数也不自动代表样本、提示和计时口径相同。

已核对该提交的 [GBV 数据说明](https://github.com/ink-polymer/speculative/blob/63635871b7176bfee5583be5d3fd01239066ea4f/docs/GBV_PAPER_EXPERIMENTS.md)，还存在以下区别：

| 项目 | GitHub GBV | 当前 Adaptive 官方协议 |
|---|---|---|
| MBPP | full/test 500 中抽 128 | sanitized/test 中抽 128 |
| MT-Bench 来源 | FastChat 固定提交的 question.jsonl | HuggingFaceH4/mt_bench_prompts |
| MT-Bench 第二轮 | 每方法使用自己的首轮回答 | 同组共享官方指定方法的首轮回答 |
| LiveCodeBench | 累计六文件 release_v6、自定义任务提示 | 官方同六文件加载及官方 format_lcb |
| 生成种子 | 17、29、43 | 0 |

GBV 文档本身也只声明题数和抽样规则对齐 DDTree，不声称完整数据与运行协议相同。因此本次仅对齐 8B 权重配对并记录差异，没有把两套结果混成同一个协议。
