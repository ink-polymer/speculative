# AdaptiveTree：修正版非 RL，DDTree 官方 T=0 实验

新增 [Qwen3-8B 独立版本](docs/ADAPTIVE_QWEN3_8B.md)：bash scripts/run_paper_t0_qwen3_8b.sh plan。包含同一套正式方法、历史控制及六项消融，默认独立输出；不传参数仅显示计划，不启动 GPU。8B 权重 revision 与已上传 GBV 配置一致。

本包以修正成本归因、保留原始六候选预算的动态 `adaptive_b128` 作为正式主方法，同时保留明确命名的历史对照、单因素消融、数学证明、正式评测和测试。`adaptive_b256` 只用于检验扩大预算上限。**没有 RL 训练、GBV、模型权重、数据文件或新 GPU 实验结果。** 文件名中的 full 指完整实验矩阵，采样数量按用户要求采用 DDTree 官方设置，并非全量数据集。

[构树与流程图](docs/ADAPTIVE_DDTREE_METHOD.md) · [论文版数学证明](docs/ADAPTIVE_DDTREE_T0_PAPER_PROOF.md) · [控制器公式与实现边界](docs/ADAPTIVE_DDTREE_T0_PROOF.md) · [完整实验说明](docs/PAPER_T0_EXPERIMENTS.md) · [官方对齐核对](docs/DDTREE_PROTOCOL_ALIGNMENT.md)

## 方法与评测

- 保留原版 DDTree best-first；正式 `adaptive_b128` 最多枚举 128 节点，在 30/45/60/80/100/128 的嵌套树间按校准接受收益与完整预算相关实测成本动态选预算并持续更新。没有 policy 网络、训练集或 checkpoint。
- 官方十数据集：GSM8K 128、MATH-500 128、AIME24 30、AIME25 30、HumanEval 164、MBPP-sanitized 128、LiveCodeBench 128、SWE-bench 128、MT-Bench 80、Alpaca 128。
- 共 1,072 题/对话，含 MT-Bench 双轮后每方法 1,152 次回答。直接执行固定版官方数据处理和 seed=0 抽样，不是全量测试集。
- 三组原始 Target/DFlash 模型：Qwen3-4B、Qwen3-8B、Qwen3-Coder-30B-A3B-Instruct。T=0、BF16、每回答最多 2,048 新 token。
- Draft 使用 FA2；Target 分 SDPA/FA2 两组；树方法仅 SDPA。固定 DDTree 对照只运行 B128，复用官方实现并与主方法保持相同最大节点预算。
- 正式主方法、历史控制与六项消融：B=128、恢复旧成本归因、恢复周期探索、去接受校准、去延迟项、预热后冻结校准。默认完整矩阵 60 个进程组、44,928 次生成调用，另加预热。
- 逐题 token 对照官方 Target-only。默认严格复现模式会在不一致时保存诊断并停止；公平整合矩阵使用 `record-bf16-mismatches` 保留并统计全部差异，不删除或筛掉样本。两种口径都不是任务准确率评分或 BF16 无条件等价保证。
- 公平整合实验可显式启用 `--method-order-policy balanced-rotation`，让所有方法在每个计时位置均衡轮换；默认仍为 `official-fixed` 以保留上游复现口径。同后端架构主表写入 `tables_controlled_sdpa.csv`，最佳后端表只作辅助对照。

## 环境与运行

进入本目录，在独立 Python 3.10/3.11 环境中先安装与 NVIDIA 驱动匹配的 CUDA PyTorch，再执行：

```bash
python -m pip install -r requirements-paper.txt
python -m pip install -e . --no-deps
# 按当前 PyTorch/CUDA 安装兼容的 flash-attn；还需可用 C++ 编译器。
bash scripts/run_paper_t0_full.sh plan
bash scripts/run_paper_t0_full.sh doctor
bash scripts/run_paper_t0_full.sh prepare
bash scripts/run_paper_t0_full.sh all
```

默认沿用官方八 GPU 进程。单卡、仅 4B 的联调必须显式缩小范围：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_paper_t0_full.sh all \
  --nproc-per-node 1 --model-index 0 --dataset gsm8k \
  --smoke-count 2 --run-dir outputs/adaptive_official_smoke
```

smoke 不能用于论文；显式单卡/模型子集会记录为协议范围或硬件偏离。正式运行可分别调用 evaluate 和 summarize。上游未公开历史 HF 快照，本包锁定本次数据和权重 revision；不能声称复原未知的作者历史快照。没有 collect/train 步骤。

成本归因修正和 B=128 六候选现已注册为正式 `adaptive_b128`。tree-build、编译、
Target 验证和 KV/commit 延迟都计入所选预算；仅 proposal 作为固定成本。旧方法明确
命名为 `adaptive_legacy`，不再占用主方法名称。正式方法/对照如下：

| 名称 | 作用 |
|---|---|
| `adaptive_b128` | 修正成本归因、B≤128、无周期探索的动态正式主方法 |
| `adaptive_b256` | 仅将候选预算上限扩到 B=256 的单因素消融 |
| `adaptive_legacy` | B≤128、旧成本归因、带周期探索的历史控制 |
| `adaptive_legacy_cost_attribution` | 保持 B≤128、关闭探索，仅恢复旧成本归因的单因素消融 |
| `adaptive_with_exploration` | 只恢复周期探索的单因素消融 |
| `adaptive_no_acceptance_calibration` | 去接受率校准 |
| `adaptive_no_latency` | 去延迟判别 |
| `adaptive_frozen_after_warmup` | 预热后冻结在线估计 |

旧的 `--experimental-cost-attribution` 与 `--experimental-extended-budgets` 参数仅为
启动脚本兼容而保留，不会添加重复方法。旧长方法名只可用于重现历史 artifact，绝不
写入新的正式 contract。报告在每行写入 `method_role`，并分别提供 `primary_rows`、
`ablation_rows` 和 `adaptive_budget_usage`。

### W&B 在线监控

W&B 是可选依赖；`requirements-paper.txt` 已固定版本。密钥只从作业环境中的
`WANDB_API_KEY` 读取，绝不能写入配置、命令行参数或仓库。交互式安全输入后再提交：

```bash
read -rsp 'WANDB API key: ' WANDB_API_KEY && printf '\n'
export WANDB_API_KEY
export WANDB_PROJECT=adaptivetree
export ADAPTIVE_MODEL_INDEX=0  # 另提交一次 1，即 Qwen3-8B
sbatch --export=ALL ../scripts/adaptive-cost-attribution.sbatch
unset WANDB_API_KEY
```

每个 GPU rank 建立一个同组 run，实时记录逐回答 decode TPOT、tokens/s、相对本次
Target 的加速比、接受长度、阶段耗时，以及 AdaptiveTree 的预算均值/最大值和
`>128` 轮次占比；不上传 prompt 或生成 token。也可直接给入口传入
`--wandb-project PROJECT [--wandb-entity ENTITY] [--wandb-group GROUP]`。为减少对延迟
实验的扰动，入口关闭 W&B 的后台系统指标、系统元数据和代码采集，只上传上述显式指标。

## 本地检查与历史结果

```bash
PYTHONPATH=src python -m pytest \
  tests/test_paper_runtime_audit.py tests/test_paper_real_forward.py \
  tests/test_paper_qwen3_8b.py tests/test_paper_official_protocol.py tests/test_paper_protocol.py \
  tests/test_paper_datasets_extended.py tests/test_config_device.py \
  tests/test_engine.py tests/test_ddtree_builder.py tests/test_ddtree_integration.py \
  tests/test_verification.py tests/test_dflash_adapter.py tests/test_vanilla_engine.py \
  -o addopts='' -q
```

本地测试不加载真实预训练模型，不能替代服务器 GPU 验收。旧截图 5.468/5.644/6.801/5.516× 对应原版非 RL 算法，但原始 2,000 条中只有 727 条与当时 AR 完全一致；不能作为已验证无损或新官方协议结果。

SOURCE_SHA256.json 记录本包当前分发文件（不含清单自身）的校验和。third_party/ddtree_pinned 保留官方源码和独立来源清单；旧 DDTree reference 及 controlled_* 文件只供回归测试，不是当前入口。来源与许可证见 [OFFICIAL_SOURCES.md](third_party/OFFICIAL_SOURCES.md) 和 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
