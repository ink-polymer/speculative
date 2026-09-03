# Adaptive DDTree：原版非 RL，DDTree 官方 T=0 实验

新增 [Qwen3-8B 独立版本](docs/ADAPTIVE_QWEN3_8B.md)：bash scripts/run_paper_t0_qwen3_8b.sh plan。包含同一套主实验及四项消融，默认独立输出；不传参数仅显示计划，不启动 GPU。8B 权重 revision 与已上传 GBV 配置一致。

本包恢复截图对应的原版延迟感知预算控制器，包含方法介绍、数学证明、正式评测、4 项消融和测试。**没有 RL 训练、GBV、模型权重、数据文件或新 GPU 实验结果。** 文件名中的 full 指完整实验矩阵，采样数量按用户要求采用 DDTree 官方设置，并非全量数据集。

[构树与流程图](docs/ADAPTIVE_DDTREE_METHOD.md) · [论文证明](docs/ADAPTIVE_DDTREE_T0_PROOF.md) · [完整实验说明](docs/PAPER_T0_EXPERIMENTS.md) · [官方对齐核对](docs/DDTREE_PROTOCOL_ALIGNMENT.md)

## 方法与评测

- 保留原版 DDTree best-first；最多枚举 128 节点，在 30/45/60/80/100/128 的嵌套树间按校准接受收益与实测成本选预算。没有 policy 网络、训练集或 checkpoint。
- 官方十数据集：GSM8K 128、MATH-500 128、AIME24 30、AIME25 30、HumanEval 164、MBPP-sanitized 128、LiveCodeBench 128、SWE-bench 128、MT-Bench 80、Alpaca 128。
- 共 1,072 题/对话，含 MT-Bench 双轮后每方法 1,152 次回答。直接执行固定版官方数据处理和 seed=0 抽样，不是全量测试集。
- 三组原始 Target/DFlash 模型：Qwen3-4B、Qwen3-8B、Qwen3-Coder-30B-A3B-Instruct。T=0、BF16、每回答最多 2,048 新 token。
- Draft 使用 FA2；Target 分 SDPA/FA2 两组；树方法仅 SDPA。固定 DDTree 对照预算 16/32/64/128/256/512/1024，采用官方回答级 decode TPOT 均值之比。
- 主方法与四项消融：去接受校准、去延迟项、去周期探索、预热后冻结校准。默认完整矩阵 60 个进程组、55,296 次生成调用，另加预热。
- 逐题 token 对照官方 Target-only；不一致保存诊断并停止，不删除失败题。不是任务准确率评分或 BF16 无条件等价保证。

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
