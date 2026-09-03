# AdaptiveTree：T=0 全量论文实验

本目录是独立运行的分层结构策略 AdaptiveTree 实验包，包含训练、主实验、
11 个独立重训消融和相关测试。**不包含 GBV 实验实现、模型权重、数据集或正式 GPU 结果。**

与仓库根目录的历史实验相互独立；运行下面的命令前先进入 `adaptivetree_paper/`。
完整协议见 [PAPER_T0_EXPERIMENTS.md](docs/PAPER_T0_EXPERIMENTS.md)。

## 实验范围

- 原始 Qwen3-4B target 与 DFlash-b16 draft 均冻结，仅训练树结构策略。
- 同状态反事实采集 → 监督预训练 → Monte Carlo actor-critic → dev 选模 → 全量配对测试。
- 主测试：GSM8K 1,319、MATH-500 500、HumanEval 164、MBPP-full 500、
  AIME25 30、LiveCodeBench release_v6 1,055、MT-Bench 80 组双轮对话。
- MBPP-sanitized test 257 题为独立附加测试，不与 MBPP-full 混写。
- 原始训练源 15,347 题；去除与评测重叠及重复样本后，约 90% train / 10% dev。
- T=0、BF16、共享 eager 后端；关闭 TF32、CUDA Graph、torch.compile。
- 逐题逐轮对齐 AR token；失败不删题。质量评分器不属于本入口。

## 获取本目录

仓库历史分支中已有较大的运行产物。可只获取本次实验包，避免下载它们：

```bash
git clone --depth 1 --filter=blob:none --sparse \
  --branch paper/adaptivetree-full-datasets-20260903 \
  https://github.com/ink-polymer/speculative.git speculative-adaptivetree
cd speculative-adaptivetree
git sparse-checkout set adaptivetree_paper
cd adaptivetree_paper
```

## 环境与入口

正式服务器使用独立 Python 3.10/3.11 环境，先安装与驱动兼容的 CUDA PyTorch，
然后在本目录执行：

```bash
python -m pip install -r requirements-paper.txt
python -m pip install -e . --no-deps
export CUDA_VISIBLE_DEVICES=0
bash scripts/run_paper_t0_full.sh plan
bash scripts/run_paper_t0_full.sh doctor
```

`plan` 只展示实验矩阵，不下载、不训练。`doctor` 检查 GPU 环境。
全量数据下载、模型加载和训练须在服务器另行运行：

```bash
bash scripts/run_paper_t0_full.sh prepare
bash scripts/run_paper_t0_full.sh all --smoke-count 2 --run-dir outputs/paper_t0_smoke
bash scripts/run_paper_t0_full.sh collect
# 阅读候选空间诊断后，再决定是否启动正式训练。
bash scripts/run_paper_t0_full.sh train
bash scripts/run_paper_t0_full.sh evaluate
bash scripts/run_paper_t0_full.sh summarize
```

smoke 仅用于联调，不能作为论文结果。完整默认矩阵包含 36 个策略 checkpoint、
609,705 次测试生成调用，另有反事实采集、训练和验证开销。
`2048` 是每轮回答的 token 上限，不是题量上限。

## 本地测试

以下测试不需要下载预训练模型或真实数据；不能替代服务器 BF16/GPU 验收：

```bash
PYTHONPATH=src python -m pytest \
  tests/test_paper_protocol.py tests/test_paper_datasets_extended.py \
  tests/test_engine.py tests/test_ddtree_builder.py tests/test_ddtree_integration.py \
  tests/test_verification.py tests/test_dflash_adapter.py tests/test_vanilla_engine.py \
  -o addopts='' -q
```

`SOURCE_SHA256.json` 记录原样复制的源码、配置、协议、依赖与测试文件的 SHA-256。
本目录新增的 README、忽略规则和来源说明不包含在该原始文件清单中。
第三方文件的来源及用途见 [OFFICIAL_SOURCES.md](third_party/OFFICIAL_SOURCES.md)。
