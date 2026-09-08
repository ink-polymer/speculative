# DFlash-SpecBlock：面向 NVIDIA GPU 的动态树推测解码

本分支把冻结的原版 AdaptiveTree 论文包（[`adaptivetree_paper/`](adaptivetree_paper/)）
与后续的正温度扩散树上块验证工程放在同一个可复现分支中。整合边界、GH200
AdaptiveTree 部分结果和新方法的运行入口见
[`docs/ADAPTIVETREE_DIFFUSION_TREE_BV_INTEGRATION.md`](docs/ADAPTIVETREE_DIFFUSION_TREE_BV_INTEGRATION.md)。

新增独立理论原型：[固定覆盖核 + 预算停止的扩散块验证](docs/DIFFUSION_CORE_STOPPING.md)。
给出可预测停止下完整序列等分布证明、精确有理数核对和有条件双基线优势；
修复两个旧反例，同时保留新构造仍输给 DDTree 的反例。**查重显示大量框架重合，
原创性尚未成立**。未接入/替换下列冻结 GPU 实验，不把理论接受数当作实测速度。

当前改进候选：[保底路径与树内续接的扩散块验证](docs/DIFFUSION_SCAFFOLD_BV.md)。
L15/B45 下保留 DFlash 完整路径、随机扩散路径及高概率填充；证明固定上下文的
DFlash 接受长度保底，并严格修复旧反例；但新版本仍有同预算输给 DDTree 的严格反例，
不能声称普遍支配。尚未证明 GPU 速度或新颖性。
`bash scripts/run_diffusion_tree.sh plan --study configs/diffusion_scaffold_t10.json` 查看新计划；
t03/t06/t10 是独立新配置，旧注册不覆盖。

上一版研究入口：**单步块扩散的树上块验证**，见 [完整推导与运行协议](docs/DIFFUSION_TREE_BV.md)。
从 DFlash 的 clean-anchor + masked-block 去噪核出发，所有候选位置随机耦合，
共享前缀合并为 trie；保留联合潜变量后共同选择接受分支、深度与纠正 token。
证明包含扩散块截断质量、完整输出分布守恒及依赖去噪交叉熵的接受下界。
`bash scripts/run_diffusion_tree.sh` 只显示 T=0.3/0.6/1.0 三项计划。
新颖性和 GPU 速度尚未确立；旧 AC-TBV 与下面的历史实验保留为独立对照。
新增 [理论对照审计](docs/DIFFUSION_TREE_THEORY_AUDIT.md)：无损的明确假设、
同时优于两者的条件定理，以及驳倒“当前方法无条件总更好”的同预算严格反例。

AdaptiveTree 新增 [Qwen3-8B 独立版本](docs/ADAPTIVE_QWEN3_8B.md)：`bash scripts/run_paper_t0_qwen3_8b.sh plan`。采用配套 8B DFlash 固定权重版本，包含同一套主实验与四项消融；无参数只显示计划，不启动 GPU。

新增 **原版 T=0 Adaptive DDTree 正式实验入口（非 RL）**：只自适应节点预算，不做策略训练。按用户确认，数据和评测采用 DDTree 官方协议：十数据集、seed=0 抽样，共 1,072 题/对话，含双轮后每方法 1,152 次回答；不是全量数据。三组模型、主实验与消融见 [运行与审计说明](docs/PAPER_T0_EXPERIMENTS.md)，[构树介绍](docs/ADAPTIVE_DDTREE_METHOD.md) 与 [论文证明](docs/ADAPTIVE_DDTREE_T0_PROOF.md)。使用 `bash scripts/run_paper_t0_full.sh plan` 查看矩阵；GPU 正式实验尚未运行。旧分层 RL 已归档，不与原版结果混用。下文其他历史实验不是这个入口的默认配置。

**三路径 GBV 论文实验（Qwen3-4B + Qwen3-8B，12 配置精简方案）**：两个模型均比较 AR、DFlash、单路径 BV、K=3 GBV 和 DDTree；4B 额外比较 K=2、K=4，共 4B 的 7 配置、8B 的 5 配置。正式温度固定 T=1，小规模上机贪心正确性检查保留。数据量继续对齐 DDTree，每个配置/种子 786 题或对话、866 次回答；三个种子共 31,176 次回答。默认 `bash scripts/run_gbv_paper.sh gbv-first` 先运行两模型的 K=3 GBV；随后 `main`、`complete` 复用已有结果补齐。详见 [正式实验说明](docs/GBV_PAPER_EXPERIMENTS.md)。下文的旧 2k / T=0 流程单独保留。

本工程把 DFlash 的 block diffusion drafter 与 SpecBlock 的动态树组织、ancestor-only 并行
验证组合为一个可审计实现，并以 NVIDIA CUDA 作为正式训练、推理和性能后端。

- **DFlash**：目标模型多层隐藏状态注入 draft layer KV，未来 token 以扩散块一次产生；
- **SpecBlock**：构建 greedy 主链和 rank-guided 兄弟分支，再用目标模型并行验证；
- **DDTree**：把同一次 block forward 的 K 个分布视作条件独立，用全局 best-first 堆在节点
  预算内挑出概率最高的候选前缀集合；单块、无需 rank head，由 `tree_mode="ddtree"` 启用；
- **GPU 优化**：PyTorch SDPA、Tensor Core/TF32、静态 KV cache、CUDA Graph，以及经过
  正确性验证后可选的 `torch.compile`；
- **严格验证**：正式模式为 greedy lossless decoding，每个样本都与目标模型逐 token
  baseline 比较。

论文与官方实现：

- DFlash paper：<https://arxiv.org/abs/2602.06036>
- DFlash code：<https://github.com/z-lab/dflash>
- SpecBlock paper：<https://arxiv.org/abs/2605.07243>
- SpecBlock code：<https://github.com/shiweijiezero/SpecBlock>
- DDTree paper：<https://arxiv.org/abs/2604.12989>
- DDTree code：<https://github.com/liranringel/ddtree>

## 草稿树拓扑：SpecBlock 与 DDTree

两种拓扑共用同一套 DFlash drafter、ancestor-only 4D mask 验证、最长 greedy 路径选择和 KV
cache 压缩，只有「如何把节点预算分配到各候选」这一步不同，因此两者的 benchmark 可直接对照。

| | `tree_mode="specblock"` | `tree_mode="ddtree"` |
|---|---|---|
| 预算分配 | 逐 slot 局部宽度（rank bucket 决定） | 全局 best-first 堆（累计 log-prob 排序） |
| rank head | 必需 | 不使用 |
| 跨块 continuation | 支持（`max_blocks>=2`） | 不适用（`max_blocks=1`） |
| 预算超限处理 | 事后 `prune` 并保护主链 | 构建期自然截断 |
| 配置 | [configs/qwen3_4b_cuda.json](configs/qwen3_4b_cuda.json) | [configs/qwen3_4b_cuda_ddtree.json](configs/qwen3_4b_cuda_ddtree.json) |

DDTree 的正确性依据：在条件独立假设下，候选前缀的对数联合概率等于各 slot log-prob 之和，
因此堆顶始终是全局最优的未生成节点。`tests/test_ddtree_builder.py` 用穷举验证了它在给定
预算下最大化期望接受 token 数，并逐节点比对了 `third_party/ddtree_official` 的官方实现。

运行 DDTree benchmark（无需先训练 rank head）：

```bash
bash scripts/run_ddtree_benchmark.sh
MAX_PROMPTS=2 MAX_NEW_TOKENS=32 bash scripts/run_ddtree_benchmark.sh  # smoke
```


## 当前状态

已实现：

1. Qwen3-4B target 与 `z-lab/Qwen3-4B-DFlash-b16` 的固定 revision 下载和结构校验；
2. DFlash 多层 feature fusion、逐层 KV injection 与增量 cache 裁剪；
3. SpecBlock 四 bucket branching、root beam、greedy 主链、give-up、hitchhike、adaptive
   beam 和 tree-budget 剪枝；
4. ancestor-only 4D attention mask、最长 greedy 接受路径和非连续 target KV 压缩；
5. 官方口径的 15 维分布摘要与 `(H+15)->256->4` rank head；
6. CUDA 设备 fail-closed、SDPA 注意力、TF32 开关和 CUDA Graph 固定 shape 验证；
7. baseline/hybrid 逐样本正确性、延迟、接受长度和加速比记录。

本节旧流程只支持 `temperature=0`。随机采样需要 proposal probability 与 rejection sampling，不能
复用 greedy 最长路径验证。

正式默认配置是 [configs/qwen3_4b_cuda.json](configs/qwen3_4b_cuda.json)：

- `device="cuda:0"`，CUDA 不可用时直接失败；
- `dtype="bfloat16"`；
- `attn_implementation="sdpa"`；
- `allow_tf32=true`；
- `rank_mode="learned"`，缺失或不匹配 checkpoint 时直接失败；
- `block_size=15`、`max_blocks=1`、`tree_budget=60`；
- 固定 target/draft revision，保证权重可追溯。

`max_blocks=1` 表示默认生产路径只启用单块树，跨块 continuation 虽已实现，但必须用
`max_blocks>=2` 的独立配置重新训练 rank head 并做正确性/性能消融。

吞吐优先的候选配置是
[configs/qwen3_4b_cuda_optimized.json](configs/qwen3_4b_cuda_optimized.json)：它在 BF16/SDPA
基础上启用 StaticCache CUDA Graph，使用仓库中已有且与 K=15、`max_blocks=1` 匹配的 learned
rank checkpoint，并把 tree budget 收紧到 30。部署前仍必须在目标 GPU 上按后文协议验证
exact match、P95/P99、峰值显存与长稳行为；该配置不是未经实测即可直接宣称的速度结论。

## 官方 DFlash 与 DDTree 对照

仓库的 `third_party/` 目录包含固定 commit 的 DFlash 和 DDTree 官方源码。为避免官方依赖与
本工程环境冲突，两者使用独立虚拟环境；统一 runner 默认读取本工程已经固定的 200 条
benchmark JSONL，确保 prompt、顺序和生成长度口径一致：

```bash
bash scripts/setup_official_references.sh
MAX_SAMPLES=2 MAX_NEW_TOKENS=32 bash scripts/run_official_comparison.sh  # smoke
bash scripts/run_official_comparison.sh                                  # 200 条正式对照
```

完整说明、commit、数据集差异和输出字段见
[docs/OFFICIAL_BASELINES.md](docs/OFFICIAL_BASELINES.md)。

## GPU 加速设计

### SDPA 与 Flash 内核

`attn_implementation="sdpa"` 交给 PyTorch 根据 GPU、dtype、head dimension 和 mask 自动
选择 Flash、内存高效或数学后端。标准因果路径通常更容易进入 Flash 内核；树验证使用自定义
四维 ancestor mask，可能安全回退。工程不会为了强制某个内核而改变 mask 语义。

### CUDA Graph

原图验证配置已映射为 `use_cuda_graphs=true`，使用固定 shape 输入和预分配 StaticCache：

- [configs/qwen3_4b_cuda_tree15_float32_graph.json](configs/qwen3_4b_cuda_tree15_float32_graph.json)
- [configs/qwen3_4b_cuda_tree15_float32_graph_mainchain.json](configs/qwen3_4b_cuda_tree15_float32_graph_mainchain.json)
- [configs/qwen3_4b_cuda_tree15_float32_graph_nopad.json](configs/qwen3_4b_cuda_tree15_float32_graph_nopad.json)

所有配置的 `cuda_graph_max_cache_len` 默认为 4096。上下文超限、动态 shape 或 capture 失败
必须明确报错或回退，不能静默返回错误输出。

### TF32 与编译

`allow_tf32=true` 用于工业吞吐，会改变严格 IEEE FP32 的数值口径。严格 FP32 对照应创建
独立配置并设为 `false`。`torch_compile_mode` 默认是 `null`；只有在目标 GPU 上验证输出、
峰值显存、重编译次数和长稳行为后，才建议尝试 `reduce-overhead` 或 `max-autotune`。

详细说明见 [docs/NVIDIA_CUDA.md](docs/NVIDIA_CUDA.md)。

## 工程结构

```text
configs/                         CUDA 正式配置、消融配置与 smoke 配置
data/                            rank-head 数据格式示例
datasets/                        benchmark prompt 与生成训练集
docs/METHOD.md                   论文逐项对应与组合边界
docs/TRACEABILITY.md             论文/官方代码/本地实现追踪矩阵
docs/NVIDIA_CUDA.md              NVIDIA CUDA 环境与性能说明
docs/BENCHMARK_REPORT.md         GPU 迁移后的重新验证协议
scripts/check_cuda_env.py        CUDA、BF16 与关键算子自检
scripts/download_models.py       下载并校验 target/draft 权重
scripts/run_full_pipeline.sh     数据、训练、benchmark 一键流程
scripts/run_ddtree_benchmark.sh  DDTree 拓扑 benchmark（无需 rank head）
src/dflash_specblock/            核心实现
src/dflash_specblock/tree.py     SpecBlock 主链 + 兄弟分支树
src/dflash_specblock/ddtree_builder.py  DDTree best-first 树
tests/                           数学、拓扑与设备契约测试
```

## 1. 安装

先按 PyTorch 官方安装选择器安装与驱动匹配的 CUDA wheel，再安装工程依赖：

```bash
conda create -n dflash-specblock python=3.11 -y
conda activate dflash-specblock

# 先安装 CUDA 版 torch，再执行：
pip install -r requirements-gpu.txt
pip install -e .

export CUDA_VISIBLE_DEVICES=0
nvidia-smi
python scripts/check_cuda_env.py
```

自检必须显示 CUDA、cuDNN、GPU、compute capability、BF16 和 Flash SDPA 状态，并以
`NVIDIA CUDA 环境检查通过` 结束。

## 2. 下载模型

```bash
python scripts/download_models.py --config configs/qwen3_4b_cuda.json
```

模型默认写入 `models/`。如果 Hugging Face 仓库要求认证，使用 `HF_TOKEN` 或脚本
`--token`，不要把 token 写入配置或提交到仓库。

## 3. 下载 benchmark 数据

```bash
python -m dflash_specblock.dataset_pipeline \
  --output-dir datasets/processed/specblock_official
```

默认 suite 包含 MT-Bench、HumanEval、MATH-500、Alpaca、NQ-Open 和 translation。小规模
联调可增加 `--max-samples-per-dataset 32`。

## 4. 生成 rank-head 训练集

```bash
python -m dflash_specblock.generate_rank_data \
  --config configs/qwen3_4b_cuda.json \
  --prompts datasets/processed/specblock_official/prompts_all.jsonl \
  --output datasets/generated/rank_train.jsonl \
  --max-new-tokens 128
```

正式数据应由固定 revision 的目标模型 greedy 输出生成，并记录 prompt 来源与生成配置。

## 5. 在 GPU 上训练 rank head

```bash
python -m dflash_specblock.train_rank_head \
  --config configs/qwen3_4b_cuda.json \
  --train-data datasets/generated/rank_train.jsonl \
  --output checkpoints/rank_head_tree15.pt \
  --epochs 3 \
  --learning-rate 2e-4 \
  --device cuda:0
```

checkpoint 会绑定模型 ID/revision、block size、`max_blocks`、hidden size、head 结构和有效
更新数。跨配置 checkpoint 不会被静默复用。

## 6. 演示

```bash
bash scripts/run_demo.sh
```

或：

```bash
python -m dflash_specblock.cli \
  --config configs/qwen3_4b_cuda.json \
  --prompt "请解释推测解码为什么能够保持 greedy 输出不变。" \
  --max-new-tokens 128
```

`configs/qwen3_4b_smoke.json` 的 heuristic ranker 只用于接口联调，不得作为论文或生产结论。

## 7. Benchmark 与 profile

```bash
bash scripts/run_benchmark.sh
bash scripts/run_vanilla_dflash_benchmark.sh
bash scripts/run_tree15_pipeline.sh
python scripts/profile_stages.py
```

新基准应写到新的 GPU 输出目录或带设备标识的文件，至少记录：

- baseline/hybrid token 数与端到端延迟；
- `wall_clock_speedup`、接受长度和 verify 次数；
- prefill、draft、verify、cache compact 分阶段耗时；
- exact match 与首个不一致位置；
- P50/P95/P99、峰值显存、软件版本和 GPU 信息。

正式测量前先 warmup，计时边界使用 CUDA event 或 `torch.cuda.synchronize()`，至少执行三次
独立重复。任何 token 不一致都应先按正确性故障处理。

完整协议见 [docs/BENCHMARK_REPORT.md](docs/BENCHMARK_REPORT.md)。

## 8. 一键全流程

```bash
export CUDA_VISIBLE_DEVICES=0
bash scripts/run_full_pipeline.sh
```

可用环境变量覆盖关键路径与规模：

```bash
CONFIG=configs/qwen3_4b_cuda.json \
MAX_SAMPLES_PER_DATASET=64 \
MAX_PROMPTS=128 \
MAX_NEW_TOKENS=128 \
RANK_EPOCHS=1 \
bash scripts/run_full_pipeline.sh
```

## 9. 历史产物

现有 `logs/`、`outputs/`、`backups/` 与部分 checkpoint 来自迁移前运行。为保留可追溯性，
这些历史内容不会机械改写为 GPU 结果，也不能用于 NVIDIA GPU 性能结论。迁移后必须在目标
GPU 上重新训练/验证 checkpoint，并重新生成 benchmark 与报告。

## 旧 SpecBlock 流程的已知限制

- 当前是算法实验实现，不是 SGLang/vLLM 生产插件；
- 仅支持 batch=1、greedy decoding；多提示词 benchmark 仍逐条运行；
- 自定义树 mask 不保证命中 Flash SDPA kernel，必须根据 profiler 实测；
- CUDA Graph 需要固定 shape 和预分配 cache；
- `max_blocks=1` 是正式默认，跨块路径收益尚需 GPU 消融；
- 不包含模型权重；heuristic 结果不可用于优于 DFlash/SpecBlock 的结论。

方法对应、公式与创新边界见 [docs/METHOD.md](docs/METHOD.md) 与
[docs/TRACEABILITY.md](docs/TRACEABILITY.md)。
