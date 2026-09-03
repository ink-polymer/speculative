# DFlash-SpecBlock：面向昇腾 910B A2 的扩散块动态树推测解码实验

## H200 三路径 GBV 论文实验：12 配置精简方案

使用 Qwen3-4B、Qwen3-8B 与各自配套的 DFlash-b16 草稿，核心为 K=3、L=15、T=1。两个模型均比较 Target AR、DFlash、单路径 BV、三路径 GBV、DDTree；4B 额外比较 K=2、K=4。保留原双向注意力、正常 Target 特征和前缀共享。

- [实验协议与运行说明](docs/GBV_PAPER_EXPERIMENTS.md)
- [两模型套件](configs/gbv_paper_suite.json)、[4B 配置](configs/gbv_paper_ddtree_counts.json)、[8B 配置](configs/gbv_paper_qwen3_8b.json)。
- 默认 `bash scripts/run_gbv_paper.sh gbv-first`：先跑两模型三路径 GBV，共 5,196 次回答。随后运行 `main`、`complete`，自动复用已有记录；最终只新增 4B 的 K=2、K=4。
- 完整方案：4B 的 7 配置、8B 的 5 配置，三个生成种子，共 252 个数据集任务、28,296 条题目/完整对话记录、31,176 次回答。
- 数据量继续对齐 DDTree：GSM8K 128、MATH-500 128、AIME25 30、HumanEval 164、MBPP 128、LiveCodeBench 128、MT-Bench 80 组双轮；每个配置/种子为 786 题或对话、866 次回答。固定选题种子 0，所有方法和两个模型共用原始题号清单。
- 当前不跑长度、单路径 token/BV、前缀共享开关或全评测集 T=0 消融；小规模上机贪心一致性、缓存/树验证、数据审计和评分检查继续保留。
- DFlash 基线使用贪心草稿与 Target 采样匹配规则。DFlash、DDTree 为同一引擎内复现，数据量和算法规则对齐不表示完整优化运行时或官方样本身份一致；DDTree 固定预算为 45 个非根节点，不代表其最优预算性能。
- CPU 回归检查通过；8B 验证包含固定配置的模型结构构造。H200 真实权重/BF16 校验、七套真实数据准备审计和正式测速仍待运行。
- MT-Bench 提供外部裁判导出/导入接口，尚未执行真实裁判评分。旧 `run_gbv_paper_full.sh` 默认转到当前两模型 `complete`；4B 全源划分仅作为独立备选配置保留。

## 既有 Ascend / SpecBlock 实验

本工程把两篇论文的核心思路组合为一个可审计的实验实现：

- **DFlash** 负责草稿生成：目标模型多层隐藏状态持续注入每个 draft layer 的 KV，未来
  token 以 block diffusion 方式一次并行产生；
- **SpecBlock** 负责草稿组织和验证：每块形成 greedy 主链与 rank-guided 兄弟分支，多个
  pending 起点在下一块批量扩展，最后用 ancestor-only 树注意力做一次目标模型验证；
- **Ascend A2** 是一等运行环境：代码使用 `torch.npu`、`npu:<id>` 与
  `ASCEND_RT_VISIBLE_DEVICES`，不依赖 CUDA/Triton 专用路径。

论文与官方实现：

- DFlash paper：<https://arxiv.org/abs/2602.06036>
- DFlash code：<https://github.com/z-lab/dflash>
- SpecBlock paper：<https://arxiv.org/abs/2605.07243>
- SpecBlock code：<https://github.com/shiweijiezero/SpecBlock>

## 当前实现状态与严格模式

已实现：

1. Qwen3-4B 目标模型与 `z-lab/Qwen3-4B-DFlash-b16` 权重自动下载；
2. DFlash 第一块的目标多层 feature fusion、逐层 KV injection 与官方 DynamicCache
   增量/裁剪语义；
3. 后续块从 pending 起点缓存的 DFlash `h(L)` 批量继续扩散，严格绕过第一块 `fc`；
4. SpecBlock 四 bucket 动态 branching、block-1 slot-0 beam、greedy 主链、兄弟候选、
   give-up、hitchhike 与 adaptive beam；
5. 按累计 log probability 且保留祖先的 tree-budget 剪枝（P0 修复：greedy 主链
   无条件保留，不参与竞争性剪枝——主链是接受长度的唯一来源且不额外占 draft 成本）；
6. 目标模型 ancestor-only 4D attention mask、单次并行树验证、逐叶枚举的最长 greedy
   接受路径；
7. DynamicCache 非连续接受路径压缩；
8. 官方实现口径的 15 维分布摘要（top-20 lse 归一化）、`(H+15)->256->4` rank head、
   块内 valid-prefix 与随机 cut 跨块 curriculum；
9. baseline token 级无损一致性断言、acceptance/时延记录和 CPU 契约测试。

当前只支持 `temperature=0`。这是有意的：随机采样的无损验证需要按 draft/target 概率执行
speculative rejection sampling，不能用 greedy 路径逻辑冒充。

正式配置 [configs/qwen3_4b_a2.json](configs/qwen3_4b_a2.json) 已设为：

- `device="npu:0"`：没有 NPU 时直接失败，不回退 CPU；
- `rank_mode="learned"`：没有 rank checkpoint 时在加载大模型前直接失败；
- `branch_factors=[2,4,10,0]`：官方 `RANK_SLOT_TOPK` 常量的 reference mapping；
- 固定 Qwen/DFlash Hugging Face commit SHA，保证权重和 remote code 可复现。

> **P1 说明（max_blocks=1）**：所有正式 NPU 配置当前使用 `max_blocks=1`，即只构建
> 单块内 slot 兄弟树，不触发跨块 `propose_continuation` 扩展。这意味着 README 第 3 条
> 描述的跨块扩展在当前生产路径上是**未启用**的。代码已实现且通过测试，但 M>=2 的价值
> 尚未经 A2 消融实验确认。若据此写论文，需明确标注实验结果来自 M=1 配置，或先恢复
> M>=2 做判定实验。

## heuristic 只用于 smoke test

DFlash 官方权重不包含 SpecBlock rank head，因此配置提供两种模式：

- [configs/qwen3_4b_smoke.json](configs/qwen3_4b_smoke.json)：heuristic，仅检查接口；
- [configs/qwen3_4b_a2.json](configs/qwen3_4b_a2.json)：learned，正式实验唯一默认配置。

**heuristic 结果不是论文复现结果，不得写入实验主表。**

## 工程结构

```text
configs/                         A2 严格配置与单独的 smoke 配置
data/                            rank-head 数据格式示例
datasets/                        下载后的 benchmark prompt 与生成训练集（默认忽略提交）
docs/METHOD.md                   论文逐项对应与组合边界
docs/TRACEABILITY.md             论文/官方代码/本地实现追踪矩阵
docs/ASCEND_910B_A2.md           CANN / torch_npu 环境说明
examples/prompts.jsonl           小型评测提示集
scripts/download_models.py       下载目标与 DFlash 权重
scripts/run_full_pipeline.sh     下载数据、生成训练集、训练、benchmark 一键流程
scripts/check_ascend_env.py      NPU 环境和 BF16 matmul 自检
scripts/run_demo.sh              单提示词演示
scripts/run_benchmark.sh         baseline 对照实验
src/dflash_specblock/
  dataset_pipeline.py            SpecBlock benchmark 下载与 prompt 统一
  dflash_adapter.py              DFlash block 接口与跨块 hidden KV 注入
  generate_rank_data.py          用 target 模型离线生成 rank-head 训练文本
  rank_head.py                   15 维摘要、rank bucket、valid-prefix
  tree.py                        SpecBlock block-iterative 动态树
  verification.py                树注意力、最长路径、KV cache 压缩
  engine.py                      端到端 draft-tree-verify 循环
tests/                           不依赖大模型的 CPU 单元测试
```

## 1. 配置 Atlas A2 环境

先阅读 [docs/ASCEND_910B_A2.md](docs/ASCEND_910B_A2.md)。核心原则是：服务器的
`torch`、`torch_npu` 和 CANN 必须来自同一配套矩阵，不要让普通 PyPI 的 torch 覆盖平台
wheel。

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0

conda create -n dflash-specblock python=3.10 -y
conda activate dflash-specblock

# 先安装与服务器 CANN 配套的 torch / torch_npu wheel，再运行：
pip install -r requirements-ascend.txt
pip install -e .
python scripts/check_ascend_env.py
```

预期最后显示 `Ascend 环境检查通过`。如果 `torch.npu.is_available()` 为 False，应先修复
驱动/CANN/torch_npu，不要把配置改成其他设备规避问题。

## 2. 下载参数

```bash
python scripts/download_models.py --config configs/qwen3_4b_a2.json
```

下载脚本锁定并下载：

- `Qwen/Qwen3-4B` -> `models/Qwen3-4B`
- `z-lab/Qwen3-4B-DFlash-b16` -> `models/Qwen3-4B-DFlash-b16`

脚本随后校验 model type、hidden size、词表、目标层数、DFlash architecture、mask token 和
block size；任何一项不匹配都会终止。

如果 Hugging Face 仓库要求身份认证，使用环境变量 `HF_TOKEN` 或脚本 `--token`；不要把
token 写进配置或提交到版本库。`models/` 已加入 `.gitignore`。

## 3. 下载 SpecBlock benchmark 数据集

本仓库新增了对 SpecBlock 论文 benchmark 口径的公开 prompt 下载与统一格式转换。当前默认
suite 包含：

- `MT-Bench`
- `HumanEval`
- `MATH-500`
- `Alpaca`
- `Natural Questions (NQ-Open)`
- `translation`，默认使用 `wmt14 de-en`，并允许按参数覆盖

下载并生成项目可直接使用的 JSONL：

```bash
python -m dflash_specblock.dataset_pipeline \
  --output-dir datasets/processed/specblock_official
```

生成后会得到：

- `datasets/processed/specblock_official/manifest.json`
- `datasets/processed/specblock_official/prompts_all.jsonl`
- 各 benchmark 的独立 JSONL 文件

如需只做小规模联调，可限制每个 benchmark 的样本数：

```bash
python -m dflash_specblock.dataset_pipeline \
  --output-dir datasets/processed/specblock_official \
  --max-samples-per-dataset 32
```

## 4. 生成 rank-head 训练集

正式配置不会自动降级到 heuristic。推荐先用目标模型自身 greedy 输出离线生成训练文本，以保
持 target alignment：

```bash
python -m dflash_specblock.generate_rank_data \
  --config configs/qwen3_4b_a2.json \
  --prompts datasets/processed/specblock_official/prompts_all.jsonl \
  --output datasets/generated/rank_train.jsonl \
  --max-new-tokens 128
```

输出 JSONL 每行至少包含：

```json
{"text": "完整的目标模型生成文本，包含上下文与回复"}
```

同时会保留源 prompt、源 benchmark 和生成耗时，便于追踪训练样本来源。

## 5. 训练正式 rank head

准备好第 4 节生成的训练文件后，直接训练：

```bash
python -m dflash_specblock.train_rank_head \
  --config configs/qwen3_4b_a2.json \
  --train-data datasets/generated/rank_train.jsonl \
  --output checkpoints/rank_head.pt \
  --epochs 3 \
  --learning-rate 2e-4
```

## 6. 运行演示

```bash
bash scripts/run_demo.sh
```

或直接执行：

```bash
python -m dflash_specblock.cli \
  --config configs/qwen3_4b_a2.json \
  --prompt "请解释推测解码为什么能够保持 greedy 输出不变。" \
  --max-new-tokens 128
```

输出包含文本、prefill/draft/verify 时延、每轮树节点数、接受 draft token 数和平均每次验证
提交长度。

如仅需检查下载和接口，可显式使用 smoke 配置；不能把其结果作为正式实验：

```bash
python -m dflash_specblock.cli \
  --config configs/qwen3_4b_smoke.json \
  --prompt "smoke test" \
  --max-new-tokens 8
```

## 7. rank-head 数据要求

JSONL 每行格式：

```json
{"text": "完整的目标模型生成文本，包含上下文与回复"}
```

建议用 Qwen3-4B 自身 greedy 输出构建训练集，以保持 target alignment。小样例只用于检查
数据格式，不能得到有意义的 rank head。

训练步骤对每条文本随机取 anchor：目标模型提供多层上下文，DFlash 一次预测后续 K 位；
真实 token 在每位 draft distribution 的 rank 被映射到 `1 / 2-4 / 5-10 / >10`，且第一处
greedy 错误之后的所有位置都被 valid-prefix mask 屏蔽。每条样本还均匀采样
`cut in {1,...,K}`，用第一块的 cached `h(L)` 训练第二块 rank 分布。

保存的 checkpoint 会记录 target/draft 模型 ID、两端 commit SHA、K、hidden size、head
结构与有效更新数。正式加载时逐项核对，旧模型或不同 K 的 rank head 不会被静默复用。

## 8. baseline 对照实验

```bash
bash scripts/run_benchmark.sh
```

结果写到 `outputs/qwen3_4b_a2.jsonl`，字段包括：

- baseline 与 hybrid 生成 token 数和墙钟时间；
- `wall_clock_speedup`；
- `average_committed_per_verify`；
- verifier 调用次数；
- `greedy_exact_match` 与 `first_mismatch_index`。

任何 token 不一致都会立即抛错并停止，禁止继续汇总速度数据。

正式报告前应：

1. 使用 learned rank head；
2. 预热 20-50 个请求；
3. 固定同一提示集、最大生成长度和 greedy 参数；
4. 同时报 acceptance length 与 drafting cost，不能只报 speedup；
5. 重复至少三次并报告均值/标准差；
6. 确认 `npu-smi info` 中没有其他任务竞争设备。

## 9. 一键全流程

如果你想把“下载模型 -> 下载 benchmark -> 生成训练集 -> 训练 rank head -> 跑 benchmark”
-> 官方任务评测”一次串完，可以直接使用：

```bash
bash scripts/run_full_pipeline.sh
```

该脚本支持通过环境变量覆盖关键路径与超参，例如：

```bash
CONFIG=configs/qwen3_4b_a2.json \
MAX_SAMPLES_PER_DATASET=64 \
MAX_PROMPTS=128 \
MAX_NEW_TOKENS=128 \
RANK_EPOCHS=1 \
bash scripts/run_full_pipeline.sh
```

默认会额外运行：

```bash
python -m dflash_specblock.official_eval \
  --config configs/qwen3_4b_a2.json \
  --prompts datasets/processed/specblock_official/prompts_all.jsonl \
  --output-dir outputs/official_eval \
  --mode hybrid
```

如果你配置了 judge 模型，还会对 `MT-Bench` 和 `Alpaca` 进行 LLM judge：

```bash
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=https://api.openai.com/v1
export OPENAI_MODEL=gpt-4o-mini
```

也可以用兼容 OpenAI Chat Completions 的自部署 judge 服务；环境变量 `JUDGE_API_KEY`、
`JUDGE_BASE_URL`、`JUDGE_MODEL` 同样可用。

## 10. 官方任务评测

统一评测入口：

```bash
python -m dflash_specblock.official_eval \
  --config configs/qwen3_4b_a2.json \
  --prompts datasets/processed/specblock_official/prompts_all.jsonl \
  --output-dir outputs/official_eval \
  --mode hybrid
```

输出文件：

- `outputs/official_eval/results.jsonl`
- `outputs/official_eval/summary.json`

当前已实现的 scorer：

- `HumanEval`: `pass@1`
- `MATH-500`: `accuracy`
- `NQ-Open`: `exact_match` 和 `f1`
- `translation`: `BLEU`
- `MT-Bench`: judge `score`
- `Alpaca`: judge `win_rate`

如果没有配置 judge，`MT-Bench` 和 `Alpaca` 会保留生成结果，但汇总分数为空。

## 11. 配置说明

默认实验参数采用 SpecBlock 的 reference 设置：`K=4`、`M=2`、tree budget 60、
bucket mapping `[2,4,10,0]`。

其中最后的 0 表示 rank>10 时，该位置不增加兄弟候选，并停止其后更深 slot 的兄弟扩展；
greedy 主链始终保留。block 1 slot 0 的 root-diversity beam 是论文算法中的显式特例。

DFlash checkpoint 的原始 block 为 16（1 个干净 anchor + 15 个未来 token）；本组合实验默认
只取 K=4，以匹配 SpecBlock 的块宽并控制树验证预算。可以提高 K，但必须同时重新训练 rank
head，并重新评估 tree budget 与 NPU 内存。

## 12. 测试

CPU 上可以验证不依赖权重的数学与拓扑：

```bash
pip install -e ".[test]"
pytest
```

测试覆盖 DFlash 增量 cache、mask-token 兼容、跨块 projection bypass、15 维摘要、严格 rank
head、valid-prefix、官方分支宽度、give-up/hitchhike、祖先剪枝、重复 token 最长路径、真实
Transformers DynamicCache 裁剪和 4D tree mask。真机测试按用户要求暂时跳过。

## 已知限制

- 这是面向算法实验的纯 PyTorch 实现，不是 SGLang/vLLM 生产插件；为兼容 NPU，没有移植
  官方 CUDA/Triton 性能 kernel，但树的语义和验证结果不因此改变。
- 仅 batch=1、greedy decoding；benchmark 支持多提示词但逐条运行。
- 唯一的方法创新是：用 DFlash diffusion block 替换 SpecBlock 原 AR/shift block，并把 cached
  DFlash `h(L)` 作为后续块逐层 KV 条件。该接口已在代码中实现并通过单元测试，但当前所有
  正式配置使用 `max_blocks=1`，跨块扩展路径未在生产中启用。其收益仍需 M>=2 的 A2 消融
  实验确认。
- 不包含模型权重和训练数据。
- `heuristic` 的分支质量不可用于得出“优于 DFlash/SpecBlock”的结论。

更完整的论文对应、公式与工程偏差见 [docs/METHOD.md](docs/METHOD.md)。
