# 原版 Adaptive DDTree：DDTree 官方 T=0 实验与消融

当前统一为 **非 RL 的原版延迟感知 Adaptive DDTree**。用户已确认连抽样数量也与官方一致，因此默认不再使用七套全量测试集。文件名中的 full 表示完整实验矩阵，不表示全量数据。

[构树介绍与流程图](ADAPTIVE_DDTREE_METHOD.md) · [论文数学证明](ADAPTIVE_DDTREE_T0_PROOF.md) · [官方对齐说明](DDTREE_PROTOCOL_ALIGNMENT.md)

## 1. 方法与官方依据

Adaptive 构树器逐字节恢复自本项目提交 `9dd67698ad828b8c3fca8659e3a388f0b2dfbdf7`：一次 DFlash block 前向，DDTree best-first 枚举最多 128 节点，再从 30/45/60/80/100/128 的嵌套前缀中选预算。初始 60；每预算预热一轮；EMA α=0.2；间隔 64 次有效决策探索。不训练 policy、reward、rank head 或 draft，不需要训练集/checkpoint。

评测对齐 DDTree 官方提交 `c96427a185677bf4133ed865dd1626a5041aef9b`。只复现其 **T=0 部分**，不将此方法扩展为 T=1 GBV。

- 官方源代码保存在 third_party/ddtree_pinned，运行前核对文件 SHA-256；仅对无结尾换行的文件补 LF。
- 数据准备直接调用官方 model/utils.py 的处理函数，不重写 prompt。
- AR 使用官方 dflash_generate(block_size=1)，DFlash 与固定 DDTree 使用官方未改动生成函数。
- Adaptive 接入同一官方 DDTree 循环，仅替换构树、加入预算反馈与诊断；Target 前向、树编译、接受规则、KV 整理、停止规则和解码计时边界沿用官方。
- 基线不是旧的 B=60 自定义 engine，也不是本项目另一个验证算法；不启用 CUDA Graph。

这是“官方协议加上待比较方法和审计”，不是声称整个程序逐字节等于上游。旧截图属于相同构树算法，但不属于新的官方评测实现。

## 2. 官方数据与样本数

| 数据集 | 官方来源、划分 | 官方测量题/对话数 |
|---|---|---:|
| GSM8K | openai/gsm8k，main/test | 128 |
| MATH-500 | HuggingFaceH4/MATH-500，test | 128 |
| AIME24 | HuggingFaceH4/aime_2024，train | 30 |
| AIME25 | MathArena/aime_2025，train | 30 |
| HumanEval | openai/openai_humaneval，test | 164 |
| MBPP | google-research-datasets/mbpp，sanitized/test | 128 |
| LiveCodeBench | code_generation_lite 的 test.jsonl 至 test6.jsonl | 128 |
| SWE-bench Lite | princeton-nlp/SWE-bench_Lite，test | 128 |
| MT-Bench | HuggingFaceH4/mt_bench_prompts，train | 80，保留两轮 |
| Alpaca | tatsu-lab/alpaca，train | 128 |

总计 **1,072 题/对话**、每方法 **1,152 次回答**。若完整源的长度大于上限，严格使用 `dataset.shuffle(seed=0).select(range(limit))`；否则维持源顺序。不用 Python random 替代 Hugging Face shuffle，不以 GBV/旧 2K 的数据替代，不把 MBPP-full 500 题混入主表。

数据准备时固定各源的不可变 revision、源索引与处理后 turns hash，之后所有模型、方法和后端使用同一份选中样本。官方脚本未记录历史 HF revisions，因此能够保证本次比较中数据相同、处理和抽样规则相同，不能无依据声称与作者当年未公开的具体快照逐字相同。

MT-Bench 按官方方式共享当前轮输入；SDPA 组的下一轮对话使用最后一个原始 DDTree 方法（预算 1024）的回答，FA2 组用 DFlash 回答。新加 Adaptive/消融不能改变这个上下文来源。跨后端汇总时再次检查输入与 token 一致性。

上游叫 train 的 AIME、MT-Bench、Alpaca 划分在这里仅作评测；此方法没有离线训练环节。这里对应官方速度/接受长度 benchmark，不包含代码测试执行、SWE 修复率或 MT-Bench judge 分数，不能把速度表当成任务质量评分。

## 3. 模型、后端和顺序

原样保留官方三对模型：

1. Qwen/Qwen3-4B + z-lab/Qwen3-4B-DFlash-b16。
2. Qwen/Qwen3-8B + z-lab/Qwen3-8B-DFlash-b16。
3. Qwen/Qwen3-Coder-30B-A3B-Instruct + z-lab/Qwen3-Coder-30B-A3B-DFlash。

4B 使用旧实验固定的 Target/Draft SHA：1cfa9a7208912126459214e8b04321603b3df60c / b74e3a329c4d963783143b1e970d95b002be72bd。8B 使用 GBV 已上传配置的同一对 SHA：b968826d9c46dd6066d109eabc6255188de91218 / 9b41424b7109f9c5413454f481b09a82b85333f4。30B 首次 prepare 解析 SHA 并锁定，不使用可变本地微调目录；不同 8B revision 的旧清单拒绝复用。

新增 [Qwen3-8B 独立入口](ADAPTIVE_QWEN3_8B.md)：bash scripts/run_paper_t0_qwen3_8b.sh plan；包含相同主实验与四项消融，8B 独立输出，不同时加载其他模型。原有完整三模型入口保持不变。

温度 0、BF16、每回答最多 2,048 新 token、thinking 关闭、seed=0、每 GPU batch=1。沿用官方脚本的 2,048，而不是 benchmark.py 单独调用时的 16,384 默认值。Draft 始终 FA2；Target 分别跑 SDPA 和 FA2 两组。树方法仅在 SDPA 组运行。遵循官方 PyTorch 默认 TF32 行为，不额外强设 TF32；采用官方 C++ cache compaction，编译失败则停下排查，不静默给不同配置冠以相同结果名。

默认 8 个 GPU 进程，样本按 range(rank, n, world_size) 划分。官方每个“数据集×模型×后端”独立启动进程组；本代码相同。控制器在对应进程内跨题保持状态，不同消融不共用统计量。Warmup 文本、长度上限 16 及基线方法顺序沿用官方；Adaptive 的预热观测在正式流开始前清空。

显式设置 --nproc-per-node 1 可用于单卡，但会记录为与官方默认并行度不同；不能宣称硬件设置完全相同。显式只选模型或数据集同样标记子集。

## 4. 主方法与消融

| 组 | 方法 |
|---|---|
| 两个 Target 后端 | 官方 Target-only、官方 DFlash |
| SDPA 固定预算 | 官方 DDTree：16/32/64/128/256/512/1024 |
| SDPA 主方法 | 原版 adaptive：六候选预算在线选择 |
| no_acceptance_calibration | 接受率校准系数固定 1 |
| no_latency | 决策时成本分母固定 1，保留预热/探索 |
| no_exploration | 关闭周期探索，保留首次预算预热 |
| frozen_after_warmup | 六预算首次观测后冻结耗时及接受率校准；计数与探索仍继续 |

消融均不重训模型。Adaptive 的构树转换与控制器开销计入官方 decode timer。控制器仍使用“草稿＋构树”及“树编译＋验证＋KV/提交”的实测毫秒更新原版标量规则；换用官方执行器后，这些观测来自它的阶段计时，而不是伪称仍为旧 engine 的 CUDA-event 数字。

默认 10 数据集 × 3 模型 × 2 后端，60 个进程组；包括 Adaptive 和消融共 **55,296 次正式生成调用**，另加预热。没有 36 个 RL checkpoint、反事实数据引擎或训练 epochs。

## 5. 计时、汇总与审计

官方 TPOT 是单次回答的 decode 时间 / 输出 token 数；排除 target prefill，推测方法在第一次 draft 后重新设置 decode_start，因此也排除首次 draft。主加速比为 **AR 的回答级 TPOT 均值 / 方法的回答级 TPOT 均值**，不是总耗时比，也不是每题 speedup 的平均。

直接复用官方 make_latex_table.py 中的均值、接受长度均值和后端择优函数：

- Target-only 与 DFlash 各自在 SDPA/FA2 中选 TPOT 均值较小者。
- DDTree 主表在七个固定预算中选最优；同时保留每预算明细。
- Adaptive/各消融与同一个官方 Target baseline 和最佳 DDTree 对比。
- 不再把三个顺序种子、重复测量或 bootstrap 当作官方原协议。

额外审计在生成计时之外：保存数据/代码/权重来源、每轮输入 hash 与原始输出；发现任一方法与官方 Target-only 输出不一致即保存诊断并停止，不筛掉失败题。跨后端输入或输出不一致也拒绝出无损比较表。此检查针对官方 Target-only 实现；BF16 实数等价、特殊 mask 清理和任务准确率仍不是由此自动证明的。

只对成功完成、hash 匹配的整个进程组续跑；失败组重新从头运行，不能跳过前半题而丢失自适应历史。拒绝覆盖已有但未经确认的 .pt。已完成组的 .pt 可用官方表格脚本读取；Adaptive 扩展表由 official_reporting.py 生成。不得加载来源不明的 pickle/.pt。

复查补充：每个 worker 在加载模型前重新检查父任务契约的哈希、当前代码/官方来源/数据清单、模型与数据集范围、进程数和输出路径，防止父任务启动后改代码却仍沿用旧实验身份。doctor 记录每张参测 GPU 的 UUID；worker 开始/结束及续跑/汇总均核对卡身份与 FA2 版本。不支持只看卡名或进程数就混合不同设备结果。缺少 UUID 的旧环境记录不能直接续用，应使用新的运行目录；不删除旧结果。

## 6. 运行

服务器使用独立 Python 3.10/3.11 环境，先安装适配驱动的 CUDA PyTorch 和 FA2 wheel，再安装 requirements-paper.txt。FA2 与 C++ 编译器/ninja 是必需条件；本地 CPU 单元测试不代表真实 GPU 通过。

```bash
python -m pip install -r requirements-paper.txt
bash scripts/run_paper_t0_full.sh plan
bash scripts/run_paper_t0_full.sh doctor
bash scripts/run_paper_t0_full.sh prepare
bash scripts/run_paper_t0_full.sh all
# 单卡、只检查 4B 的显式 smoke，不是官方完整矩阵：
CUDA_VISIBLE_DEVICES=0 bash scripts/run_paper_t0_full.sh all \
  --nproc-per-node 1 --model-index 0 --smoke-count 2 \
  --run-dir outputs/adaptive_ddtree_smoke
# 正式执行也可分阶段：
bash scripts/run_paper_t0_full.sh evaluate
bash scripts/run_paper_t0_full.sh summarize
```

数据目录 datasets/ddtree_official_t0；结果 outputs/adaptive_ddtree_official_t0。plan 不加载模型、不训练。旧 v2 RL、v3 全量受控协议和旧结果目录不能混用。非默认 controlled_* 代码仅作历史回归参考，不是当前论文入口。

## 7. 证明与历史结果

[数学证明](ADAPTIVE_DDTREE_T0_PROOF.md) 对应原版嵌套预算方法，证明代理质量性质以及在明确计算假设下的 T=0 贪心等价；不证明吞吐必然提高，也不是随机采样无偏性或 RL 收敛证明。

旧截图属于原版非 RL 算法，但原始 2,000 条只有 **727 条**与当时 AR 完全一致，**1,273 条不一致**。它不作为这套官方协议的新结果或已验证无损结果。本次只修改、核对代码和文稿，未启动服务器 GPU 实验；尚无新加速比。
