# 扩散草稿多路径 GBV：正式实验代码

这套代码以 K=3、L=15 的 GBV 为主方法，评测 Qwen3-4B 和 Qwen3-8B 两组 Target 与各自配套的 DFlash-b16 扩散草稿模型。4B 做主实验及详细消融，8B 做主实验及贪心一致性对照。主实验、消融、评分和汇总共用 `src/gbv_experiments/`，入口是 `scripts/gbv_paper.py`。尚未产生 H200 正式实验结果；本机测试不能作为论文速度结果。

## 1. 固定协议

默认套件为 `configs/gbv_paper_suite.json`，调度以下两个配置；七套数据的评测题数保持 DDTree 数量协议。`configs/gbv_paper_full.json` 仅保留为 **4B 全量源划分**的独立备选。

| 模型组 | Target | Draft | 配置文件 |
|---|---|---|---|
| 4B | Qwen/Qwen3-4B | z-lab/Qwen3-4B-DFlash-b16 | gbv_paper_ddtree_counts.json |
| 8B | Qwen/Qwen3-8B | z-lab/Qwen3-8B-DFlash-b16 | gbv_paper_qwen3_8b.json |

固定 revision：

- 4B Target：`1cfa9a7208912126459214e8b04321603b3df60c`；Draft：`b74e3a329c4d963783143b1e970d95b002be72bd`。
- 8B Target：`b968826d9c46dd6066d109eabc6255188de91218`；Draft：`9b41424b7109f9c5413454f481b09a82b85333f4`。

8B 配置来自固定版本的 [Target config](https://huggingface.co/Qwen/Qwen3-8B/blob/b968826d9c46dd6066d109eabc6255188de91218/config.json) 和 [Draft config](https://huggingface.co/z-lab/Qwen3-8B-DFlash-b16/blob/9b41424b7109f9c5413454f481b09a82b85333f4/config.json)。两者隐藏维度均为 4096、词表大小 151936；Draft 使用 Target 的层 `[1,9,17,25,33]`。源配置与哈希保存在 `experiments/gbv_paper/qwen3_8b_metadata.json`；meta tensor 测试只验证结构可构造与接口维度，不替代真实权重加载和 H200 验证。

两组统一设置：

- 模型权重 BF16；概率计算 FP64；Target/Draft 使用 SDPA；batch size 1；关闭 thinking、TF32；不使用 top-k、top-p、重复惩罚或量化。
- 生成种子 17、29、43。每题由原始 ID 派生种子；当前运行阶段内的方法顺序按题随机轮换。两模型依次加载，分别保存结果，不同时常驻显存。
- 每个回答轮次最多生成 2,048 个 token，包含首个 Target token；遇到 EOS 结束当前回答。截断回答保留并评分，另报截断比例。MT-Bench 两轮分别应用此上限。
- 块宽为 16：1 个已知 anchor + 15 个未来位置。`length=L` 不含 anchor；长度消融仍计算完整 16 位置，只取前 L 个未来位置。
- 主实验 Target 温度 T=1；GBV、BV、token 拒绝采样与 DDTree 的草稿概率温度也为 1。DFlash 匹配基线始终以 argmax 构造草稿。T=0 额外对照仅包含 AR 与 K=3 GBV；该 GBV 仍从温度 1 的草稿分布采样。
- 所有正式配置保留 Draft 原双向注意力、正常 Target 特征和 Draft KV 复用。注意力因果化、特征置零不列入实验。
- 无提示截断，超出上下文容量报错。模型加载、tokenizer 与评分不计入生成时间。

## 2. 数据及训练/测试边界

默认使用固定评测子集。题数取自 [DDTree 官方运行脚本](https://github.com/liranringel/ddtree/blob/master/run_benchmark.sh)，只对齐本套实验已有的七个数据集。AIME25 的源仓库把比赛题存放在名为 `train` 的划分中；本实验将这 30 题全部用于评测，不用于训练。

| 数据集 | 来源 | 配置 / 划分 | 数据源完整条数 | 每个方法、每个生成种子的评测条数 |
|---|---|---|---:|---:|
| GSM8K | openai/gsm8k | main / test | 1,319 | 128 |
| MATH-500 | HuggingFaceH4/MATH-500 | 默认 / test | 500 | 128 |
| AIME25 | MathArena/aime_2025 | 默认 / train（仅评测） | 30 | 30 |
| HumanEval | openai/openai_humaneval | 默认 / test | 164 | 164 |
| MBPP | google-research-datasets/mbpp | full / test | 500 | 128 |
| LiveCodeBench | livecodebench/code_generation_lite | test / 累计 release_v6 | 1,055 | 128 |
| MT-Bench | lm-sys/FastChat | 官方 question.jsonl | 80 组双轮对话 | 80 组双轮对话 |
| 合计 | | | 3,648 题或对话 | **786 题或对话，866 次回答** |

抽样遵循 [DDTree 官方基准代码](https://github.com/liranringel/ddtree/blob/master/benchmark.py)：源集合大于预定题数时使用 `shuffle(seed=0).select(range(n))`；题数等于源集合大小时保持原顺序。选题种子固定为 **0**，与生成种子 17、29、43 分离；所有基线、GBV 和消融复用同一份题号清单，不随方法或生成种子重抽。代码通过 Hugging Face `Dataset.shuffle` 的实际输出核对该抽样规则。

这里对齐的是**评测数量与抽样规则**，未宣称与 DDTree 论文使用完全相同的题目、提示或推理协议。只有数据源版本与原始顺序也一致，抽出的题号才会一致。例如本配置仍从 MBPP `full/test` 的 500 题中选 128 题，DDTree 官方加载器使用 `sanitized/test`；二者题数相同不代表样本相同。LiveCodeBench 的时间窗口、提示处理和 MT-Bench 对话历史等也以本文代码为准。数量对齐后仍须按实际运行协议比较速度和质量，不能把官方表中数值当成本代码重跑结果。

来源：[GSM8K](https://huggingface.co/datasets/openai/gsm8k/blob/main/README.md)、[MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500)、[AIME25](https://huggingface.co/datasets/MathArena/aime_2025/blob/main/README.md)、[HumanEval](https://github.com/openai/human-eval)、[MBPP](https://huggingface.co/datasets/google-research-datasets/mbpp/blob/main/README.md)、[LiveCodeBench 官方加载器](https://huggingface.co/datasets/livecodebench/code_generation_lite/blob/main/code_generation_lite.py)、[MT-Bench 官方题目](https://github.com/lm-sys/FastChat/blob/587d5cfa1609a43d192cedb8441cac3c17db105d/fastchat/llm_judge/data/mt_bench/question.jsonl)。七个数据集覆盖 [DFlash 论文主表](https://arxiv.org/html/2602.06036v1) 的任务名称；这不表示版本、提示、生成上限和硬件与其原始实验完全相同，本文应报告本节的实际协议。

`mbpp_sanitized` 的 257 条 test 也受支持，但它是另一配置；不得把从它抽出的 128 题与从 `full/test` 抽出的 128 题混写。若另做该配置，需复制配置，将 `datasets` 和 `evaluation.counts` 中的 `mbpp` 一并换成 `mbpp_sanitized`，并更换数据目录和输出目录。AIME24 的 30 题保留为可选配置，默认主实验不包含。

LiveCodeBench 的抽样母集固定为累计 `release_v6`：按 `test.jsonl`、`test2.jsonl` 至 `test6.jsonl` 的顺序读取六个官方文件并核查共 1,055 题，再固定选取 128 题；不只取新增的 v6 题目，也不另做日期筛选。`lite` 数据源本身经过官方测试用例精简，本代码保留每道入选题的全部公开和隐藏用例，不再截取。论文应写明 `code_generation_lite / release_v6，固定 seed=0 抽取 128 题`；不能与其他时间窗口或完整版测试用例的分数直接混用。入选题的测试用例独立保存并校验哈希，评分时按需加载，避免同时把所有隐藏测试放入内存。

MT-Bench 固定为 FastChat commit `587d5cfa1609a43d192cedb8441cac3c17db105d` 的全部 80 组对话，保留官方题号 81–160。第二轮输入依次包含第一轮用户问题、当前方法生成的第一轮回答、第二轮用户问题。每个方法使用自己的对话历史，生成器重新编码完整历史，并记录每轮输入与随机种子的哈希；不拼入参考答案。

数据准备先遍历源划分并核查完整条数与 ID 唯一性，再保存入选题目、原始 ID 及数据源提供的答案、参考实现和测试用例；首次准备解析并记录 Hugging Face revision，MT-Bench 使用上述固定 GitHub commit。因此减少生成题数不保证按相同比例减少源文件下载量。清单分别记录源条数 `source_count`、评测条数 `count`、原始行号 `selected_source_indices`、题号 `selected_source_ids`、抽样规则、提示格式与文件 SHA-256。固定子集的覆盖标记为 `fixed_evaluation_subset`；不能写成 `full_evaluation_split`。重新运行复用已准备清单，条数、题号、顺序或文件哈希不符时报错。

**生成器只读取 `prompt` / `user_turns`，评分资料保存在独立的 `evaluation` 字段或文件中**。数学答案、隐藏测试及参考实现不进入提示。MBPP 提示包含第一条公开断言以指定函数接口，其他标准断言隐藏，评分要求全部标准断言通过；challenge tests 保留但不计入主分数。LiveCodeBench 使用完整题面与提供的 starter code，评分执行入选题的全部保留测试。论文需注明这些提示协议。

本套实验不进行训练、微调、拟合或从测试分数选择 checkpoint；Target 和 Draft 均冻结。旧目录中的 rank-head、PPO、策略训练数据及 checkpoint 不参与本套实验。因而不需要为了 GBV 另外构造训练集。若以后加入学习型模块，应单独提供其训练数据和验证集，再使用 `audit --training-jsonl ...` 做题目重叠检查。已有测试集不能充当选超参数的验证集；本配置应在正式结果产生前固定，消融表展示预定网格的全部结果。

`audit` 核查数据完整性、数学标准答案可解析性、HumanEval/MBPP 参考代码非空、评测题重复及指定训练文件与评测题的规范化文本重叠。LiveCodeBench 不含统一参考程序，MT-Bench 不含可自动判定所有回答的标准答案；二者检查完整题目、轮次、测试结构与文件哈希，并在审计中明确记录参考答案检查的适用范围。精确文本比对不能排除改写题或预训练语料污染。公开预训练模型的全部训练语料并未在此独立审计，论文不能据此声称已证明“训练/测试完全无污染”。

## 3. 围绕三路径 GBV 的主实验和消融

4B、8B 均执行以下 T=1 主实验，主方法固定为 K=3、L=15。

| 配置名 | 方法 | 回答的问题 |
|---|---|---|
| target_t1 | Target AR | 相同 Target 的质量、时间与加速比基线 |
| dflash_match | DFlash 贪心草稿 + Target 采样前缀匹配 | 相对于 DFlash 匹配规则的端到端表现 |
| dflash_bv | 单路径 BV，K=1 | 增加到三路径后的收益与成本 |
| gbv | 三路径 GBV，K=3 | 核心方法的速度、质量和接受长度 |
| ddtree | 前缀概率树，45 个非根节点预算 | 相同最大候选位置额度下，不同候选组织/验证方式的表现 |

`dflash_match` 按 [DDTree 中的 DFlash 实现](https://github.com/liranringel/ddtree/blob/c96427a185677bf4133ed865dd1626a5041aef9b/dflash.py) 构造贪心草稿，依次接受与 Target 采样 token 匹配的最长前缀，并提交首次不匹配处的 Target token。以前名为 `dflash_token` 的标准 p/q 拒绝采样并不等于这个规则，本版将其明确改名为 `single_token_rejection`，仅作为验证机制对照。

这里的 DFlash 与 DDTree 是**同一实验引擎内的方法复现**。匹配规则和数量协议对齐，不表示复现了官方完整优化运行时：本框架统一 SDPA 与 FP64 概率运算，计时包含首轮 Draft；官方的注意力后端选择、优化和计时设置可能不同。论文表中应注明“本框架复现”，不能填入官方表格数值并当作本机重跑结果。

DDTree 预算由 60 调整为 45 个非根节点，对应 GBV 的 3×15 个候选位置上限；根 anchor 另计。前缀共享后 GBV 实际节点可能少于 45，且两种方法构树与校正成本不同，因此该对照不是严格等实际 FLOPs 或等耗时。报告同时给出实际树节点、接受长度和 forward 次数；此配置也不代表官方 DDTree 经预算搜索得到的最优性能。

详细消融仅在 4B 执行：

| 对照 | 取值 | 解释范围 | 新增配置数 |
|---|---|---|---:|
| 候选数 K | 1、2、3、4 | 多候选的接受收益能否抵消开销；K=3 的取舍 | 2 |
| 验证长度 L | 4、8、15 | 三路径下验证前缀长度的收益与成本；完整 Draft 块宽固定 | 2 |
| 单路径验证机制 | token 拒绝采样 / BV，均 K=1 | 固定草稿分布，考察块验证的贡献 | 1 |
| 前缀共享 | 开 / 关，均 K=3、L=15 | 相同候选多重集下共享计算的开销收益 | 1 |

K=1 的 GBV 在分布构造上退化为单路径 BV，直接复用 `dflash_bv`；K=3、L=15 与开启共享均复用主方法记录。单路径机制对照不应写成 K=3 下 BV 贡献的直接测量。共享关闭时保留原候选及其抽样重数，不改变候选采样规则。全部结果按预定网格报告，不根据测试分数挑选有利设置。

此外，两个模型各增加 T=0 的 AR、K=3 GBV 两个配置，检查真实完整评测上的贪心 token 一致率并报告速度；这不是所有方法的 T=0 主表。主表回答的重点是 T=1 下三路径 GBV 的表现。

不再调度双向注意力因果化、Target 特征置零、Draft 缓存开关、概率精度切换、K=8 或额外中间温度。源码中的旧开关仅供兼容和内部回归检查，不增加正式实验任务。

| 累计阶段 | 4B 配置数 | 8B 配置数 | 数据集×配置×种子任务数 | 题目/完整对话记录 | 回答生成次数 |
|---|---:|---:|---:|---:|---:|
| gbv-first | 1 | 1 | 42 | 4,716 | 5,196 |
| main | 5 | 5 | 210 | 23,580 | 25,980 |
| complete | 13 | 7 | 420 | 47,160 | 51,960 |

这里的一个任务表示一种模型、一种配置、一个数据集、一个生成种子的整批评测，不是一遍训练。各阶段计数为累计范围；后续复用已有结果，不能把三行相加。每个配置/种子均为相同 786 题或对话、866 次回答；MT-Bench 两个回答合为一条记录。

## 4. H200 环境与分阶段启动

建议独立 Python 3.11 环境，安装与服务器驱动兼容的 CUDA PyTorch，记录实际环境：

```bash
python3.11 -m venv .venv-gbv
source .venv-gbv/bin/activate
python -m pip install torch==2.9.1
python -m pip install -r requirements-gbv-paper.txt
docker build -t gbv-code-eval:py311 experiments/gbv_paper

# 只打印两模型完整矩阵，不下载模型或运行 GPU 推理。
PYTHON_BIN=.venv-gbv/bin/python bash scripts/run_gbv_paper.sh plan

# 默认阶段：先跑 4B 和 8B 的三路径 GBV。
PYTHON_BIN=.venv-gbv/bin/python bash scripts/run_gbv_paper.sh gbv-first

# 同一输出目录：保留 GBV 结果，补齐两个模型的 T=1 主实验。
PYTHON_BIN=.venv-gbv/bin/python bash scripts/run_gbv_paper.sh main

# 同一输出目录：补齐 4B 消融及两个模型的 T=0 对照。
PYTHON_BIN=.venv-gbv/bin/python bash scripts/run_gbv_paper.sh complete
```

未指定阶段时默认 `gbv-first`。每个阶段依次执行代码测试、统一数据准备/审计、可用参考答案评分自检，再对各模型进行 GPU checkpoint 预检、未完成题目生成、评分与阶段汇总。预检包含小规模 AR 对照、树分支及缓存回退检查；这些是正确性门槛，不计入正式主表。任何必需步骤失败即停止，不跳题。

GBV 优先阶段尚无全量 AR 对照，报告只呈现吞吐、接受长度与可评分任务的质量；加速比和相对 AR 的质量差值留空，整体报告标记未完成。`main` 阶段补齐基线后即可查看完整 T=1 主表，但完整预定矩阵仍待 `complete`。只有最终阶段要求所有配置生成和评分齐全，并绘制主实验和消融图。

默认共享数据目录为 `datasets/gbv_paper_ddtree_counts`。输出目录：

```text
outputs/gbv_paper_two_models/
  plan_gbv-first.json / plan_main.json / plan_complete.json
  data_audit.json / gold_audit.json
  qwen3_4b/   # 4B 独立 manifest、results、scores、阶段报告
  qwen3_8b/   # 8B 独立 manifest、results、scores、阶段报告
  phase_completed_<phase>.json
```

首次运行即在每个模型的 manifest 中固定该模型的完整矩阵。阶段只是调度过滤条件，不改变实验身份，因此可安全续跑。代码、环境、模型、完整配置或数据清单改变时拒绝复用旧结果；本版须使用新输出目录，不能续写旧 22 配置实验。不同阶段没有跨阶段随机交错，正式比较应维持同一专用 GPU、功率/时钟设置及后台负载；按题交错仅作用于同阶段尚未完成的方法，不能据此声称完全消除了跨阶段时间漂移。

只调度 8B，仍复用同一套件中对应模型的结果目录：

```bash
PYTHON_BIN=.venv-gbv/bin/python bash scripts/run_gbv_paper.sh gbv-first --model-ids qwen3_8b
```

可用 `SUITE`、`DATA_DIR`、`OUTPUT`、`DEVICE`、`CODE_BACKEND` 环境变量修改位置或运行设备。旧 `CONFIG` 变量不再用于两模型入口；设置它会明确报错。`run_gbv_paper_full.sh` 为兼容入口，直接执行当前套件的 `complete`，名称中的 full 不表示全源数据。

底层命令仍支持单模型配置，例如查看 8B 的计划，或只调度其 GBV：

```bash
python scripts/gbv_paper.py plan --config configs/gbv_paper_qwen3_8b.json
python scripts/gbv_paper.py run --config configs/gbv_paper_qwen3_8b.json \
  --only-variants gbv --output outputs/gbv_paper_two_models/qwen3_8b
```

直接调用 `run` 前须完成 `prepare`、`audit`、`validate-gold` 和相应 `check-model`。优先使用套件脚本自动执行这些步骤。`--only-variants` 仅过滤本轮调度，保留完整 manifest；`--groups` 会建立不同的实验集合，应使用独立目录，不能用于上述阶段续跑。`--smoke` 仅用于诊断，要求独立 smoke 目录，正式报告拒绝其结果。

全源划分备选使用 `configs/gbv_paper_full.json`，并为所有准备、审计、预检、生成、评分步骤指定独立的数据和输出目录。它仅包含 4B 的 13 配置：每个配置/种子 3,648 题或对话、3,728 次回答，三种子合计 142,272 条记录、145,392 次回答，不属于默认两模型套件。固定子集与全源划分须分别报告。

代码题默认在禁网 Docker 容器中执行，镜像 ID 进入评分记录。没有 Docker 时可在专用隔离环境显式使用 `CODE_BACKEND=process`；该模式仅限制子进程资源，不提供容器文件系统/网络隔离。正式结果须记录实际后端，不能混合不同后端的评分文件。

`validate-gold` 检查全部入选题中可用的官方数学答案和参考实现，失败则保存题号并停止，不能删除失败题来凑齐实验。LiveCodeBench 和 MT-Bench 没有统一参考解，该项标为不适用；它们仍需通过数据结构与评分接口检查。指定本地模型时将配置中的 Target/Draft 改为本地目录，程序记录文件哈希。

## 5. 计时、评分和统计

- **TTFT / prefill_ms**：Target 处理输入并采样首 token，包含该阶段所需的 Target 特征提取。
- **decode_ms**：从首 token 之后到最后一个 token 提交；包含第一次及后续 Draft、采样、树构造、验证、校正和 KV 整理。
- **decode throughput**：按回答轮次计算 `sum(generated_tokens - 1) / sum(decode_seconds)`。因此 MT-Bench 一组双轮对话的 decode token 数为两轮总 token 数减 2。不把每轮首 token 加入 decode 分子。
- **端到端吞吐**：全部生成 token 数 / prefill 与 decode 总时长。
- **加速比**：相同数据、ID、种子和 Target 温度配对后的 AR 时间/token 除以待测方法时间/token。随机采样下两个方法的回答长度可以不同，同时报告长度与截断率。
- MT-Bench 按完整对话汇总两轮生成时间、token 数和 forward 次数，显存取两轮峰值的最大值；TTFT 与截断率按实际回答轮次统计。不同方法在 T>0 时的第二轮历史可能不同，该差异属于各自真实多轮生成过程。它不是固定相同第二轮输入的微基准。
- 模型计时使用 CUDA 同步边界；预热不计入正式时间。默认不启用分阶段 profiling，避免混入插桩成本。`--profile` 使用独立目录，其 CUDA event 时长包含该事件区间的主机调度间隙，不能解释为纯 kernel 时间。
- 同时报告每轮接受数、实际提交数、树节点数、Target/Draft forward 次数、内存与延迟分位数。显存数值包含已经加载的两套权重；AR 在此公平交错框架中也保持 Draft 驻留，该数值不能当作独立 AR 部署的最低显存。
- 数学评分固定 [Math-Verify](https://github.com/huggingface/Math-Verify) 0.8.0，处理数学等价；不以简单字符串相等代替。HumanEval/MBPP/LiveCodeBench 使用实际测试执行得到每条生成的 pass/fail，报告 pass@1 的种子平均，不把 3 个种子中的任意成功称为 pass@1。
- LiveCodeBench 复用未修改的[官方执行器](https://github.com/LiveCodeBench/LiveCodeBench/blob/28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24/lcb_runner/evaluation/testing_util.py)，固定 commit `28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24`。每个测试用例限时 6 秒，总任务时间上限随用例数增加，内存 4 GiB；HumanEval/MBPP 每题限时 10 秒、内存 2 GiB。Python 版本、资源限制和容器镜像是本实验协议的一部分。
- MT-Bench 默认写入 `passed=null, metric=not_scored`，其准确率、质量差值和相应置信区间留空，不能当作 0 分或已通过。它的外部裁判分数为独立的 1–10 分指标，不混入数学 accuracy 或代码 pass@1。
- 速度和质量差值的置信区间使用按题聚类的配对 bootstrap，同题的不同种子整体重采样。不能仅因质量差异“不显著”就宣称有限精度实现严格无偏。
- T=0 报告与 AR 的 token 一致率；T>0 不要求相同 seed 逐 token 一致。不同算法消耗随机数的顺序不同。

### MT-Bench 外部裁判接口

每个模型完整生成后可分别导出 [FastChat 单答案评分格式](https://github.com/lm-sys/FastChat/tree/587d5cfa1609a43d192cedb8441cac3c17db105d/fastchat/llm_judge)。每个配置、种子对应一个 `model_answer/*.jsonl`，每题包含两个回答；导出同时保存官方题目和预测哈希。这些命令只导出文件和导入已有裁判结果，不发起付费 API 请求。

```bash
python scripts/gbv_paper.py export-mtbench \
  --run-dir outputs/gbv_paper_two_models/qwen3_4b --data-dir datasets/gbv_paper_ddtree_counts \
  --output outputs/gbv_paper_two_models/qwen3_4b/mtbench_export

# 使用上述回答按 FastChat 协议完成外部裁判后，导入其单答案评分 JSONL。
python scripts/gbv_paper.py import-mtbench-judgments \
  --run-dir outputs/gbv_paper_two_models/qwen3_4b \
  --export-dir outputs/gbv_paper_two_models/qwen3_4b/mtbench_export \
  --judgments /path/to/single_answer_judgments.jsonl \
  --output outputs/gbv_paper_two_models/qwen3_4b/mtbench_quality
```

上例为 4B，8B 将路径中的 `qwen3_4b` 替换为 `qwen3_8b`。

导入要求同一个裁判模型、每个配置/种子的全部 80 题及两个轮次，拒绝重复、遗漏、预测哈希变化和无法解析的 `-1` 分数。产物包括逐题逐轮原始分数、第一轮/第二轮/整体均分及裁判身份与输入哈希。论文使用该质量指标时，应另行固定并说明裁判模型、评分提示、参考答案和裁判采样设置；当前代码没有宣称裁判评分已经执行。

## 6. 代码正确性与数值处理

GBV 实现沿用本项目的扩散条件分布 → K 次候选抽样 → 前缀树 Target 验证 → 字典序选择 → 所选分布校正 → BV 提交的流程。扩散 forward 内部保留双向注意力；树 Target 的每个节点只读取已接受上下文及自身祖先。重复候选可以共享计算节点，但不会从 K 次抽样中删掉。

原先直接计算两项近似相等的幂之差可能在长前缀中产生全零概率行。新内核在缩放后用非负多项式计算这个差商，并逐层更新相对质量，主配置用 FP64。遇到不合法概率报错，不静默替换为其他分布。这个实现减少已知的消减误差，不等于机器浮点与实数严格相等。

本地测试包括小词表下的有理数全枚举、枚举实际 BV 采样分支验证最终联合输出分布、深前缀数值退化回归，以及真实小型 Qwen3/DFlash 的树/顺序 logits、特征、缓存、EOS 和长度边界校验。数据测试覆盖 AIME 原始题号、LiveCodeBench 六文件加载与私有测试解码/执行、MT-Bench 第二轮历史、分轮 token 计数、完整对话续跑和裁判导出/导入。选题测试对照实际 Hugging Face shuffle，核查原始题号与顺序、源条数、全量/子集目录隔离、入选题完整测试用例及固定子集从准备到评分汇总的流程；缺题、重复题号或替换题号不能生成完整报告。这些使用本地构造的数据，不替代真实数据准备与审计。H200 上另执行真正 checkpoint 的 greedy、树分支及缓存回退校验。新增测试覆盖 DFlash 匹配验证的联合输出分布、T=1 贪心草稿规则、8B 固定配置的 meta 构造，以及先 GBV 后补齐的断点续跑。CPU 测试通过不能替代 H200/BF16 校验；后者不通过时不得生成正式速度结果。

GBV/BV 的算法归属需保留：[Thomas 与 Pal，2026](https://arxiv.org/abs/2602.16961)、项目 vendored 的 BV 文献引用及 [DFlash](https://arxiv.org/abs/2602.06036)。本实验框架和数值实现调整不构成对原 GBV 算法原创性的声明。

## 7. 结果文件与复现

| 文件 | 用途 |
|---|---|
| datasets/gbv_paper_ddtree_counts/manifest.json | 源版本与完整条数、选题种子、入选原始行号和题号、文件哈希 |
| data_audit.json / gold_audit.json | 数据结构、训练文件重叠及标准答案评分检查 |
| gpu_preflight_<phase>.json | H200 环境、实际模型校验及误差记录 |
| run_manifest.json | 模型、数据、源代码、环境、参数和完整任务集合的身份 |
| results.jsonl | 每题每种子的原始 token、文本、时间与轮次统计；MT-Bench 含两个 turn_results |
| scores.jsonl / scoring_manifest.json | 原始评分、预测哈希、评分器身份 |
| errors.jsonl | 运行错误，错误不会被伪装成完成的样本 |
| stage_completed_*.json | 指定方法阶段生成完成，另记完整实验是否完成 |
| completed.json | 该模型全部预定生成均完成的标记 |
| phase_completed_<phase>.json | 套件中所选模型的本阶段生成与评分记录均齐全 |
| report_<phase>/summary.csv / summary.json | 逐数据集、逐方法的统计与置信区间 |
| report_complete/table.tex / *.pdf / *.png | 论文表格及主实验、各消融组的图 |
| mtbench_export/ | 官方裁判格式的完整双轮回答、问题与导出清单 |
| mtbench_quality/ | 导入后的独立裁判分数、分轮均分和裁判清单 |

断点续跑复用同一命令。程序用 `(variant, dataset, source_id, seed)` 去重，只修复未写完的最后一行；中间损坏、重复键或模型/数据/代码/环境改变都会报错。MT-Bench 两轮完成后才写入一条记录；第二轮中断会从该对话第一轮重新生成，汇总拒绝缺轮的记录。生成和评分共享输出目录写锁，防止两个进程同时写入；不要并发启动同一套件输出根目录的两个阶段。运行期间不要修改代码或配置。

正式汇总要求预定评测集合中的所有方法、题目及生成种子均完成生成与评分，并将运行题号清单与数据准备清单核对；默认子集完成后的标记仍为 `fixed_evaluation_subset`。`--allow-partial` 和 `--performance-only` 仅用于诊断，输出明确标记其状态。评测结果应按数据集逐项呈现，不把不同任务的分数混成一个未经定义的综合准确率。
