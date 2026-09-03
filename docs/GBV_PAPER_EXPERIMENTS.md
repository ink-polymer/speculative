# 扩散草稿多路径 GBV：正式实验代码

这套代码评测固定 Qwen3-4B Target 与固定 DFlash-b16 扩散草稿模型上的多路径 GBV。主实验、消融、评分和汇总共用 `src/gbv_experiments/`，入口是 `scripts/gbv_paper.py`。尚未产生 H200 正式实验结果；本机测试不能作为论文速度结果。

## 1. 固定协议

配置：`configs/gbv_paper_full.json`。

- Target：`Qwen/Qwen3-4B`，revision `1cfa9a7208912126459214e8b04321603b3df60c`。
- Draft：`z-lab/Qwen3-4B-DFlash-b16`，revision `b74e3a329c4d963783143b1e970d95b002be72bd`。
- 模型权重 BF16；主配置概率计算 FP64；SDPA；batch size 1；关闭 thinking、TF32；不使用 top-k、top-p、重复惩罚或量化。
- 随机种子 17、29、43。每题由原始 ID 派生随机种子；方法顺序按题随机轮换，避免一种方法总处于冷启动或后期热状态。
- 每个回答轮次最多生成 2,048 个 token，包含首个 Target token；遇到 EOS 结束当前回答。到达上限的回答保留并评分，单独报告截断比例。MT-Bench 的两个回答分别使用此上限，第一轮 EOS 不会取消第二轮。
- 块宽为 16：1 个已知 anchor + 15 个未来位置。本文代码中的 `length=L` 不含 anchor。改变 L 时仍计算 checkpoint 的完整 16 位置，只取前 L 个未来位置，避免同时改变输入掩码布局。
- 无提示截断；超出上下文容量直接报错。模型加载、tokenizer 和文本评分不计入生成时间。

模型配套关系和关闭 thinking 的用法见 [DFlash 模型卡](https://huggingface.co/z-lab/Qwen3-4B-DFlash-b16/blob/b74e3a329c4d963783143b1e970d95b002be72bd/README.md)。

## 2. 数据及训练/测试边界

“全量”指下列固定评测集合中的每一题，没有 2,000 条或其他本地抽样上限。AIME25 的源仓库把比赛题存放在名为 `train` 的划分中；本实验将这 30 题全部用于评测，不用于训练。

| 数据集 | 来源 | 配置 / 划分 | 每个方法、每个种子的题数 |
|---|---|---|---:|
| GSM8K | openai/gsm8k | main / test | 1,319 |
| MATH-500 | HuggingFaceH4/MATH-500 | 默认 / test | 500 |
| AIME25 | MathArena/aime_2025 | 默认 / train（仅评测） | 30 |
| HumanEval | openai/openai_humaneval | 默认 / test | 164 |
| MBPP | google-research-datasets/mbpp | full / test | 500 |
| LiveCodeBench | livecodebench/code_generation_lite | test / 累计 release_v6 | 1,055 |
| MT-Bench | lm-sys/FastChat | 官方 question.jsonl | 80 组双轮对话 |
| 合计 | | | 3,648 题或对话，3,728 次回答 |

来源：[GSM8K](https://huggingface.co/datasets/openai/gsm8k/blob/main/README.md)、[MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500)、[AIME25](https://huggingface.co/datasets/MathArena/aime_2025/blob/main/README.md)、[HumanEval](https://github.com/openai/human-eval)、[MBPP](https://huggingface.co/datasets/google-research-datasets/mbpp/blob/main/README.md)、[LiveCodeBench 官方加载器](https://huggingface.co/datasets/livecodebench/code_generation_lite/blob/main/code_generation_lite.py)、[MT-Bench 官方题目](https://github.com/lm-sys/FastChat/blob/587d5cfa1609a43d192cedb8441cac3c17db105d/fastchat/llm_judge/data/mt_bench/question.jsonl)。七个数据集覆盖 [DFlash 论文主表](https://arxiv.org/html/2602.06036v1) 的任务名称；这不表示版本、提示、生成上限和硬件与其原始实验完全相同，本文应报告本节的实际协议。

`mbpp_sanitized` 的 257 条 test 也受支持，但它是另一配置；不得把它的分数与 full 500 的分数混写。可复制配置，把 `mbpp` 换成 `mbpp_sanitized`，同时更换数据目录和输出目录。AIME24 的 30 题保留为可选配置，默认主实验不包含。

LiveCodeBench 固定为累计 `release_v6`：读取 `test.jsonl`、`test2.jsonl` 至 `test6.jsonl` 全部六个官方文件，共 1,055 题，不只取新增的 v6 题目，也不另做日期筛选。`lite` 数据源本身经过官方测试用例精简，本代码保留该数据源提供的全部公开和隐藏用例，不再截取。论文应写明 `code_generation_lite / release_v6`；不能与其他时间窗口或完整版测试用例的分数直接混用。每题的测试用例独立保存并校验哈希，评分时按需加载，避免同时把所有隐藏测试放入内存。

MT-Bench 固定为 FastChat commit `587d5cfa1609a43d192cedb8441cac3c17db105d` 的全部 80 组对话，保留官方题号 81–160。第二轮输入依次包含第一轮用户问题、当前方法生成的第一轮回答、第二轮用户问题。每个方法使用自己的对话历史，生成器重新编码完整历史，并记录每轮输入与随机种子的哈希；不拼入参考答案。

数据准备保留题目、原始 ID 及数据源提供的答案、参考实现和测试用例；首次准备解析并记录 Hugging Face revision，MT-Bench 使用上述固定 GitHub commit。清单固定实际条数、提示格式与文件 SHA-256。**生成器只读取 `prompt` / `user_turns`，评分资料保存在独立的 `evaluation` 字段或文件中**。数学答案、隐藏测试及参考实现不进入提示。MBPP 提示包含第一条公开断言以指定函数接口，其他标准断言隐藏，评分要求全部标准断言通过；challenge tests 保留但不计入主分数。LiveCodeBench 使用完整题面与提供的 starter code，评分执行全部保留的测试。论文需注明这些提示协议。

本套实验不进行训练、微调、拟合或从测试分数选择 checkpoint；Target 和 Draft 均冻结。旧目录中的 rank-head、PPO、策略训练数据及 checkpoint 不参与本套实验。因而不需要为了 GBV 另外构造训练集。若以后加入学习型模块，应单独提供其训练数据和验证集，再使用 `audit --training-jsonl ...` 做题目重叠检查。已有测试集不能充当选超参数的验证集；本配置应在正式结果产生前固定，消融表展示预定网格的全部结果。

`audit` 核查数据完整性、数学标准答案可解析性、HumanEval/MBPP 参考代码非空、评测题重复及指定训练文件与评测题的规范化文本重叠。LiveCodeBench 不含统一参考程序，MT-Bench 不含可自动判定所有回答的标准答案；二者检查完整题目、轮次、测试结构与文件哈希，并在审计中明确记录参考答案检查的适用范围。精确文本比对不能排除改写题或预训练语料污染。公开预训练模型的全部训练语料并未在此独立审计，论文不能据此声称已证明“训练/测试完全无污染”。

## 3. 主实验和消融

主实验 T=1：Target AR、单路径 DFlash token verification、单路径 DFlash BV、K=3 的 GBV、DDTree（树预算 60）。DDTree 是同一计时框架内的前缀概率树复现；其节点预算与 GBV 的候选数不同，论文必须同时报告节点数和接受长度，不能称为相同计算预算对照。也不能把这里的时间直接写成其他项目的官方跑分。

| 消融组 | 取值 / 对照 | 控制关系 |
|---|---|---|
| paths | K=1,2,3,4,8 | L=15，T=1，其他固定 |
| lengths | L=4,8,15 | K=3，T=1，完整 draft 块宽不变 |
| temperatures | T=0,0.2,0.6,1 | 每个温度有同温度 AR 基线 |
| block_verification | 单路径 token vs 单路径 BV | 两者 K=1，避免把候选数变化算成验证收益 |
| prefix_sharing | 合并 / 不合并共享前缀 | 保留全部 K 个候选的抽样重数 |
| draft_cache | 保留 / 重新计算 Draft context KV | Target KV 始终按已提交路径保留 |
| bidirectional_attention | 双向 / 因果 draft mask | checkpoint 相同，是推理时移除可见边的干预 |
| target_features | 正常 / 置零 Target 特征 | checkpoint 相同，是推理条件移除 |
| probability_precision | FP64 / FP32 概率运算 | 权重始终 BF16，均使用稳定计算式 |

注意：因果 mask 和特征置零会改变既有预训练模型的输入条件，不能据此推断“重新训练一个因果模型一定差多少”。这里没有因果模型重训实验。也没有通过故意省略 GBV 校正来制造更快的有偏方法。

相同参数组合会去重，保留所属的所有消融组。默认共 22 个配置、3 个种子，合计 **240,768 条题目/完整对话记录，246,048 次回答生成**；仅 `main` 为 54,720 条记录、55,920 次回答生成。每一个配置都使用相同的七套全量题目。MT-Bench 的两个回答算一条完整记录，避免把多轮对话重复计为不同题目。

## 4. H200 环境与启动

建议独立 Python 3.11 环境。PyTorch CUDA wheel 应与服务器驱动兼容，正式环境记录实际安装版本。

```bash
python3.11 -m venv .venv-gbv
source .venv-gbv/bin/activate
python -m pip install torch==2.9.1
python -m pip install -r requirements-gbv-paper.txt
docker build -t gbv-code-eval:py311 experiments/gbv_paper
PYTHON_BIN=.venv-gbv/bin/python bash scripts/run_gbv_paper_full.sh
```

默认脚本依次执行：本地代码测试 → 七套全量数据准备 → 数据审计 → 所有可用官方参考答案的评分自检 → GPU 实际 checkpoint 校验 → 全量生成 → 评分 → 汇总。MT-Bench 的默认汇总包括性能，外部裁判质量分数通过下述独立接口导入。任何必需步骤失败即停止，不跳题，不把失败样本作为成功实验记录。

代码题默认在禁网容器中执行，固定内存、CPU 时间与进程上限。Docker 镜像 ID 进入评分记录。若服务器没有 Docker，可在专用隔离环境明确设置 `CODE_BACKEND=process`；该模式只有子进程资源限制，不提供文件系统或网络隔离，不能当作容器沙箱。论文记录实际后端，不能混合两个后端的评分文件。评测容器不挂载模型目录或账户凭据。

如果可用的官方参考答案或参考实现不能通过评分，`validate-gold` 会保存失败题目并停止；需逐题核查官方数据、提示处理和运行依赖。LiveCodeBench 和 MT-Bench 在该步骤标注无统一参考解，`passed` 为 `null`，不伪造参考解评分成功。LiveCodeBench 执行器另有标准输入、函数调用、隐藏用例、超时与语法错误的本地回归测试。

分步运行：

```bash
python scripts/gbv_paper.py plan --output outputs/gbv_paper_full/plan.json
python scripts/gbv_paper.py prepare --data-dir datasets/gbv_paper_full
python scripts/gbv_paper.py audit --data-dir datasets/gbv_paper_full --output outputs/gbv_paper_full/data_audit.json
python scripts/gbv_paper.py validate-gold --output outputs/gbv_paper_full/gold_audit.json
python scripts/gbv_paper.py check-model --output outputs/gbv_paper_full/gpu_preflight.json
python scripts/gbv_paper.py run --output outputs/gbv_paper_full
python scripts/gbv_paper.py score --run-dir outputs/gbv_paper_full
python scripts/gbv_paper.py report --run-dir outputs/gbv_paper_full --output outputs/gbv_paper_full/report
```

只运行主实验或单个完整消融组：

```bash
python scripts/gbv_paper.py run --groups main --output outputs/gbv_main_full
python scripts/gbv_paper.py run --groups paths --output outputs/gbv_paths_full
```

两条命令均是全量数据。若一开始计划运行整套，直接使用同一个完整配置，避免重复运行共同基线。`--smoke` 仅用于安装诊断，会明确标记为 smoke 且要求单独的 smoke 输出目录，正式报告拒绝此类结果。

若此前已用四数据集版本生成数据或结果，请使用新的 `DATA_DIR` 和 `OUTPUT`。程序会拒绝把不同数据集、提示格式或代码版本的记录混入同一实验目录。

指定已有本地模型时，把配置中的 `model.target` / `model.draft` 改为目录。程序记录权重及配置文件哈希；不要把微调模型伪装成指定的公开 checkpoint。

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

完整生成后可导出 [FastChat 单答案评分格式](https://github.com/lm-sys/FastChat/tree/587d5cfa1609a43d192cedb8441cac3c17db105d/fastchat/llm_judge)。每个配置、种子对应一个 `model_answer/*.jsonl`，每题包含两个回答；导出同时保存官方题目和预测哈希。这些命令只导出文件和导入已有裁判结果，不发起付费 API 请求。

```bash
python scripts/gbv_paper.py export-mtbench \
  --run-dir outputs/gbv_paper_full --data-dir datasets/gbv_paper_full \
  --output outputs/gbv_paper_full/mtbench_export

# 使用上述回答按 FastChat 协议完成外部裁判后，导入其单答案评分 JSONL。
python scripts/gbv_paper.py import-mtbench-judgments \
  --run-dir outputs/gbv_paper_full \
  --export-dir outputs/gbv_paper_full/mtbench_export \
  --judgments /path/to/single_answer_judgments.jsonl \
  --output outputs/gbv_paper_full/mtbench_quality
```

导入要求同一个裁判模型、每个配置/种子的全部 80 题及两个轮次，拒绝重复、遗漏、预测哈希变化和无法解析的 `-1` 分数。产物包括逐题逐轮原始分数、第一轮/第二轮/整体均分及裁判身份与输入哈希。论文使用该质量指标时，应另行固定并说明裁判模型、评分提示、参考答案和裁判采样设置；当前代码没有宣称裁判评分已经执行。

## 6. 代码正确性与数值处理

GBV 实现沿用本项目的扩散条件分布 → K 次候选抽样 → 前缀树 Target 验证 → 字典序选择 → 所选分布校正 → BV 提交的流程。扩散 forward 内部保留双向注意力；树 Target 的每个节点只读取已接受上下文及自身祖先。重复候选可以共享计算节点，但不会从 K 次抽样中删掉。

原先直接计算两项近似相等的幂之差可能在长前缀中产生全零概率行。新内核在缩放后用非负多项式计算这个差商，并逐层更新相对质量，主配置用 FP64。遇到不合法概率报错，不静默替换为其他分布。这个实现减少已知的消减误差，不等于机器浮点与实数严格相等。

本地测试包括小词表下的有理数全枚举、枚举实际 BV 采样分支验证最终联合输出分布、深前缀数值退化回归，以及真实小型 Qwen3/DFlash 的树/顺序 logits、特征、缓存、EOS 和长度边界校验。新增数据测试覆盖 AIME 原始题号、LiveCodeBench 六文件加载与私有测试解码/执行、MT-Bench 第二轮历史、分轮 token 计数、完整对话续跑和裁判导出/导入。H200 上另执行真正 checkpoint 的 greedy、树分支及缓存回退校验。CPU 测试通过不能替代 H200/BF16 校验；后者不通过时不得生成正式速度结果。

GBV/BV 的算法归属需保留：[Thomas 与 Pal，2026](https://arxiv.org/abs/2602.16961)、项目 vendored 的 BV 文献引用及 [DFlash](https://arxiv.org/abs/2602.06036)。本实验框架和数值实现调整不构成对原 GBV 算法原创性的声明。

## 7. 结果文件与复现

| 文件 | 用途 |
|---|---|
| data_audit.json / gold_audit.json | 数据结构、训练文件重叠及标准答案评分检查 |
| gpu_preflight.json | H200 环境、实际模型校验及误差记录 |
| run_manifest.json | 模型、数据、源代码、环境、参数和完整任务集合的身份 |
| results.jsonl | 每题每种子的原始 token、文本、时间与轮次统计；MT-Bench 含两个 turn_results |
| scores.jsonl / scoring_manifest.json | 原始评分、预测哈希、评分器身份 |
| errors.jsonl | 运行错误，错误不会被伪装成完成的样本 |
| completed.json | 所有预定生成均完成的标记 |
| report/summary.csv / summary.json | 逐数据集、逐方法的统计与置信区间 |
| report/table.tex / *.pdf / *.png | 论文表格及主实验、各消融组的图 |
| mtbench_export/ | 官方裁判格式的完整双轮回答、问题与导出清单 |
| mtbench_quality/ | 导入后的独立裁判分数、分轮均分和裁判清单 |

断点续跑复用同一命令。程序用 `(variant, dataset, source_id, seed)` 去重，只修复未写完的最后一行；中间损坏、重复键或模型/数据/代码/环境改变都会报错。MT-Bench 两轮完成后才写入一条记录；第二轮中断会从该对话第一轮重新生成，汇总拒绝缺轮的记录。生成和评分共享输出目录写锁，防止两个进程同时写入。运行期间不要修改代码或配置。

正式汇总要求全量生成与全量评分；`--allow-partial` 和 `--performance-only` 仅用于诊断，输出明确标记其状态。全量评测结果应按数据集逐项呈现，不把不同任务的分数混成一个未经定义的综合准确率。
