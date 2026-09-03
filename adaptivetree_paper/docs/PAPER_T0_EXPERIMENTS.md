# T=0 分层结构策略：正式实验代码

本入口实现最近数学证明中的**分层结构策略 AdaptiveTree**，包括反事实预训练、强化学习、主实验和独立重训的消融。它不是此前的 latency-aware AdaptiveTree，也不是 T>0 的三路径 GBV；旧实验结果不导入本流程。

当前交付是代码和本地 CPU 测试。尚未下载并验收全量数据、加载正式权重或进行服务器 GPU 实验；不能据此声称已经加速、已经训练收敛，或所有 BF16 输出必然一致。

## 1. 入口与文件

- `scripts/run_paper_t0_full.sh`：统一入口；不传阶段默认执行 `all`。
- `configs/paper_t0_full.json`：数据、训练、消融和测量配置。
- `configs/paper_t0_model.json`：冻结的原始 target/draft、revision 和共享执行后端。
- `src/dflash_specblock/paper/data.py`：全量数据、去重、划分和来源检查。
- `structure.py`：状态、policy 和逐层构树。
- `runtime.py`：共享 eager 推理及同状态反事实验证。
- `training.py`：预训练、在线 actor-critic、断点及验证集选模。
- `evaluation.py`：配对测试、逐 token 检查、bootstrap 和表格。
- `tests/test_paper_protocol.py`：新流程的边界、隔离、完整性及端到端测试。
- `tests/test_paper_datasets_extended.py`：AIME25、LiveCodeBench、MT-Bench、多轮历史与完整数据清单测试。

新流程不改动旧 `engine.py`、`ddtree_builder.py`、`verification.py`，也不更改嵌套仓库中的实验代码。

## 2. 方法的准确实现

每轮仅一次原始 DFlash forward，输入始终是训练规格的 1 个 anchor + 15 个 mask，得到 `[15,V]` logits。策略只控制树，target/draft 全部冻结。

状态共 84 维：每个 slot 的 top-1 概率、top-1/top-2 差、归一化熵、top-8 概率和（60 维）；已有 target 上下文最后位置的多层拼接 hidden 经 RMS 归一化和平均池化（16 维）；过去最多 8 轮的接受/分歧/规模/时间统计（6 维）；前缀长度和剩余生成额度（2 维）。target 信号只来自已经计算、已经提交的前缀，没有额外 target forward，也不读取本轮尚未验证的 target logits。

策略是 CPU 上的 `84 → 64 → 64`、Tanh MLP，输出独立 categorical heads 和一个 value head：

| 控制量 | 可选值 |
| --- | --- |
| 总节点预算 B | 30、45、60、90、120 |
| 最大深度 D | 4、8、12、15 |
| 每层保留配额 m | 1、2、4、8、12、16、24、32 |
| 每个父节点的候选宽度 w | 1、2、4、8 |

从根开始逐层构建：上一层保留的每个父节点，与本层 top-w token 组合；按累计 draft log-prob 排序；保留不超过该层配额、剩余总预算和实际候选数的节点。只有已保留节点继续扩展。固定并列规则为 token 路径字典序，保证前缀闭合；根不计入 B。代码用同层等价的累计原始 logit 排序，避免减去共同归一化项时破坏精确并列。

B/D/m/w 都是上限，不承诺实际填满。深度可能因预算用完而提前结束。策略不选择 token 本身，不能突破单块 15 层或加入跨 block continuation。数学证明的通用合法空间较大；上述是本实验选用的有限子空间。

共享 `TargetTreeVerifier` 一次计算 anchor + 所有节点；ancestor-only mask、最长 greedy 匹配、bonus token 和 KV 压缩保持不变。输出按 EOS/剩余长度截断；若继续，bonus 作为下一轮尚未缓存的 anchor。

## 3. 学习目标与反事实数据

### 同状态反事实预训练

只在训练 prompt 上用固定 DDTree-60 的轨迹采集状态。相同 draft logits 构造 DDTree-60/200/400，以及 8 个随机分层动作；每棵树在同一 target 缓存的独立副本上验证，随机候选顺序、每项重复 3 次。复制缓存和文件 I/O 在离线计时之外；候选标签取 `draft + build + verify` 的中位耗时。该离线标签**不含策略/特征提取开销，不可当成在线加速比**。

DDTree-60 是每个状态的配对 baseline。预训练选取分层候选中使 `(C-C0) - rho*(L-L0)` 最大的动作，`rho` 由训练反事实 baseline 的总提交量/总时间计算。C 为按 EOS/额度截断后的实际提交数。最佳分层候选仍可能比 DDTree 差；不会伪造一个“总能改进”的标签。

每组反事实保存实际 token、接受/提交/节点数和延迟。所有候选提交内容必须是同一 greedy 后续的前缀。`sampled_best_committed_ratio` 仅是**已采样树的逐状态事后最好接受量诊断**，不是全局 oracle 天花板、不是可部署策略、更不是吞吐加速；DDTree-200/400 也仅用于诊断。

固定 B/D/m/w 的四个消融分别采集受限动作空间；其余版本共享完整动作空间的反事实数据以控制训练材料。训练 3 个随机种子共享该离线语料，因此跨种子差异不包含独立重采反事实数据的方差。

### 在线强化学习

使用**完整 episode 的 Monte Carlo actor-critic（REINFORCE + learned value baseline）**，不是 PPO，不使用截断重要性比或 replay 旧轨迹。每个 prompt 完整生成一次后更新；训练采样动作，验证/测试各 head 取 argmax。超过所选 D 的 head 被边缘化、不计入 log-prob。

主方法每轮 reward 为负延迟（秒），prefill、同步、Python 统计等外层剩余时间计入最后一轮，保证一个 episode 的 reward 总和恰好等于负端到端时间。`gamma=1`，actor 损失按 episode 内轮次求和而非除以可变轮次数；value 使用 MSE。默认 Adam、lr=0.0003、value 系数 0.5、梯度范数裁剪 1。预训练 3 epoch，在线 RL 2 epoch。

这与反事实预训练的配对 surrogate 是两个不同阶段：**在线阶段不是继续使用局部配对比值 reward**。在同一 prompt 的 greedy 输出完全一致、输出总数不因策略变化时，最小化整套 prompt 的总生成时间与最大化聚合吞吐目标一致。策略推理、特征提取、构树和同步都在在线 reward 内；优化器更新、checkpoint I/O 不属于部署解码延迟。

## 4. 主实验与消融

所有方法使用同一个原始 Qwen3-4B target、原始 DFlash-b16 draft、chat 模板、BF16、eager attention、生成限制和 EOS 规则；TF32/CUDA Graph/torch.compile 均关闭。模型 revision 固定在配置中，本地同名微调目录不会覆盖原始模型。树方法共用现有 eager 验证器。

| 方法 ID | 含义 |
| --- | --- |
| ar | 同 target 的逐 token greedy |
| dflash | 仓库内 DFlash 单链实现、标准 causal 验证 |
| ddtree | 原有 best-first 构树，**固定 B=60** |
| acceptance_budget_control | 仅历史接受量控制 B；初始 60，之后 `clip(int(30+30*A/15),30,60)` |
| static_layered | 无策略网络，B=60、D=15、各层 m=w=4 |
| full | 本次完整分层结构策略 |
| fixed_budget | 独立训练，B 固定为 60 |
| fixed_depth | 独立训练，D 固定为 15 |
| fixed_quotas | 独立训练，各层 m 固定为 4 |
| fixed_width | 独立训练，各层 w 固定为 4 |
| no_target | 独立训练，target 16 维置零 |
| no_history | 独立训练，历史 6 维置零 |
| draft_only | 独立训练，上述 22 维全置零；仍保留长度/剩余额度 |
| acceptance_reward | 独立训练，在线 reward 只奖励未截断接受数 |
| local_ratio_reward | 独立训练，在线 reward 为当轮提交量/当轮耗时 |
| no_pretrain | 从随机初始化直接在线 RL |
| no_online_rl | 仅反事实监督预训练，不做在线 RL |

`full` 加 11 个消融，共 12 个策略版本 × 3 个种子 = 36 个独立 checkpoint。不能把 full 的 checkpoint 测试时临时关掉某项当作独立重训消融。信息消融仍计算共享特征后置零，因此测的是信息贡献，不是删除特征计算的工程加速。

`acceptance_budget_control` 是明确命名的简单控制组，**不是旧的 latency-aware AdaptiveTree**。本入口绕过旧通用 engine 的自动预算缩放，防止把“DDTree-60”实际跑成动态 30–60。

这些是共享后端下的**受控移植实现**，不是未经修改的官方 runtime；不得写成官方程序的原样测速，也不得把此处的公共 eager 验证器当作新增 GBV 验证算法。

## 5. 全量数据与隔离

| 用途 | 数据来源/配置 | split | 强制核对的原始题数 |
| --- | --- | --- | ---: |
| 训练源 | openai/gsm8k，main | train | 7,473 |
| 训练源 | EleutherAI/hendrycks_math，全部 7 科 | train | 7,500 |
| 训练源 | google-research-datasets/mbpp，full | train | 374 |
| 主测试 | openai/gsm8k，main | test | 1,319 |
| 主测试 | HuggingFaceH4/MATH-500 | test | 500 |
| 主测试 | openai/openai_humaneval | test | 164 |
| 主测试 | google-research-datasets/mbpp，full | test | 500 |
| 主测试 | MathArena/aime_2025 | train（仅作测试，见下文） | 30 |
| 主测试 | livecodebench/code_generation_lite，release_v6 | test | 1,055 |
| 主测试 | lm-sys/FastChat，MT-Bench 固定题库 | 完整 question.jsonl | 80 组 × 2 轮 |
| 独立附加测试 | google-research-datasets/mbpp，sanitized | test | 257 |

MBPP 的数量以当前仓库 README 元数据中的 splits 为准：full test=500、sanitized test=257；974/427 是整个配置的规模，不能把 train/dev 混进去称为“全量测试”。[MBPP split 元数据](https://huggingface.co/datasets/google-research-datasets/mbpp/blob/main/README.md)。其余来源：[GSM8K](https://huggingface.co/datasets/openai/gsm8k)、[MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500)、[HumanEval](https://huggingface.co/datasets/openai/openai_humaneval)、[MATH 训练集](https://huggingface.co/datasets/EleutherAI/hendrycks_math)。

新增数据的固定范围：

- **AIME25**：AIME I/II 合计 30 题。源仓库唯一 split 名为 `train`，本实验显式标记 `usage=evaluation_only`，只进入测试；训练源白名单仍只有前述三种，不能因上游 split 名称而把 AIME25 加入训练。[数据来源](https://huggingface.co/datasets/MathArena/aime_2025)。
- **LiveCodeBench**：代码生成任务固定为 `release_v6`，覆盖 2023-05 至 2025-04 的全部 1,055 题，不再按日期、难度或平台筛选。严格读取同一数据 commit 下的 `test.jsonl` 至 `test6.jsonl`，不使用会变化的 `release_latest`，也不执行远端数据集 Python 脚本。使用官方默认的 `code_generation_lite`：题目覆盖完整，但不声称包含非 lite 版的所有私有测试用例。[官方版本与 lite 说明](https://github.com/LiveCodeBench/LiveCodeBench#dataset-versions)。
- **MT-Bench**：使用官方 `question.jsonl`，固定 FastChat commit `587d5cfa1609a43d192cedb8441cac3c17db105d`，完整 80 组双轮对话。第一轮生成后，将本方法自己的回答作为 assistant 消息，再追加原始第二轮 user 问题；不拼接两轮 user 文本，不注入参考答案或借用 AR 的回答。[固定题库](https://github.com/lm-sys/FastChat/blob/587d5cfa1609a43d192cedb8441cac3c17db105d/fastchat/llm_judge/data/mt_bench/question.jsonl)。

主测试现为 **7 个数据集、3,648 条题目/对话记录**；附加 sanitized 单独列行后，共 **3,905 条记录、3,985 个生成轮次**（MT-Bench 每组两轮），不宣称 full/sanitized 彼此不重叠。没有失败过滤、匹配子集统计或 2K prompt 上限。`max_new_tokens=2048` 是每个回答轮次的输出 token 上限，不是题目数；长题可能达到该上限，不能将全题量覆盖解释为答案不限长度。

训练源合计 15,347 题，在任何采集/拟合之前完成：

1. 固定数据仓库 commit SHA；以这个 revision 下载完整指定 split。
2. 用 NFKC、大小写和空白归一化后的 prompt 与原始问题哈希查重；移除训练源中与任何测试集相同的题目、与 MT-Bench 任一轮 user 问题相同的题目，以及 MBPP 跨配置共享测试 task ID。
3. 对训练源内部按 prompt 去重，同组不能跨训练/验证。
4. 对保留 prompt 哈希固定划分约 90% train、10% dev。全部保留题目进入其中之一，实际数量在 manifest 中公开；没有为了快而再采样训练集。
5. 只在 train 采反事实和更新权重，只在 dev 选择最低总延迟 checkpoint。测试集不参与 reward 拟合、早停、选 checkpoint 或超参数搜索。

每次启动重算文件哈希、数量、来源、prompt/reference 一致性和 train/dev/test 交集；训练/验证行必须来自白名单中的官方训练源。测试角色不单靠源 split 是否叫 `test` 判断，而是核对数据集、来源、配置、split 和 `usage`。官方参考答案只作为独立审计字段保存，不喂给策略或 target。LiveCodeBench 处理后的记录仅保存公开题面、starter code 和来源元数据，不将大型私有测试载入训练/推理记录，也不反序列化其 pickle；完整原始文件留在固定 revision 的 Hugging Face 缓存，后续如需 grader 应按该 revision 独立加载。

这里的去污染是**显式文本/ID 去重**，不等于语义近重复检测，也不能保证基础模型预训练从未见过这些题。论文必须保留这一限制。不要查看 test 加速比后继续用同一 test 调参，再将其当成未触碰的最终评测。

## 6. 计时、统计与输出

默认训练种子 17/29/43；每个种子下，每题每种方法重复 3 次。每题内方法顺序和 prompt 顺序随机化；每次生成新建 cache，不复用其他方法已生成的 KV。独立合成 prompt 预热 2 轮，不用测试题预热。

MT-Bench 每轮均从该方法自己的完整 chat history 冷启动 prefill，使用相同 Qwen chat 模板；不跨回答轮次复用 KV。两个回答的生成时间和 token 数求和，第二轮不是免费计算。保存每轮输入 token 哈希、输出 token、耗时和指标；必须逐轮与 AR 对应，不能只拼平 token 后比较。报告区分“80 组对话”和“160 个回答轮次”，bootstrap 以完整对话为配对单位，不能拆散有依赖的两轮。

主指标为总生成 token / 总端到端时间，加速比为相同 prompt、相同重复次数下 AR 总时间 / 方法总时间；同时报告相对 DDTree。计时含 prefill、draft、策略/特征、构树、target 验证、KV 处理、同步和推理循环记账，不含模型加载、tokenization、最终文本 decode、离线采集或磁盘写入。因此它是单请求 batch=1 的生成墙钟吞吐，不是 SGLang serving/QPS 结果。

每题每方法每重复的**完整 token ID 序列**必须与 AR 相同。任何 mismatch、缺行、缺方法、缺重复、无效延迟、哈希变化或上游 checkpoint 身份变化，立即失败。禁止只对成功匹配的子集报加速。理论实数等价性不能代替 BF16 GPU 数值检查；失败后需单独排查并重新冻结实验，不能只替某个方法换精度/backend。

输出在 `outputs/paper_t0_full/`：

- `contract.json` / `gpu_preflight.json`：代码指纹、配置、数据、模型 revision、软件版本、GPU/驱动。
- `counterfactual/*/prompts/`：原始反事实状态/标签；`complete.json` 含完整哈希与诊断。
- `training/<seed>/<variant>/`：`latest.pt`、仅 dev 选出的 `best.pt`、各轮 dev 分数、逐训练 prompt 学习曲线及完成记录。
- `evaluation/<seed>/prompts/`：逐题/方法/重复原始时间、token、动作、节点与显存记录。
- `tables.json` / `tables.csv` / `tables.md`：每数据集、每种子的全部结果。
- `comparison.md`：各数据集并列、方法分行，展示跨种子的平均 AR 加速比 ± 样本标准差。

`tables.json` 另含每个种子内按 prompt 配对的 2,000 次 bootstrap 95% 区间；它不同于跨种子标准差，不能混写。吞吐采用比值的总量统计，不平均逐题加速比。GPU 显存为 PyTorch peak allocated，包括同进程共享驻留模型，不是 target-only 的整机最低显存。

本流程检查无损 token 等价性，**不输出伪造的任务准确率、pass@1 或 MT-Bench judge 分数**。新增三者仍用于统一 T=0 条件下的吞吐/等价性实验，不冒充官方默认温度与评分流程。数学答案评分、执行生成代码、付费 LLM judge 都不在这里自动开展；若论文需要质量分，应对保存输出使用另行核验且隔离执行的 grader，并明确 prompt/停止/提取规则。

## 7. 运行与恢复

在新的 Python 3.10/3.11 环境，先根据服务器驱动安装合适的 PyTorch CUDA wheel，再执行：

```bash
python -m pip install -r requirements-paper.txt
python -m pip install -e . --no-deps
export CUDA_VISIBLE_DEVICES=0
bash scripts/run_paper_t0_full.sh plan
bash scripts/run_paper_t0_full.sh doctor
```

如需指定解释器，可设 `PAPER_PYTHON=/absolute/path/to/venv/bin/python`。`plan` 不下载或训练；`doctor` 只检查环境。默认 CUDA:0 指可见设备中的第一张卡。不要与别的训练任务共享 GPU；本工具的设备锁只能阻止本工具自身的并发，不能阻止外部程序抢 GPU。

先准备数据并运行明确标记、独立目录的 smoke：

```bash
bash scripts/run_paper_t0_full.sh prepare
bash scripts/run_paper_t0_full.sh all --smoke-count 2 --run-dir outputs/paper_t0_smoke
```

smoke 仍先核验全部原始 splits，之后才取小子集联调；结果标记 `publication_eligible=false`，不能复用进正式目录。正式配置下可以分阶段运行：

```bash
bash scripts/run_paper_t0_full.sh collect
# 先阅读 counterfactual/*/complete.json 的候选空间诊断，再决定是否值得训练。
bash scripts/run_paper_t0_full.sh train
bash scripts/run_paper_t0_full.sh evaluate
bash scripts/run_paper_t0_full.sh summarize
```

也可以显式 `all` 连续执行，但不会承诺候选空间足够好。**完整默认矩阵需要 36 个策略 checkpoint、609,705 次测试生成调用**（包含 MT-Bench 两轮），还不含反事实采集、训练和验证；它不是快速预实验。先在服务器 smoke 中估算时间/存储预算，再启动全量；代码不会自动少跑数据。

重新运行同一阶段会按完整 prompt 恢复；采集和评测按题原子写入，训练每题保存 policy/optimizer/采样 RNG。预训练中途崩溃会从该阶段起点重新执行。代码、数据、配置、模型、环境或上游反事实/权重改变时拒绝混用旧目录，必须新建 run。不要通过删除契约或失败记录绕过检查。

本次扩展后的数据 manifest / 实验协议版本为 2：新增测试题也参与训练源去污染检查，所以旧四集 manifest 和旧结果不能直接续写成七集实验。若已有旧目录，请使用新的 `--data-dir datasets/paper_t0_seven_full --run-dir outputs/paper_t0_seven_full`；旧文件不会被覆盖。

## 8. 验收边界

本地测试覆盖逐层构树独立参考、并列概率、非法动作、未激活 head、信息消融、整段延迟 reward、真实 DynamicCache 的小型 causal attention 模型、多轮/EOS/长度边界、反事实缓存隔离，以及采集→预训练→RL→选模→评测→汇总/恢复全链路。

2026-09-03 本地验收：协议及扩展数据专项测试 **50 项通过**。用合成数据按实际全量数量检查了 `prepare → manifest → load_data`，核对 15,347 条训练源及 3,905 条评测记录；这不是下载真实全量题库的验收。小模型完整训练/评测测试也包含双轮 MT-Bench 形式，逐轮核对 token 和聚合计时。另用 Transformers 自带的随机初始化小型 `Qwen3ForCausalLM`、eager attention、真实 DynamicCache 和 FP64，对照 AR 检查了 DFlash、DDTree、两个控制组及随机结构策略的多轮输出；未下载预训练权重。测试环境为 Python 3.13.5、PyTorch 2.7.1 CPU、Transformers 4.57.1、huggingface-hub 0.36.0；依赖兼容修正仅安装在临时测试虚拟环境。这不替代正式 Python 3.10/3.11 + CUDA/BF16 验收。

运行本次相关测试：

```bash
PYTHONPATH=src python -m pytest tests/test_paper_protocol.py tests/test_paper_datasets_extended.py tests/test_engine.py tests/test_ddtree_builder.py tests/test_ddtree_integration.py tests/test_verification.py tests/test_dflash_adapter.py tests/test_vanilla_engine.py -o addopts='' -q
```

仓库还有独立的 `tests/gbv_paper/`，与根目录测试存在同名模块，应单独调用，不能把混合收集错误当作方法测试失败。根目录旧测试 `test_retrieve_indices_keeps_host_control_paths_on_cpu` 请求 CUDA tensor，但现有 `DraftTree.retrieve_indices(device=...)` 会遵循指定设备，在 CPU-only 本机失败；本交付不改动这个旧接口。新流程的验证器读取 CPU `retrieve_paths()`，不依赖该 CUDA 路径。

服务器仍必须实际通过：依赖导入、固定权重加载、完整数据校验、BF16 mask/KV 数值检查、smoke 全链路、无其他 GPU 作业条件下的正式全量评测。自动 `publication_eligible` 只表示程序的完整性/一致性检查通过，不表示无需人工检查实验设计或论文结论。
