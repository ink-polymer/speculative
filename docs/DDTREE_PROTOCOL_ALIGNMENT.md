# DDTree 官方评测对齐核对（已确认采用官方抽样）

最新要求是数据和评测也与 DDTree 一致。以下以官方固定提交 `c96427a185677bf4133ed865dd1626a5041aef9b` 为对齐对象，不能把此前受控七数据集协议直接称为“完全相同”。原版 Adaptive 构树算法的恢复与本项评测迁移是两个独立事项。

用户已明确选择“和官方完全一致”。当前入口已经采用右列官方设置；下表保留迁移前差异，避免再把旧全量配置当成官方协议。完整运行方法见 [当前实验说明](PAPER_T0_EXPERIMENTS.md)。

## 迁移前差异与本次对齐目标

| 项目 | 此前本项目协议 | DDTree 官方设置 |
|---|---|---|
| MBPP | full/test 500；sanitized 另列 | sanitized/test；直接用 prompt 字段 |
| MT-Bench | FastChat GitHub question.jsonl | HuggingFaceH4/mt_bench_prompts 的 train；prompt 字段作为两轮 turns |
| LiveCodeBench | 同六个数据文件，但自定义了提示语 | test.jsonl 至 test6.jsonl；必须逐字复用官方 format_lcb |
| 数据集集合 | 七套主数据＋sanitized 附表 | 运行脚本列出十套：另含 AIME24、SWE-bench Lite、Alpaca |
| 样本数 | 全量 | 多数上限 128；AIME24/25 各 30、HumanEval 164、MT-Bench 80 |
| 抽样 | 顺序种子打乱全量 | 超出上限时 dataset.shuffle(seed=0).select(range(max_samples)) |
| 主加速比 | AR 总墙钟时间 / 方法总墙钟时间 | 各次回答 decode TPOT 的均值之比 |
| 计时边界 | 含 prefill、首轮 draft | 去除 target prefill；推测方法首次 draft 后重置 decode_start |
| 固定 DDTree 对照 | B=60 及邻近预算 | 默认扫描 16/32/64/128/256/512/1024，表格选最佳预算 |
| Attention | Target/Draft 均 SDPA | draft 固定 FA2；target 分 SDPA 与 FA2 两批，树只在 SDPA 批运行 |
| 表格 baseline | 同批 AR | AR/DFlash 各自在 SDPA/FA2 两批中按均值 TPOT 取较优结果 |
| MT-Bench 后续上下文 | 各方法自己的前一轮回答 | 所有方法共享当前输入；下一轮采用最后一个推测方法的回答 |
| 预热与顺序 | 额外重复和随机顺序 | Warmup 文本、最多 16 token；固定 seed=0；按脚本顺序运行 |

官方 run_benchmark.sh 将输出上限设为 2,048；benchmark.py 单独调用的默认值却是 16,384。复现实验脚本时应采用前者，不能只抄 Python 默认参数。

官方的 AR 基线通过 `dflash_generate(block_size=1)` 执行，而不是本项目的独立 baseline_greedy。计时实现和汇总必须一起对齐，不能只把总耗时比改名成 TPOT。官方 T=0 benchmark 未逐题强制 token 一致性；本项目仍需保留独立的正确性审计，且不能将失败过滤后冒充完整实验。

## 已确认的处理

采用官方 seed=0 抽样，不保留全量作为默认正式评测。十套数据共 1,072 题/对话；包括 MT-Bench 双轮后每方法 1,152 次回答。默认保留官方三组模型和八进程，温度限定为本方法适用的 T=0。官方未锁定历史 Hugging Face revision，因此本项目固定本次复现实验快照，不能凭空保证与作者历史数据快照逐字一致。

默认入口已切换到 official.py、official_data.py、official_worker.py 和 official_reporting.py；原版 Adaptive 构树器本身保持不变。新增的是原版 Adaptive 与消融、来源锁定和失败即停的审计。GPU 尚未运行，不能宣称已有新结果。

原始依据：[数据处理](https://github.com/liranringel/ddtree/blob/c96427a185677bf4133ed865dd1626a5041aef9b/model/utils.py)、[benchmark](https://github.com/liranringel/ddtree/blob/c96427a185677bf4133ed865dd1626a5041aef9b/benchmark.py)、[运行脚本](https://github.com/liranringel/ddtree/blob/c96427a185677bf4133ed865dd1626a5041aef9b/run_benchmark.sh)、[汇总口径](https://github.com/liranringel/ddtree/blob/c96427a185677bf4133ed865dd1626a5041aef9b/make_latex_table.py)。已通过原始 GitHub 提交重新核对数据处理、benchmark、DDTree 生成及表格脚本，不仅依赖本地文件的“official”名称。
