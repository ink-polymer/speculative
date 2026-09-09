# AdaptiveTree / DDTree / DFlash 正式实验矩阵

本文件是本轮新服务器实验的唯一范围合同。机器可读版本为
`configs/formal_experiment_matrix.json`。实验分为两个不可混合的协议：

1. **T=0 主实验**：正式修正版 AdaptiveTree、DDTree、DFlash 与消融；
2. **T=1 随机基线**：只比较已经支持随机采样的 Target、DFlash、DDTree。

本轮不运行树状块验证，也不把 `ddtree_lazy_projection` 改名成树状块验证。
AdaptiveTree 当前只注册在 T=0，因此 T=1 表不能出现 AdaptiveTree 列。

## 1. 模型范围

| 层级 | Target / Draft | 状态 |
|---|---|---|
| 主矩阵 | Qwen3-4B / Qwen3-4B-DFlash-b16 | 必跑，revision 固定 |
| 主矩阵 | Qwen3-8B / Qwen3-8B-DFlash-b16 | 必跑，revision 固定 |
| 扩展 | Qwen3-Coder-30B-A3B-Instruct / 对应 DFlash | 本轮不跑 |

30B 不是“漏跑”：单张 H20 尚未通过同时装载 Target/Draft 的显存预检，当前
MoE 路径也未完成相同实现审计。正式结论只能覆盖 4B 与 8B；今后在合适硬件上
完成预检后，30B 应使用新的独立运行目录与合同。

## 2. T=0 正式主实验

### 数据与规模

沿用冻结 DDTree 官方抽样：GSM8K 128、MATH-500 128、AIME24 30、AIME25
30、HumanEval 164、MBPP sanitized 128、LiveCodeBench 128、SWE-bench Lite
128、MT-Bench 80、Alpaca 128。共 1,072 个问题/对话；MT-Bench 保留两轮，
所以每个方法、每个模型共 1,152 次回答。

### 方法

| 角色 | 注册键 | 定义 |
|---|---|---|
| Target | `baseline` | Target-only |
| DFlash | `dflash` | 官方 DFlash |
| DDTree | `ddtree_tb16` … `ddtree_tb1024` | B=16/32/64/128/256/512/1024 |
| 正式主方法 | `adaptive` | 修正成本归因、候选预算扩至 B=256、关闭周期探索 |
| 历史对照 | `adaptive_legacy` | 修改前的旧 AdaptiveTree |
| 预算对照 | `adaptive_b128` | 修正成本归因，但预算上限保持 B=128 |
| 成本归因消融 | `adaptive_legacy_cost_attribution` | 保持 B≤256 和关闭探索，只恢复旧计时归因 |
| 探索对照 | `adaptive_with_exploration` | 新主方法恢复周期探索 |
| 接受率消融 | `adaptive_no_acceptance_calibration` | 在新主方法上移除接受率校准 |
| 延迟消融 | `adaptive_no_latency` | 在新主方法上移除实测延迟项 |
| 在线更新消融 | `adaptive_frozen_after_warmup` | 在新主方法上预热后冻结估计 |

`no_exploration` 不再作为主方法旁边的消融键，因为关闭探索已经是新
`adaptive` 的组成部分；探索的作用由 `adaptive_with_exploration` 做反向对照。

### 公平条件

- T=0、BF16、thinking 关闭、最多 2,048 新 token；数据抽样 seed 与生成 seed
  均为 0。
- Draft 固定 FlashAttention 2。Target 分别执行 SDPA 与 FA2；架构主表只使用
  同一 Target=SDPA 后端，跨后端择优表只作辅助结果。
- 固定 DDTree、DFlash、Target 与全部 AdaptiveTree 变体共享同一模型 revision、
  输入、停止规则、计时边界和样本。
- 正式计时使用 `balanced-rotation`，使各方法在每个执行位置上的次数差不超过 1。
- 在 BF16 执行中观察到的贪心 token 分歧使用 `record-bf16-mismatches`
  保留全部样本并报告逐方法精确输出率；该名称只描述执行精度，不断言分歧成因。
  只要存在分歧，就不能声称严格无损，也不能只筛选相同输出样本后报告总体加速比。
  若 MT-Bench 首轮历史方法的输出跨后端不同，第二轮上下文差异会单独计数；此时
  跨后端择优表不是同上下文配对，只能作辅助结果。公平架构主表始终使用同一个
  SDPA run，因而 Target、DFlash、DDTree 与全部 AdaptiveTree 方法仍共享逐轮输入。
- 主表保留上游“一次完整 pass”的定义。它包含每方法每模型 1,152 个配对回答；
  不把重复的确定性 seed 假装成统计重复。若另外做硬件稳定性复验，应重新初始化
  控制器并使用独立运行目录，逐次报告，不与主表静默合并。
- 因此本轮 T=0 只报告描述性点估计，不计算置信区间；T=1 的配对聚簇 bootstrap
  不能移用于 T=0，也不能用来包装确定性重复。

T=0 两模型合计生成调用数为 43,776：每轮 SDPA 有 2 个基础方法、7 个固定
DDTree 和 8 个 AdaptiveTree/消融方法；FA2 只运行 Target 与 DFlash。

## 3. T=1 随机采样基线

T=1 是独立的随机采样稳健性基线，不是 AdaptiveTree 的跨温度实验。

| 项目 | 冻结设置 |
|---|---|
| 模型 | 同一 4B/8B revision 对 |
| 方法 | Target、DFlash、DDTree |
| 温度/提议 | Target=1.0；DFlash Draft 保持 greedy argmax；DDTree 构树温度=1.0 |
| 后端 | Target=SDPA、Draft=SDPA |
| 数据 | 8 个已注册数据集，DDTree 官方数量 |
| 随机重复 | seed 17、29、43 |
| 顺序 | `balanced_rotation`，seed 20260909 |
| 统计 | 与同模型/数据/样本/seed 的 Target 成对；按 source_id 聚簇，10,000 次 bootstrap |

八个数据集为 GSM8K、MATH-500、AIME24、AIME25、HumanEval、MBPP sanitized、
LiveCodeBench、MT-Bench，共 816 个问题/对话、896 个生成轮次。三种方法、三颗
随机 seed、两个模型合计 16,128 次生成。

SWE-bench 与 Alpaca 没有被“忘记”：当前 T=1 质量评测管线没有注册这两项。
SWE-bench 需要独立的软件仓库执行环境，Alpaca 需要外部主观评审；在评分合同
完成前加入速度表会制造不完整的质量对照。它们仍保留在 T=0 官方速度矩阵中。

这里的 DFlash `draft_temperature` 必须为空：实现固定从 masked-block logits 取
argmax，再用 T=1 Target 分布做标准 matching verification；该字段即使误填为 1
也不会改变当前实现，但会错误描述方法。DDTree 才使用 T=1 Draft 概率分布构树。

随机采样即使使用相同 seed，也不要求不同算法逐 token 完全相同。T=1 禁止使用
greedy exact-match 作为通过条件；应报告吞吐、接受长度、任务质量、生成长度，
以及与 Target 成对的置信区间。三颗 seed 是输出分布重复，不是三个独立 GPU
硬件重复；bootstrap 必须把同一 source_id 的所有 seed 放在同一个簇中。

冻结配置中的评分后端默认值是 Docker；当前服务器可显式选择 `process`。两者都必须
先通过真实评分器自检，最终报告会把实际后端与 doctor、运行清单和评分清单三方绑定。
Docker 绑定不可变镜像 ID；process 分别绑定不解引用的虚拟环境调用路径、真实二进制
路径与 SHA-256、`pyvenv.cfg`，以及实际子解释器的 Python/NumPy/SymPy 版本。若编排
进程是 root，process 评分必须使用专用非 root 执行身份，不能在 root 权限下运行参赛代码。

## 4. 明确延期的内容

- **树状块验证**：本轮不运行、不出表、不作论文结论。`ddtree_lazy_projection`
  只能作为 DDTree 实现优化对照，不能用作树状块验证的别名。
- **AdaptiveTree at T=1**：当前生成器明确只支持 T=0。若未来扩展，需要新的随机
  接受/残差算法和分布正确性测试，不能只解除温度检查。
- **30B**：等待更大显存或经过验证的模型并行方案及 MoE 兼容性预检。
- **跨协议总加速比**：T=0 与 T=1 的数据、随机性和 Draft 后端不同，禁止把速度
  求平均或合并成一个数字。

## 5. 执行前审计

任何正式运行目录创建前，都应先执行：

```bash
python scripts/audit_formal_experiment_matrix.py \
  --matrix configs/formal_experiment_matrix.json \
  --output outputs/formal-matrix-audit.json

PYTHONPATH=src pytest -q tests/gbv_paper/test_formal_experiment_matrix.py
```

审计会检查 canonical 方法名、B=256 候选、模型 revision、数据数量、后端、三颗
T=1 seed、均衡顺序、聚簇 bootstrap 和延期声明。任一项不一致即返回非零状态，
禁止以“完整正式实验”名义启动。

T=1 的两份冻结配置为：

- `configs/adaptive_block_qwen3_4b.json`
- `configs/adaptive_block_qwen3_8b.json`

完整注册范围共 59,904 次生成调用。这个数字不包括 warmup、失败后从组头重跑，
也不包括明确延期的树状块验证或 30B。
