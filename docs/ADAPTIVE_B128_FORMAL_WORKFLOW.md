# 动态 Adaptive B128 正式实验全流程

## 1. 唯一正式身份

- 主方法：`adaptive_b128`，method schema v4，架构 `guarded_raw_prefix_v7`。
- 候选预算：30/45/60/80/100/128；B128 是安全臂；EMA α=0.2。
- 动态性：先运行安全臂，利用其已接受路径反事实估计较小前缀；只有乐观成本界仍可能获益时才探测一个候选，并继续在线更新。
- 成本口径：Draft 是共享固定成本；tree build、tree compile、Target verify 和 KV/commit 全部计入所选预算。
- 每 32 轮重新评估一次；其余轮次复用安全决策。控制器仍可在证据满足门槛时切换预算，不是固定 B128。
- `adaptive_b256` 仅为扩大候选上限的单因素消融；旧组合逻辑只以 `adaptive_legacy` 历史对照出现。

旧 schema v2/v3、旧 `adaptive`/B256 主方法及任何旧服务器计时都不得改名或导入本轮。

## 2. 正式矩阵

### T=0：主实验与消融

- 模型：固定 revision 的 Qwen3-4B 与 Qwen3-8B Target/DFlash Draft 对。
- 数据：GSM8K 128、MATH-500 128、AIME24 30、AIME25 30、HumanEval 164、MBPP 128、LiveCodeBench 128、SWE-bench 128、MT-Bench 80×2 轮、Alpaca 128。
- 每模型每方法 1,152 个回答；最大新 token 2,048；T=0；BF16；seed=0。
- 同后端主表：Target、DFlash、固定 DDTree B128（官方实现）、`adaptive_b128` 和全部消融均以 Target=SDPA 比较；Draft 固定 FA2。
- FA2 Target 的 Target/DFlash 结果仅供上游“最佳后端”辅助表，不能进入架构主结论。
- 方法按数据集做 balanced rotation，避免固定执行位置偏差。

消融以 `adaptive_b128` 为锚点：

| 键 | 只改变什么 |
|---|---|
| `adaptive_b256` | 候选预算上限从 128 扩至 256 |
| `adaptive_legacy_cost_attribution` | 恢复旧成本分区 |
| `adaptive_with_exploration` | 恢复每 64 轮周期探索 |
| `adaptive_no_acceptance_calibration` | 关闭接受率校准 |
| `adaptive_no_latency` | 去掉预算间实测延迟判别 |
| `adaptive_frozen_after_warmup` | 预热后冻结在线估计 |
| `adaptive_legacy` | 修改前 B128 组合逻辑；只作历史控制，不作单因素结论 |

### T=1：独立随机基线

只运行 Target、DFlash、DDTree-B45，覆盖同一 4B/8B 模型、8 个支持的数据集和 seed 17/29/43。三者共享 SDPA/SDPA、T=1、样本、顺序、长度上限和评分器；概率计算使用 FP64。当前 AdaptiveTree 没有注册 T=1 采样律，所以不得把 `adaptive_b128` 放进 T=1 表，也不得把 T=0/T=1 加速比合并。

树状块验证和 30B 在本轮均为 `deferred_not_run`。

## 3. 新服务器执行状态机

每台新 GPU 都使用全新的源码目录和输出目录。推荐：

```bash
export INTEGRATED_PYTHON=/root/autodl-tmp/envs/speculative/bin/python
export INTEGRATED_RUN_DIR=/root/autodl-tmp/outputs/adaptive-b128-t0-t1-h20-001
export ADAPTIVE_DATA_DIR=/root/autodl-tmp/data/adaptive-t0-formal
export SAMPLING_DATA_DIR=/root/autodl-tmp/data/sampling-t1-formal
export CODE_BACKEND=process
export GBV_PROCESS_PYTHON=/opt/gbv-code-eval/bin/python
```

按以下顺序执行：

1. `bash scripts/run_integrated_fresh_server.sh plan`：确认 2 个模型、T=0/T=1 范围、46,080 次 generation call 及 B128 primary。
2. `bash scripts/run_integrated_fresh_server.sh audit`：静态核对方法角色、revision、数据量、后端、随机律和禁止声明。
3. `bash scripts/run_integrated_fresh_server.sh doctor`：核对 GPU UUID、CUDA/FA2/C++、模型兼容性、评分沙箱和分布律测试；GPU 有外来计算进程时失败。
4. `bash scripts/run_integrated_fresh_server.sh start`：再次执行 matrix audit、fairness audit 和 doctor 后才后台启动。它先准备/锁定数据，再完成 T=1 数据与答案审计、4B/8B 真实模型预检、T=0 双后端全方法 smoke，全部通过后才进入正式计时。
5. `bash scripts/run_integrated_fresh_server.sh status`：查看 PID、退出状态和日志尾部。
6. 断连不影响后台进程。若机器或进程异常退出且 contract、源码、数据、环境、GPU UUID 都未变化，执行 `resume`；任一身份变化必须换新输出目录，不能硬续跑。

启动顺序固定为预检 → T=0 4B → T=0 8B → T=1 4B → T=1 8B → 重算并汇总。单卡上串行运行，避免模型并发和显存争抢。

## 4. 发布门禁

完成标志只在下列条件全部满足后生成：

- 每个模型、数据集、后端、seed 和方法记录数完整，无重复、无跳题；
- 原始 artifact、completion marker、数据、配置、源码和环境 SHA-256 全部匹配；
- GPU UUID、Python/CUDA/包版本和评分器身份与 doctor 一致；
- 每个数据集内方法计时位置次数最大差不超过 1；
- T=0 主表只使用同一 SDPA Target 后端并报告逐方法 exact-output rate；
- T=1 质量、长度、接受 token 与配对聚簇 bootstrap 由原始记录重算；
- 树状块验证无结果、T=0/T=1 无合并统计、旧服务器结果导入为 false。

T=0 当前采用 `record-bf16-mismatches`：所有 BF16 token 分歧会保留并计数，不删样本，但因此不得声称“严格无损”。若论文需要严格无损结论，应另建新 contract 用 `strict` 复验；不能用事后筛选的一致子集代替总体结论。

## 5. 结果读取

- `adaptive/<model>/tables.json`：T=0 同后端主表、消融、预算使用与数值差异。
- `sampling_t1/<model>/report/summary.json`：T=1 性能与任务质量。
- `report/integrated_results.json` / `.md`：最终分协议汇总。
- `completed.json`：仅表示所有门禁和重算完成，不代表严格无损门通过。

比较 `adaptive_b128` 与 DDTree 时，应分别报告各数据集和模型的 TPOT、相对 Target 加速、相对该数据集最佳固定 DDTree 加速、接受长度、预算分布和 exact-output rate；不能只挑赢的数据集，也不能把不同后端的最优值拼成主表。
