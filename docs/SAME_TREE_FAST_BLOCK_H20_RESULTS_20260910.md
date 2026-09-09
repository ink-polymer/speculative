# H20 同树快速块验证结果（2026-09-10）

## 结论与边界

修正 DFlash 元数据后的最终资格试跑 `r2` 通过预先声明的严格速度门槛：树状块验证相对 DDTree 为 **1.02527×**，95% CI 为 **[1.00731, 1.04353]**；相对 DFlash 为 **1.19706×**，95% CI 为 **[1.16337, 1.23211]**。DDTree 相对 DFlash 也为正，**1.16756× [1.13370, 1.20289]**。

这些归档行按 r2 方法标签计算后通过了两个严格速度门槛；但由于下文说明的
旧 observer 路由证据局限，r2 本身不能独立把这些速度行升格为“已绑定实际 fused callable”
的证据。它**不是完整正式实验**，也不是 H200 结果。完整的 4B/8B、8 数据集、2048 token、
质量评测矩阵尚需在新服务器从头运行；正式矩阵完成前 `claim_tree_block_results_from_pilots` 和
`claim_tree_block_results_before_completed_validated_full_matrix` 均保持 `false`。

## 最终 r2 实验设计

| 项目 | 固定设置 |
|---|---|
| GPU | NVIDIA H20，约 102 GB |
| Target | `Qwen/Qwen3-4B`，revision `1cfa9a7208912126459214e8b04321603b3df60c` |
| Draft | `z-lab/Qwen3-4B-DFlash-b16`，revision `b74e3a329c4d963783143b1e970d95b002be72bd` |
| 精度与后端 | Target/Draft BF16；SDPA/SDPA；TF32 关闭；概率计算 FP64 |
| 温度 | Target T=1；DDTree/树状块 Draft T=1；DFlash Draft 为官方贪心 argmax |
| DDTree 结构 | L=15，B=45 Draft 节点，46 个 Target 概率行 |
| 数据 | 7 个官方数据集，每个固定前 8 条，共 56 个 source |
| 重复 | 3 个生成种子：17、29、43 |
| 配对规模 | 168 个 source-seed 组；每组 3 种方法；共 504 条记录 |
| 生成长度 | 最多 256 个新 token |
| 执行顺序 | 每数据集、每方法在位置 0/1/2 各出现 8 次 |
| 主统计量 | 先对同一 source 的 3 个种子求 log-speedup 均值，再对数据集等权几何平均 |
| 置信区间 | source 聚类 bootstrap，10,000 次 |

### 方法身份

| 显示名称 | 内部方法 | Draft 提议 | 验证方式 |
|---|---|---|---|
| DFlash | `dflash` | 官方贪心 masked block | 官方 DFlash 路径 |
| DDTree | `ddtree` | T=1 probability tree | 对全部 Target 行批量 multinomial |
| 树状块验证 | `ddtree_fused_scan` | 与 DDTree 完全相同的 T=1 probability tree | 单个持久 CUDA block，只扫描实际到达的 FP64 行 |

候选方法在实验设计上只替换 DDTree 的验证器。r2 归档记录显示两者准备的父节点、
树 token 和整个 `[46, 151936]` FP64 Target 概率张量相同。但 r2 的旧 observer 在 verifier
调用前触发，所以这份归档只能证明所记录的同树输入和拟调用路由，不能独立证明
fused callable 实际被调用并成功返回。两种采样器实现同一祖先 categorical law，但 RNG
算法映射随机流的方式不同，因此同 seed 下不要求逐条生成路径一致。

## 最终 r2 汇总结果

| 方法 | 记录数 | Decode tokens | Decode 时间 (ms) | 吞吐 (tok/s) | 平均接受 Draft/轮 | 平均提交 token/轮 | 轮数 |
|---|---:|---:|---:|---:|---:|---:|---:|
| DFlash | 168 | 37,468 | 289,669.309 | 129.3475 | 3.3169 | 4.2461 | 8,824 |
| DDTree | 168 | 37,644 | 237,160.933 | 158.7277 | 4.8238 | 5.7218 | 6,579 |
| 树状块验证 | 168 | 37,627 | 231,000.779 | 162.8869 | 4.8148 | 5.7210 | 6,577 |

总吞吐仅用于描述。正式的速度判断使用逐 source 配对、数据集等权的几何平均和聚类 bootstrap，而不是直接用上表总吞吐相除。

### 严格配对速度门槛

| 比较 | 等权数据集几何平均 | 95% CI | CI 下界 > 1 | 结论 |
|---|---:|---:|---:|---|
| 树状块验证 / DDTree | 1.025267× | [1.007312, 1.043535] | 是 | 通过 |
| 树状块验证 / DFlash | 1.197062× | [1.163375, 1.232113] | 是 | 通过 |
| DDTree / DFlash | 1.167562× | [1.133701, 1.202890] | 是 | 通过 |

### 分数据集配对速度比

| 数据集 | 树状块 / DDTree | 树状块 / DFlash | DDTree / DFlash |
|---|---:|---:|---:|
| GSM8K | 1.024503× | 1.155398× | 1.127765× |
| MATH-500 | 1.020881× | 1.209284× | 1.184549× |
| AIME 2025 | 1.060527× | 1.230671× | 1.160433× |
| HumanEval | 1.004993× | 1.215397× | 1.209359× |
| MBPP | 1.015947× | 1.069089× | 1.052307× |
| LiveCodeBench | 1.031855× | 1.154056× | 1.118428× |
| MT-Bench | 1.019049× | 1.366019× | 1.340484× |

## 正确性与公平性证据

| 检查 | 规模或条件 | 结果 |
|---|---|---|
| 完整词表 inverse-CDF 对照 | 1,024 seeds，词表 151,936，最大深度 15 | 路径、token、bonus 全部一致；通过 |
| r2 同树输入记录 | 46×151,936 FP64 概率张量 | 父节点、树 token、Target 行完全一致；旧 observer 不独立证明成功调用路由 |
| 方法顺序平衡 | 7 数据集 × 3 方法 × 3 位置 | 每个 cell 8 条；通过 |
| 配对完整性 | 168 组 | 同 prompt、同 sampling seed、每组 3 方法；通过 |
| 官方精度 | BF16，SDPA/SDPA，TF32=false | 通过 |
| 官方 DFlash 控制 | Draft temperature=`null`，greedy argmax | 通过 |
| 独立统计复算 | 直接读取 504 条原始记录 | 三个 point estimate 与 CI 逐浮点值一致；通过 |
| 五个记录源码哈希 | engine、两个 verifier、脚本、配置 | 通过；精确 r2 engine 快照已保存；不代表传递依赖闭包 |
| 服务器整合回归 | CUDA/引擎/公平性/正式矩阵相关测试 | 257 项通过，exit 0 |

服务器测试排除了一个仅在 root 环境下必然拒绝的 process-scorer 身份测试；该测试验证的是评分 Python 不应解析到 `/root`，与 CUDA 验证器和本次速度结果无关。失败日志和排除后通过日志均保留，没有覆盖或删除。

## 验证器微基准与冻结选择

| CUDA block threads | 同真实 DDTree 状态下相对官方 verifier | 95% CI | 用途 |
|---:|---:|---:|---|
| 256 | 2.272687× | [1.769848, 2.902985] | 开发候选 |
| 512 | 2.736974× | [2.093909, 3.550846] | 调优 |
| 640 | 2.886713× | [2.229388, 3.705701] | 调优最优 |
| 736 | 2.833484× | [2.206063, 3.614305] | 调优 |
| **640（最终冻结）** | **2.877643×** | **[2.220888, 3.695913]** | 正式候选 |

微基准使用 12 个真实 DDTree 状态、每状态 7 个交错重复、每重复 100 次调用。它只证明验证器内核速度，不替代端到端结果。

## 保留的失败结果

在最终 640-thread 版本冻结前，早期 256-thread 候选做过 18 个手写 prompt、6 次重复的 held-out gate：

| 比较 | Speedup | 95% CI | 严格门槛 |
|---|---:|---:|---|
| 早期树状块 / DDTree | 1.021797× | [0.958755, 1.087070] | 失败 |
| 早期树状块 / DFlash | 1.263377× | [1.188238, 1.338277] | 通过 |

这个失败结果被完整保留。它发生在调优前，且 prompt 随后已参与开发判断，因此不能把它重新包装成最终 held-out 证据。最终 `r2` 是固定官方数据子集上的资格试跑，也仍不能替代完整正式矩阵。

## r1 与 r2 的关系

| 运行 | 树状块/DDTree | 树状块/DFlash | 状态 |
|---|---:|---:|---|
| r1 | 1.025328× [1.007531, 1.043660] | 1.201680× [1.168474, 1.236133] | 行为使用官方 DFlash 贪心，但清单把 Draft T 错记为 1.0；保留、不作为最终版本 |
| r2 | 1.025267× [1.007312, 1.043535] | 1.197062× [1.163375, 1.232113] | 元数据改为 `null` 并增加 fail-closed 官方控制检查；最终资格试跑 |

## 已注册的完整重跑矩阵

| 协议 | 模型 | 数据 | 方法 | 长度/重复 | 状态 |
|---|---|---|---|---|---|
| T=0 | Qwen3-4B、Qwen3-8B | 10 个 AdaptiveTree 官方集合 | Target、DFlash、DDTree B=16..1024、修正后的 AdaptiveTree、控制和消融 | 2048 token；完整 pass | 已注册，需新服务器重跑 |
| T=1 | Qwen3-4B、Qwen3-8B | GSM8K、MATH-500、AIME24/25、HumanEval、MBPP-sanitized、LiveCodeBench、MT-Bench | Target、DFlash、DDTree-B45、同树块验证 | 2048 token；3 seeds | 已注册，需新服务器重跑 |

T=0 和 T=1 是不同生成协议，禁止合并为一个跨温度加速比。Qwen3-Coder-30B 在单 H20 未通过显存与 MoE 兼容性预检，因此仍明确标记为 deferred，不能暗示已经完成。

## 已知限制

1. r2 每个数据集仅 8 条，而非完整注册样本数；不含 AIME24，且使用 MBPP 而非正式矩阵的 MBPP-sanitized。
2. r2 最多生成 256 token，而正式矩阵是 2048 token。
3. r2 没有任务质量评分，不能据此声称质量等价或“严格无损”。
4. BF16 并行树 Target 行与逐 token 自回归 Target 可能有数值差异；候选和 DDTree 之间的树及 Target 行相同，但不能把这一点外推成对自回归 Target 的逐位一致。
5. 结果只适用于记录的 NVIDIA H20 软件/时钟环境，不能改名为 H200 或宣称跨硬件普遍成立。
6. r2 的原始行、报告和五份记录源码可独立复算，但五文件哈希不是完整运行源码的
   传递依赖闭包，且当时的 prepared-data 目录/清单未随归档保存。因此这不是自包含的
   端到端重跑包；新正式运行必须重新准备并哈希完整输入。
7. r2 的旧 observer 在 verifier 调用前触发，未绑定成功返回的实际 callable 和该次
   调用前 RNG 状态。因此它不能单独作为 fused-scan 真实路由证明；新正式
   witness 必须在 verifier 成功返回后记录实际 callable、调用前 RNG 及输出哈希。

## 复核命令

从仓库根目录运行：

```bash
python scripts/validate_same_tree_pilot.py \
  results/pilots/20260909/h20_same_tree_fast_verifier/h20-same-tree-block-official-pilot-20260910-r2 \
  --engine-source results/pilots/20260909/h20_same_tree_fast_verifier/reproduction/r2_source/engine.py
```

预期输出包含 `integrity: PASS_PINNED_R2_BYTES_AND_VALIDATED_FIELDS`、
`strict_speed_gate.passed: true`、504 条记录、168 个配对组，以及三组与上表完全一致的速度比和置信区间。

原始证据位于 `results/pilots/20260909/h20_same_tree_fast_verifier/`：每次运行的 `manifest.json`、`rows.json`、`report.json`、完整日志、调优 gate、全词表审计、服务器测试和精确 r2 `engine.py` 快照均已保留。
