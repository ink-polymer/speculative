# DFlash 基准分支迁移到 DDTree — 结果说明

日期：2026-08-24
项目：`/Users/brucez/Desktop/speculative-benchmark-official-dflash-2k-20260824`
计算后端：**NVIDIA CUDA（保持不变）**

---

## 一、结论摘要

已把原本只支持 SpecBlock 草稿树的实现，扩展为**双拓扑架构**：SpecBlock 与 DDTree 共用同一套
DFlash drafter、ancestor-only 树验证、KV cache 压缩与 CUDA 后端，仅「如何把节点预算分配给候选
前缀」这一步不同。因此两者的 benchmark 可以直接对照，无需切换代码路径或环境。

三项要求的达成情况：

| 要求 | 状态 | 依据 |
|---|---|---|
| 1. 保证原始文件结构不出错 | 达成 | 新增文件为主，`tree_mode` 默认 `specblock`；既有 28 份配置与所有既有脚本零改动；既有 64 项测试全部保持通过，无新增失败 |
| 2. 完整迁移到 DDTree，依旧使用 CUDA | 达成 | 树构建逐节点等价于官方 `build_ddtree_tree`（11 组参数化用例）；所有配置 `device="cuda:0"`、`dtype="bfloat16"`，未触碰任何设备/环境代码 |
| 3. 可选的工程优化/结构创新 | 达成（2 项工程优化落地，1 项结构改动降级为消融开关） | 见第五节，含理论依据与取舍说明 |

**需要重点说明的一点**：迁移过程中我原本设计了一个结构性改动（预留完整 greedy 链），但在理论
分析中**证明它不成立为「改进」**，因此把它降级为默认关闭的消融开关，而不是当作创新交付。详见
第五节 5.3——这是本次工作中信息量最高的部分。

---

## 二、DDTree 与 SpecBlock 的方法差异

两者解决同一个问题：给定目标模型单次可并行验证的节点预算 `B`，应该验证哪些候选前缀。归纳偏置
不同：

| | SpecBlock | DDTree |
|---|---|---|
| 预算分配 | **局部**：先连 greedy 主链，再由 rank head 预测的 bucket 决定每个 slot 展开几个兄弟 | **全局**：所有候选前缀按累计 log-prob 进同一个最大堆，弹出前 `B` 个 |
| 额外模型 | 需要训练 rank head（`(H+15)->256->4` MLP） | 不需要 |
| 跨块扩展 | 支持 `max_blocks>=2` | 单块（`max_blocks=1`） |
| 预算超限 | 事后 `prune` + 保护主链 | 构建期自然截断 |
| 宽度形状 | 由分类器决定，逐位置离散 | 由概率连续决定，自适应 |

DDTree 的关键洞察：把一次 block forward 得到的 `K` 个 slot 分布视作条件独立，则候选前缀
`v = (t_1..t_d)` 的对数联合概率可加：

```text
s(v) = Σ_{i≤d} log q_i(t_i)
```

在可加性下，官方的两条堆推入规则（sibling: `(d, r) → (d, r+1)`；child: `(d, r) → (d+1, 0)`）
保证 `s` 沿两条边单调不增，于是任一未生成节点的分数都不高于其已生成前驱，**堆顶恒为全局最优的
未生成节点**。这也解释了为什么 DDTree 可以在构建期直接按预算截断，而 SpecBlock 必须事后剪枝。

---

## 三、代码改动清单

### 3.1 新增文件

| 文件 | 作用 |
|---|---|
| `src/dflash_specblock/ddtree_builder.py` | DDTree best-first 树构建器（核心） |
| `tests/test_ddtree_builder.py` | 官方等价性、拓扑不变量、**最优性穷举证明** |
| `tests/test_ddtree_integration.py` | 端到端无损性、rank head 跳过验证 |
| `configs/qwen3_4b_cuda_ddtree.json` | DDTree 正式配置（CUDA / BF16 / K=15 / budget=60） |
| `configs/qwen3_4b_cuda_ddtree_reserve_chain.json` | 消融配置（默认不使用） |
| `scripts/run_ddtree_benchmark.sh` | DDTree benchmark runner（无需 rank head） |

### 3.2 修改文件（均为向后兼容的增量）

| 文件 | 改动 | 对既有行为的影响 |
|---|---|---|
| `config.py` | 新增 `tree_mode`、`ddtree_reserve_greedy_chain` 两个字段及交叉校验 | 无：`tree_mode` 默认 `"specblock"`，既有配置解析结果不变 |
| `tree.py` | 新增 `DraftTree.preset_ancestor_mask()` | 无：仅新增方法，不改动既有逻辑 |
| `dflash_adapter.py` | `propose_first()` 新增 `compute_rank: bool = True` | 无：默认值保持原行为 |
| `engine.py` | 读取 `tree_builder.requires_rank` 决定是否算 rank | 无：SpecBlock builder 无该属性，`getattr` 默认 `True` |
| `cli.py` | 按 `tree_mode` 装配对应 builder | 无：`specblock` 分支逻辑原样保留 |
| `__init__.py` | 导出 `DDTreeBuilder` | 无：纯新增导出 |
| `README.md` / `docs/METHOD.md` / `docs/TRACEABILITY.md` | 补充 DDTree 章节与追踪矩阵 | 文档 |

### 3.3 结构保真度

- **官方源码零修改**：`third_party/ddtree_official` 与 `third_party/dflash_official` 未改动一行。
  等价性测试通过按源码单独编译 `build_ddtree_tree` 来比对，绕开其对 flash-attn / loguru 的
  导入（测试环境无 GPU），比对的仍是官方逐字节源码。
- **既有配置零改动**：`configs/` 下原有 28 份配置（含 19 份 `*tree15*`、A2 系列、smoke）全部保持
  原样，本次只新增 2 份 DDTree 配置。
- **既有脚本零改动**：DDTree 使用独立的新 runner，不侵入 `run_full_pipeline.sh` 等既有流程。
- **验证层零改动**：`verification.py`（ancestor mask / 最长 greedy 路径 / KV 压缩 / CUDA Graph）
  完全复用，未因 DDTree 增加任何分支。

本次改动仅涉及 15 个文件：9 个新增（1 实现 + 2 测试 + 2 配置 + 1 脚本，及 3 份文档更新），
6 个既有源文件的向后兼容增量。`third_party/`、`device.py`、`verification.py`、`models.py`、
`rank_head.py`、`benchmark.py` 均未改动。

---

## 四、CUDA 后端保持说明

按要求，本次迁移**未触碰任何设备或环境配置**：

- `device.py`（CUDA 解析、TF32 策略、CUDA event 计时）未改动；
- 所有新增配置为 `"device": "cuda:0"`、`"dtype": "bfloat16"`、`"attn_implementation": "sdpa"`、
  `"allow_tf32": true`，与既有 CUDA 正式配置完全对齐；
- `requirements-gpu.txt`、`requirements-ascend.txt`、`docs/ASCEND_910B_A2.md` 未改动；
- `GraphedTargetTreeVerifier`（CUDA Graph + StaticCache）对 DDTree 直接可用，因为 DDTree 产出
  的是同一种 `DraftTree`，节点数受同一个 `tree_budget` 约束，固定 shape 假设依然成立。

DDTree 的堆构建、visibility 递推是 host 侧 Python/numpy 控制流（与官方一致），大张量计算
（topk、logsumexp、target forward）仍在 GPU 上。

---

## 五、工程优化与结构改动

### 5.1 优化一：visibility 矩阵注入，消除重复推导

DDTree 建树时为了确定父子关系，已经用「父行继承」递推出了完整的祖先可见性矩阵：

```python
visibility[i, :i] = visibility[parent, :i]   # 继承父行
visibility[i, i]  = True                      # 自身可见
```

而 `DraftTree.ancestor_mask()` 的通用实现会为每个节点重新上溯父链，对每个 `(node, ancestor)`
对做一次 Python 级张量赋值。既然结果必然相同，验证阶段的这次重算是纯粹浪费。

新增 `preset_ancestor_mask()` 把已算好的矩阵注入缓存。**安全性**：任何 `add_node` / `prune`
都会触发 `_invalidate_caches()` 丢弃该缓存，不存在陈旧风险；方法本身校验形状、dtype 与 device。
**等价性**：`test_preset_ancestor_mask_matches_generic_derivation` 断言注入值与通用推导逐元素
相同。

理论依据：这是标准的**计算复用**（common subexpression elimination），不改变任何数学定义。

### 5.2 优化二：DDTree 路径跳过 rank head

DDTree 的预算分配完全由 log-prob 决定，从不读取 rank head 的输出。但原 `propose_first()` 无条件
执行两件事：在 151K 词表上做 `topk(20)`，以及跑一次 rank head MLP。

`compute_rank=False` 跳过这两步。**这不是近似或精度妥协**——跳过的是结果不会被任何下游代码消费
的计算。`BlockProposal.rank_logits` 退化为零张量占位，仅满足 `validate()` 的形状契约。

由 `DDTreeBuilder.requires_rank = False` 声明式驱动，engine 通过
`getattr(tree_builder, "requires_rank", True)` 读取。SpecBlock builder 未声明该属性，自动回退
`True`，行为不变（`test_specblock_still_computes_rank_head` 显式保护这一点）。

理论依据：**死代码消除**（dead code elimination）。这两步位于解码关键路径上，每个 decode round
都会执行。

### 5.3 结构改动：`reserve_greedy_chain` —— 为什么它没有被当作创新交付

这是本次工作中最需要如实说明的部分。

**最初动机（看起来合理）**：当 draft 置信度偏低或随深度衰减时，best-first 会把预算优先花在浅层
兄弟上。此时树中最长的 greedy 前缀可能远短于 `K`，单轮接受长度存在硬上界。既然 DFlash 的
`K` 个 top-1 token 在 block forward 中已经算出、不额外增加 draft 成本，先无条件铺满 greedy 链
再分配剩余预算，似乎能提高单轮接受长度上界。

这个直觉与本项目此前在 SpecBlock 侧做的 P0 修复（`prune` 保护主链）同源——那里主链确实会被
浅层兄弟挤掉，保护它是正确的。

**但对 DDTree 而言这个类比不成立**。在自一致假设（target 分布 ≈ draft 分布）下，节点 `v` 被
走到的概率为 `exp(s(v))`，于是：

```text
E[接受 draft token 数] = Σ_{v ∈ 树} exp(s(v))
```

可行解是**前缀封闭**的节点集合（保留某节点必须保留其整条祖先链）。而 best-first 恰好是在按
`exp(s(v))` 降序贪心地选取前缀封闭集合——它选出的就是使该求和最大的 `B` 个节点。

我用穷举验证了这一点：`test_best_first_maximizes_expected_acceptance_against_brute_force` 对
小规模实例（K=3，每 slot top-3）枚举**所有**前缀封闭子集，确认 best-first 的选集在所有测试
参数下都达到最优值。

**推论**：`reserve_greedy_chain` 挤占的正是 best-first 判定为更有价值的兄弟节点。它换来更深的
greedy 链，但在期望接受数上**至多持平，通常略差**。这不是实现问题，而是目标函数决定的。

因此我做了三件事，而不是把它包装成创新：

1. **默认关闭**（`ddtree_reserve_greedy_chain: false`），仅作为消融配置保留；
2. **写进测试**：`test_reserve_greedy_chain_does_not_beat_official_expected_acceptance` 显式断言
   它不优于官方策略，防止未来有人误当作加速改进来引用；
3. **写进文档**：`docs/METHOD.md` §4.4 与 `docs/TRACEABILITY.md`「唯一创新边界」都标注它是
   消融开关而非改进。

**它唯一可能有收益的情形**：自一致假设失效，且 target 实际比 draft 更偏向 greedy 链（即 draft
的 top-1 命中率被自身 `q` 低估）。这是纯经验问题，必须在真实 GPU 上以 τ 和墙钟时间实测判定，
不能仅凭理论宣称。保留开关正是为了让这个假设可被实验检验。

### 5.4 这条最优性结论对后续研究方向的约束

这是比单个开关更有价值的产出：**在 DDTree 的条件独立框架内，任何仅仅「重新分配同一节点预算」
的启发式都不可能提升期望接受数**。要真正提升 τ，必须改变问题本身，可行方向包括：

- **提升 `q_i` 质量**：更好的 drafter 或针对后段 slot 的额外训练（DFlash 的置信度随 mask 位置
  衰减是已观察到的现象）；
- **放宽条件独立假设**：引入 slot 间依赖后，`s(v)` 不再可加，best-first 的最优性证明失效，
  重新分配才可能有收益。这与「block-as-node tree SD」的思路方向一致；
- **改变验证机制**：例如 partial acceptance，允许接受路径中间的部分匹配，这改变的是「接受什么」
  而非「验证什么」，不受上述结论约束。

---

## 六、验证状态

### 6.1 已完成的静态与逻辑验证

| 项目 | 结果 |
|---|---|
| 与官方 `build_ddtree_tree` 逐节点等价 | 11 组参数化用例通过（token / parent / depth / visibility 全一致），覆盖 budget=1/7/30/60/200、K=3/4/8/15、随机分布与尖峰分布 |
| best-first 最优性 | 穷举验证通过（所有前缀封闭子集） |
| 拓扑不变量 | parent < index、depth 与父链一致、`slot_index == depth - 1`、深度受 `block_size` 约束 |
| 累计分数正确性 | 与路径上各 slot log-prob 之和逐节点吻合（tol 1e-4） |
| 端到端无损性 | mock 模型下 DDTree 与逐 token greedy baseline 输出完全一致（含 `reserve_greedy_chain` 开启时） |
| rank head 跳过 | DDTree 调用次数 = 0；SpecBlock 调用次数 > 0 |
| 配置交叉校验 | 4 类非法组合全部被拦截并给出中文原因 |
| 全量回归 | 94 passed（既有 64 项全部保持通过 + 新增 30 项：builder 22 + 集成 8） |

关于 2 项失败：`test_cuda_runtime_configures_tf32` 与
`test_retrieve_indices_keeps_host_control_paths_on_cpu` 在本机报
`Torch not compiled with CUDA enabled`。这两项**迁移前后同样失败**，根因是本机 macOS 的 torch
为 CPU-only 构建，且两项测试均不涉及本次改动的任何代码路径。

### 6.2 尚未完成、必须在 GPU 上补做的验证

以下结论**不能**仅凭本地逻辑验证得出，须在目标 NVIDIA GPU 上实测：

1. **真实 τ 对照**：DDTree vs SpecBlock vs 官方 DDTree，在固定的 200 条 benchmark JSONL 上；
2. **墙钟加速比**：包含 host 侧堆构建开销的端到端延迟（堆构建是 Python 控制流，其占比只能实测）；
3. **`reserve_greedy_chain` 的经验判定**：即 5.3 中自一致假设是否成立；
4. **两项工程优化的实际收益**：visibility 注入与 rank head 跳过节省的绝对毫秒数；
5. **CUDA Graph 路径**：`use_cuda_graphs=true` 下 DDTree 的 exact match、P95/P99 与峰值显存。

建议的执行顺序：

```bash
# 冒烟：先确认链路通
MAX_PROMPTS=2 MAX_NEW_TOKENS=32 bash scripts/run_ddtree_benchmark.sh

# 正式：200 条对照
bash scripts/run_ddtree_benchmark.sh

# 消融：验证 5.3 的假设
CONFIG=configs/qwen3_4b_cuda_ddtree_reserve_chain.json \
  OUTPUT=outputs/benchmark_ddtree_reserve_chain.jsonl \
  bash scripts/run_ddtree_benchmark.sh

# 官方对照（独立环境，需 flash-attn）
bash scripts/run_official_comparison.sh
```

---

## 七、如何使用

```bash
# DDTree 正式配置（无需先训练 rank head）
python -m dflash_specblock.cli \
  --config configs/qwen3_4b_cuda_ddtree.json \
  --prompt "your prompt"

# 原 SpecBlock 路径完全不受影响
python -m dflash_specblock.cli \
  --config configs/qwen3_4b_cuda.json \
  --prompt "your prompt"
```

切换拓扑只需改 `tree_mode`。配置层会拒绝不自洽的组合，例如 `ddtree` + `max_blocks=2`（DDTree
是单块方法）、`ddtree` + `rank_mode=learned`（DDTree 不使用 rank head），并给出中文原因。
