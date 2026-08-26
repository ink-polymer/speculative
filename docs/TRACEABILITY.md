# 方法与代码追踪矩阵

本文件用于区分“原方法”“NVIDIA GPU 性能实现”和“本项目唯一创新”，防止把工程优化误写成论文原文。

## 固定参考版本

- DFlash 论文：arXiv:2602.06036；官方源码参考 commit
  `94e4abc5e0c31b67bc1a9d30f1cc34ece28a8756`。
- SpecBlock 论文：arXiv:2605.07243；官方源码参考 commit
  `b7938c556e5dc42236362dcc4eb37a1edb562c70`。
- DDTree 论文：arXiv:2604.12989；官方源码 vendored 于 `third_party/ddtree_official`，
  commit `c96427a185677bf4133ed865dd1626a5041aef9b`。
- Qwen target revision：`1cfa9a7208912126459214e8b04321603b3df60c`。
- DFlash checkpoint revision：`b74e3a329c4d963783143b1e970d95b002be72bd`。

## DFlash 对应

| 原方法要求 | 本地实现 | 不变量 |
|---|---|---|
| 多层 target hidden 拼接 | `extract_target_context` | hidden tuple layer id 加 1 |
| `fc + RMSNorm` feature fusion | checkpoint 官方 forward | 不在本地重复实现 |
| 每层 KV injection | checkpoint 官方 decoder/attention | `target_hidden` 传入每层 |
| clean anchor + mask block | `_noise_embedding` | 本实验 K=4，总输入长度 K+1 |
| block 内双向并行 | `is_causal=False` | 一次 forward 产生 K 个分布 |
| draft DynamicCache 裁剪 | `draft_first_raw` | 每轮裁回 target 已验证前缀 |

## SpecBlock 对应

| 原方法要求 | 本地实现 | 不变量 |
|---|---|---|
| K=4、M=2、tree=60 | 正式 JSON 配置 | 默认固定 |
| 四个 rank bucket | `target_rank_buckets` | 1 / 2-4 / 5-10 / >10 |
| 15 维分布摘要 | `distribution_summary` | top10 LP + 3 gaps + p1 + entropy，top-20 lse 口径 |
| rank head | `DFlashRankHead` | `(H+15)->256->4`，无 bias，输入 detach |
| per-slot branching | `SpecBlockTreeBuilder` | block1 slot0 beam；其余按 bucket |
| give-up / hitchhike | `_expand_rows` | 保留 greedy chain；控制 pending |
| expansion buffer | `_PendingStart` | token、累计 LP、cached h(L) 成对保存 |
| adaptive beam | `_adaptive_beam` | 每块 forward 前先裁 pending |
| tree budget | `DraftTree.prune` | 累计 LP 剪枝且祖先闭包 |
| tree attention | `build_tree_attention_mask` | 旧前缀 + anchor + 自身祖先可见 |
| greedy verify | `select_greedy_path` | 枚举叶路径，最长匹配，追加 bonus |
| target KV 裁剪 | `_compact_cache` | 接受路径重排为连续 cache 后 crop |
| valid-prefix | `valid_prefix_mask` | 块内首错之后屏蔽；块间不加 filter |
| cross-block curriculum | `train_rank_head` | cut 从 1..K 均匀采样 |

## DDTree 对应

`tree_mode="ddtree"` 时启用，实现位于 `src/dflash_specblock/ddtree_builder.py`。等价性由
`tests/test_ddtree_builder.py::test_matches_official_build_ddtree_tree` 逐节点比对官方
`build_ddtree_tree` 保证（token / parent / depth / visibility 全部一致）。

| 原方法要求 | 本地实现 | 不变量 |
|---|---|---|
| 条件独立可加分数 | `build_from_logits` | `s(v) = Σ log q_i(t_i)`，FP32 topk + logsumexp |
| 全局 best-first 堆 | `heapq` + `(-logw, ranks, ...)` | 堆元素顺序与官方逐字段一致 |
| sibling 推入 | `push_sibling` | `(depth, rank+1)`，父节点不变 |
| child 推入 | `push_child` | `(depth+1, rank=0)`，受 `depth_limit` 约束 |
| 预算截断 | `while ... node_count < budget` | 构建期截断，不调用 `prune` |
| 深度上限 | `depth_limit = min(K, logits 行数)` | 深度 `d` 取自 slot `d-1` |
| visibility 递推 | `_to_draft_tree` | 父行继承，与逐节点上溯父链等价 |
| 树验证 / KV 压缩 | 复用 `verification.py` | 与 SpecBlock 完全共用，未修改 |

DDTree 不使用 rank head、不使用跨块 continuation，因此 SpecBlock 表中的 rank bucket、15 维
摘要、give-up / hitchhike、adaptive beam、`prune` 均不参与该路径。

## Latency-aware DDTree 对应

`tree_mode="ddtree_adaptive"` 复用同一条 DDTree best-first 节点顺序，但一次枚举到最大预算，
再从嵌套前缀中选择本轮验证树。它不改变候选 token、不增加 draft forward，也不放宽 target
greedy 验证条件。

| 设计要求 | 本地实现 | 不变量 |
|---|---|---|
| 当前上下文置信度 | `mass_by_budget` | 只累加当前 proposal 的 prefix probability |
| 硬件感知成本 | `observe` + per-budget verify EWMA | 只读取已完成轮次计时 |
| 在线校准 | `_acceptance_scale` | 用真实接受数校准 surrogate，不参与接受判定 |
| 嵌套预算 | `_select_node_count` | 只截取 best-first 前缀，保持祖先闭包 |
| 无额外模型前向 | `manages_budget=True` | engine 仍只调用一次 `propose_first` |

对应测试为 `tests/test_ddtree_adaptive.py`：验证所选树严格等于最大 DDTree 的前缀、warmup
顺序、实测吞吐选择以及非法预算拒绝。

## 唯一创新边界

SpecBlock 原来的 AR/shift drafter 被 DFlash diffusion drafter 替换。第一块保持 DFlash 不变；
后续块携带 pending 位置的 DFlash `h(L)` 和候选 token，绕过 target-feature projection `fc`
（保留 `hidden_norm`，对应官方 `condition_norm`），再把 `h(L)` 作为 DFlash 每层 KV 条件生成
下一 diffusion block。除此之外，rank bucket、树拓扑、budget、ancestor-only验证、最长路径
接受和 bonus token 均按 SpecBlock 执行。

基础 DDTree 路径不含方法创新：树构建逐节点复现官方实现，其余组件复用本工程既有的验证与
cache 管理。本次迁移新增的两处改动都严格限定在工程层面，不改变树的数学定义：

1. `DraftTree.preset_ancestor_mask`：把 DDTree 建树时已算出的 visibility 直接注入缓存，避免
   验证阶段重复推导。结果与通用推导逐元素相同（`test_preset_ancestor_mask_matches_generic_derivation`）。
2. `propose_first(compute_rank=False)`：DDTree 不读取 rank 输出，跳过 top-20 摘要与 rank head
   前向。跳过的是纯粹未被消费的计算（`test_engine_skips_rank_head_for_ddtree`），SpecBlock
   路径行为不变（`test_specblock_still_computes_rank_head`）。

`ddtree_reserve_greedy_chain` 是**消融开关而非改进**，默认关闭。理由见
`docs/METHOD.md` §4.4：best-first 已被证明在期望接受数上最优，任何仅重新分配同一预算的
启发式都不可能超过它。该开关只在自一致假设失效时可能有收益，须在 GPU 上以 τ 和墙钟实测判定。

`LatencyAwareDDTreeBuilder` 是本项目新的性能策略：它不声称在固定 B 下优于 DDTree，而是
优化 DDTree 未决定的“每轮 B 取多少”。理论保证仅覆盖嵌套树合法性；端到端加速是否提升仍须
在同一 T=0、同一硬件、同一数据集协议下实测。

SDPA、CUDA Graph、`torch.compile` 与可选的 CUDA/Triton kernel 只属于性能实现；它们不得
改变上述拓扑、mask 语义、cache 更新规则或 greedy lossless 验证不变量。
