# 方法与代码追踪矩阵

本文件用于区分“原方法”“NVIDIA GPU 性能实现”和“本项目唯一创新”，防止把工程优化误写成论文原文。

## 固定参考版本

- DFlash 论文：arXiv:2602.06036；官方源码参考 commit
  `94e4abc5e0c31b67bc1a9d30f1cc34ece28a8756`。
- SpecBlock 论文：arXiv:2605.07243；官方源码参考 commit
  `b7938c556e5dc42236362dcc4eb37a1edb562c70`。
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

## 唯一创新边界

SpecBlock 原来的 AR/shift drafter 被 DFlash diffusion drafter 替换。第一块保持 DFlash 不变；
后续块携带 pending 位置的 DFlash `h(L)` 和候选 token，绕过 target-feature projection `fc`
（保留 `hidden_norm`，对应官方 `condition_norm`），再把 `h(L)` 作为 DFlash 每层 KV 条件生成
下一 diffusion block。除此之外，rank bucket、树拓扑、budget、ancestor-only验证、最长路径
接受和 bonus token 均按 SpecBlock 执行。

SDPA、CUDA Graph、`torch.compile` 与可选的 CUDA/Triton kernel 只属于性能实现；它们不得
改变上述拓扑、mask 语义、cache 更新规则或 greedy lossless 验证不变量。
