# 方法设计与论文对应关系

## 1. DFlash 部分

本工程复用 DFlash 官方 Qwen3 draft checkpoint。每轮第一块执行以下过程：

1. 从目标模型浅层到深层均匀选取隐藏状态并拼接；
2. 使用 DFlash checkpoint 自带的 `fc + RMSNorm` 融合目标上下文；
3. 将“当前已验证锚点 token + K 个 mask token”送入扩散 drafter；
4. 融合后的目标上下文在每个 draft layer 中作为额外 Key/Value；
5. K 个未来位置在一次 forward 中并行产生 logits；
6. 独立 DynamicCache 只保留已验证前缀，当前 anchor/mask KV 在验证前裁掉，与官方
   `past_key_values_draft.crop(start)` 对齐。

这对应 DFlash 论文第 4.1 节和附录 A.3。官方实现参考：
<https://github.com/z-lab/dflash/blob/main/dflash/model.py>。

## 2. SpecBlock 部分

树构建严格采用 SpecBlock 的拓扑语义，而不是对 K 个位置做笛卡尔积：

- 每个 block 先产生一条长度 K 的 greedy 主链；
- block 1 的 slot 0 使用 `beam_width`，其余位置由 `b(bucket)` 附加兄弟候选；
- `[2,4,10,0]` 模式遇到 give-up bucket 后停止更深位置的兄弟扩展，但保留免费 greedy 链；
- pending 候选进入 expansion buffer，并立即按官方 adaptive beam
  `min(beam_width, max(1, (budget - nodes) // K))` 裁剪；
- 下一块只对裁剪后的 pending 起点做一次批量 forward；
- 超过 tree budget 后按累计 log probability 剪枝，但必须保留祖先；
- 目标模型通过 ancestor-only tree attention 一次并行验证所有节点；
- greedy 模式枚举全部根到叶路径，选择目标 argmax 的最长匹配前缀，并追加 bonus token；
- target DynamicCache 先把接受路径 KV 搬到连续前缀，再调用 `crop` 更新内部长度。

正式默认映射 `[2,4,10,0]` 直接取自官方代码常量
`benchmarks_hf/algorithms/_specblock_base.py` 中的 `RANK_SLOT_TOPK`（对应 rank
1 / 2-4 / 5-10 / >10 四个 bucket 的 per-slot top-k，含 greedy 自身）。官方实现参考：
<https://github.com/shiweijiezero/SpecBlock>。

## 3. 两者结合处

- 原SpecBlock 的 AR/shift drafter 被 DFlash block diffusion drafter 替换。
- 第一块使用目标模型多层特征，保持 DFlash 的 KV injection。
- 后续块使用前一 DFlash 块起点位置缓存的最后层 hidden state，对齐 SpecBlock
  `use_draft_condition`：绕过只适用于目标多层特征的 `fc`，但仍执行 `hidden_norm`
  （官方对应 `condition_norm`），再把该状态作为每层 KV 条件。这是唯一新增的方法接口。
- rank head 输入为 DFlash 最后层 hidden state 与官方定义的 15 维分布摘要：top-10
  log-prob、top1 与 rank-2/3/5 的 gap、top1 probability、entropy。归一化常数与 entropy
  都按官方实现在 top-20 上计算，保证特征口径与 rank head 权重一致。
- rank head 严格采用 `(H+15)->256->4` 无 bias MLP；训练使用四个 bucket、块内 valid-prefix
  mask 与均匀随机 cut 的跨块 curriculum。按官方要求，块之间不额外施加「前缀全对」过滤。

## 4. DDTree 部分

对应论文 *Accelerating Speculative Decoding with Block Diffusion Draft Trees*
（<https://arxiv.org/abs/2604.12989>），官方实现
<https://github.com/liranringel/ddtree>，固定 commit 见
`third_party/OFFICIAL_SOURCES.md`。本工程实现位于
`src/dflash_specblock/ddtree_builder.py`，由 `tree_mode="ddtree"` 启用。

### 4.1 与 SpecBlock 的分工差异

SpecBlock 与 DDTree 都在解决同一个问题——给定目标模型一次可并行验证的节点预算 `B`，
应该把预算分配给哪些候选前缀——但归纳偏置不同：

- SpecBlock 做**局部宽度分配**：先连出 greedy 主链，再由 rank head 预测的 bucket 决定每个
  slot 展开几个兄弟。宽度决策是逐位置的，依赖一个额外训练的分类器。
- DDTree 做**全局预算分配**：把一次 block forward 得到的 `K` 个 slot 分布视作条件独立，
  于是任一候选前缀的对数联合概率可加，直接用一个全局最大堆按累计 log-prob 弹出前 `B` 个
  节点。不需要 rank head，也不需要跨块 continuation。

### 4.2 正确性与最优性

记 slot `i` 的 draft 分布为 `q_i`，候选前缀 `v = (t_1..t_d)` 的分数为
`s(v) = Σ_{i≤d} log q_i(t_i)`。官方 `build_ddtree_tree` 的两条推入规则是：

1. **sibling**：弹出 `(depth, rank)` 后推入 `(depth, rank+1)`，父节点不变；
2. **child**：弹出节点后推入其 `(depth+1, rank=0)` 子节点。

由于每个 slot 的 topk 按 log-prob 降序，`s` 沿这两条边都单调不增，所以任一尚未生成的节点
分数都不高于其已生成的前驱（父节点或前一兄弟）。堆顶因此始终是全局最优的未生成节点，
最先弹出的 `B` 个节点就是 `s` 最大的 `B` 个候选前缀。这也解释了 DDTree 可以在构建期直接
按预算截断，而 SpecBlock 需要事后 `prune`。

进一步地，在「target 分布与 draft 分布一致」的自一致假设下，节点 `v` 被走到的概率为
`exp(s(v))`，于是

```text
E[接受 draft token 数] = Σ_{v ∈ 树} exp(s(v))
```

可行解是前缀封闭的节点集合（保留某节点必须保留其整条祖先链）。`tests/test_ddtree_builder.py`
的 `test_best_first_maximizes_expected_acceptance_against_brute_force` 对小规模实例穷举所有
前缀封闭子集，确认 best-first 的选集达到该目标的最优值。

**这条结论对后续改进方向有约束意义**：任何仅仅「重新分配同一预算」的启发式，在该目标下都
不可能超过官方策略，最多持平。要真正提升 τ，必须改变问题本身——例如提高 `q_i` 的质量、
放宽条件独立假设（引入 slot 间依赖），或改变验证机制而非改变分配顺序。

### 4.3 与本工程既有组件的复用关系

DDTree 只替换「树构建」这一步，产出与 SpecBlock 相同的 `DraftTree`，因此以下组件完全复用、
未作任何修改：ancestor-only 4D additive mask、最长 greedy 接受路径选择、非连续 target KV
压缩、CUDA Graph/StaticCache 验证器、DFlash drafter 与其增量 draft cache 裁剪。

无损性由目标模型的验证语义保证，与草稿树的形状无关：验证阶段只接受「target argmax 与草稿
token 一致」的最长路径，树的形状只影响能接受多长，不影响接受什么。

### 4.4 `ddtree_reserve_greedy_chain` 消融开关

配置项 `ddtree_reserve_greedy_chain`（默认 `false`）会先无条件铺满 all-rank-0 的 greedy 链，
再把剩余预算交给同样的 best-first 分配。sibling/child 推入规则不变，因此产出的仍是一棵合法
DDTree。

动机：当 draft 置信度偏低或随深度衰减时，best-first 会把预算优先花在浅层兄弟上，此时树中
最长的 greedy 前缀可能明显短于 `K`，单轮接受长度存在硬上界。保留完整 greedy 链可以把这个
上界恢复到 `K`。

但由 4.2 的最优性结论，这个开关**在自一致假设下不会提升期望接受数**，因为它挤占的正是
best-first 认为更有价值的兄弟节点。`tests/test_ddtree_builder.py` 中的
`test_reserve_greedy_chain_does_not_beat_official_expected_acceptance` 显式断言了这一点。

它唯一可能带来收益的情形是自一致假设不成立、且 target 实际比 draft 更「偏向 greedy 链」
（即 draft 的 top-1 命中率被 `q` 低估）。这是一个经验问题，必须在真实 GPU 上以 τ 和墙钟
时间实测判定，不能仅凭理论宣称。因此默认关闭，仅作为消融配置
`configs/qwen3_4b_cuda_ddtree_reserve_chain.json` 保留。

## 5. 必须说明的实验边界

DFlash 官方 checkpoint 不含 SpecBlock rank head，因此存在两种模式：

- `heuristic`：仅用于权重下载后验证工程、CUDA 算子和树验证是否连通，不属于论文严格结果；
- `learned`：先运行 `train_rank_head.py` 训练 rank head，才用于正式 acceptance/speed 实验。

`tree_mode="ddtree"` 不受这条限制：它不使用 rank head，因此 `rank_mode` 固定为
`heuristic`（占位，其输出不被读取），可以直接进入正式 acceptance/speed 实验。配置层会拒绝
`ddtree` + `learned` 的组合，避免误以为需要先训练分类器。

当前实验协议支持 greedy lossless verification。随机采样下的树 speculative sampling 需要保存每个
候选节点的 draft proposal probability 并进行 rejection sampling，不应把 greedy 验证直接套到
temperature > 0，因此本版本主动拒绝该配置。
