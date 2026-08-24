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

## 4. 必须说明的实验边界

DFlash 官方 checkpoint 不含 SpecBlock rank head，因此存在两种模式：

- `heuristic`：仅用于权重下载后验证工程、CUDA 算子和树验证是否连通，不属于论文严格结果；
- `learned`：先运行 `train_rank_head.py` 训练 rank head，才用于正式 acceptance/speed 实验。

当前实验协议支持 greedy lossless verification。随机采样下的树 speculative sampling 需要保存每个
候选节点的 draft proposal probability 并进行 rejection sampling，不应把 greedy 验证直接套到
temperature > 0，因此本版本主动拒绝该配置。
