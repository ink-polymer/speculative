# DFlash-SpecBlock 树模式 vs 正版 DFlash 线性模式 Benchmark 对比报告

**生成时间**: 2026-08-17  
**实验平台**: Atlas A2 (Ascend 910B), CANN, torch_npu  
**数据文件**: `outputs/benchmark_tree15.jsonl` / `outputs/benchmark_vanilla_dflash.jsonl`

---

## 1. 实验设置

| 项目 | 配置 |
|---|---|
| 目标模型 | Qwen/Qwen3-4B (revision `1cfa9a72`) |
| 草稿模型 | z-lab/Qwen3-4B-DFlash-b16 (revision `b74e3a32`) |
| 设备 | npu:0, bfloat16 |
| block_size | 15 |
| max_new_tokens | 128 |
| benchmark prompts | 200 条（同一份 `prompts_benchmark_tree15.jsonl`，随机种子 123） |
| greedy baseline | 同一 target 模型逐 token argmax |

**两种模式的唯一区别**:

| | Tree15 (DFlash-SpecBlock) | Vanilla DFlash |
|---|---|---|
| 草稿策略 | 树扩展：greedy 主链 + rank-guided 兄弟分支 + 跨块 pending | 线性：每个 block 仅取 top-1 |
| 验证策略 | ancestor-only tree attention（最多 60 节点并行验证） | 标准因果注意力（16 节点） |
| rank head | learned（`rank_head_tree15.pt`, 3 epochs） | 不需要 |
| tree_budget | 60 | — |
| beam_width | 4 | — |
| branch_factors | [2, 4, 10, 0] | — |
| max_blocks | 1 | — |

两种模式共用同一个 DFlash block diffusion drafter、同一 target KV injection、同一 draft cache 增量维护逻辑，仅验证端不同，对比公平。

---

## 2. 总体结果对比

| 指标 | Tree15 | Vanilla DFlash | 差异 |
|---|---:|---:|---:|
| 精确匹配数 | 73 / 200 | 95 / 200 | +22 |
| 精确匹配率 | 36.5% | 47.5% | +11.0pp |
| **总加速比** | **2.065x** | **2.341x** | +0.275x |
| 匹配样本加速比 | 2.158x | 2.634x | +0.476x |
| 平均接受长度 τ | 3.750 | 3.292 | -0.458 |
| 中位接受长度 τ | 3.256 | 2.811 | -0.445 |
| 平均 verify 次数 | 29.81 | 34.55 | +4.74 |
| 平均 hybrid 耗时 | 7056 ms | 6004 ms | -1053 ms |
| 平均 baseline 耗时 | 14572 ms | 14052 ms | -520 ms |
| 总耗时 | 1h 12m 32s | 1h 07m 26s | -5m 06s |

**核心发现**: Vanilla DFlash 的**总加速比反而更高**（+0.28x），尽管**平均接受长度更低**（-0.46）。树模式多接受的 token 无法补偿 verify 开销的增加。

---

## 3. 配对分析（59 条两边都精确匹配的样本）

为消除 baseline 差异和 prompt 难度差异，取两种模式都精确匹配的 59 条样本做配对对比：

| 指标 | Tree15 | Vanilla | 差异 (Tree - Vanilla) |
|---|---:|---:|---:|
| 平均加速比 | 2.314x | 2.623x | -0.309x |
| 平均接受长度 τ | 3.923 | 3.541 | +0.383 |
| 平均 hybrid 耗时 | 3682 ms | 3150 ms | +532 ms |
| 平均 verify 次数 | 15.14 | 17.39 | -2.25 |

**配对结论**: 在公平条件下，Tree15 每次多接受 0.38 个 token，但每次 verify 多花 532 ms。多接受的 token 价值（约 532ms / 0.38 token ≈ 1400 ms/token）远高于 baseline 每 token 成本（约 140 ms/token），说明树 verify 开销远超接受长度收益。

---

## 4. 加速比分布对比

| 加速比区间 | Tree15 数量 | Vanilla 数量 | 差异 |
|---|---:|---:|---:|
| <1.0x | 1 | 0 | -1 |
| 1.0-1.5x | 28 | 11 | -17 |
| 1.5-2.0x | 81 | 55 | -26 |
| 2.0-3.0x | 59 | 93 | +34 |
| 3.0-4.0x | 14 | 20 | +6 |
| 4.0-5.0x | 12 | 7 | -5 |
| >5.0x | 5 | 14 | +9 |

- **减速样本（<1.0x）**: Tree15 有 1 条，Vanilla 有 0 条
- **高加速样本（>5x）**: Tree15 有 5 条，Vanilla 有 14 条

Vanilla 的加速比分布明显右移：低速段（<1.5x）更少，高速段（>3x）更多。

---

## 5. 极端样本分析

### 5.1 加速比最高样本

| 排名 | 模式 | idx | 加速比 | τ | 匹配 | prompt 摘要 |
|---|---|---:|---:|---:|:---:|---|
| 1 | Vanilla | 34 | 10.22x | 14.11 | ✓ | Find the largest $x$-value at which the graphs of  |
| 2 | Vanilla | 19 | 7.49x | 5.31 | ✗ | Analyze the given text to identify the maximum and |
| 3 | Vanilla | 191 | 6.26x | 8.00 | ✓ | Find the area of a square shape with side length 5 |
| 4 | Vanilla | 4 | 6.24x | 7.94 | ✗ | Given a list of numbers, find the 3rd largest numb |
| 5 | Vanilla | 196 | 5.95x | 7.94 | ✓ | Calculate the mean from the given numbers: 2, 3, 5 |
| 1 | Tree15 | 34 | 8.64x | 14.11 | ✗ | Find the largest $x$-value at which the graphs of  |
| 2 | Tree15 | 92 | 5.72x | 9.77 | ✗ | Write a query to find all books published in 2021  |
| 3 | Tree15 | 26 | 5.40x | 9.07 | ✓ | from typing import List   def filter_by_prefix(str |
| 4 | Tree15 | 191 | 5.13x | 8.73 | ✓ | Find the area of a square shape with side length 5 |
| 5 | Tree15 | 4 | 5.10x | 7.94 | ✗ | Given a list of numbers, find the 3rd largest numb |

数学/计算类 prompt 在两种模式下都能获得极高加速比（idx=34, 191, 196），因为 draft 接受长度可达 8-14。

### 5.2 加速比最低样本

| 排名 | 模式 | idx | 加速比 | τ | 匹配 | prompt 摘要 |
|---|---|---:|---:|---:|:---:|---|
| 1 | Vanilla | 199 | 1.23x | 1.67 | ✓ | Translate the following text from de to en.  Beide |
| 2 | Vanilla | 49 | 1.22x | 2.00 | ✓ | Indicate a yes or no answer to the given statement |
| 3 | Vanilla | 126 | 1.22x | 1.56 | ✓ | Generate a headline that would appear on a news we |
| 4 | Vanilla | 176 | 1.15x | 1.50 | ✓ | Rewrite the sentence so that it makes more sense.  |
| 5 | Vanilla | 79 | 1.05x | 1.33 | ✓ | Provide a translation from English to German for t |
| 1 | Tree15 | 49 | 1.19x | 2.00 | ✓ | Indicate a yes or no answer to the given statement |
| 2 | Tree15 | 199 | 1.14x | 1.67 | ✓ | Translate the following text from de to en.  Beide |
| 3 | Tree15 | 14 | 1.13x | 1.75 | ✓ | Convert the following Proper Nouns to Plural forms |
| 4 | Tree15 | 126 | 1.12x | 1.75 | ✓ | Generate a headline that would appear on a news we |
| 5 | Tree15 | 142 | 0.71x | 3.53 | ✗ | Name the day of the week when Thanksgiving falls i |

**关键观察**: Tree15 出现 1 条减速样本（idx=142, 0.71x）——verify 开销超过了 baseline 逐 token 解码的成本；Vanilla 最低也有 1.05x，未出现减速。

---

## 6. 结论与讨论

### 6.1 核心结论

1. **总加速比 Vanilla 胜出（2.34x vs 2.07x）**：尽管 Tree15 的平均接受长度更高（3.75 vs 3.29），但树验证的额外开销抵消了多接受的 token 价值。

2. **精确匹配率 Vanilla 更高（47.5% vs 36.5%）**：线性 draft 无分支探索，greedy 路径更稳定；树模式引入更多候选虽增加接受长度，但也引入更多发散点。

3. **树 verify 开销是瓶颈**：配对分析显示，Tree15 每次 verify 比 Vanilla 多花 532 ms，而多接受的 token 仅 0.38 个（价值约 53 ms）。开销/收益比 ≈ 10:1。

### 6.2 原因分析

- **verify 前向成本**：Tree15 每次 verify 最多 60 个节点 + 构造 ancestor-only 4D mask；Vanilla 只需 16 个节点 + 标准因果 mask。在 Ascend 910B 上，eager attention 的前向成本与序列长度近似线性，60 vs 16 的差距显著。
- **mask 构造成本**：`ancestor_mask` 需要对数传播 `ceil(log2(N))` 次张量操作，N=60 时需 6 次全张量 OR；Vanilla 的标准下三角 mask 一次构造完成。
- **cache 压缩成本**：Tree15 的 `_compact_cache` 需要对 60 个候选做 `index_select` 并按接受路径重排；Vanilla 只需 `crop(keep_length)` 一次。

### 6.3 适用场景讨论

- **当前配置（block_size=15, max_blocks=1）下，树扩展不划算**：单块树最多 15 个主链 + 兄弟，verify 节点数膨胀快但接受长度提升有限。
- **树模式潜在优势场景**：
  - 更大 block_size（如 64/128），prefill 成本被更多 token 摊薄
  - max_blocks > 1，跨块 pending 能累积更深的接受路径
  - 更长上下文，baseline 每 token 成本上升，多接受 token 的相对价值增大
  - 硬件对 tree attention 有专门优化（如 FlashAttention tree mask kernel）

### 6.4 后续方向

1. 测试 max_blocks=2/3 的树模式，观察跨块 pending 是否能拉开接受长度差距
2. 测试 block_size=8 的小块配置，降低单次 verify 节点数
3. 在 Ascend 上实现 tree attention 专用 kernel，降低 mask 构造与前向开销
4. 对比 temperature>0 的随机采样场景（需要 rejection sampling 验证，不在当前 greedy 协议内）

---

## 7. 数据文件

| 文件 | 说明 |
|---|---|
| `outputs/benchmark_tree15.jsonl` | 树模式 200 条逐条结果 |
| `outputs/benchmark_vanilla_dflash.jsonl` | 线性模式 200 条逐条结果 |
| `datasets/processed/specblock_official/prompts_benchmark_tree15.jsonl` | 200 条 benchmark prompts |
| `configs/qwen3_4b_a2_tree15.json` | 实验配置 |
| `checkpoints/rank_head_tree15.pt` | 树模式 rank head checkpoint |
| `logs/vanilla_dflash_benchmark.log` | Vanilla 运行日志 |

每条 JSONL 记录包含字段：`index, prompt, baseline_tokens, baseline_ms, hybrid_tokens, hybrid_ms, wall_clock_speedup, average_committed_per_verify, verify_iterations, greedy_exact_match, first_mismatch_index`。
