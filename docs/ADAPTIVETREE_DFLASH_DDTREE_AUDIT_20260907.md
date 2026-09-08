# AdaptiveTree：DFlash / DDTree 性能反转核查

核查日期：2026-09-07。仅诊断；未修改服务器源码、环境、结果或 Slurm 作业。

## 结论边界

已确认当前保存的结果中 DFlash-best 比 DDTree-best 快，不是将最差 DDTree
预算拿来比较造成的。后端选优放大了部分差距，但不能解释全部差距。
树验证、CPU 建树、同步及 KV 整理是源码可定位的额外成本；本次尚未完成
现有 `stage_times` 的读取，不能把其中某一项宣称为已测定的主导根因。

继续读取分阶段数据时 SSH 共享连接消失，BatchMode 返回认证失败；可见终端
也已退回本机。未发送验证码、重启连接、取消或重提任务。恢复登录后可继续
只读提取结果，不需要为了这一步重跑 GPU 实验。

## 对象、状态与协议

- 服务器根目录：`/nobackup/proj/disk/naiss2026-3-658/personal/jiage91/fuyile`。
- Adaptive 作业：2072993，结果：`runs/adaptive-record-8b-unpinned`。
- 最后成功查询：2026-09-07 10:04:43 CEST / 16:04:43 北京时间；作业运行于 n141。
- 当时 GSM8K、MATH500、AIME24、AIME25、HumanEval 双后端已完成；MBPP
  SDPA 已完成、FA2 日志至 101/128。其余四个数据集尚未完成。本表仅用双后端
  完成的五个数据集、480 个回答，不能称为完整实验最终表。
- 当前只跑 Qwen3-8B 与对应 DFlash-b16：BF16、T=0、seed=0、最多 2048 新 token。
- 实际设备：NVIDIA GH200 120GB，单卡；平台 aarch64；PyTorch 2.9.1+cu130、
  Transformers 4.57.1、FlashAttention 2.8.3、CUDA 13.0、驱动 580.159.04。
- 运行身份：`f21f543f5e6c1f3efa4aca317586c05285ea081915b3ed424d9a5d5fecf6c592`。
- 指标：每个回答的 decode time / output tokens，然后对回答取算术平均。
  不含目标模型 prefill 和首轮 draft；不是完整请求端到端延迟，也不是 pooled TPOT。
- DFlash/AR 独立选 SDPA 与 FA2 中数据集均值较快者；DDTree 选 SDPA 下
  16、32、64、128、256、512、1024 中最快预算。所有 draft 都使用 FA2。
- 该后端与预算选择遵循作者协议，而不是专门给 DFlash 加的优惠。
  论文用 8 张 H200 分片评估数据，每个 worker 处理自己的模型与请求，不能将其
  解释成一个请求用了八卡张量并行。因此单卡数量本身不能解释反转。
  参见 [DDTree 论文附录 B](https://arxiv.org/html/2604.12989v1#A2)。

## 已复核的部分结果

单位 ms/token，越低越好。完成标记的 SHA256、运行身份和回答数通过只读
提取程序核验；本次汇总沿用已取得的结果，没有重新生成回答。

| 数据集 | DDTree 最佳预算 | DDTree SDPA | DFlash SDPA | DFlash 最佳后端 | DFlash-best | DDTree / DFlash-best |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| GSM8K | 512 | 9.020510 | 8.662218 | SDPA | 8.662218 | 1.0414 |
| MATH500 | 512 | 8.433959 | 7.863218 | FA2 | 6.743452 | 1.2507 |
| AIME24 | 512 | 9.038375 | 8.844335 | FA2 | 8.324887 | 1.0857 |
| AIME25 | 256 | 8.935807 | 9.297756 | FA2 | 7.753764 | 1.1524 |
| HumanEval | 512 | 8.790641 | 8.340657 | SDPA | 8.340657 | 1.0540 |

同用 SDPA 时 DFlash 仍胜 4/5；AIME25 是 DDTree 胜，取 FA2 后才反转。
MATH500 的 DFlash 相对加速由同后端 1.0726× 增至选优后 1.2507×。
这些是描述性点估计，未建立配对置信区间，不能把小差距直接写成显著性结论。

预算搜索也确实跑了：例如 GSM8K 的 DDTree 从 B16 的 11.440870，随预算扩大
改善到 B512 的 9.020510，再到 B1024 的 9.132207。MATH500 对应为 10.577864、
8.433959、8.642238。既不能说没有调预算，也不能说增大预算必然更快。

## 源码层面确定与待验证的原因

1. **验证工作量不同。** DFlash 每轮输入 16 个位置；DDTree 用 B 个草稿节点加
   一个根节点。当前最佳预算多为 512，即通常验证 513 个位置，而不是 16 个。
   这不是 32 倍延迟的断言：GPU 并行使成本非线性，但仍必须用更长接受长度
   抵消额外工作。
2. **树形路径额外成本明确存在。** 固定版本 DDTree 的 `build_ddtree_tree`
   将 top-k 概率及 token 拷到 CPU，运行 Python heap，构造 NumPy 可见性矩阵；
   `compile_ddtree_tree` 将节点和可见性矩阵拷回 GPU，形成自定义 attention mask；
   接受后通过索引整理 KV。DFlash 是连续块验证和 KV crop。
3. **SDPA 标签不证明具体 kernel。** 连续因果块和任意树 mask 可能走不同路径，
   但本次未取得 profiler kernel trace，不能声称已证实 DDTree 回退到了 math kernel。
4. **同步计时不同但来自固定上游代码。** 两者都在阶段边界调用
   `torch.cuda.synchronize()`；DDTree 多出建树、编译和复制阶段。额外同步的影响
   需要独立实验，不能从源码直接推算毫秒或将其指认为唯一原因。
5. **不是已发现的 C++ 回退故障。** 已读日志显示 C++ tail cache compaction 成功
   加载；对应 worker 要求扩展成功才继续。未找到 DDTree 使用 Python 回退的证据。
6. **平台与数值轨迹是混杂因素。** GH200/Grace 与论文 H200 平台不同，依赖版本
   也未建立逐项等同；只能记录差异，不能归因成“GH200 一定更慢”。

两个本地固定源文件的 SHA256 与运行 contract 记录一致：

- `third_party/ddtree_pinned/ddtree.py`：
  `8d0ecd9d07a7266d3825eefd176811502313a0da7cedfcdc8aec1a4971c3a99b`
- `third_party/ddtree_pinned/dflash.py`：
  `297e5126980ff7bad7650ce7ede9dea9b11b1b3485154ab5b9a2a3f899b33159`

固定提交为 `c96427a185677bf4133ed865dd1626a5041aef9b`。以上源码检查不是对
SSH 断开后远端文件状态的重新认证。

另一个值得检查的外部线索：官方仓库的第三方复现报告称，在 8×H200 上接受长度
合理但速度未复现，并怀疑 SDPA 树 mask 效率。该报告不是作者确认的根因，也不是
本机 kernel 证据。[复现问题 #3](https://github.com/liranringel/ddtree/issues/3)。

## 当前结果不能支持的论文结论

运行策略是 `record-bf16-mismatches`，不是严格逐 token 一致性门禁。独立读取
原始 token 存储，复核了 7680 条方法回答记录；五个数据集内：

- Adaptive 与同后端 AR 不同的回答：365/480。
- 各数据集最佳 DDTree 与 SDPA AR 不同的回答：363/480。
- 最佳后端 DFlash 与相应后端 AR 不同的回答：350/480。
- 两后端 AR 自身不同的回答：375/480。

这里的“不同”是整条回答至少一个 token 不同，不是任务错误率或 token 错误率。
差异可能改变输出长度、后续上下文和执行工作量，不能不经排查一概归为正常 BF16
误差。当前速度结果只能作为该运行条件下的性能观察，不能证明严格 lossless，
也不能据此证明在 T>0 时无偏。还没有任务评分支持质量不下降。

当前每题固定方法顺序、单 seed，没有随机顺序重复计时；控制器跨题保持状态。
同后端各方法共享输入，但 MT-Bench 后续回合的 SDPA 历史取 DDTree1024 输出，
FA2 历史取 DFlash 输出；在允许 token 差异时会产生跨后端输入混杂。

此外五数据集的 no_exploration 相对完整 Adaptive 几何平均加速为 1.3436×，
frozen_after_warmup 为 1.1534×。这不是正式消融最终结论，但目前不支持
“所有完整组件都必要且提升性能”的叙述。

## GBV 必须与这轮实验分开

原 GBV 不能再被当成已经成功的速度架构。在 2089362 开发 pilot 中，Qwen3-4B、
T=1、3 个手写提示、每方法 3 次重复、最多 96 token，pooled decode TPOT 为：
GBV 13.3161、DDTree 9.2433、DFlash 10.3062 ms/token。
新 protected 候选也为 14.7737，没有超过基线。

这些是失败/对照证据，不是完整数据集结论；不能与上面的 8B/T=0/均值 TPOT
直接横向比较。原 GBV 正式作业 2057806 最后仍运行于 n464，日志至 12024/16506，
本次没有取得其完整最终统计。

## 恢复登录后的最小诊断步骤（尚未执行完）

1. 不重跑模型，只读已完成 PT 的 `stage_times`、`decode_rounds`、输出长度与接受
   长度，对 DFlash-SDPA、DFlash-FA2、全部 DDTree 预算分别计算每轮/每 token 成本。
   注意 `tree_build_copy/heap/visibility` 已包含在 `tree_build`，不得重复相加；
   `round_timestamps` 是累计时间，不可直接当每轮耗时取平均。
2. 对总 decode 时间与阶段之和作闭合检查，首轮 draft 排除规则保持一致；同时给出
   官方 mean-response TPOT 和 pooled TPOT，不能混用二者推导接受收益。
3. 按相同 prompt 成对分析，报告长度、EOS、首次 token 分歧及 2k 截断情况。
   只取 token 一致子集也有选择偏差，不能替代全体结果。
4. 若保存的阶段数据仍不能定位具体 kernel，再申请独立小型 profiling/重放实验。
   固定相同上下文与提案，对照连续块、树 mask、节点预算与实际 SDPA kernel；
   随机化方法顺序并重复。该步骤不能修改或替换当前正式结果。

本次未提交该 profiling、未改基线、未改变统计规则来制造 DDTree 获胜。
