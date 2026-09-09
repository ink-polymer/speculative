# Adaptive DDTree 历史论文证明核校记录（2026-09-03）

> 本文档对应旧版 `adaptive_legacy`（B<=128、旧成本归因、周期探索），只作历史证明追溯。
> 它不定义或完整证明当前修正后的 canonical AdaptiveTree。当前实验范围以
> [正式实验矩阵](FORMAL_EXPERIMENT_MATRIX.md) 和
> [统一运行与审计说明](INTEGRATED_ADAPTIVE_TREE_BLOCK_EXPERIMENT.md) 为准。

历史证明正文见 [论文版数学证明](ADAPTIVE_DDTREE_T0_PAPER_PROOF.md)。本文档是核校说明，
不属于证明正文，也不能替代当前实现的控制器契约、单因素消融检查与 GPU 结果审计。

## 对应代码与范围

- 对应历史分支 `codex/adaptivetree-official-t0-qwen3-8b-20260903` 的实验代码提交
  `c9b711882e5d63a094ed3040060385030dabc712`；当时的后续提交只增加证明、入口链接和校验清单，
  不改变该历史生成算法。
- 覆盖旧版非 RL、T=0、单次块草稿、最大 128 节点和六个嵌套预算；不是当前 canonical
  `B<=256` budget-aware tree-build cost attribution 控制器的完整证明，也不是 T=1 采样证明。
- [构树器](../src/dflash_specblock/ddtree_builder.py)：核对完整词表归一化、固定最大预算 top-k、兄弟/子节点入堆、前缀截断及概率质量近似。
- [控制器](../src/dflash_specblock/paper/controller.py)：历史核校只覆盖当时的原版决策和四项消融；
  当前 8 方法注册表、6 项消融/控制及 `adaptive_legacy_cost_attribution` 的单因素约束由新契约另行验证。
- [实际生成循环](../src/dflash_specblock/paper/adaptive_official.py)：核对接受长度扣除锚点、bonus 尚未写入 KV、EOS/长度截断，以及反馈的耗时边界。
- [固定版官方验证](../third_party/ddtree_pinned/ddtree.py)：核对位置为 start+depth、ancestor-only mask、唯一子词元查找、接受路径 KV 压缩。

## 数学核校

1. 将“固定槽位分布下的辅助随机序列”与 target 的真实贪心序列分开；不假定二者分布相等。
2. best-first 证明给出唯一前驱与堆前沿论证，处理并列优先级；最优性只限定于该次保留候选空间中的代理期望。
3. 节点质量是期望长度，允许大于 1；代理质量的递减边际不推出真实吞吐最优。
4. 路径等价证明包含各层 KV，而不只比较最终 logits；跨轮归纳明确“缓存包含全部已确认前缀，但不含末尾锚点”。
5. 覆盖零草稿接受、最大深度接受、首锚点终止、EOS、输出上限和计算后再截断；同父词元唯一是显式条件。
6. 将精确计算定理与实际浮点充分条件分开。logit 间隔条件使用实际运行缓存的总误差，未把单轮误差界错误地当成所有轮次的保证。
7. 显式注明有限 mask 哨兵、质量裁剪、top-k 并列和额外 mask 词元删除的适用边界。

## 当时已执行的检查

- 独立发布包的 169 项测试再次全部通过（36.44 秒）。其中 16 项真实小型 Qwen3/DFlash 测试包含 160 次输出对比，覆盖 CPU FP32/BF16、固定树、Adaptive 和消融。
- 既有构树测试包含小候选空间的穷举代理最优性检查；这些有限实例的测试用于发现错误，不代替数学证明。
- 使用本机 VS Code LaTeX Workshop 安装中的 MathJax，逐项转换了论文证明的 136 处行内公式和 24 个公式块，零解析错误。
- 证明采用 Markdown 的单/双美元分隔符，不用代码块包裹公式；上述检查不等同于检查用户所有 VS Code 预览插件的界面表现。
- 工作区与发布包中的论文版证明逐字一致；提交前自有文档差异检查通过。固定版第三方源码保留官方原始空白，不为格式检查修改来源文件。

## 当前仍不能据此声称的结论

该记录没有运行真实 8B checkpoint 或 H200/CUDA/FA2/C++ 正式验收，也没有新的速度或任务精度
结果。169 项历史测试、MathJax 检查及理想计算证明均不构成实际 BF16 完整数据实验必然无损的
保证；也不能推出 corrected canonical AdaptiveTree 的成本归因正确、吞吐更高或 T=1 采样无偏。

新服务器正式实验只运行 Qwen3-4B/8B 的 T=0/T=1 矩阵；30B 延期，树状块验证仅进入
独立 T=1，不属于 AdaptiveTree 的 T=0 证明。canonical AdaptiveTree 的 `B<=256`、无周期探索
和 budget-aware 成本归因必须由当前源码契约、测试、doctor 及完整新跑结果共同支持，不能引用
本文档替代。
