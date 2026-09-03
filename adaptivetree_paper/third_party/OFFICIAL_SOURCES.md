# 固定官方来源

当前正式执行器直接使用 DDTree 官方提交 [c96427a](https://github.com/liranringel/ddtree/tree/c96427a185677bf4133ed865dd1626a5041aef9b)。

ddtree_pinned 目录保存 benchmark.py、run_benchmark.sh、make_latex_table.py、ddtree.py、dflash.py、distributed.py、model 模块、requirements.txt 及 MIT LICENSE。运行时检查该目录 SOURCE_SHA256.json；仅补齐缺失的结尾 LF，不改动官方逻辑。

- 数据处理、prompt、采样、原始基线和 TPOT 函数均来自该版本。
- paper/adaptive_official.py 从其 DDTree 循环派生，仅接入原版 Adaptive 构树、反馈与诊断；保留版权来源。
- DDTreeBuilder/LatencyAwareDDTreeBuilder 恢复自本项目 [9dd6769](https://github.com/ink-polymer/speculative/tree/9dd67698ad828b8c3fca8659e3a388f0b2dfbdf7)。
- 旧 ddtree_official/ddtree.py 只供历史 builder 等价回归；当前不从该目录加载生成代码。
- 不分发模型权重和数据集；下载时记录不可变 revision。上游未固定其历史 HF 快照，不能据此保证与作者历史快照完全相同。
- 无 GBV 实验或新正式性能结果；CPU 契约测试不能替代 BF16/GPU 验证。
