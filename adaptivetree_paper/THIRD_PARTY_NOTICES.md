# Third-party notices

本包的 DDTree 官方运行代码来自 [liranringel/ddtree](https://github.com/liranringel/ddtree/tree/c96427a185677bf4133ed865dd1626a5041aef9b)，MIT License，Copyright 2026 Liran Ringel。完整许可保存在 third_party/ddtree_pinned/LICENSE。paper/adaptive_official.py 是该版本 ddtree_generate 的派生实现，保留同一许可及来源说明。

历史 builder 回归引用的 third_party/ddtree_official/ddtree.py 保留其目录中的 MIT LICENSE。其他独立 SpecBlock 语义实现不复制未声明许可证的 SpecBlock 源文件；它们不是当前论文方法的创新声明。

其他依赖与参考：

- [DFlash](https://github.com/z-lab/dflash)：MIT License。
- [SpecBlock](https://github.com/shiweijiezero/SpecBlock)：算法参考。
- [PyTorch](https://github.com/pytorch/pytorch)：BSD-style license。
- [FlashAttention](https://github.com/Dao-AILab/flash-attention)：BSD-3-Clause；本正式协议需要 FA2。
- NVIDIA CUDA Toolkit：受 NVIDIA CUDA Toolkit EULA 约束。

Qwen 和 DFlash 权重及数据集各自遵循模型卡/数据卡许可；不随本包分发。详细固定版本见 [官方来源](third_party/OFFICIAL_SOURCES.md)。
