# Third-party notices

本工程是独立语义移植，没有复制 SpecBlock 未声明许可证的源文件。算法和接口设计参考：

- DFlash paper: <https://arxiv.org/abs/2602.06036>
- DFlash code: <https://github.com/z-lab/dflash>（MIT License）
- SpecBlock paper: <https://arxiv.org/abs/2605.07243>
- SpecBlock code: <https://github.com/shiweijiezero/SpecBlock>
- PyTorch: <https://github.com/pytorch/pytorch>（BSD-style license）
- NVIDIA CUDA Toolkit: <https://developer.nvidia.com/cuda-toolkit>（受 NVIDIA CUDA Toolkit
  EULA 约束）
- FlashAttention（可选性能实现）: <https://github.com/Dao-AILab/flash-attention>
  （BSD-3-Clause License）

为便于审计，本工程固定参考 DFlash commit
`94e4abc5e0c31b67bc1a9d30f1cc34ece28a8756` 与 SpecBlock commit
`b7938c556e5dc42236362dcc4eb37a1edb562c70`。本工程使用 PyTorch CUDA/SDPA 重现算法语义；
CUDA Graph、编译和注意力 kernel 优化不改变方法定义。

下载的 Qwen 与 DFlash 权重分别受其模型卡许可证约束，权重不会打包进本工程。
