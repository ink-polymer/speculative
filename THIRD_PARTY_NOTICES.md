# Third-party notices

本工程是独立语义移植，没有复制 SpecBlock 未声明许可证的源文件。算法和接口设计参考：

- DFlash paper: <https://arxiv.org/abs/2602.06036>
- DFlash code: <https://github.com/z-lab/dflash>（MIT License）
- SpecBlock paper: <https://arxiv.org/abs/2605.07243>
- SpecBlock code: <https://github.com/shiweijiezero/SpecBlock>
- Ascend Extension for PyTorch: <https://github.com/Ascend/pytorch>（BSD-style license）

为便于审计，本工程固定参考 DFlash commit
`94e4abc5e0c31b67bc1a9d30f1cc34ece28a8756` 与 SpecBlock commit
`b7938c556e5dc42236362dcc4eb37a1edb562c70`。Ascend 版本使用纯 PyTorch 张量操作重现算法
语义，不移植 CUDA/Triton 性能 kernel。

下载的 Qwen 与 DFlash 权重分别受其模型卡许可证约束，权重不会打包进本工程。
