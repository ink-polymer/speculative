# NVIDIA CUDA 环境与加速说明

## 1. 支持范围

本工程的正式训练、推理和基准设备为 NVIDIA GPU，设备字符串使用 `cuda:<id>`。正式配置
采用 BF16；建议使用计算能力 8.0 或更新的 GPU，以获得 BF16、Tensor Core、SDPA 高效后端
和 CUDA Graph 支持。CPU 仅适合轻量开发检查，不是正式性能后端。

运行前需要：

1. 与 GPU 匹配的 NVIDIA 驱动；
2. CUDA 版 PyTorch，而不是仅 CPU 的 wheel；
3. 与 PyTorch、驱动和 Python 版本兼容的 Transformers 依赖。

版本与安装命令应以 PyTorch 官方安装选择器和 NVIDIA 驱动兼容矩阵为准：

- <https://pytorch.org/get-started/locally/>
- <https://docs.nvidia.com/deploy/cuda-compatibility/>

## 2. 安装与自检

```bash
conda create -n dflash-specblock python=3.11 -y
conda activate dflash-specblock

# 先按 PyTorch 官方安装选择器安装匹配的 CUDA wheel，再安装工程依赖。
pip install -r requirements-gpu.txt
pip install -e .

export CUDA_VISIBLE_DEVICES=0
nvidia-smi
python scripts/check_cuda_env.py
```

自检会在 `cuda:0` 上执行 BF16 matmul、四维 additive mask、softmax、`topk`、
`logsumexp` 和非连续 `index_select`，并输出 CUDA runtime、cuDNN、GPU 名称、计算能力、
BF16 支持和 PyTorch Flash SDPA 开关。任何一项失败都应先修复环境，再运行正式任务。

## 3. 配置开关

正式 JSON 配置包含以下 GPU 字段：

- `device: "cuda:0"`：没有 CUDA 时直接失败，不静默切到 CPU；
- `attn_implementation: "sdpa"`：使用 PyTorch scaled dot product attention 调度；
- `allow_tf32: true`：允许 Ampere 及更新 GPU 用 TF32 加速 FP32 matmul；
- `torch_compile_mode: null`：默认关闭编译，按目标环境验证后可设为
  `"reduce-overhead"` 或 `"max-autotune"`；
- `use_cuda_graphs`：仅固定 shape 的图验证配置启用；
- `cuda_graph_max_cache_len: 4096`：CUDA Graph 静态 KV cache 的容量上限。

`allow_tf32=true` 会改变严格 IEEE FP32 的数值口径。需要做逐位或高精度对照时，应建立独立
配置并设为 `false`；不能把启用 TF32 的结果描述为严格 FP32 基准。

## 4. 注意力后端

PyTorch SDPA 会根据 dtype、shape、mask 和 GPU 能力自动选择 Flash、内存高效或数学后端。
标准因果注意力通常最容易进入 Flash 内核；ancestor-only 树 mask 是自定义四维 mask，未必
满足所有 Flash 内核约束。工程必须保留语义正确的 SDPA/数学回退，不能为了强制使用某个
内核而改变树的可见性。

如需尝试外部 FlashAttention-2，应在目标模型、PyTorch、CUDA 与编译器版本完全匹配的
Linux 环境中单独安装，并分别验证标准因果路径和树 mask 路径。不能仅凭包已安装就宣称
树验证进入了 FlashAttention 内核。

## 5. CUDA Graph 与 `torch.compile`

CUDA Graph 适合重复执行、地址与 shape 固定的 verify forward。capture 前必须完成模型
加载、缓存分配和足够 warmup；replay 期间不得发生动态分配或依赖 Python 控制流改变 shape。
配置变化、上下文超过 `cuda_graph_max_cache_len` 或 capture 失败时，应明确回退 eager 路径。

`torch.compile` 和 CUDA Graph 不应未经验证同时叠加。推荐顺序：

1. 先验证 eager + SDPA 的正确性；
2. 独立测量 CUDA Graph；
3. 独立测量 `torch.compile`；
4. 只有在精确输出、峰值显存和长稳测试都通过后再组合。

## 6. 工业级基准要求

- 固定模型 revision、prompt 集、随机种子、dtype 和所有配置；
- 先 warmup，再测量；计时边界必须包含 `torch.cuda.synchronize()` 或 CUDA event；
- 同时记录驱动、CUDA runtime、PyTorch、Transformers、GPU 型号、显存和功耗状态；
- 分开报告 prefill、draft、verify、cache compact、端到端延迟、吞吐和峰值显存；
- 至少报告 P50/P95/P99，并进行长时间稳定性与 OOM 恢复测试；
- 用相同目标模型逐 token greedy 结果验证 lossless exact match。

## 7. 常见问题

- `torch.cuda.is_available() == False`：检查 NVIDIA 驱动、容器 `--gpus` 映射和 CUDA 版
  PyTorch wheel；
- `no kernel image` 或非法指令：wheel 的 CUDA 架构与 GPU 不兼容；
- CUDA Graph capture 失败：检查动态 shape、capture 期间分配、CPU-GPU 同步和 cache 容量；
- SDPA 退回数学实现：检查 dtype、head dimension、mask 形状和 PyTorch 版本；
- OOM：先减小 `max_new_tokens`、`tree_budget` 或 graph cache 容量，再考虑量化或并行化。
