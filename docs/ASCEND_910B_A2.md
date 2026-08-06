# 昇腾 910B / Atlas A2 环境说明

## 1. 软件栈

昇腾 PyTorch 不是 CUDA PyTorch。运行前必须已经安装：

1. Atlas A2 对应的 NPU 驱动与固件；
2. 与驱动匹配的 CANN Toolkit、ops、NNAL；
3. 与 CANN 严格配套的 PyTorch 与 `torch_npu` wheel。

不要直接执行 `pip install torch` 覆盖服务器镜像自带版本。华为官方版本矩阵：
<https://github.com/Ascend/pytorch#ascend-auxiliary-software>。

## 2. 每次登录后的环境初始化

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0
npu-smi info
```

如果 CANN 安装在其他位置，请将第一行替换为实际 `set_env.sh` 路径。

## 3. 创建环境

以下命令假设服务器已经提供配套 `torch` 和 `torch_npu` wheel；具体文件名以管理员提供的
CANN 版本为准。

```bash
conda create -n dflash-specblock python=3.10 -y
conda activate dflash-specblock

# 示例：先安装管理员提供且与 CANN 配套的两个 wheel。
# pip install /path/to/torch-*.whl
# pip install /path/to/torch_npu-*.whl

pip install -r requirements-ascend.txt
pip install -e .
python scripts/check_ascend_env.py
```

`torch_npu` 2.5.1 及更新版本通常支持自动加载；工程仍会显式尝试导入 `torch_npu`，以兼容
较老的 A2 镜像。

环境自检会在 `npu:0` 上实际运行 BF16 matmul、四维 additive attention mask、softmax、
`topk`、`logsumexp` 和 `index_select`。任何一项失败都表示当前软件栈还不能运行本工程，
不应继续正式实验。

## 4. 本工程的设备约束

- 设备字符串使用 `npu:0`；
- 同步使用 `torch.npu.synchronize()`；
- 可见卡使用 `ASCEND_RT_VISIBLE_DEVICES`；
- 不依赖 CUDA event、CUDA graph、Triton 或 FlashAttention CUDA kernel；
- 树拓扑和祖先 mask 使用纯 PyTorch，优先正确性与可移植性；
- 目标模型强制 eager attention，以确保自定义 4D ancestor-only mask 被保留。

正式测速前应先完成 20-50 次 warmup，并用 `npu-smi info` 确认没有其他任务占卡。

## 5. 常见问题

- `ImportError: libascendcl.so`：没有 source CANN 的 `set_env.sh`，或 CANN 安装不完整。
- `torch.npu.is_available() == False`：驱动/CANN/torch_npu 版本不匹配，先修复环境，不要修改代码绕过。
- 算子报 `not supported`：先确认版本矩阵，再尝试更新到同一 CANN 分支的较新 `torch_npu` patch。
- OOM：先减小 `max_new_tokens` 和 `tree_budget`；Qwen3-4B + DFlash-4B draft 在 BF16 下仍需为
  KV cache、目标多层特征和树验证留出空间。
