"""在下载大模型前验证 NVIDIA 驱动、CUDA PyTorch 与关键 GPU 算子。"""

from __future__ import annotations

import sys

import torch


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA 不可用：请检查 NVIDIA 驱动、容器 GPU 映射和 CUDA 版 PyTorch。"
        )
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    if not torch.cuda.is_bf16_supported():
        raise SystemExit("当前 GPU 不支持 BF16；正式配置需要 Ampere 或更新架构。")

    left = torch.randn((64, 64), dtype=torch.bfloat16, device=device)
    right = torch.randn((64, 64), dtype=torch.bfloat16, device=device)
    result = left @ right

    # 树验证真正依赖的不只是 matmul：还需要 4D additive mask、softmax、topk、
    # logsumexp 和非连续 index_select。这里用小张量提前暴露 CUDA 算子或精度问题。
    query = torch.randn((1, 2, 4, 8), dtype=torch.bfloat16, device=device)
    key = torch.randn((1, 2, 4, 8), dtype=torch.bfloat16, device=device)
    scores = torch.matmul(query, key.transpose(-2, -1))
    allowed = torch.tril(torch.ones((4, 4), dtype=torch.bool, device=device))
    tree_mask = torch.full(
        (1, 1, 4, 4),
        torch.finfo(torch.bfloat16).min,
        dtype=torch.bfloat16,
        device=device,
    )
    tree_mask.masked_fill_(allowed, 0)
    probabilities = torch.softmax(scores + tree_mask, dim=-1)
    logits = torch.randn((4, 128), dtype=torch.bfloat16, device=device)
    top_values = torch.topk(logits, k=10, dim=-1).values
    log_z = torch.logsumexp(logits.float(), dim=-1)
    selected = logits.index_select(0, torch.tensor([0, 2], device=device))
    torch.cuda.synchronize(device)
    properties = torch.cuda.get_device_properties(device)
    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA runtime: {torch.version.cuda}")
    print(f"cuDNN: {torch.backends.cudnn.version()}")
    print(f"GPU: {properties.name}")
    print(f"Compute capability: {properties.major}.{properties.minor}")
    print(f"BF16 supported: {torch.cuda.is_bf16_supported()}")
    print(f"Flash SDP enabled: {torch.backends.cuda.flash_sdp_enabled()}")
    is_finite = bool(result.isfinite().all().item())
    print(f"BF16 matmul: shape={tuple(result.shape)}, finite={is_finite}")
    print(
        "Tree ops: "
        f"mask_softmax_finite={bool(probabilities.isfinite().all().item())}, "
        f"topk={tuple(top_values.shape)}, logsumexp={tuple(log_z.shape)}, "
        f"index_select={tuple(selected.shape)}"
    )
    print("NVIDIA CUDA 环境检查通过。")


if __name__ == "__main__":
    main()
