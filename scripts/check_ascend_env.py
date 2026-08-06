"""在下载大模型前快速验证 torch_npu、CANN 动态库和基础 NPU 算子。"""

from __future__ import annotations

import sys

import torch


def main() -> None:
    try:
        import torch_npu  # noqa: F401
    except ImportError as exc:
        raise SystemExit("未安装 torch_npu，或其版本与当前 Python 不匹配") from exc

    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise SystemExit("torch.npu 不可用：请检查驱动、CANN set_env.sh 与版本矩阵")
    torch.npu.set_device("npu:0")
    left = torch.randn((64, 64), dtype=torch.bfloat16, device="npu:0")
    right = torch.randn((64, 64), dtype=torch.bfloat16, device="npu:0")
    result = left @ right

    # 树验证真正依赖的不只是 matmul：还需要 4D additive mask、softmax、topk、
    # logsumexp 和非连续 index_select。这里用小张量提前暴露 torch_npu 算子缺口。
    query = torch.randn((1, 2, 4, 8), dtype=torch.bfloat16, device="npu:0")
    key = torch.randn((1, 2, 4, 8), dtype=torch.bfloat16, device="npu:0")
    scores = torch.matmul(query, key.transpose(-2, -1))
    allowed = torch.tril(torch.ones((4, 4), dtype=torch.bool, device="npu:0"))
    tree_mask = torch.full(
        (1, 1, 4, 4),
        torch.finfo(torch.bfloat16).min,
        dtype=torch.bfloat16,
        device="npu:0",
    )
    tree_mask.masked_fill_(allowed, 0)
    probabilities = torch.softmax(scores + tree_mask, dim=-1)
    logits = torch.randn((4, 128), dtype=torch.bfloat16, device="npu:0")
    top_values = torch.topk(logits, k=10, dim=-1).values
    log_z = torch.logsumexp(logits.float(), dim=-1)
    selected = logits.index_select(0, torch.tensor([0, 2], device="npu:0"))
    torch.npu.synchronize()
    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    print(f"torch_npu: {torch_npu.__version__}")
    print(f"NPU: {torch.npu.get_device_name(0)}")
    is_finite = bool(result.isfinite().all().item())
    print(f"BF16 matmul: shape={tuple(result.shape)}, finite={is_finite}")
    print(
        "Tree ops: "
        f"mask_softmax_finite={bool(probabilities.isfinite().all().item())}, "
        f"topk={tuple(top_values.shape)}, logsumexp={tuple(log_z.shape)}, "
        f"index_select={tuple(selected.shape)}"
    )
    print("Ascend 环境检查通过。")


if __name__ == "__main__":
    main()
