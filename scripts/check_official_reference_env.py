#!/usr/bin/env python3
"""Fail-closed CUDA/dependency check for the two official reference environments."""

from __future__ import annotations

import argparse
import importlib.util


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--implementation", choices=("dflash", "ddtree"), required=True)
    args = parser.parse_args()

    import torch
    import transformers

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this Python environment")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("The selected NVIDIA GPU does not support BF16")
    if args.implementation == "ddtree" and importlib.util.find_spec("flash_attn") is None:
        raise RuntimeError("DDTree requires flash-attn")

    device = torch.device("cuda:0")
    value = torch.randn((64, 64), device=device, dtype=torch.bfloat16)
    result = value @ value
    torch.cuda.synchronize(device)
    print(
        f"{args.implementation}: OK; torch={torch.__version__}; "
        f"transformers={transformers.__version__}; gpu={torch.cuda.get_device_name(device)}; "
        f"dtype={result.dtype}"
    )


if __name__ == "__main__":
    main()
