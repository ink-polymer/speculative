"""昇腾设备抽象。

这里显式使用 ``torch.npu``，并只保留 CPU 作为单元测试回退。工程不把 NPU 伪装成其他设备，
也不调用其他加速后端的 event、graph 或 synchronize API。
"""

from __future__ import annotations

import importlib
import re
import time
from dataclasses import dataclass

import torch


def _load_torch_npu() -> bool:
    try:
        importlib.import_module("torch_npu")
    except ImportError:
        return False
    return hasattr(torch, "npu")


def npu_available() -> bool:
    return _load_torch_npu() and bool(torch.npu.is_available())


def resolve_device(requested: str = "auto") -> torch.device:
    requested = requested.lower()
    if requested == "auto":
        return torch.device("npu:0") if npu_available() else torch.device("cpu")
    if re.fullmatch(r"npu(?::\d+)?", requested):
        if not npu_available():
            raise RuntimeError(
                "请求了 NPU，但 torch.npu 不可用。请检查驱动、CANN、torch 与 torch_npu 版本。"
            )
        device = torch.device(requested)
        torch.npu.set_device(device)
        return device
    if requested == "cpu":
        return torch.device("cpu")
    raise ValueError("本工程只接受 auto、cpu 或 npu:<id> 设备字符串")


def synchronize(device: torch.device) -> None:
    if device.type == "npu":
        torch.npu.synchronize(device)


def dtype_from_name(name: str) -> torch.dtype:
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    try:
        return mapping[name]
    except KeyError as exc:
        raise ValueError(f"未知 dtype: {name}") from exc


@dataclass
class DeviceTimer:
    """包含 NPU 异步同步的墙钟计时器。"""

    device: torch.device
    started_at: float = 0.0
    stopped_at: float = 0.0

    def __enter__(self) -> "DeviceTimer":
        synchronize(self.device)
        self.started_at = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        synchronize(self.device)
        self.stopped_at = time.perf_counter()

    @property
    def elapsed_ms(self) -> float:
        end = self.stopped_at or time.perf_counter()
        return (end - self.started_at) * 1000.0
