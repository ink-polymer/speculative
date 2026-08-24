"""NVIDIA CUDA device helpers and accurate accelerator timing."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

import torch


def cuda_available() -> bool:
    """Return whether PyTorch can use at least one CUDA device."""
    return bool(torch.cuda.is_available() and torch.cuda.device_count() > 0)


def resolve_device(requested: str = "auto") -> torch.device:
    """Resolve ``auto``/``cpu``/``cuda[:id]`` and fail closed for explicit CUDA."""
    normalized = requested.strip().lower()
    if normalized == "auto":
        if not cuda_available():
            return torch.device("cpu")
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        return device
    if normalized == "cpu":
        return torch.device("cpu")
    if re.fullmatch(r"cuda(?::\d+)?", normalized):
        if not cuda_available():
            raise RuntimeError(
                "CUDA was requested, but torch.cuda is unavailable. Check the NVIDIA driver, "
                "the CUDA-enabled PyTorch build, and GPU visibility."
            )
        index = 0 if ":" not in normalized else int(normalized.rsplit(":", 1)[1])
        device_count = int(torch.cuda.device_count())
        if index >= device_count:
            raise ValueError(
                f"CUDA device index {index} is out of range; {device_count} device(s) are visible"
            )
        device = torch.device("cuda", index)
        torch.cuda.set_device(device)
        return device
    raise ValueError("device must be one of auto, cpu, cuda, or cuda:<id>")


def configure_cuda_runtime(device: torch.device, allow_tf32: bool = True) -> None:
    """Configure the process-wide CUDA floating-point performance policy."""
    if device.type != "cuda":
        return
    if not isinstance(allow_tf32, bool):
        raise TypeError("allow_tf32 must be a boolean")

    precision = "tf32" if allow_tf32 else "ieee"
    if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
        # PyTorch 2.9+ forbids mixing this API with the legacy allow_tf32 flags.
        torch.backends.cuda.matmul.fp32_precision = precision
        torch.backends.cudnn.fp32_precision = precision
    else:
        torch.set_float32_matmul_precision("high" if allow_tf32 else "highest")
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def dtype_from_name(name: str) -> torch.dtype:
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    try:
        return mapping[name]
    except KeyError as exc:
        raise ValueError(f"unknown dtype: {name}") from exc


@dataclass
class DeviceTimer:
    """CUDA-event timer with an opt-in wall clock for host-heavy stages."""

    device: torch.device
    use_cuda_events: bool = True
    started_at: float = 0.0
    stopped_at: float = 0.0
    _start_event: object | None = field(default=None, init=False, repr=False)
    _end_event: object | None = field(default=None, init=False, repr=False)
    _elapsed_ms: float = field(default=0.0, init=False, repr=False)

    def __enter__(self) -> "DeviceTimer":
        if self.device.type == "cuda" and self.use_cuda_events:
            with torch.cuda.device(self.device):
                self._start_event = torch.cuda.Event(enable_timing=True)
                self._end_event = torch.cuda.Event(enable_timing=True)
                self._start_event.record()
        else:
            self.started_at = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self.device.type == "cuda" and self.use_cuda_events:
            if self._start_event is None or self._end_event is None:
                raise RuntimeError("CUDA timer events were not initialized")
            with torch.cuda.device(self.device):
                self._end_event.record()
                self._end_event.synchronize()
                self._elapsed_ms = float(self._start_event.elapsed_time(self._end_event))
        else:
            self.stopped_at = time.perf_counter()
            self._elapsed_ms = (self.stopped_at - self.started_at) * 1000.0

    @property
    def elapsed_ms(self) -> float:
        if self._elapsed_ms:
            return self._elapsed_ms
        if self.device.type == "cuda" and self.use_cuda_events:
            return 0.0
        end = self.stopped_at or time.perf_counter()
        return (end - self.started_at) * 1000.0
