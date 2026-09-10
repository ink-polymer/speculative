"""Optional one-launch KV-cache compaction for AdaptiveTree.

The verifier appends the same tree window to every key/value cache tensor.
Official DDTree compacts those tensors one at a time.  This module preserves
the exact in-place operation while batching equal-shaped contiguous tensors
into one Triton launch.  Unsupported layouts take the unchanged fallback.
"""
from __future__ import annotations

import math

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - exercised on installations without Triton
    triton = None
    tl = None


if triton is not None:
    @triton.jit(
        do_not_specialize=("past_length", "sequence_length", "keep_count"),
        do_not_specialize_on_alignment=(
            "past_length", "sequence_length", "keep_count"),
    )
    def _gather_cache_list_kernel(
        pointer_table,
        keep_indices,
        scratch,
        past_length,
        sequence_length,
        inner_size: tl.constexpr,
        outer_size: tl.constexpr,
        keep_count,
        tensor_count: tl.constexpr,
        element_bytes: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        per_tensor = outer_size * keep_count * inner_size
        total = tensor_count * per_tensor
        mask = offset < total
        tensor_index = offset // per_tensor
        remainder = offset % per_tensor
        outer_index = remainder // (keep_count * inner_size)
        remainder = remainder % (keep_count * inner_size)
        kept_index = remainder // inner_size
        inner_index = remainder % inner_size
        address = tl.load(pointer_table + tensor_index, mask=mask, other=0)
        if element_bytes == 2:
            cache = tl.cast(address, tl.pointer_type(tl.uint16))
        else:
            cache = tl.cast(address, tl.pointer_type(tl.uint32))
        source_position = past_length + tl.load(
            keep_indices + kept_index, mask=mask, other=0)
        source_offset = (
            (outer_index * sequence_length + source_position) * inner_size
            + inner_index
        )
        value = tl.load(cache + source_offset, mask=mask)
        tl.store(scratch + offset, value, mask=mask)

    @triton.jit(
        do_not_specialize=("past_length", "sequence_length", "keep_count"),
        do_not_specialize_on_alignment=(
            "past_length", "sequence_length", "keep_count"),
    )
    def _scatter_cache_list_kernel(
        pointer_table,
        scratch,
        past_length,
        sequence_length,
        inner_size: tl.constexpr,
        outer_size: tl.constexpr,
        keep_count,
        tensor_count: tl.constexpr,
        element_bytes: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        per_tensor = outer_size * keep_count * inner_size
        total = tensor_count * per_tensor
        mask = offset < total
        tensor_index = offset // per_tensor
        remainder = offset % per_tensor
        outer_index = remainder // (keep_count * inner_size)
        remainder = remainder % (keep_count * inner_size)
        kept_index = remainder // inner_size
        inner_index = remainder % inner_size
        address = tl.load(pointer_table + tensor_index, mask=mask, other=0)
        if element_bytes == 2:
            cache = tl.cast(address, tl.pointer_type(tl.uint16))
        else:
            cache = tl.cast(address, tl.pointer_type(tl.uint32))
        destination_position = past_length + kept_index
        destination_offset = (
            (outer_index * sequence_length + destination_position) * inner_size
            + inner_index
        )
        value = tl.load(scratch + offset, mask=mask)
        tl.store(cache + destination_offset, value, mask=mask)


class BatchedTritonCacheCompactor:
    """Reuse pointer staging buffers and compact compatible cache tensors."""

    def __init__(self, maximum_tensors: int, device: torch.device,
                 maximum_copy_elements: int = 0, element_size: int = 0):
        self.enabled = bool(
            triton is not None and device.type == "cuda" and maximum_tensors > 0
        )
        self.maximum_tensors = int(maximum_tensors)
        self.host_pointers = None
        self.device_pointers = None
        self.host_pointer_array = None
        self.scratch = None
        self.batched_calls = 0
        if self.enabled:
            self.host_pointers = torch.empty(
                self.maximum_tensors, dtype=torch.int64, device="cpu",
                pin_memory=True)
            self.device_pointers = torch.empty(
                self.maximum_tensors, dtype=torch.int64, device=device)
            self.host_pointer_array = self.host_pointers.numpy()
            if maximum_copy_elements > 0 and element_size in (2, 4):
                scratch_dtype = torch.uint16 if element_size == 2 else torch.uint32
                self.scratch = torch.empty(
                    maximum_copy_elements, dtype=scratch_dtype, device=device)

    def compact(self, tensors, past_length: int, keep_indices) -> bool:
        """Return True after an exact batched compaction, else request fallback."""
        if not self.enabled or not tensors:
            return False
        tensor_count = len(tensors)
        first = tensors[0]
        if (tensor_count > self.maximum_tensors or not first.is_cuda
                or not first.is_contiguous() or first.ndim < 2
                or first.element_size() not in (2, 4)
                or keep_indices.device != first.device
                or keep_indices.dtype != torch.long
                or keep_indices.ndim != 1):
            return False
        sequence_length = int(first.shape[-2])
        inner_size = int(first.shape[-1])
        if past_length < 0 or past_length > sequence_length:
            return False
        keep_count = int(keep_indices.numel())
        current_length = sequence_length - past_length
        if keep_count == 0 or keep_count == current_length:
            self.batched_calls += 1
            return True
        if keep_count > current_length:
            return False
        signature = (first.device, first.dtype, tuple(first.shape))
        if any((not tensor.is_contiguous()
                or (tensor.device, tensor.dtype, tuple(tensor.shape)) != signature)
               for tensor in tensors):
            return False
        outer_size = first.numel() // (sequence_length * inner_size)
        self.host_pointer_array[:tensor_count] = [
            tensor.data_ptr() for tensor in tensors
        ]
        device_pointers = self.device_pointers[:tensor_count]
        device_pointers.copy_(
            self.host_pointers[:tensor_count], non_blocking=True)
        total = tensor_count * outer_size * keep_count * inner_size
        scratch_dtype = (torch.uint16 if first.element_size() == 2
                         else torch.uint32)
        if (self.scratch is None or self.scratch.numel() < total
                or self.scratch.dtype != scratch_dtype
                or self.scratch.device != first.device):
            self.scratch = torch.empty(
                total, dtype=scratch_dtype, device=first.device)
        block = 256
        grid = (math.ceil(total / block),)
        _gather_cache_list_kernel[grid](
            device_pointers,
            keep_indices,
            self.scratch,
            past_length=past_length,
            sequence_length=sequence_length,
            inner_size=inner_size,
            outer_size=outer_size,
            keep_count=keep_count,
            tensor_count=tensor_count,
            element_bytes=first.element_size(),
            BLOCK=block,
        )
        _scatter_cache_list_kernel[grid](
            device_pointers,
            self.scratch,
            past_length=past_length,
            sequence_length=sequence_length,
            inner_size=inner_size,
            outer_size=outer_size,
            keep_count=keep_count,
            tensor_count=tensor_count,
            element_bytes=first.element_size(),
            BLOCK=block,
        )
        self.batched_calls += 1
        return True
