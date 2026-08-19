"""测试 NPU graph mode 对 verify forward 的加速效果。

固定 shape + padding,让 graph 可 capture。
对比:
1. eager forward (dynamic shape) - 当前实现
2. graph replay (fixed shape + padding)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
import torch_npu

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from dflash_specblock.device import resolve_device, synchronize
from dflash_specblock.config import ExperimentConfig


def bench_eager(model, input_ids, mask, position_ids, cache, cache_position, device,
                warmup=3, iters=20):
    """测试动态 shape 的 eager forward。"""
    for _ in range(warmup):
        cache.crop(input_ids.shape[1])
        _ = model(input_ids=input_ids, attention_mask=mask,
                  position_ids=position_ids, past_key_values=cache,
                  cache_position=cache_position, use_cache=True, return_dict=True)
    synchronize(device)
    start = time.perf_counter()
    for _ in range(iters):
        cache.crop(input_ids.shape[1])
        _ = model(input_ids=input_ids, attention_mask=mask,
                  position_ids=position_ids, cache_position=cache_position,
                  past_key_values=cache, use_cache=True, return_dict=True)
    synchronize(device)
    return (time.perf_counter() - start) / iters * 1000


def bench_graph(model, input_ids, mask, position_ids, cache, cache_position, device,
                warmup=3, iters=20):
    """测试固定 shape 的 graph replay。"""
    # Capture graph
    torch.npu.synchronize(device)
    # Warmup before capture
    for _ in range(3):
        cache.crop(input_ids.shape[1])
        _ = model(input_ids=input_ids, attention_mask=mask,
                  position_ids=position_ids, past_key_values=cache,
                  cache_position=cache_position, use_cache=True, return_dict=True)
    cache.crop(input_ids.shape[1])
    synchronize(device)

    # Capture
    graph = torch.npu.NPUGraph()
    static_cache = DynamicCache()
    # Need to pre-populate static cache with same content
    # For graph capture, we need static tensors
    static_input_ids = input_ids.clone()
    static_mask = mask.clone()
    static_position_ids = position_ids.clone()
    static_cache_position = cache_position.clone()

    try:
        with torch.npu.graph(graph):
            static_output = model(
                input_ids=static_input_ids,
                attention_mask=static_mask,
                position_ids=static_position_ids,
                past_key_values=cache,
                cache_position=static_cache_position,
                use_cache=True,
                return_dict=True,
            )
        synchronize(device)

        # Replay
        start = time.perf_counter()
        for _ in range(iters):
            graph.replay()
        synchronize(device)
        return (time.perf_counter() - start) / iters * 1000
    except Exception as e:
        print(f"  Graph capture failed: {type(e).__name__}: {e}")
        return None


def main():
    project = Path(__file__).resolve().parent.parent
    config = ExperimentConfig.from_json(str(project / "configs/qwen3_4b_a2_tree15_float32.json"))
    device = resolve_device(config.device)
    dtype = torch.float32

    print(f"Loading Qwen3-4B (eager) on {device}")
    tokenizer = AutoTokenizer.from_pretrained(config.target_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        config.target_path, trust_remote_code=True, dtype=dtype,
        low_cpu_mem_usage=True, attn_implementation="eager",
    ).eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)

    prompt = "Explain quantum computing in simple terms."
    input_ids_full = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    past_length = input_ids_full.shape[1]

    # Prefill
    cache = DynamicCache()
    _ = model(input_ids=input_ids_full, past_key_values=cache, use_cache=True, return_dict=True)
    cache.crop(past_length)

    cur_len = 31
    input_ids = torch.randint(0, 151936, (1, cur_len), dtype=torch.long, device=device)
    position_ids = torch.arange(past_length, past_length + cur_len,
                                dtype=torch.long, device=device).unsqueeze(0)
    cache_position = position_ids.clone()

    minimum = torch.finfo(dtype).min
    total = past_length + cur_len
    mask = torch.full((1, 1, cur_len, total), minimum, dtype=dtype, device=device)
    mask[..., :past_length] = 0
    causal = torch.tril(torch.ones(cur_len, cur_len, dtype=torch.bool, device=device))
    mask[..., past_length:].masked_fill_(causal, 0)

    print(f"\nTesting cur_len={cur_len}, past_length={past_length}")

    eager_ms = bench_eager(model, input_ids, mask, position_ids, cache, cache_position, device)
    print(f"  Eager forward:    {eager_ms:.1f} ms/iter")

    graph_ms = bench_graph(model, input_ids, mask, position_ids, cache, cache_position, device)
    if graph_ms is not None:
        print(f"  Graph replay:     {graph_ms:.1f} ms/iter")
        print(f"  Speedup:          {eager_ms/graph_ms:.2f}x")


if __name__ == "__main__":
    main()
