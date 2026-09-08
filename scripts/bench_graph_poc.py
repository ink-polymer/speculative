"""POC: CUDA Graph + StaticCache 加速 verify forward。

测试三种模式:
1. DynamicCache + eager (当前实现)
2. StaticCache + eager (固定 shape)
3. StaticCache + CUDA Graph (graph replay)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache, StaticCache

from dflash_specblock.device import resolve_device, synchronize
from dflash_specblock.config import ExperimentConfig


def bench(fn, device, warmup=3, iters=20):
    for _ in range(warmup):
        fn()
    synchronize(device)
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    synchronize(device)
    return (time.perf_counter() - start) / iters * 1000


def main():
    project = Path(__file__).resolve().parent.parent
    config = ExperimentConfig.from_json(str(project / "configs/qwen3_4b_cuda_tree15_float32.json"))
    device = resolve_device(config.device)
    dtype = torch.float32

    print(f"Loading Qwen3-4B on {device}")
    tokenizer = AutoTokenizer.from_pretrained(config.target_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        config.target_path, trust_remote_code=True, dtype=dtype,
        low_cpu_mem_usage=True, attn_implementation="eager",
    ).eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)

    prompt = "Explain quantum computing. " * 5
    input_ids_full = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    past_length = input_ids_full.shape[1]
    max_cache_len = past_length + 256

    cur_len = 31
    input_ids = torch.randint(0, 151936, (1, cur_len), dtype=torch.long, device=device)
    position_ids = torch.arange(past_length, past_length + cur_len,
                                dtype=torch.long, device=device).unsqueeze(0)

    minimum = torch.finfo(dtype).min
    total = past_length + cur_len
    mask_4d = torch.full((1, 1, cur_len, max_cache_len), minimum, dtype=dtype, device=device)
    mask_4d[..., :past_length] = 0
    mask_4d[..., past_length:total] = 0
    causal = torch.tril(torch.ones(cur_len, cur_len, dtype=torch.bool, device=device))
    mask_4d[..., past_length:total].masked_fill_(~causal, minimum)

    # === 1. DynamicCache + eager (baseline) ===
    print("\n1. DynamicCache + eager (current)")
    dyn_cache = DynamicCache()
    _ = model(input_ids=input_ids_full, past_key_values=dyn_cache, use_cache=True, return_dict=True)

    def dyn_forward():
        dyn_cache.crop(past_length)
        return model(input_ids=input_ids, attention_mask=mask_4d,
                     position_ids=position_ids, past_key_values=dyn_cache,
                     use_cache=True, return_dict=True)

    dyn_ms = bench(dyn_forward, device)
    print(f"   {dyn_ms:.1f} ms/iter")

    # === 2. StaticCache + eager ===
    print("\n2. StaticCache + eager")
    static_cache = StaticCache(model.config, max_cache_len=max_cache_len)

    # Prefill into static cache
    cache_position_prefill = torch.arange(0, past_length, dtype=torch.long, device=device)
    _ = model(input_ids=input_ids_full, past_key_values=static_cache,
              cache_position=cache_position_prefill, use_cache=True, return_dict=True)

    static_input_ids = input_ids.clone()
    static_mask = mask_4d.clone()
    static_position_ids = position_ids.clone()
    static_cache_position = torch.arange(past_length, past_length + cur_len,
                                         dtype=torch.long, device=device)

    def static_forward():
        return model(input_ids=static_input_ids, attention_mask=static_mask,
                     position_ids=static_position_ids, past_key_values=static_cache,
                     cache_position=static_cache_position, use_cache=True, return_dict=True)

    try:
        static_ms = bench(static_forward, device)
        print(f"   {static_ms:.1f} ms/iter")
    except Exception as e:
        print(f"   ERROR: {type(e).__name__}: {e}")
        static_ms = None

    # === 3. StaticCache + CUDA Graph ===
    if static_ms is not None:
        print("\n3. StaticCache + CUDA Graph")
        try:
            # Warmup
            for _ in range(3):
                _ = static_forward()
            synchronize(device)

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                graph_output = model(
                    input_ids=static_input_ids,
                    attention_mask=static_mask,
                    position_ids=static_position_ids,
                    past_key_values=static_cache,
                    cache_position=static_cache_position,
                    use_cache=True,
                    return_dict=True,
                )
            synchronize(device)

            def graph_replay():
                graph.replay()
                return graph_output

            graph_ms = bench(graph_replay, device)
            print(f"   {graph_ms:.1f} ms/iter")

            print(f"\n{'='*50}")
            print(f"SUMMARY (cur_len={cur_len}, past={past_length})")
            print(f"{'='*50}")
            print(f"  DynamicCache + eager:  {dyn_ms:.1f} ms")
            print(f"  StaticCache + eager:   {static_ms:.1f} ms")
            print(f"  StaticCache + graph:   {graph_ms:.1f} ms")
            print(f"  Graph speedup vs dyn:  {dyn_ms/graph_ms:.2f}x")
        except Exception as e:
            print(f"   ERROR: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
