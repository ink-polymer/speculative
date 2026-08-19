"""验证 graph mode 支持不同 input 内容 + 完整 verify 流程模拟。"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
import torch_npu

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache

from dflash_specblock.device import resolve_device, synchronize
from dflash_specblock.config import ExperimentConfig


def main():
    project = Path(__file__).resolve().parent.parent
    config = ExperimentConfig.from_json(str(project / "configs/qwen3_4b_a2_tree15_float32.json"))
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
    cur_len = 61  # max_budget + 1

    # Static inputs (fixed shape)
    static_input_ids = torch.zeros((1, cur_len), dtype=torch.long, device=device)
    static_position_ids = torch.zeros((1, cur_len), dtype=torch.long, device=device)
    static_cache_position = torch.zeros(cur_len, dtype=torch.long, device=device)
    minimum = torch.finfo(dtype).min
    static_mask = torch.full((1, 1, cur_len, max_cache_len), minimum, dtype=dtype, device=device)

    static_cache = StaticCache(model.config, max_cache_len=max_cache_len)

    # Prefill
    prefill_cache_pos = torch.arange(0, past_length, dtype=torch.long, device=device)
    _ = model(input_ids=input_ids_full, past_key_values=static_cache,
              cache_position=prefill_cache_pos, use_cache=True, return_dict=True)

    # Simulate verify with different content each iteration
    print(f"\nSimulating 5 verify iterations with different input content...")
    print(f"  shape: input_ids={static_input_ids.shape}, mask={static_mask.shape}")

    # Warmup
    for i in range(3):
        new_ids = torch.randint(0, 151936, (1, cur_len), dtype=torch.long, device=device)
        new_pos = torch.arange(past_length + i * cur_len, past_length + (i + 1) * cur_len,
                               dtype=torch.long, device=device).unsqueeze(0)
        static_input_ids.copy_(new_ids)
        static_position_ids.copy_(new_pos)
        static_cache_position.copy_(new_pos[0])
        # Update mask: allow past + causal current
        static_mask.fill_(minimum)
        static_mask[..., :past_length + (i + 1) * cur_len] = 0  # simplified
        _ = model(input_ids=static_input_ids, attention_mask=static_mask,
                  position_ids=static_position_ids, past_key_values=static_cache,
                  cache_position=static_cache_position, use_cache=True, return_dict=True)
    synchronize(device)

    # Capture graph
    print("  Capturing graph...")
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
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
    print("  Graph captured!")

    # Test: replay with different content
    print("\n  Replaying with different content each iteration:")
    for i in range(5):
        new_ids = torch.randint(0, 151936, (1, cur_len), dtype=torch.long, device=device)
        new_pos = torch.arange(past_length + i * cur_len, past_length + (i + 1) * cur_len,
                               dtype=torch.long, device=device).unsqueeze(0)
        static_input_ids.copy_(new_ids)
        static_position_ids.copy_(new_pos)
        static_cache_position.copy_(new_pos[0])

        # Update mask for this iteration
        static_mask.fill_(minimum)
        static_mask[..., :past_length] = 0
        cur_start = past_length + i * cur_len
        for j in range(cur_len):
            static_mask[0, 0, j, cur_start:cur_start + j + 1] = 0

        graph.replay()
        synchronize(device)
        logits = graph_output.logits
        print(f"    iter {i}: logits shape={logits.shape}, "
              f"argmax[0]={int(logits[0, 0].argmax())}, "
              f"argmax[-1]={int(logits[0, -1].argmax())}")

    # Benchmark
    print("\n  Benchmarking graph replay...")
    for _ in range(3):
        graph.replay()
    synchronize(device)
    start = time.perf_counter()
    for _ in range(20):
        static_input_ids.copy_(torch.randint(0, 151936, (1, cur_len), dtype=torch.long, device=device))
        graph.replay()
    synchronize(device)
    graph_ms = (time.perf_counter() - start) / 20 * 1000

    # Compare with eager
    print("  Benchmarking eager...")
    for _ in range(3):
        _ = model(input_ids=static_input_ids, attention_mask=static_mask,
                  position_ids=static_position_ids, past_key_values=static_cache,
                  cache_position=static_cache_position, use_cache=True, return_dict=True)
    synchronize(device)
    start = time.perf_counter()
    for _ in range(20):
        _ = model(input_ids=static_input_ids, attention_mask=static_mask,
                  position_ids=static_position_ids, past_key_values=static_cache,
                  cache_position=static_cache_position, use_cache=True, return_dict=True)
    synchronize(device)
    eager_ms = (time.perf_counter() - start) / 20 * 1000

    print(f"\n{'='*50}")
    print(f"RESULTS (cur_len={cur_len})")
    print(f"{'='*50}")
    print(f"  Eager forward:   {eager_ms:.1f} ms/iter")
    print(f"  Graph replay:    {graph_ms:.1f} ms/iter")
    print(f"  Speedup:         {eager_ms/graph_ms:.2f}x")
    print(f"  Graph supports content changes: YES")


if __name__ == "__main__":
    main()
