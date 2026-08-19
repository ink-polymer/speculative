"""扫描不同 verify token 数下的 forward 时间,确定固定开销 vs 可变开销。

用于决定最优 tree_budget:在固定开销主导时,budget 越大越好。
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


def bench_forward(model, input_ids, mask, position_ids, cache, cache_position, device,
                  warmup=3, iters=15):
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


def build_causal_mask(past_len, cur_len, dtype, device):
    total = past_len + cur_len
    minimum = torch.finfo(dtype).min
    mask = torch.full((1, 1, cur_len, total), minimum, dtype=dtype, device=device)
    if past_len:
        mask[..., :past_len] = 0
    causal = torch.tril(torch.ones(cur_len, cur_len, dtype=torch.bool, device=device))
    mask[..., past_length:].masked_fill_(causal, 0)
    return mask


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

    prompt = "Explain quantum computing in simple terms. " * 3
    input_ids_full = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    past_length = input_ids_full.shape[1]
    print(f"past_length (prompt tokens): {past_length}")

    cache = DynamicCache()
    _ = model(input_ids=input_ids_full, past_key_values=cache, use_cache=True, return_dict=True)
    cache.crop(past_length)

    token_counts = [1, 4, 8, 16, 31, 61, 91, 121]
    results = {}

    print(f"\n{'cur_len':>8} {'total_len':>10} {'ms/iter':>10} {'ms/token':>10}")
    print("-" * 42)
    for cur_len in token_counts:
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

        try:
            ms = bench_forward(model, input_ids, mask, position_ids, cache,
                              cache_position, device, warmup=3, iters=15)
            results[cur_len] = ms
            print(f"{cur_len:>8} {total:>10} {ms:>10.1f} {ms/cur_len:>10.2f}")
        except Exception as e:
            print(f"{cur_len:>8} {total:>10} ERROR: {type(e).__name__}: {e}")
            results[cur_len] = None

    if results.get(1) and results.get(31):
        fixed = results[1]
        var_per_tok = (results[31] - results[1]) / 30
        print(f"\n固定开销 (cur_len=1): {fixed:.1f} ms")
        print(f"可变开销: {var_per_tok:.2f} ms/token")
        print(f"cur_len=31 时固定占比: {fixed/results[31]*100:.1f}%")

        print("\n预测不同 tree_budget 下的 wall 时间:")
        print(f"{'budget':>8} {'verify_ms':>10} {'acceptance_est':>16} {'iters_est':>10} {'wall_est':>10}")
        for budget, acceptance in [(30, 2.7), (60, 3.2), (90, 4.0), (120, 5.0), (180, 6.5)]:
            cur = budget + 1
            verify_ms = fixed + var_per_tok * cur
            tokens = 128
            iters = tokens / acceptance
            wall = iters * (verify_ms + 25)
            print(f"{budget:>8} {verify_ms:>10.1f} {acceptance:>16.1f} {iters:>10.1f} {wall:>10.0f}")


if __name__ == "__main__":
    main()
