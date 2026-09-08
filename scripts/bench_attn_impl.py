"""测试 Qwen3-4B 在 NVIDIA GPU 上不同 attn_impl + mask 组合下的 forward 性能。

目标:确认 SDPA fast path 是否能显著快于当前 eager + 4D additive mask。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from transformers import AutoModelForCausalLM, AutoTokenizer

from dflash_specblock.device import resolve_device, synchronize
from dflash_specblock.config import ExperimentConfig


def build_causal_4d_mask(past_len, cur_len, dtype, device):
    total = past_len + cur_len
    minimum = torch.finfo(dtype).min
    mask = torch.full((1, 1, cur_len, total), minimum, dtype=dtype, device=device)
    if past_len:
        mask[..., :past_len] = 0
    causal = torch.tril(torch.ones(cur_len, cur_len, dtype=torch.bool, device=device))
    mask[..., past_len:].masked_fill_(causal, 0)
    return mask


def build_tree_4d_mask(past_len, cur_len, dtype, device):
    """Build a deterministic binary-tree ancestor mask and tree-depth positions."""
    total = past_len + cur_len
    minimum = torch.finfo(dtype).min
    allowed = torch.zeros((cur_len, cur_len), dtype=torch.bool, device=device)
    allowed[0, 0] = True
    depths = torch.zeros(cur_len, dtype=torch.long, device=device)
    for node_index in range(cur_len - 1):
        row = node_index + 1
        allowed[row, 0] = True
        current = node_index
        depth = 0
        while current >= 0:
            allowed[row, current + 1] = True
            depth += 1
            current = -1 if current == 0 else (current - 1) // 2
        depths[row] = depth
    mask = torch.full((1, 1, cur_len, total), minimum, dtype=dtype, device=device)
    if past_len:
        mask[..., :past_len] = 0
    mask[0, 0, :, past_len:].masked_fill_(allowed, 0)
    return mask, past_len + depths.unsqueeze(0)


@torch.inference_mode()
def bench_one(
    model,
    input_ids,
    attention_mask,
    position_ids,
    cache,
    cache_position,
    past_length,
    device,
    warmup=3,
    iters=10,
):
    synchronize(device)
    for _ in range(warmup):
        _ = model(input_ids=input_ids, attention_mask=attention_mask,
                  position_ids=position_ids, past_key_values=cache,
                  cache_position=cache_position, use_cache=True, return_dict=True)
        cache.crop(past_length)
    synchronize(device)
    start = time.perf_counter()
    for _ in range(iters):
        _ = model(input_ids=input_ids, attention_mask=attention_mask,
                  position_ids=position_ids, past_key_values=cache,
                  cache_position=cache_position, use_cache=True, return_dict=True)
        cache.crop(past_length)
    synchronize(device)
    return (time.perf_counter() - start) / iters * 1000


@torch.inference_mode()
def main():
    project = Path(__file__).resolve().parent.parent
    config = ExperimentConfig.from_json(str(project / "configs/qwen3_4b_cuda_tree15.json"))
    device = resolve_device(config.device)
    dtype = (
        torch.bfloat16
        if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    target_path = config.target_path

    print(f"Loading Qwen3-4B on {device} dtype={dtype}")
    tokenizer = AutoTokenizer.from_pretrained(target_path, trust_remote_code=True)

    from transformers import DynamicCache

    prompt = "What is the meaning of life?"
    input_ids_full = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    past_length = input_ids_full.shape[1]

    cur_len = 31  # anchor + 30 tree nodes
    input_ids = None
    causal_position_ids = torch.arange(
        past_length, past_length + cur_len, dtype=torch.long, device=device
    ).unsqueeze(0)
    cache_position = torch.arange(
        past_length, past_length + cur_len, dtype=torch.long, device=device
    )

    causal_mask = build_causal_4d_mask(past_length, cur_len, dtype, device)
    tree_mask, tree_position_ids = build_tree_4d_mask(
        past_length, cur_len, dtype, device
    )

    results = {}
    for label, attn_impl in [("eager", "eager"), ("sdpa", "sdpa")]:
        print(f"\n=== Loading model with attn_implementation={attn_impl} ===")
        model = AutoModelForCausalLM.from_pretrained(
            target_path, trust_remote_code=True, dtype=dtype,
            low_cpu_mem_usage=True, attn_implementation=attn_impl,
        ).eval().to(device)
        for param in model.parameters():
            param.requires_grad_(False)

        if input_ids is None:
            input_ids = torch.randint(
                0,
                int(model.config.vocab_size),
                (1, cur_len),
                dtype=torch.long,
                device=device,
            )

        for mask_name, mask, position_ids in [
            ("None", None, causal_position_ids),
            ("4D causal", causal_mask, causal_position_ids),
            ("4D binary-tree ancestor", tree_mask, tree_position_ids),
        ]:
            cache = DynamicCache()
            _ = model(input_ids=input_ids_full, past_key_values=cache,
                      use_cache=True, return_dict=True)
            cache.crop(past_length)

            try:
                ms = bench_one(
                    model,
                    input_ids,
                    mask,
                    position_ids,
                    cache,
                    cache_position,
                    past_length,
                    device,
                    warmup=3,
                    iters=10,
                )
                key = f"{attn_impl} + {mask_name}"
                results[key] = ms
                print(f"  {key:30s}: {ms:7.1f} ms/iter")
            except Exception as e:
                key = f"{attn_impl} + {mask_name}"
                results[key] = f"ERROR: {type(e).__name__}: {e}"
                print(f"  {key:30s}: ERROR — {type(e).__name__}: {e}")

        del model
        torch.cuda.empty_cache()
        synchronize(device)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    baseline_key = "eager + 4D binary-tree ancestor"
    baseline_ms = results.get(baseline_key, 0)
    if isinstance(baseline_ms, (int, float)):
        for k, v in results.items():
            if isinstance(v, (int, float)):
                ratio = v / baseline_ms if baseline_ms else 0
                speedup = baseline_ms / v if v else 0
                print(f"  {k:30s}: {v:7.1f} ms  ({speedup:.2f}x vs baseline)")
            else:
                print(f"  {k:30s}: {v}")


if __name__ == "__main__":
    main()
