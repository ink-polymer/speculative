"""扫描不同 cur_len 下的 forward 时间(无 cache),确定固定开销。"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from transformers import AutoModelForCausalLM, AutoTokenizer

from dflash_specblock.device import resolve_device, synchronize
from dflash_specblock.config import ExperimentConfig


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

    token_counts = [1, 4, 8, 16, 31, 61, 91, 121, 181, 241]
    results = {}

    print(f"\n{'cur_len':>8} {'ms/iter':>10} {'ms/token':>10}")
    print("-" * 32)
    for cur_len in token_counts:
        input_ids = torch.randint(0, 151936, (1, cur_len), dtype=torch.long, device=device)

        # Warmup
        for _ in range(3):
            _ = model(input_ids=input_ids, use_cache=False, return_dict=True)
        synchronize(device)

        # Measure
        start = time.perf_counter()
        iters = 10
        for _ in range(iters):
            _ = model(input_ids=input_ids, use_cache=False, return_dict=True)
        synchronize(device)
        ms = (time.perf_counter() - start) / iters * 1000
        results[cur_len] = ms
        print(f"{cur_len:>8} {ms:>10.1f} {ms/cur_len:>10.2f}")

    if results.get(1) and results.get(31):
        fixed = results[1]
        var_per_tok = (results[31] - results[1]) / 30
        print(f"\n固定开销 (cur_len=1): {fixed:.1f} ms")
        print(f"可变开销: {var_per_tok:.2f} ms/token")
        if results.get(31):
            print(f"cur_len=31 时固定占比: {fixed/results[31]*100:.1f}%")

        print("\n预测不同 tree_budget 下的 wall 时间 (128 tokens):")
        print(f"{'budget':>8} {'verify_ms':>10} {'accept_est':>10} {'iters':>8} {'wall_ms':>10}")
        for budget, acceptance in [(30, 2.7), (60, 3.2), (90, 4.0), (120, 5.0), (180, 6.5), (240, 8.0)]:
            cur = budget + 1
            verify_ms = fixed + var_per_tok * cur
            iters = 128 / acceptance
            wall = iters * (verify_ms + 25)
            print(f"{budget:>8} {verify_ms:>10.1f} {acceptance:>10.1f} {iters:>8.1f} {wall:>10.0f}")


if __name__ == "__main__":
    main()
