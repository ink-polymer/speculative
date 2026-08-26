# Qwen3-4B 2K benchmark comparison

All comparable runs use the same deterministic 2,000-prompt suite assembled
from GSM8K, HumanEval, MATH500, and MBPP. Decoding is greedy with thinking
disabled, BF16 weights, and `max_new_tokens=2048` on one RTX 4090.

## Overall results

| Method | Prompts | Mean speedup | Mean acceptance length | Decode throughput |
|---|---:|---:|---:|---:|
| Official DFlash | 2,000 | 4.288x | 6.778 | 251.71 tok/s |
| Official DDTree | 2,000 | 5.257x | 8.656 | 308.60 tok/s |
| DFlash-SpecBlock (ours) | 2,000 | 4.515x | 7.502 | 268.39 tok/s |
| Optimized DDTree (ours, 2026-08-25) | 2,000 | **5.525x** | 8.436 | 212.68 tok/s |

The optimized DDTree run also reports a median per-prompt speedup of 5.416x
and an aggregate wall-clock speedup of 5.542x. Its target-only baseline on the
new server was 38.11 tok/s, so absolute throughput should not be compared
directly with runs collected on the previous server; relative speedup is the
primary cross-server metric.

## Per-dataset speedup

| Dataset | Prompts | Official DFlash | Official DDTree | DFlash-SpecBlock (ours) | Optimized DDTree (ours) |
|---|---:|---:|---:|---:|---:|
| GSM8K | 1,169 | 4.146x | 5.105x | 4.365x | **5.266x** |
| HumanEval | 149 | 4.241x | 5.281x | 4.454x | **5.592x** |
| MATH500 | 450 | 5.066x | 5.952x | 5.295x | **6.356x** |
| MBPP | 232 | 3.836x | 4.875x | 4.091x | **5.173x** |

## Optimized DDTree run manifest

- Target: `Qwen/Qwen3-4B`
- Draft: `z-lab/Qwen3-4B-DFlash-b16`
- Block size: 15
- Tree budget: 60
- Dataset SHA256: `c65f52a856f9ee60d7264596ce91b84e1ffdeb27e5d800a5f4453f1243a5ea27`
- Result SHA256: `b2cab1196501fc1bd6de71963f759a47ea69a9f8ead8de58983643055dc2c07f`
- Exact matches: 728 / 2,000
- BF16 mismatches: 1,272 / 2,000

The BF16 exact-match diagnostic is not the paper's losslessness metric.
Single-token and tree-shaped attention use different tensor shapes, and
near-tied logits can therefore make different greedy choices in finite
precision.

The raw optimized DDTree records are stored at
`outputs/final_2k/optimized_ddtree_2k_20260825.jsonl`.
