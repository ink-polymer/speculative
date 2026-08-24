# Official DFlash 2K benchmark results

All methods use the same deterministic 2,000-prompt suite, BF16, greedy
decoding, thinking disabled, and `max_new_tokens=2048` on one RTX 4090.
Speedup follows the official per-request time-per-output-token aggregation.

| Method | Speedup | Mean acceptance length | Decode throughput |
|---|---:|---:|---:|
| DFlash | 4.288x | 6.778 | 251.71 tok/s |
| DDTree | 5.257x | 8.656 | 308.60 tok/s |
| DFlash-SpecBlock (ours) | 4.515x | 7.502 | 268.39 tok/s |

## Per-dataset speedup

| Dataset | Prompts | DFlash | DDTree | Ours |
|---|---:|---:|---:|---:|
| GSM8K | 1,169 | 4.146x | 5.105x | 4.365x |
| HumanEval | 149 | 4.241x | 5.281x | 4.454x |
| MATH500 | 450 | 5.066x | 5.952x | 5.295x |
| MBPP | 232 | 3.836x | 4.875x | 4.091x |

Raw per-prompt records are stored in:

- `qwen3_4b_dflash_ddtree_bf16.jsonl`
- `qwen3_4b_own_bf16.jsonl`

The BF16 bit-exact diagnostic is not the paper's losslessness metric. Different
single-token and block attention shapes can change a greedy decision at
near-tied logits in finite precision.
