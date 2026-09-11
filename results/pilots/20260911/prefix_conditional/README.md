# Prefix-conditioned tree-block WIP (2026-09-11)

This directory is a development snapshot, not formal-matrix evidence.

## Frozen controls

- Model/config: `configs/adaptive_block_qwen3_4b.json`
- Precision and attention: BF16, SDPA, TF32 disabled
- Temperature: 1.0
- Tree budget / draft length: 45 / 15
- Per decoding round: at most one draft forward and one target forward
- CUDA Graph: disabled
- Timing: official end-to-end TPOT scope, including tree construction and verification
- Workload: 3 synthetic development prompts, 64 generated tokens, 1 repeat

## Current result

The best point in this scan is `prefix_rescored_fused_p2_x0.75`:

| Method | TPOT (ms) | Tokens committed / round | Speedup vs DDTree |
|---|---:|---:|---:|
| DFlash | 9.2604 | 3.7679 | 0.620× |
| DDTree | 5.7411 | 7.0000 | 1.000× |
| Prefix-rescored + fused scan, p2, x0.75 | 5.9337 | 7.8458 | 1.003× (95% CI 0.886–1.095) |

The candidate did **not** pass the required 1.2×-over-DDTree gate. The raw
report therefore keeps `formal_complete=false` and the full benchmark matrix
must not be launched or cited as completed from this snapshot.

The experiment used checkpoint SHA-256
`cd22a8aee0f499cace96aba0d6ec76b911abd92bc62b9061b95d697428705b49`.
The checkpoint itself is intentionally not committed to Git.

## Files

- `prefix-r128-p270-fused-scan-20260911-a/report.json`: aggregate statistics,
  paired comparisons, provenance hashes, and diagnostic stage profiles.
- `prefix-r128-p270-fused-scan-20260911-a/rows.json`: per-prompt raw rows.
- `prefix-r128-p270-fused-scan-20260911-a.log`: original console log.

The GPU implementation was accompanied by 33 passing prefix-conditional tests;
the fused scan method automatically uses the batched reference verifier for CPU
tests while retaining the CUDA fused verifier in GPU benchmarks.
