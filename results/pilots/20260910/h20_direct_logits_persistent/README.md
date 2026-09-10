# H20 direct-logits persistent pilot

This is negative development evidence and is not part of the formal matrix.

- Fixed implementation commit: `1bdf26dbaed165a9eff0f2e5e71fd404b6a56d59`
- GPU: NVIDIA H20
- Workload: 3 synthetic prompts, 64 generated tokens, 5 balanced repeats
- Controls: Qwen3-4B, T=1, L15/B45, BF16 model/logits, SDPA/SDPA,
  TF32 disabled, FP64 posterior arithmetic
- CUDA/configuration tests: 10 passed
- Repeated outputs within each method: identical
- Overall gate: failed

| Method | Pooled tokens/s | Reported equal-prompt comparison |
|---|---:|---:|
| DDTree | 168.439 | reference |
| Earlier fused scan | 131.930 | direct/fused = 0.929x |
| Direct-logits persistent | 123.999 | direct/DDTree = 0.726x |
| DFlash | 105.593 | DDTree/DFlash = 1.642x |

The direct-logits candidate visited only 5.46 of 46 probability rows per
round on average, but its serial FP64 exponential/CDF work inside one CUDA
block outweighed the saved full-matrix normalization.  Its DDTree speedup was
0.7258 with 95% CI [0.6785, 0.7550], so it must not proceed to confirmation.

`direct_logits_quick_c966f46.log` retains the first compile-gate failure
(`CUDART_INF` unavailable under CUDA 12.8).  Commit `1bdf26d` replaced that
nonportable sentinel; `direct_logits_quick_1bdf26d.log` contains the passing
tests and completed balanced benchmark.
