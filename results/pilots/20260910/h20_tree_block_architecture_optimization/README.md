# H20 T=1 tree-block architecture optimization (development only)

Date: 2026-09-10. Model: Qwen3-4B. All end-to-end screens used the
official BF16 SDPA model execution, FP64 sampling probabilities, Target
temperature 1, Draft tree temperature 1 unless the proposal-only tree scoring
temperature is explicitly named, L=15, and at most B=45. CUDA Graph was not
used. These synthetic prompts are development data and are not formal results.

## What was tested

1. `ddtree_sparse_exit_fused_scan`: route through outgoing child events and
   perform a full-vocabulary correction scan only at the first exit.
2. `ddtree_same_draw_fused`: preserve DDTree's exact multinomial primitive and
   RNG state, but follow the selected path on CUDA.
3. Proposal-only tree scoring temperatures from 0.20 to 2.00 at B=45.
4. A joint scan of proposal temperature 0.50 and B in
   `{24, 28, 32, 36, 40, 45}`.

## Results

| experiment | comparison | speedup | 95% CI | decision |
|---|---|---:|---:|---|
| fixed real DDTree states | sparse-exit verifier / DDTree verifier | 8.2760x | [7.9339, 8.6532] | kernel succeeds |
| fixed real DDTree states | sparse-exit / old fused-scan verifier | 2.9080x | [2.2339, 3.8132] | kernel succeeds |
| same multinomial microgate | same-draw fused / DDTree verifier | 0.9776x | [0.9759, 0.9792] | reject |
| 12-prompt end-to-end | sparse-exit / DDTree | 0.9579x | [0.8804, 1.0417] | reject |
| 12-prompt end-to-end | sparse-exit / DFlash | 1.2815x | [1.1314, 1.4326] | passes only DFlash control |
| 12-prompt tree screen | proposal temperature 0.50 / DDTree | 0.9569x | [0.8669, 1.0739] | reject |
| 3-prompt joint screen | best temperature/budget / DDTree | 0.9712x | [0.8647, 1.1376] | reject |

The sparse-exit verifier is much faster in isolation, but selection/correction
was only about 1% of the old fused method's GPU time. The Target tree forward
remained roughly 78% of the end-to-end official timing scope. Different exact
random-number mappings also changed the number of decoding rounds on small
prompt sets: sparse-exit averaged 12.0 rounds while DDTree happened to average
11.25 in the 12-prompt screen. This path variance is why verifier microbenchmarks
must not be presented as end-to-end speedups.

## Decision

None of these candidates replaces the registered formal
`ddtree_fused_scan` method. No formal matrix was started from these development
screens. The code remains as an explicitly named experimental implementation
and the complete rows/reports are retained here so the negative result is
reproducible.
