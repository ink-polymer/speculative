# DDTree official timing and stage audit

## Finding

The previous `gbv_experiments.Engine` timing was more conservative than the
vendored DDTree/DFlash paper timing in two material ways:

| Item | Vendored DDTree/DFlash | Previous local `decode_ms` |
|---|---|---|
| Target prompt prefill | excluded | excluded |
| Anchor sampled by Target prefill | included in output-token denominator | excluded from `decode_tokens` |
| First DFlash Draft forward | excluded by resetting `decode_start` after it | included |
| Later Draft forwards | included | included |
| Tree construction/compilation | included | included |
| Target verification | included | included |
| Posterior sampling/tree traversal/KV commit | included | included |
| Tokenization, dataset loading, text decoding, serialization | excluded | excluded |

The source evidence is in `third_party/ddtree_pinned/ddtree.py`: Target prefill
ends at lines 333--347, the initial decode clock is created at line 349, and it
is reset after the first Draft forward at lines 363--379. The returned token
count and TPOT are computed at lines 457--459. DFlash has the same reset at
`third_party/ddtree_pinned/dflash.py:68--85` and the same denominator at lines
123--125. `cuda_time()` synchronizes CUDA before reading `perf_counter`.

The paper table does not pool all milliseconds and tokens. It computes an
unweighted arithmetic mean of `time_per_output_token` over responses in
`third_party/ddtree_pinned/make_latex_table.py:47--48`, then divides baseline
mean TPOT by method mean TPOT.

Consequently, the archived H20 `r2` measurements are not retroactively changed.
They did not retain the first-Draft boundary timestamp, so an exact official-
scope value cannot be reconstructed from their aggregate totals. A fresh run is
required.

## New dual timing contract

Fresh engine records now retain both contracts:

- `decode_ms`, `decode_tokens`: the existing conservative interval, including
  the first Draft forward and excluding the prefill-sampled anchor from its
  token denominator;
- `official_scope_decode_ms`, `official_scope_output_tokens`, and
  `official_scope_time_per_output_token_ms`: reset immediately after the first
  Draft forward and count the anchor, matching the vendored start/end and
  denominator convention.

The official-data pilot also flattens multi-turn examples into per-turn TPOTs
before taking the arithmetic mean, matching the upstream response aggregation.
The registered paired geometric-mean/cluster-bootstrap endpoint remains the
fail-closed gate. Official-scope numbers are reported alongside it and are not
used to decide whether a candidate passes.

This is scope compatibility, not a claim of instruction-for-instruction timing
identity. The upstream implementation synchronizes at every stage boundary;
the local primary path does not reproduce every internal stage synchronization.
That distinction is recorded as
`internal_official_stage_synchronizations_reproduced=false` in every result.

## Stage decomposition

With `profile=True`, fresh diagnostics distinguish:

| Stage | Contents |
|---|---|
| `prefill` | Target prompt forward, anchor sampling, initial features |
| `draft_prefill` | first Draft forward, excluded by upstream TPOT |
| `first_draft_boundary_sync` | synchronization used to place the upstream timer reset |
| `draft` | all later Draft forwards and Draft proposal probabilities |
| `tree_build` | DDTree/proposal topology construction |
| `tree_compile` | verification IDs, positions, and tree attention mask |
| `verify` | Target tree forward; for eager variants also full LM head and FP64 posterior rows |
| `select_and_correct` | posterior sampling, tree walk, and algorithm-specific correction |
| `stop_check` | length/EOS clipping of newly selected tokens |
| `commit` | hidden-feature selection, KV-cache compaction, and round bookkeeping |
| `target_decode` | cached one-token Target baseline loop |

`stage_profiles.json` is generated only after all unprofiled primary rows finish.
It records both host intervals and CUDA-event intervals, ranks every stage, and
names the longest one. Host time on CUDA mainly measures Python submission and
orchestration; CUDA-event rankings are the appropriate GPU-hotspot view. The
synchronized total remains the authoritative wall clock.

## What the existing evidence says about the bottleneck

The archived H20 `r2` run did not enable stage profiling, so no exact stage
percentages should be attributed to that run. Older 2026-09-08 diagnostic
profiles do consistently identify Target verification as the dominant stage:

| Model/method, T=1 | Target verify | Draft | Commit | Tree build | Original sampler/correction |
|---|---:|---:|---:|---:|---:|
| Qwen3-4B DFlash | 81.8% | 13.2% | 3.5% | 0.7% | 0.8% |
| Qwen3-4B DDTree B=45 | 81.0% | 13.5% | 3.5% | 1.4% | 0.6% |
| Qwen3-8B DFlash | 81.8% | 13.5% | 3.4% | 0.6% | 0.7% |
| Qwen3-8B DDTree B=45 | 81.1% | 13.7% | 3.4% | 1.3% | 0.5% |

Percentages above are recomputed from CUDA-event stage totals in
`results/optimization/20260908/qwen3_4b_terminal_mass_summary.json` and
`results/optimization/20260908/qwen3_8b_support_summary.json`. They are
diagnostic snapshots, not the new candidate's formal result.

This explains why a 2.877643x verifier-only microbenchmark produced only a
1.025267x H20 end-to-end gain. Under the simplifying Amdahl assumption that the
verifier is the only changed component, the observed pair implies that only
about 3.78% of DDTree end-to-end time was accelerated. Even making that portion
free would cap speedup near 1.0393x. Therefore further gains must reduce work in
the much larger Target verification region, not only make the tree walk faster.

The current development candidates follow that evidence:

- `ddtree_lazy_softmax_fused_scan` keeps the complete Target LM-head result but
  avoids FP64 normalization for unreachable leaf rows;
- `ddtree_lazy_projection_fused_scan` also avoids LM-head projection for
  unreachable leaves, subject to a required BF16 numerical audit because the
  GEMM batch shape changes.

Their fresh timing and stage profiles must be measured on the target GPU before
choosing either candidate. No speed claim is made from code inspection alone.
