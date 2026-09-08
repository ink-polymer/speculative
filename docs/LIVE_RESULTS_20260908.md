# Live experiment status — 2026-09-08

This page contains sanitized metrics and scheduler identifiers only. Model
weights, prompts, caches, mismatch token dumps, and large tensor captures are not
uploaded.

## Scheduler snapshot

| Job | Model | Experiment | State |
|---:|---|---|---|
| 2142136 | Qwen3-4B | diffusion scaffold pilot, T=0.3/0.6/1.0 | COMPLETED |
| 2142137 | Qwen3-8B | diffusion scaffold pilot, T=0.3/0.6/1.0 | COMPLETED |
| 2142850 | Qwen3-4B | AdaptiveTree 10-dataset matrix | RUNNING on n141 |
| 2142851 | Qwen3-8B | first resume attempt | FAILED: shared prepare lock |
| 2142956 | Qwen3-8B | second resume attempt | FAILED: locked GPU UUID differed |
| 2143068 | Qwen3-8B | fresh 10-dataset 350GB rerun | RUNNING on n555 |
| 2144508 | Qwen3-4B | diffusion scaffold fixed-grid optimization | COMPLETED |
| 2144509 | Qwen3-8B | diffusion scaffold fixed-grid optimization | COMPLETED |
| 2145591 | Qwen3-4B | shared-probability scaffold optimization rerun | COMPLETED |
| 2145624 | Qwen3-8B | shared-probability scaffold optimization rerun | COMPLETED |
| 2146158 | Qwen3-4B | diffusion support-width 8/16/32/64 scan | COMPLETED |
| 2146157 | Qwen3-8B | diffusion support-width 8/16/32/64 scan | COMPLETED |
| 2147349 | Qwen3-4B | terminal-mass tree-block kernel/shape scan | COMPLETED |
| 2147350 | Qwen3-8B | terminal-mass tree-block kernel/shape scan | COMPLETED |
| 2147882 | model-independent | terminal-mass exact-law/engine gate | COMPLETED |
| 2148200 | Qwen3-8B partial | AdaptiveTree aggregate audit | COMPLETED |
| 2149121 / 2149122 | Qwen3-4B / 8B | original same-tree verifier replay | COMPLETED |
| 2150535 / 2150536 | Qwen3-4B / 8B | sparse-complement verifier replay | COMPLETED |
| 2151204 / 2151205 | Qwen3-4B / 8B | joint-event verifier replay | COMPLETED |
| 2151759 / 2151994 | Qwen3-4B / 8B | fused-CUDA verifier replay | COMPLETED |

The pilots run the full `tests/gbv_paper` gate before loading checkpoints.
Pilot metrics are diagnostic and use three fixed development prompts; they are
not formal benchmark claims.

## Diffusion scaffold pilot results

Speedup is baseline decode TPOT divided by `diffusion_full` decode TPOT.
Values above 1 are faster. Each cell uses 9 timing records from three fixed
development prompts and three repeats on one NVIDIA GH200 120GB.

| Model | T | diffusion_full TPOT (ms) | vs DFlash | vs DDTree | Checkpoint gate |
|---|---:|---:|---:|---:|---|
| Qwen3-4B | 0.3 | 13.367 | 1.095x | 0.748x | FAIL: DFlash TV=0.1441 |
| Qwen3-4B | 0.6 | 10.787 | 1.383x | 0.854x | FAIL: DFlash TV=0.0944 |
| Qwen3-4B | 1.0 | 10.504 | 1.181x | 1.006x | FAIL: DFlash TV=0.0603 |
| Qwen3-8B | 0.3 | 12.976 | 1.002x | 0.808x | FAIL: DDTree TV=0.1446 |
| Qwen3-8B | 0.6 | 10.116 | 1.191x | 0.845x | FAIL: DDTree TV=0.0966 |
| Qwen3-8B | 1.0 | 13.649 | 0.845x | 0.597x | FAIL: DDTree TV=0.0619 |

All six runs have `pilot_complete=true`,
`within_method_repeat_equal=true`, and `formal_complete=false`. The complete
unit-test directory passed before GPU execution. The checkpoint gate correctly
failed because a positive-temperature baseline check failed before the candidate:
DFlash for 4B and DDTree for 8B. These numbers must not be presented as validated
speed superiority.

Complete per-method pilot metrics:

- [Qwen3-4B T=0.3](../results/pilots/20260908/qwen3_4b_t03.json)
- [Qwen3-4B T=0.6](../results/pilots/20260908/qwen3_4b_t06.json)
- [Qwen3-4B T=1.0](../results/pilots/20260908/qwen3_4b_t10.json)
- [Qwen3-8B T=0.3](../results/pilots/20260908/qwen3_8b_t03.json)
- [Qwen3-8B T=0.6](../results/pilots/20260908/qwen3_8b_t06.json)
- [Qwen3-8B T=1.0](../results/pilots/20260908/qwen3_8b_t10.json)

## Diffusion scaffold optimization in progress

Jobs 2144508 and 2144509 are practical follow-up runs for Qwen3-4B and
Qwen3-8B. For each of T=0.3, 0.6, and 1.0, the fixed grid covers ten
`(labelled paths, block length, tree budget)` settings and both terminal-mass
and ancestral continuation backends, for twenty candidates total. Every
temperature also remeasures canonical L=15/B=45 DFlash and DDTree controls on
the same GPU.

The scan reports weighted decode TPOT, speedup versus both controls, accepted
draft tokens, committed tokens, mean tree nodes, repeat equality, and CUDA stage
profiles for the best two candidates at each temperature. Selection and timing
use the same three development prompts, so this is explicitly a diagnostic
optimization rather than a held-out or formal result. The strict checkpoint
equivalence audit remains a separate mandatory gate and is not weakened by the
optimizer.

The second pass in jobs 2145591 and 2145624 removes a measured implementation
bottleneck without changing the sampling law: the diffusion block verifier and
its Target-only continuation now reuse one normalized FP64 Target-probability
tensor instead of normalizing the same tree rows again. Exact finite-law
enumeration tests and the full `tests/gbv_paper` suite pass before the paired GPU
rerun. The rerun will determine whether removing the measured 6--7 ms of
per-round `select_and_correct` overhead is sufficient to cross the DDTree TPOT
baseline at T=0.6 and T=1.0.

The shared-probability rerun improved the best Qwen3-4B result to 0.976x versus
DDTree at T=0.6 and 0.987x at T=1.0; Qwen3-8B remained at 0.835x and 0.807x.
Jobs 2146158 and 2146157 therefore run a focused third pass over proposal
support widths 8, 16, 32, and 64. The hypothesis is that wider retained support
will reduce early correction exits at higher temperatures, improving committed
tokens per Target tree forward enough to offset the small sparse-transport cost.

Qwen3-4B job 2146158 completed successfully. Its best single configuration
across all three temperatures is `k1_l15_b60_s64_ancestral`: **1.244x versus
DFlash and 1.042x versus DDTree** by three-temperature geometric mean, with
repeat equality at every temperature. This is the first scanned diffusion
configuration to exceed both controls on the aggregate development metric.
Its per-temperature DDTree speedups are 1.212x at T=0.3, 0.874x at T=0.6, and
1.068x at T=1.0. Temperature-specific selection raises the best observed
DDTree ratios to 1.291x, 0.986x, and 1.079x respectively; T=0.6 has therefore
not crossed yet. The sanitized complete summary is
[available here](../results/optimization/20260908/qwen3_4b_support_summary.json).

Qwen3-8B job 2146157 also completed successfully. The fastest candidates by
temperature reach 1.052x, 0.898x, and 0.828x versus DDTree at T=0.3, 0.6, and
1.0. The best single configuration across all temperatures is
`k1_l15_b45_s64_terminal`, at 1.250x versus DFlash but only 0.889x versus
DDTree. Wider proposal support is therefore not sufficient to remove the 8B
high-temperature gap, and the next active test is the direct terminal-mass
DDTree execution optimization in job 2147350. The sanitized complete summary
is [available here](../results/optimization/20260908/qwen3_8b_support_summary.json).

Jobs 2147349 and 2147350 add the requested direct tree-block path for both
models. They compare canonical DFlash and DDTree with seven terminal-mass tree
shapes and three exact verification kernels. In particular, L=15/B=45 keeps the
DDTree probability tree and model work fixed and changes only the sampler from
the official batched ancestral posterior to a terminal-mass block draw. The
other shapes are an explicitly diagnostic throughput search. The two completed
performance jobs inherited the generic optimizer's diffusion test list. A
separate post-run server gate, job 2147882, subsequently passed the terminal-mass
exact-law, probability-law, fairness, optimization, and full engine tests with
exit code zero. The batch script is fixed so future `terminal` scans run this
method-specific gate before loading checkpoints. Speed measurements remain
development-prompt results until a frozen held-out run passes all gates.

Both terminal-mass jobs completed successfully and both models now have one
repeat-equal configuration that beats DDTree at every tested temperature:

| Model | Single configuration | T=0.3 vs DDTree | T=0.6 vs DDTree | T=1.0 vs DDTree | 3-T geomean vs DDTree | 3-T geomean vs DFlash |
|---|---|---:|---:|---:|---:|---:|
| Qwen3-4B | `tm_l15_b60_dense` | 1.509x | 1.021x | 1.145x | **1.208x** | **1.354x** |
| Qwen3-8B | `tm_l15_b45_dense` | 1.153x | 1.043x | 1.040x | **1.077x** | **1.423x** |

For Qwen3-8B, the displayed configuration freezes the L15/B45 tree and model
controls, but it does not couple the realized stochastic path to DDTree. For
Qwen3-4B, the strict same-tree `tm_l15_b45_dense` observation has a 1.184x
three-temperature aggregate but a 0.976x T=0.6 point; the displayed B60 row
also changes proposal budget. Temperature-specific end-to-end selection reaches
1.517x/1.044x/1.145x for 4B and 1.220x/1.087x/1.040x for 8B. These observations
are reinterpreted by the fixed-state audit immediately below.

**Superseding architecture audit:** these end-to-end rows do not establish a
verifier-architecture speedup. The methods consume random numbers differently
and took different realized acceptance paths; repeat equality only establishes
within-method reproducibility. A 252-pair-per-model replay on identical real
tree tensors found that the original terminal-mass verifier reaches only
0.290x (4B) and 0.308x (8B) of DDTree verifier speed. Sparse complement, one
joint terminal event draw, and a one-launch CUDA traversal improved the kernel,
but the fastest completed version still reached only 0.614x and 0.667x. All
three temperature gates failed. The prior architecture-win interpretation is
therefore withdrawn; see the
[full architecture and AdaptiveTree audit](ARCHITECTURE_ADAPTIVE_AUDIT_20260908.md).

Sanitized complete summaries: [Qwen3-4B](../results/optimization/20260908/qwen3_4b_terminal_mass_summary.json)
and [Qwen3-8B](../results/optimization/20260908/qwen3_8b_terminal_mass_summary.json).
These are still diagnostic development-prompt measurements, not a held-out or
formal speed claim. The strict checkpoint-equivalence and frozen-dataset gates
remain required before publication.

## AdaptiveTree results available now

Only the Qwen3-8B T=0 run has completed paired SDPA/FlashAttention artifacts,
for seven of ten datasets. Speedup is baseline decode TPOT divided by method
decode TPOT, so values above 1 are faster.

| Dataset | Adaptive vs DFlash | Adaptive vs best DDTree | no_exploration vs DFlash | no_exploration vs best DDTree |
|---|---:|---:|---:|---:|
| GSM8K | 1.324x | 1.378x | 1.408x | 1.467x |
| MATH-500 | 0.910x | 1.138x | 1.185x | 1.482x |
| AIME 2024 | 0.984x | 1.068x | 1.403x | 1.523x |
| AIME 2025 | 0.792x | 0.913x | 1.313x | 1.513x |
| HumanEval | 1.107x | 1.166x | 1.481x | 1.561x |
| MBPP | 0.959x | 1.004x | 1.272x | 1.332x |
| LiveCodeBench | 1.251x | 1.425x | 1.361x | 1.551x |
| **7-dataset geometric mean** | **1.032x** | **1.143x** | **1.343x** | **1.488x** |

AdaptiveTree wins 3/7 datasets against DFlash and 6/7 against tuned DDTree.
The `no_exploration` ablation wins all seven stored aggregate comparisons and is
therefore retained in both the resumed 8B run and the new 4B run. This variant
only disables scheduled exploration; it still performs warmup, latency EWMA,
acceptance calibration, and adaptive budget selection.

These speed rows are diagnostic only. The run used
`record-bf16-mismatches`: 354/1,472 backend responses were exact across every
stored method, 1,118 had at least one method mismatch, and 522/736 paired prompts
had different SDPA and FlashAttention target-baseline tokens. Accordingly,
`publication_gate_passed=false` and `strict_lossless_claim_eligible=false`.

The follow-up exact-output subset audit confirms the direction but not the
original magnitude. On the five datasets with adequately populated identical
Adaptive/`no_exploration` outputs, `no_exploration` is 1.099x faster by geometric
mean. DDTree is 0.895x as fast as DFlash on the corresponding adequately
populated exact-output subsets. Forced exploration is inefficient, the full
controller often drifts away from B=128, and its global fixed-cost EWMA absorbs
budget-dependent tree-build time. Detailed counts, stage evidence, limitations,
and sanitized JSON are in the
[audit report](ARCHITECTURE_ADAPTIVE_AUDIT_20260908.md).

The old Qwen3-8B run is retained unchanged. It cannot be resumed on an arbitrary
GH200 because its immutable contract pins a physical GPU UUID. Job 2143068 uses
a new result directory and will run all ten datasets on one consistently recorded
GPU instead of weakening the hardware contract.

Missing AdaptiveTree work: a complete fresh Qwen3-8B matrix and all Qwen3-4B
datasets. The page will be updated when new validated artifacts are available.
