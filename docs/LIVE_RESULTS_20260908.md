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
| 2145591 | Qwen3-4B | shared-probability scaffold optimization rerun | PENDING: Priority |
| 2145624 | Qwen3-8B | shared-probability scaffold optimization rerun | PENDING: Priority |

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
therefore retained in both the resumed 8B run and the new 4B run.

These speed rows are diagnostic only. The run used
`record-bf16-mismatches`: 354/1,472 backend responses were exact across every
stored method, 1,118 had at least one method mismatch, and 522/736 paired prompts
had different SDPA and FlashAttention target-baseline tokens. Accordingly,
`publication_gate_passed=false` and `strict_lossless_claim_eligible=false`.

The old Qwen3-8B run is retained unchanged. It cannot be resumed on an arbitrary
GH200 because its immutable contract pins a physical GPU UUID. Job 2143068 uses
a new result directory and will run all ten datasets on one consistently recorded
GPU instead of weakening the hardware contract.

Missing AdaptiveTree work: a complete fresh Qwen3-8B matrix and all Qwen3-4B
datasets. The page will be updated when new validated artifacts are available.
