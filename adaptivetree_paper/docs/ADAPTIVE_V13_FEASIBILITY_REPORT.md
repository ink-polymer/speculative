# AdaptiveTree v13: 1.20x feasibility gate

Date: 2026-09-11

Status: **development-only; rejected as the formal main method**

## Decision

AdaptiveTree v13 does not reach the required 1.20x speedup over the equal-node-cap
official DDTree baseline.  The best tested v13 candidate is 1.0269x DDTree and is
slower than the fixed Adaptive B192 control measured in the same run (1.0332x).
The formal experiment remains stopped.

The runs in this report include draft inference, tree construction, tree
compilation, verification, controller work, and commit work in TPOT.  They use the
same target model, draft model, prompts, maximum output length, warm-up protocol,
and B192 verifier matrix for the candidate and its primary DDTree control.

## Development protocol

- Hardware: one NVIDIA H20 96 GB.
- Target: pinned `Qwen3-4B` revision.
- Draft: pinned `Qwen3-4B-DFlash-b16` revision; K=15 draft horizon.
- Workload: MT-Bench development subset, 8 responses, one repeat, 128 output
  tokens per response.
- Primary baseline: official DDTree B192.
- Main acceptance rule: TPOT speedup >=1.20x, with no response-level output
  mismatch against the equal-cap reference.
- These short runs are screening experiments, not paper-grade measurements.

## Hard maximum-depth sweep

Each candidate retains the B192 node cap but forbids paths deeper than the stated
limit.  Shortening the tree reduces useful deep paths faster than it saves tree
work.

| Maximum depth | Mean TPOT (ms) | Mean accepted length | Speedup vs DDTree | Exact response match | Gate |
|---:|---:|---:|---:|---:|:---:|
| 6  | 5.3236 | 6.2720 | 0.7493x | 4/8 | Fail |
| 8  | 4.7703 | 7.2233 | 0.8550x | 7/8 | Fail |
| 10 | 4.5487 | 7.6298 | 0.8879x | 5/8 | Fail |
| 12 | 4.0945 | 8.2194 | 0.9687x | 6/8 | Fail |
| 14 | 3.9872 | 8.5760 | 1.0133x | 7/8 | Fail |
| 15 (fixed control, depth-14 run) | 3.9131 | 8.8383 | 1.0325x | 8/8 | Fail |

The response mismatches make depths 6--14 invalid for promotion even before the
speed threshold is considered.  They are retained as diagnostic results.

## Frozen rank-head calibration

**Superseded implementation note (v14):** v13 changed calibrated scores but did
not re-sort token IDs before passing them to the official best-first enumerator.
That enumerator requires descending score order.  The v13 rank-head rows below
therefore remain useful negative diagnostics, but they are not a valid test of
a fully functioning learned reranker.  v14 fixes the ordering contract and tests
a more direct target/draft ratio head; it still fails the performance gate.

The existing rank head predicts four target-token rank buckets.  v13 redistributes
the draft model's rank mass using this prediction and geometrically blends it with
the original DDTree log-probability score.  The checkpoint is frozen during
inference.

Checkpoint SHA-256:
`8993d0e42e94a0b055161b63523ab6610303069dd151f19a229698cceadb10a5`

| Calibration strength | Mean TPOT (ms) | Mean accepted length | Candidate speedup | Same-run fixed-control speedup | Exact response match | Gate |
|---:|---:|---:|---:|---:|---:|:---:|
| 0.25 | 3.9315 | 8.8623 | 1.0269x | 1.0332x | 6/8 | Fail |
| 0.50 | 4.1643 | 8.6149 | 0.9733x | 1.0307x | 7/8 | Fail |
| 1.00 | 4.2640 | 8.2429 | 0.9280x | 1.0342x | 6/8 | Fail |

At strength 0.25, accepted length rises by only 0.024 token while tree-build cost
roughly doubles (0.1216 vs 0.0611 ms/output-token).  The head was trained for the
SpecBlock rank-bucket task, not for selecting the complete B192 tree, so this
direct reuse is not an adequate learned reranker.  Its training/evaluation split
would also require a formal leakage audit before any paper claim.

## Larger official DDTree budgets

This sweep tests whether a substantially larger proposal pool can supply enough
acceptance to make a learned selector plausible.  It does not apply a hindsight
oracle; each row is the actual official DDTree result at that node cap.

| DDTree budget | Mean TPOT (ms) | Mean accepted length | Speedup vs DDTree B192 |
|---:|---:|---:|---:|
| 192  | 4.1840 | 8.8383 | 1.0000x |
| 256  | 4.6709 | 8.6683 | 0.8958x |
| 512  | 6.9395 | 9.2499 | 0.6029x |
| 1024 | 12.5187 | 9.4421 | 0.3342x |

Increasing verification width produces at most 0.604 additional accepted token in
this screen, while verification latency grows sharply.  It cannot supply the
approximately 10.8 accepted-token level estimated for a 1.20x result under this
kernel and model pair.

## Why controller-only tuning cannot close the gap

In the budget sweep, official DDTree B192 takes 4.1840 ms/token.  A 1.20x candidate
must take at most 3.4867 ms/token.  The fixed Adaptive candidate takes 4.0742
ms/token.  Even the impossible optimistic calculation that deletes all measured
tree-build, tree-compile, and commit time leaves approximately 3.9167 ms/token, or
only 1.0682x DDTree.  The remaining gap is in draft/verification rounds, not in
the Python controller.

Earlier response-level hindsight sweeps agree with this diagnosis: choosing among
the tested temperature, depth-discount, and depth-bias arms has an upper bound of
about 1.003x, 1.013x, and 1.016x respectively.  Reinforcement learning over only
those actions cannot produce a genuine 1.20x result; it needs a new source of
proposal quality or a different verification contract.

## Fair next routes

1. Train or obtain a longer-horizon draft model, then run both AdaptiveTree and
   official DDTree with that identical draft model and identical verifier budget.
   This preserves the claim “faster than DDTree,” but is a new training project
   and has no guaranteed 1.20x outcome.
2. Keep the current B16 draft and define the target as 1.20x over official DFlash,
   while continuing to report DDTree separately.  This is a different research
   claim and must not be presented as 1.20x over DDTree.
3. Train a tree-specific policy on disjoint training prompts.  Its action must
   directly optimize selected-node utility under measured verification cost, and
   evaluation must be frozen before the held-out test suite.  With the current
   B16 proposal ceiling, this is scientifically valid but still unlikely to close
   the full gap by itself.

No v13 result should replace the accepted main candidate until it passes the
1.20x gate on the complete held-out matrix with confidence intervals and exact
output validation.
