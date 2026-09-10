# AdaptiveTree v14: candidate ratio transport

Date: 2026-09-11

Status: **development-only; rejected**

## Outcome

The candidate-ratio transport architecture does not improve AdaptiveTree and
does not approach the required 1.20x speedup over official DDTree B192.  The
final fair smoke at calibration strength 0.10 produces 0.9959x DDTree, while the
unchanged fixed Adaptive B192 control produces 1.0362x in the same run.

| Method | Mean TPOT (ms) | Mean accepted length | Speedup vs DDTree B192 | Exact response match |
|---|---:|---:|---:|---:|
| Official DDTree B192 | 3.9035 | 8.9637 | 1.0000x | 4/4 |
| Fixed Adaptive B192 control | 3.7671 | 8.9637 | 1.0362x | 4/4 |
| v14 ratio transport, strength 0.10 | 3.9197 | 8.8066 | 0.9959x | 4/4 |

This is a four-response, one-repeat, 128-token development smoke on one NVIDIA
H20.  It includes draft, learned correction, tree construction, compilation,
verification, controller, and commit work in TPOT.  It is intentionally not a
formal result.

## Architecture

The frozen target and DFlash models remain unchanged.  A low-rank head receives
each DFlash slot hidden state and the target output embeddings of the top-45
draft candidates.  It predicts centered target-minus-draft logit corrections
only for those candidates.  The implementation then:

1. applies a bounded calibration strength;
2. exactly adjusts the full-vocabulary normalization for the changed top-45
   probability mass;
3. re-sorts calibrated scores and token IDs together; and
4. passes the ordered probabilities to the same B192 official best-first tree
   enumerator and verifier.

The re-sort is required by the enumerator contract.  Its absence was discovered
by the new architecture test and explains why the earlier v13 rank calibration
was not a valid learned-reranker test.

## Training record

- Training rows: 580 target-model-generated texts.
- Updates: 580, one epoch.
- Final mean Smooth-L1 loss: 3.902512.
- Head shape: hidden size 2560, rank 32, top-45 candidates.
- Training data include MT-Bench and Math-500 source prompts; the screening gate
  uses GSM8K train prompts.  A stricter source-ID and text-overlap audit would be
  mandatory before any formal use.
- Checkpoint SHA-256:
  `70036f30d81c8ba7ef2fef898c882cda50b818f1df6a8efa9085230ba17b5771`.

## Strength screen

| Strength | Candidate speedup vs DDTree | Same-run fixed-control speedup | Mean accepted length | Exact response match | Decision |
|---:|---:|---:|---:|---:|:---:|
| 0.05 | 1.0071x | 1.0329x | 8.8625 | 3/4 | Reject |
| 0.10 (final fair run) | 0.9959x | 1.0362x | 8.8066 | 4/4 | Reject |
| 1.00 | 0.6542x | 1.0423x | 5.9448 | 3/4 | Reject |

The preliminary strength-0.05 and strength-1.00 rows were recorded before the
control path was exempted from the experimental re-sort.  Candidate behavior is
unchanged; the final strength-0.10 run is the clean fairness result.  All raw
artifacts are retained.

## Decision

Do not expand this v14 screen or replace the main method.  The learned correction
reduces acceptance and adds about 0.074 ms/output-token of tree-build work versus
the fixed Adaptive control in the final run.  More controller tuning or RL over
this action space cannot credibly supply the missing 20%.

Reaching a genuine 1.20x over DDTree now requires changing proposal capacity,
most plausibly a longer-horizon or jointly trained draft model, and then applying
that identical draft to both AdaptiveTree and DDTree.  Redefining the baseline as
DFlash may produce a 1.20x number but would be a different claim.
