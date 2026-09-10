# AdaptiveTree v17: residual slot MLP

Date: 2026-09-11

Status: **development-only; rejected**

## Outcome

The position-wise rank-512 SwiGLU residual adapter improves held-out token
cross-entropy but does not improve DDTree tree acceptance.

| Residual strength | Speedup vs official DDTree B192 | Fixed Adaptive control | Mean accepted length | Exact response match |
|---:|---:|---:|---:|---:|
| 1.00 | 1.0175x | 1.0341x | 8.8415 | 2/4 |
| -1.00 diagnostic | 1.0318x | 1.0721x | 8.7003 | 3/4 |

The official DDTree B192 acceptance is 8.9637.  Both measurements are
four-response, one-repeat, 128-token H20 screens with adapter inference included
in TPOT.  Neither passes output equality or the 1.20x threshold.

## Training

- Disjoint training rows: 1,500 (`translation`, `nq_open`, `alpaca`).
- Initial validation cross-entropy: 6.724950.
- Epoch validation cross-entropy: 6.045463, 5.838805, 5.740504.
- Best checkpoint: epoch 3, 4,047 updates.
- Checkpoint SHA-256:
  `320645ef8738c8224e49c8fd164e7de2e423a36bac1133bcf31ae7964a8d9fb6`.

## Decision

Reject the token-CE proposal-adapter family.  Better per-token likelihood does
not translate into better fixed-budget prefix-tree coverage.  The next
architecture must add proposal horizon/capacity or optimize tree utility
directly.  The formal suite remains stopped.
