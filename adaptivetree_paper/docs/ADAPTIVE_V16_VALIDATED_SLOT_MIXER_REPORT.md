# AdaptiveTree v16: validated causal slot mixer

Date: 2026-09-11

Status: **development-only; rejected**

## Training protocol

v16 corrects the v15 training protocol with a lower learning rate, a fixed 10%
validation split, early stopping, and best-checkpoint restoration.  Training
uses 1,500 target-generated rows from only `translation`, `nq_open`, and
`alpaca`; sources used by the intended formal suite are excluded.

- Training-data SHA-256:
  `0b77dc8c3b3d34b6149911de6e9d7a183f55285cd238e0f250748144f6b0d674`.
- Learning rate: 2e-5.
- Updates: 4,047 over three epochs.
- Validation loss: 6.501758, 6.369461, 6.250108.
- Best checkpoint: epoch 3.
- Checkpoint SHA-256:
  `3bbcbb0b813240cea18c8ed13aa4a25574e15f10fdf027b4044fe4bc0dcde6e1`.

## Development gate

| Residual strength | Speedup vs official DDTree B192 | Fixed Adaptive control | Mean accepted length | Exact response match |
|---:|---:|---:|---:|---:|
| 1.00 | 0.9965x | 1.0366x | 8.6838 | 4/4 |
| -0.50 diagnostic | 0.9934x | 1.0222x | 8.8399 | 3/4 |

Both measurements are four-response, one-repeat, 128-token H20 screens and
include mixer inference in TPOT.  The unmodified DDTree acceptance is 8.9637.

## Decision

Reject the causal slot mixer.  Validation token cross-entropy improves, but the
fixed-budget tree acceptance does not.  The reverse-direction diagnostic also
fails, so the issue is not merely residual sign or strength.  A follow-up must
change the adapter architecture or optimize tree utility directly.  The formal
suite remains stopped until a candidate exceeds 1.20x.
