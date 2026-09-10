# AdaptiveTree v15: causal slot mixer

Date: 2026-09-11

Status: **development-only; rejected**

## Outcome

The frozen low-rank causal mixer does not improve the B16 DFlash proposal.  It
is applied before the target LM head and its complete runtime is charged to the
draft stage.

| Mixer strength | Candidate TPOT speedup vs DDTree B192 | Fixed Adaptive B192 control | Mean accepted length | Exact response match |
|---:|---:|---:|---:|---:|
| 0.10 | 1.0085x | 1.0429x | 8.8546 | 3/4 |
| 1.00 | 0.9830x | 1.0346x | 8.6145 | 3/4 |

Both rows are four-response, one-repeat, 128-token H20 development screens.
Official DDTree B192 accepts 8.9637 tokens on the same prompts.  Neither result
passes output equality or the 1.20x performance requirement.

## Training

- Architecture: strictly causal depthwise slot convolution with a rank-64
  residual bottleneck and kernel size 3.
- Training rows and updates: 580.
- Learning rate: 2e-4; one epoch.
- Final running mean cross-entropy: 5.334062.
- Checkpoint SHA-256:
  `f37dd1d34e2eb51b578c365766049d0f511ea7ecb749c245d06de19496741729`.

The running loss rose after the early updates, so v15 saves a useful negative
architecture result but not a well-selected trained checkpoint.  Any follow-up
must use a lower learning rate, a disjoint validation split, and best-checkpoint
selection.  The full formal suite remains stopped.
