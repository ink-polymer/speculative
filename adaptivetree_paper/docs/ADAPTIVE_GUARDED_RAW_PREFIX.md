# AdaptiveTree guarded raw-prefix development audit

Date: 2026-09-10

Status: development candidate; **not promoted to a formal result**.

## Architecture

`adaptive_b128` now uses `guarded_raw_prefix_v3`:

1. Enumerate the official DDTree best-first heap once at the maximum node cap.
2. Materialize the selected verifier prefix directly, without constructing a
   `DraftTree` object and converting it back to official tensors.
3. Reuse pinned host buffers and NumPy/tree workspaces across rounds.
4. Treat the maximum budget as a safe arm.  Each smaller budget receives one
   initial latency sample; it receives additional samples only after showing a
   15% preliminary latency saving.
5. Require three samples, 8% latency saving, and 3% predicted tokens/ms gain
   before a smaller budget can challenge the safe arm.
6. Re-evaluate once every 32 rounds; other rounds skip proposal-mass scoring.
7. Use counterfactual accepted-path observations from a max-budget tree to
   update every nested prefix without additional target-model calls.

All draft, tree construction, tree compilation, target verification, commit,
and controller overhead remains inside reported decode TPOT.

## Development protocol

- Model: Qwen3-4B target and its pinned DFlash-b16 draft.
- Hardware: one NVIDIA H20.
- Data: deterministic samples from `openai/gsm8k/main/train`; the registered
  formal protocol uses GSM8K test and is not used for tuning here.
- Generation: temperature 0, up to 256 new tokens, balanced cyclic method
  positions, official C++ KV-cache compaction.
- Node comparisons: both the official B128 reference and equal-cap B192
  controls are retained.

## Results

### Repeated B128 gate (16 prompts x 2 repeats)

| Method | Mean TPOT (ms) | Speedup vs DDTree B128 | Mean acceptance | Exact match vs equal-cap DDTree |
|---|---:|---:|---:|---:|
| DDTree B128 | 4.1102 | 1.0000x | 8.8567 | 100% |
| Previous Adaptive B128 | 4.1460 | 0.9913x | 8.7983 | 81.25% |
| Raw fixed B128 diagnostic | 4.0990 | 1.0027x | 8.8567 | 100% |
| Guarded raw Adaptive B128 | 4.1063 | 1.0009x | 8.8402 | 93.75% |

The new dynamic B128 candidate removes the old regression, but the measured
0.09% lead is within normal timing noise and does not establish superiority.

### Extended-cap gates (16 prompts, two independent runs)

| Run | DDTree B128 (ms) | DDTree B192 (ms) | Adaptive B192 (ms) | Adaptive B192 vs DDTree B128 | Adaptive B192 vs DDTree B192 |
|---|---:|---:|---:|---:|---:|
| v9 | 4.1214 | 4.0162 | 4.0242 | 1.0242x | 0.9980x |
| v10 | 3.9275 | 3.8391 | 3.8893 | 1.0098x | 0.9871x |

B192 consistently beats the official fixed B128 operating point, but equal-cap
Adaptive B192 does not beat DDTree B192.  Therefore the observed B192 gain is
primarily a node-cap selection result, not evidence that AdaptiveTree has
surpassed DDTree at equal capacity.

### Rejected candidates

- A fully reserved 15-node greedy spine reduced acceptance and was removed.
- Proposal temperatures 0.70, 0.85, 1.15, and 1.30 all lost to the original
  probability ordering and were not promoted.
- Fixed B80 and B100 lost to B128 on the held-out gate.

## Claim boundary and next gate

Do not claim that AdaptiveTree is faster than DDTree at equal node capacity
from these data.  A defensible statement is that the raw-prefix implementation
eliminates the previous Adaptive B128 regression, and that B192 is a better H20
operating point than the official B128 setting on this development sample.

Before changing the formal primary, repeat an equal-cap B192 comparison on a
second held-out dataset and Qwen3-8B, then require a confidence interval above
1.0.  The old formal run was stopped and its artifacts were preserved; no v3
development artifact may be resumed into that output directory.
