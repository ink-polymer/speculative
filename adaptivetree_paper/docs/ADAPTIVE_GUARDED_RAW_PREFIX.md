# AdaptiveTree guarded raw-prefix development audit

Date: 2026-09-10

Status: development gate passed; **not itself a formal benchmark result**.

## Architecture

`adaptive_b128` uses `guarded_raw_prefix_batched_commit_v10`:

1. Enumerate the exact official DDTree best-first prefix at the B128 node cap.
2. Reuse pinned host buffers for the GPU-to-CPU top-k metadata transfer.
3. Build token, depth, parent, score, and visibility tensors in a compiled CPU
   extension instead of constructing Python nodes and converting them back.
4. Follow the verified target path directly over the compact token/parent
   representation, without materializing 129 Python child dictionaries.
5. Write accepted indices into reusable pinned-host and GPU buffers instead of
   constructing and transferring the same index tensor twice per round.
6. Compact all equal-shaped target KV tensors with a two-phase Triton
   gather/scatter.  The temporary buffer prevents in-place source/destination
   races; unsupported devices or layouts use the official per-tensor path.
7. Treat B128 as the safe arm. Smaller budgets are screened using
   counterfactual acceptance observations from the accepted B128 path and are
   not blindly sampled during startup.
8. Permit at most one smaller-budget pilot when an optimistic latency bound
   predicts at least 3% utility gain. Require three measurements, at least 8%
   latency saving, and at least 3% measured tokens/ms gain before promotion.
9. Re-evaluate every 32 rounds; all other rounds use the cached safe arm.

The compiled tree extension has a semantics-identical Python fallback, and the
batched cache compactor has the official per-tensor fallback. Draft, tree
construction, tree compilation, target verification, commit, and online
controller overhead all remain inside reported decode TPOT. Kernel compilation
occurs during the same untimed warmup used by the official benchmark.

## Development protocol

- Model: pinned Qwen3-4B target and DFlash-b16 draft.
- Hardware: one NVIDIA H20.
- Data: deterministic held-out examples from `openai/gsm8k/main/train`; formal
  GSM8K test examples were not used for architecture selection.
- Generation: temperature 0, up to 256 new tokens, and balanced cyclic
  execution positions. DDTree uses its official C++ KV-cache compaction;
  AdaptiveTree records whether every round used the bit-exact batched backend.
- Fairness: DDTree and AdaptiveTree both use at most 128 draft nodes. Tree and
  controller time are included. The strict gate also requires bit-identical
  greedy outputs against official DDTree for every response.

## Passing paired gate (16 prompts × 4 repeats)

| Method | Mean TPOT (ms) | Speedup vs DDTree | Mean acceptance | Decode rounds | Exact output |
|---|---:|---:|---:|---:|---:|
| Official DDTree B128 | 4.1449 | 1.0000x | 8.8567 | 1,764 | 100% |
| Guarded dynamic Adaptive B128 v10 | 4.0383 | **1.0264x** | 8.8567 | 1,764 | 100% |

The guarded method passes the preregistered development rule of exact outputs
and at least 1% lower mean TPOT. Tree build is 0.04626 versus 0.07623
ms/output-token, and commit is 0.07128 versus 0.15392 ms/output-token. All
1,764 Adaptive rounds report the batched backend. Across 64 paired responses,
the mean DDTree-minus-Adaptive difference is 0.10668 ms/output-token; Adaptive
wins 59 pairs and the paired bootstrap 95% interval is [0.06945, 0.14092] ms.
This establishes an equal-cap system-path advantage on the held-out gate; it
does not claim a higher-quality tree, because acceptance is intentionally
identical here.

Two immediately preceding independent 16×4 runs of the same CUDA path measured
1.0264x and 1.0226x; both gates passed. The table uses the canonical rerun whose
source identity exactly matches this branch, rather than selecting among runs.

## Rejected candidates

- A fully reserved 15-node greedy spine reduced acceptance.
- Fixed depth penalties from -0.05 through -0.40 reduced acceptance or TPOT.
- Fixed depth bonuses from +0.02 through +0.20 did not improve TPOT.
- Proposal temperatures 0.70, 0.85, 1.15, and 1.30 all lost to the official
  probability ordering.
- Fixed B80 and B100 lost to B128.
- A contextual B100 guard preserved exactness only by selecting B100 on 42 of
  882 rounds and improved TPOT by just 0.11%, so it was not promoted.
- A new 0.60--1.50 proposal-temperature scan changed tree topology; although
  1.10 raised acceptance on the four-prompt smoke set, it failed the strict
  token-output identity requirement and was rejected.
- Progressive exact top-k (33 → 65 → 128 with a boundary sentinel) rarely
  required expansion, but `torch.topk(33)` was not faster on H20 and the extra
  guard increased measured tree-build time, so it was removed.
- B192 beat official DDTree B128 but did not beat DDTree B192 consistently;
  this was a node-cap gain and is not used as evidence of equal-cap superiority.

## Claim boundary

The development evidence supports promoting v10 into the formal experiment
candidate. Cross-model, cross-dataset, and temperature-specific formal runs
must be regenerated from scratch. A formal paper claim should report those
results and must distinguish system-path speedup from tree-quality gains.
