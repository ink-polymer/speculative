# AdaptiveTree guarded raw-prefix development audit

Date: 2026-09-10

Status: development gate passed; **not itself a formal benchmark result**.

## Architecture

`adaptive_b128` uses `guarded_raw_prefix_v7`:

1. Enumerate the exact official DDTree best-first prefix at the B128 node cap.
2. Reuse pinned host buffers for the GPU-to-CPU top-k metadata transfer.
3. Build token, depth, parent, score, and visibility tensors in a compiled CPU
   extension instead of constructing Python nodes and converting them back.
4. Follow the verified target path directly over the compact token/parent
   representation, without materializing 129 Python child dictionaries.
5. Treat B128 as the safe arm. Smaller budgets are screened using
   counterfactual acceptance observations from the accepted B128 path and are
   not blindly sampled during startup.
6. Permit at most one smaller-budget pilot when an optimistic latency bound
   predicts at least 3% utility gain. Require three measurements, at least 8%
   latency saving, and at least 3% measured tokens/ms gain before promotion.
7. Re-evaluate every 32 rounds; all other rounds use the cached safe arm.

The compiled extension has a semantics-identical Python fallback. Draft,
tree construction, tree compilation, target verification, commit, and online
controller overhead all remain inside reported decode TPOT. Extension
compilation occurs during untimed warmup, as does official DDTree's C++ cache
compaction compilation.

## Development protocol

- Model: pinned Qwen3-4B target and DFlash-b16 draft.
- Hardware: one NVIDIA H20.
- Data: deterministic held-out examples from `openai/gsm8k/main/train`; formal
  GSM8K test examples were not used for architecture selection.
- Generation: temperature 0, up to 256 new tokens, balanced cyclic execution
  positions, and official C++ KV-cache compaction.
- Fairness: DDTree and AdaptiveTree both use at most 128 draft nodes. Tree and
  controller time are included. The strict gate also requires bit-identical
  greedy outputs against official DDTree for every response.

## Passing paired gate (16 prompts × 2 repeats)

| Method | Mean TPOT (ms) | Speedup vs DDTree | Mean acceptance | Decode rounds | Exact output |
|---|---:|---:|---:|---:|---:|
| Official DDTree B128 | 3.9953 | 1.0000x | 8.8567 | 882 | 100% |
| Guarded dynamic Adaptive B128 | 3.9525 | **1.0108x** | 8.8567 | 882 | 100% |

The guarded method passes the preregistered development rule of exact outputs
and at least 1% lower mean TPOT. Its tree-build stage is 0.04494 ms/output-token
versus 0.07404 for official DDTree, a 39.3% reduction. The 32 paired TPOT
differences have mean 0.04282 ms, standard error 0.01259 ms, and t=3.40; the
Adaptive method wins 23 of 32 pairs. The result establishes
an equal-cap implementation/online-controller advantage on this held-out gate;
it does not claim a higher-quality tree, because acceptance is intentionally
identical here.

## Rejected candidates

- A fully reserved 15-node greedy spine reduced acceptance.
- Fixed depth penalties from -0.05 through -0.40 reduced acceptance or TPOT.
- Fixed depth bonuses from +0.02 through +0.20 did not improve TPOT.
- Proposal temperatures 0.70, 0.85, 1.15, and 1.30 all lost to the official
  probability ordering.
- Fixed B80 and B100 lost to B128.
- Progressive exact top-k (33 → 65 → 128 with a boundary sentinel) rarely
  required expansion, but `torch.topk(33)` was not faster on H20 and the extra
  guard increased measured tree-build time, so it was removed.
- B192 beat official DDTree B128 but did not beat DDTree B192 consistently;
  this was a node-cap gain and is not used as evidence of equal-cap superiority.

## Claim boundary

The development evidence supports promoting v7 into the formal experiment
candidate. Cross-model, cross-dataset, and temperature-specific formal runs
must be regenerated from scratch. A formal paper claim should report those
results and must distinguish system-path speedup from tree-quality gains.
