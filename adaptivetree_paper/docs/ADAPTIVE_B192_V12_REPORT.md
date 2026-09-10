# AdaptiveTree dynamic B192 v12 development report

Date: 2026-09-10  
Hardware: NVIDIA H20 96 GB  
Status: development-only; not approved as the formal primary method

## Fair comparison contract

- The reference and candidate both have a maximum of 192 draft nodes.
- DDTree is the pinned official implementation at B192.
- `adaptive_fixed_b192_control` uses the same raw-tree, top-k transfer,
  controller bookkeeping, and batched cache-commit path as the dynamic method,
  but is forced to select B192 every round.
- `adaptive_dynamic_b192_v12` chooses before heap enumeration. Its tested
  policy uses a 0.95 current-confidence threshold, 0.90 historical support,
  periodic B192 refreshes, and an acceptance fallback. The retained default
  floor is B160 because B128 was more numerically sensitive.
- Prompt selection is fixed, method order is cyclically balanced, and proposal,
  tree construction, compilation, target verification, controller, and cache
  commit time are all included in TPOT.

## Main 16 x 4 x 256 validation

| Method | TPOT (ms) | Speedup vs DDTree B192 | Acceptance length | Decode rounds | Exact sequence vs B192 |
|---|---:|---:|---:|---:|---:|
| Official DDTree B192 | 3.811942 | 1.0000x | 9.1908 | 1708 | 100.0% |
| Fixed Adaptive B192 control | 3.737497 | 1.0199x | 9.1908 | 1708 | 100.0% |
| Dynamic Adaptive B160/B192 | 3.734779 | 1.0207x | 9.2252 | 1712 | 87.5% |

The dynamic controller selected B192 in 1556 rounds and B160 in 156 rounds.
Its paired TPOT delta versus the fixed Adaptive control was -0.002718 ms/token,
with a prompt-cluster bootstrap 95% interval of [-0.026730, 0.021174]. It won
38 of 64 paired runs. The interval crosses zero, so the dynamic-node component
does not have a demonstrated speed advantage.

The fixed Adaptive control's delta versus official DDTree B192 was -0.074445
ms/token, with a prompt-cluster bootstrap 95% interval of
[-0.114989, -0.043862]. It won 50 of 64 paired runs. This approximately 1.99%
gain is attributable to the raw-tree and batched cache-commit system path, not
to dynamic node selection.

## B128/B160/B192 validation

| Method | TPOT (ms) | Speedup vs DDTree B192 | Budget counts | Exact sequence vs B192 |
|---|---:|---:|---|---:|
| Official DDTree B192 | 3.897889 | 1.0000x | 192 only | 100.0% |
| Fixed Adaptive B192 control | 3.791167 | 1.0282x | 192: 1708 | 100.0% |
| Dynamic Adaptive B128/B160/B192 | 3.786370 | 1.0295x | 192: 1556; 160: 84; 128: 72 | 87.5% |

The same two prompts diverged in every repeat. Because B128 was present in
every divergent dynamic run, it was removed from the retained default policy.
Removing it did not restore strict sequence equality and did not produce a
statistically identifiable dynamic speed gain.

## Numerical-sensitivity control

The pinned official DDTree was run directly at B160 and B192 on the same 16
prompts. DDTree B160 matched DDTree B192 on only 11/16 prompts and was slower:
3.872493 versus 3.844604 ms/token. The dynamic method matched B192 on 14/16.
This shows that strict token identity across node counts is sensitive to BF16
verification batch shape and is not an Adaptive-only cache bug. Formal studies
that compare different node counts should report task accuracy as well as exact
sequence sensitivity.

## Proposal-temperature sweep

| Tree proposal temperature | TPOT (ms) | Acceptance length | Decode rounds | Exact sequence vs B192 |
|---:|---:|---:|---:|---:|
| 0.8 | 4.251611 | 8.4016 | 260 | 62.5% |
| 1.0 fixed control | 4.010640 | 8.8383 | 244 | 100.0% |
| 1.2 | 4.047194 | 8.6457 | 250 | 75.0% |

Both directions reduced acceptance and performance. Proposal temperature 1.0
is retained.

## Decision

Do not replace the formal primary method with dynamic B192 v12. The fair
equal-cap result supports the fixed B192 raw-system implementation, but the
dynamic node selector itself adds no statistically demonstrated speedup on H20.
The v12 implementation and artifacts are retained on an isolated experimental
branch so the negative result is reproducible and cannot silently alter the
accepted v10 branch.
