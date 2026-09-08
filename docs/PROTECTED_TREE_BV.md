# Baseline-protected tree block verification: development candidate

Status: real-arithmetic argument and executable finite-state oracles; GPU speed
and publication novelty are not established. This replaces neither official
DDTree nor the running AdaptiveTree experiment.

## Proposal contract

Choose K distinct first tokens S before sampling an independent shared suffix
z from the actual product proposal Q. Keep separate target/KV rows for each
root even when suffix token IDs coincide. The target sees the resulting K
branches in one tree-masked forward pass. For K=3 and length L=15 there are 45
draft nodes, plus the clean anchor. Roots outside S are sampled directly with
their target mass. Inside S write alpha[a] = p(root=a)/p(S).

Q is the distribution actually used by the drafter, not its argmax or a
renormalization chosen after observing z. Each p_a(x|s) must be the target
conditional on that specific root and suffix prefix. The theorem does not
certify that BF16 tree-masked logits equal sequential target logits.

## Forward flow

At prefix s maintain an early-root ordinary-BV floor e_a and an actual score
w_a. Initially both equal alpha[a]. For each next token x reserve

    B_a(x) = min(e_a * p_a(x), alpha[a] * Q(x)).
    C_a(x) = w_a * p_a(x).
    D_a(x) = C_a(x) - B_a(x).
    U(x)   = Q(x) - sum_a B_a(x).
    F_a(x) = B_a(x) + D_a(x) * min(1, U(x)/sum_b D_b(x)).

Use zero additional flow when the denominator is zero. On the actual proposed
token z update e'_a=B_a(z)/Q(z) and w'_a=F_a(z)/Q(z). Only K-vectors participate
in the sequential recurrence; vocabulary-wide residuals are batched. At the
end of the proposed suffix set Q identically zero, making the last row a bonus
row. The residual is R_a(x)=C_a(x)-F_a(x).

Induction: w_a >= e_a >= 0 and sum_a w_a <= 1. B fits both target capacities
and source Q; D and U are nonnegative. Therefore 0 <= F_a <= C_a and sum_a F_a
<= Q. The next scores obey the same invariants, with w'_a >= e'_a. This is a
floor on each branch, not just on a root-averaged heuristic.

## Backward selection and exactness

Reuse the layer-wise backward flow-sampling construction (LV; see the prior
attribution and reduction in RM_BV_NOVELTY_AUDIT.md). At depth j let
s_j=sum_a w_a, r_j=sum_{a,x} R_a(x), and t_j=1-s_j+r_j. The local success
probability is r_j/t_j and failure probability (1-s_j)/t_j. Select the deepest
successful depth, then draw (root, correction) jointly proportional to R at
that depth. A zero t_j is an unreachable row and is completed arbitrarily.
The implementation draws depth, correction, and root in three categorical
draws. In particular, the root is NOT drawn from the old RM target posterior.

Why this works: condition on the proposal prefix through depth j. Backward
induction gives expected probability of no later success equal to

    sum_x Q(x) * (1 - sum_a w'_a(x))
      = 1 - sum_{a,x} F_a(x) = 1 - s_j + r_j = t_j.

Thus the unconditional mass of selecting (j,a,x), averaged over future
proposals, is exactly t_j*(r_j/t_j)*(R_a(x)/r_j)=R_a(x). At the last row the
same identity holds because F=0 and t=1. Multiply by proposal-prefix mass and
p(S). Residual exits plus continuation flow partition C_a pointwise; these
terms telescope across depths to the target autoregressive law. Outside-S
mass completes the root distribution. Repeating rounds at the emitted
history preserves the joint target law in exact arithmetic. Stopping at EOS
or a deterministic output-length cap preserves the corresponding stopped law.

## What improves, and what does not follow

The probability of reaching any suffix prefix, averaged over later proposal
tokens and verifier randomness, is its proposal mass times sum_a w_a. Since
w_a >= e_a at every possible prefix, every emitted-length survival probability
and the expected committed length are at least those of early-root ordinary
BV on the SAME shared-suffix proposal. This repairs both recorded RM failures:

| Exact small case | Old RM | Early-root BV | Protected |
| --- | ---: | ---: | ---: |
| With zero probabilities | 3.167586 | 3.240000 | 3.240000 |
| Strictly positive | 3.1789025 | 3.249040 | 3.249340 |

These are toy expected lengths, not measured tokens/s. They imply neither
dominance over official DDTree's different tree proposal nor lower latency.
The additional flow work can outweigh acceptance gains. Finite-precision
behavior, memory traffic, tree masks and KV compaction need separate tests.

## Validation and experiment paths

- `test_protected_tree_bv.py`: independent Fraction forward oracle, complete
  target continuation law, exhaustive branches of the actual sampler, all
  emitted-length tails, zero support, recorded failures and long prefixes.
- Shared engine tests include the protected method for greedy matching,
  stochastic repeatability, cache reuse/compaction, EOS and length caps.
- `protected_pilot.py` runs CPU/CUDA flow parity, real checkpoint captures,
  same-state verifier replay, paired end-to-end diagnostic timings, greedy
  checks and detailed checkpoint audit. It uses three hand-written development
  prompts, not the evaluation data. It never marks formal completion.
- `terminal_formal.py --study configs/protected_tree_formal.json` retains
  immutable registration, complete paired groups/resume, correctness gates,
  three seeds and three timing repeats, two latency metrics, paired prompt
  bootstrap, objective quality scoring and provenance. It compares eleven
  methods including target, DDTree, DFlash, GBV, old RM and shared-tree controls.
  The existing seven datasets and fixed source selection remain unchanged.

The formal template remains H200. A GH200 pilot is explicitly labeled GH200;
it cannot pass the H200 runtime gate. No automatic multi-day formal job is
launched by the pilot. Docker code scoring and strict checkpoint gates must
be satisfied before a formal completion report can be produced.

This forward allocation is a research candidate, not a certified first-in-
literature result. Generic block verification, shared proposals and LV backward
sampling are not claimed as new. A current, specific literature comparison
and convincing controlled GPU evidence are still required.
