# Atom-coupled tree block verification (AC-TBV)

后续研究入口为 [DIFFUSION_TREE_BV.md](DIFFUSION_TREE_BV.md)：利用单步块扩散
核的全深度随机 trie。本文实现继续作为 `atom_control`，不代表新方法的实验或证明。

Status, 2026-09-07: executable research candidate. No H200/GH200 speed result,
publication novelty certificate, or claim of universal dominance. This is an
inference-algorithm architecture, not a newly trained neural-network architecture.
Existing official baselines and remote running experiments are not replaced.

Local validation completed: 732 tests, zero failures/errors/skips (31.130s),
recorded in `.artifacts/atom-tree-integration-tests.xml`. This includes the
new verifier, existing GBV suite and synthetic formal-run/resume/report tests.
The latter are orchestration checks, not measured model speed or quality.
Shell syntax, CLI/plan and CPU-only numerical-probe execution were checked.
The H200 formal plan contains 63,666 records / 70,146 generation turns;
none have been run. SSH was still disconnected at final local validation.

## What is different from the failed shared-suffix candidate

The proposal retains a small latent probability space at every draft position.
The same atom can map to different tokens under different first-token branches.
Thus branch suffixes are correlated but need not be identical. Their JOINT
proposal law, not just each path's marginal probability, enters block verification.
Branch, accepted depth and correction jointly describe an exit from the tree;
there is no greedy path choice followed by an assumed IID proposal correction.

The default proposer uses existing frozen DFlash logits, top-3 first tokens,
and top-8 normalized suffix supports. Shifted inverse CDFs couple the branch
proposals. Instead of assigning root a the same offset a/K at every depth, the
main method cycles multiplicative permutations of the K low-discrepancy strata.
This preserves every per-position DFlash marginal while preventing one root from
being locked to one quantile band for its whole path. The old fixed a/K schedule
is retained as an explicit ablation. The CDF partition has at most K*R+1 padded atom columns.
The actual partition widths define the recorded proposal, including floating
rounding. Supports and shifts are selected BEFORE the latent suffix draws.
The generic interface accepts different marginal proposals per branch; the
current default does NOT provide a trained branch-conditioned drafter.

This changes the random tree family as well as its verifier. The default still
has K=3 branches of L=15 nodes: 45 draft nodes plus a clean anchor. All target
conditionals are obtained in one ordinary ancestor-masked target forward;
there is one DFlash forward per round. Equal token IDs under different roots
retain separate KV rows. The final correction is processed as the next anchor.

## Exact-arithmetic construction

Fix the committed context and preselected distinct roots S. Let Z=p(S), and
alpha_a=p(a)/Z within S. Outside S, emit one target token with its original mass.
For each suffix position j, the latent atom H_j has distribution s_j(h); branch a
maps it to token g_aj(h). These finite mappings are fixed before drawing H.
The API currently represents a product law over positions, not an arbitrary
history-adaptive joint distribution. Each branch marginal is computed from
the actual atoms:

    q_aj(x) = sum_{h:g_aj(h)=x} s_j(h).

After the sampled latent prefix, obtain p_aj(x) on branch a's OWN token prefix.
Lift the supported target mass to atoms:

    l_aj(h) = p_aj(g_aj(h)) s_j(h) / q_aj(g_aj(h)).

For zero q, take the lift as zero; all such target mass stays in the exit
residual. Several atoms may map to the same token and must all be summed.
Initialize e_a=w_a=alpha_a. At each position:

    B_a(h) = min(e_a l_a(h), alpha_a s(h))
    C_a(h) = w_a l_a(h)
    D_a(h) = C_a(h)-B_a(h)
    U(h)   = s(h)-sum_a B_a(h)
    F_a(h) = B_a(h)+D_a(h) min(1, U(h)/sum_b D_b(h)).

A zero extra-capacity denominator means zero added flow. Update on the drawn h:
e'_a=B_a(h)/s(h), w'_a=F_a(h)/s(h). By induction w>=e>=0, sum(w)<=1,
F<=C and sum_a F_a(h)<=s(h). This reserves the early-root ordinary-BV flow
in the lifted space before pooling remaining capacity.

The target-space exit residual is

    R_aj(x) = w_aj p_aj(x) - sum_{h:g_aj(h)=x} F_aj(h).

It is nonnegative, including outside-proposal tokens. Let r_j=sum_{a,x}R_aj(x)
and f_j=1-sum_a w_aj. The terminal row has F=0, r=sum(w), f=1-sum(w).
Using the existing LV backward-flow construction, select the deepest successful
row, with local success r_j/(r_j+f_j). Its endpoint probability is

    e_j = [r_j/(r_j+f_j)] product_{t>j}[f_t/(r_t+f_t)].

Zero-denominator rows are unreachable and may be completed arbitrarily.
Sample (branch a, depth j) jointly with weight Z e_j R_aj(Sigma)/r_j, or take
the outside-root event with mass 1-Z. Only then construct and sample the ONE
selected full-vocabulary residual R_aj. Two categorical draws suffice.

Proof outline: conditional on a latent prefix, the expected failure of all
later rows is

    sum_h s_j(h)[1-sum_a w'_a(h)] = 1-sum_{a,h}F_aj(h) = f_j+r_j.

Backward induction therefore gives unconditional exit mass R_aj(x), after
averaging future atoms. For each branch prefix, its residual exits and outgoing
mapped atom flows partition w_aj*p_aj pointwise. Summing over latent histories
that map to the same token prefix telescopes to the target autoregressive law.
The outside-root event completes the first-token distribution. Repeated rounds
compose with target continuation; applying EOS/length stopping preserves the
corresponding stopped law. This is an exact-arithmetic argument, not a proof
that BF16 tree logits equal sequential logits or that finite RNG is exact.

No-pool F=B is an ordinary early-root BV control on the SAME marginal proposals.
The pointwise protected floor implies every accepted-length survival probability
is at least that control's, after integrating latent/verifier randomness. It
does not dominate DDTree or the old shared-suffix tree family universally.

## Independent checks and falsification

`tests/gbv_paper/test_atom_tree_bv.py` implements a Fraction forward-conservation
oracle independent of the production backward sampler. It exhausts latent
draws and actual categorical choices, completes outputs with Target to compare
the FULL joint string law, and checks protected length tails. It includes
zero source mass, zero target/root coverage, duplicate atom mappings, differing
branch proposals, empty suffixes, greedy generation, EOS, caps, KV reuse and
non-shared-prefix execution with a tiny real Qwen/DFlash model.

Two binary examples use exactly the same four draft nodes, marginal proposals
and target evaluations. Expected emitted tokens, including correction:

| Target second-token rule | Aligned coupled BV | Stratified coupled BV |
| --- | ---: | ---: |
| 90% token 0 independent of first token | 2.6 | 3.0 |
| 90% copy of first token | 3.0 | 2.6 |

Both gains and failures are retained. These are distributional mechanism checks,
not GPU speedups or evidence that arbitrary diversification always helps.

The verifier's flow bookkeeping uses O(K*M*C) intermediates. It still must read
target logits and normalize over the full vocabulary: O(K*M*V) work remains,
and the target LM head still outputs full logits. The production path avoids
materializing every full probability/residual row; one exit row is expanded.
It currently creates a temporary FP64 scaled-logit tensor for normalizers.
No kernel fusion, new compiler backend or trained adapter is used.

## Prior art / claim boundary

Block acceptance and residual correction are established by
[Block Verification](https://arxiv.org/abs/2403.10444). Multi-path verification
already appears in [GBV](https://arxiv.org/abs/2602.16961),
[SpecTr-GBV](https://arxiv.org/abs/2604.25925), and
[UniVer](https://arxiv.org/html/2605.04543v1). Backward layer-flow sampling is
attributed to [Layer Verification](https://chulheeyun.github.io/publication/cha2026layer/).
CDF coupling/stratification is not claimed as an invention. The present object
to investigate is this specific joint atom proposal plus protected mapped flow
and sparse exit implementation. The limited literature review has NOT established
priority or excluded a reduction to prior frameworks. No top-conference claim.

## Running and experimental scope

New method: `Variant(method="atom_tree_bv", paths=3, length=15, tree_budget=45)`.
Four mechanism ablations: `atom_tree_bv_fixed` (old depth-invariant stratification),
`atom_tree_bv_aligned` (coupling only),
`atom_tree_bv_no_pool` (joint excess-flow sharing), `atom_tree_ancestral`
(ordinary target traversal on the same proposed tree). The aligned control
shares the same truncated marginal proposals, unlike the legacy full-q control.

`configs/atom_tree_formal.json` reuses the previous seven datasets and fixed
samples, freezes Qwen3-8B checkpoints, T=1, three seeds, three timing repeats,
and 2048-token cap. There are ten methods: Target, DFlash matching, DDTree,
the full candidate, four ablations, failed protected control, and old GBV.
The shared-engine baseline names do NOT imply executing the unmodified official
benchmark. Official DDTree/DFlash runners remain separate; their numbers must
not be relabeled as shared-engine controls or mixed across TPOT definitions.

The existing staged runner supplies provenance, complete-group resume, strict
checkpoint gates, objective code/math scoring and paired prompt-cluster CIs.
The new success gate requires BOTH DDTree and DFlash decode-speed CI lower bounds
above one. MT-Bench has no external judge and no quality claim. A complete formal
report is impossible until all gates, three repeats, scoring and artifacts exist.

The formal registration is H200. Arrhenius's GH200 pilot explicitly records GH200
without changing that registration. Restored SSH and an isolated source snapshot
are required before submitting `scripts/atom-tree-pilot.sbatch`; it only runs
three hand-written diagnostic prompts (96-token cap), not the formal matrix.

Local check:

```sh
.artifacts/gbv-test-venv/bin/python -m pytest tests/gbv_paper -q
GBV_FORMAL_PYTHON=.artifacts/gbv-test-venv/bin/python bash scripts/run_terminal_mass_formal.sh plan --study configs/atom_tree_formal.json
```

Do not advance from the pilot to a multi-day run merely because the code works.
First measure complete decode/e2e latency, branch coverage, accepted tokens,
actual GPU identity, same-state replay, memory and strict correctness evidence.
