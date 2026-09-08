# Branch-Recycling Tree Block Verification

Status: experimental design and isolated implementation. No novelty claim is made.

## 1. Objective

The existing `tree_gbv_full` verifier uses a verified finite tree to select one
root-to-leaf path, applies GBV's selected-distribution correction, and then runs
single-path Block Verification (BV). All off-path nodes are discarded after the
path is selected.

The `tree_gbv_prefix_recycle` variant uses the same transition on a
root-diversified finite prefix tree.  This changes only the finite Draft
proposal before Target verification; the selected-leaf correction and every
recycled BV segment use the same exact conditional laws described below.

Branch-Recycling Tree Block Verification (BRBV) keeps the same finite tree,
Target/Draft temperatures, GBV path law, and BV kernel. Its only new transition
is this: when BV's correction token is already a verified child of the current
tree prefix, that child becomes the anchor of another conditional GBV/BV segment
inside its remaining subtree. No new Target forward is required.

```text
finite draft tree + one masked Target verification
                         |
                         v
              conditional GBV path selection
                         |
                         v
                 Block Verification
                         |
             +-----------+------------+
             | correction is a child? |
             +-----------+------------+
                    yes /   \ no
                       v     v
           commit child      emit final correction
           and restrict tree
                  |
                  +----> next conditional GBV/BV segment
```

The loop advances by at least one tree level whenever it recycles a correction,
so it terminates in at most the tree depth. If a correction reaches a leaf, the
already verified Target row at that leaf supplies one final Target token.

## 2. Probability objects

For a current accepted prefix `s` and its remaining subtree:

1. `R_s` is the original finite-tree leaf distribution conditioned on `s`.
2. `Gamma_s` is GBV's path-selection rule applied to `K` IID paths from `R_s`.
3. `R_s^Gamma` is the exactly computed selected-path distribution.
4. BV receives one sampled path, Target rows on that path, and the conditional
   rows of `R_s^Gamma`.

All leaf selection and proposal probabilities are recomputed after restricting
to the correction child. A correction is never treated as a draft acceptance
unless the exact `(parent node, token)` edge exists in the already verified
tree.

## 3. Losslessness argument

Let `B_s` denote one valid GBV-plus-BV kernel at prefix `s`. By the GBV and BV
results, `B_s` emits a nonempty Target-distributed continuation ending in a
correction token. Condition on every possible emitted continuation `z`.

- If the final token of `z` is not a verified child, BRBV returns `z`, exactly
  as the ordinary kernel does.
- If it is a verified child, BRBV composes `B_s` with another valid conditional
  kernel `B_{sz}` on the corresponding subtree.
- At a leaf it composes with one direct draw from the verified Target row.

Composition with a valid Target-conditional kernel cannot change the law of the
already emitted prefix and produces the correct Target law for all later tokens.
Induction on remaining tree depth therefore gives the Target autoregressive law
for the complete BRBV output. The branch-dependent decision to continue is a
deterministic function of the emitted prefix and fixed verified tree, so it adds
no selection bias.

This is an algorithmic real-arithmetic statement. The implementation uses FP64
probabilities and retains explicit finite/nonnegative checks; it does not claim
that floating-point arithmetic is identical to exact real arithmetic.

## 4. Architecture invariants

- Target and Draft temperatures are unchanged and recorded separately.
- The tree is constructed before Target verification and is never expanded from
  Target probabilities.
- Each continuation stays inside the subtree identified by the emitted
  correction node.
- Selected leaf masses are normalized inside that subtree and converted to
  exact sparse prefix conditionals before BV.
- A committed node id uniquely determines its full prefix and cache row.
- Every recycle advances depth; output length is at most tree depth plus one.
- The default `tree_gbv_full` path remains available as the non-recycling
  reference.

## 5. Difference from checked related work

| Method | Core tree verification operation | Difference in BRBV |
|---|---|---|
| Block Verification, arXiv:2403.10444 | Joint verification of one chain | BRBV composes unchanged BV kernels across correction-hit subtrees. |
| Traversal Verification, arXiv:2505.12398 | Leaf-to-root trials, sibling rejection updates, and backtracking | BRBV is forward-only and never performs the traversal residual-update loop. |
| GBV, arXiv:2602.16961 | Select one of multiple paths, then single-path BV | BRBV re-enters a conditional GBV/BV kernel only after the emitted correction reaches another verified branch. |
| SpecTr-GBV, arXiv:2604.25925 | Optimal transport over multi-draft token blocks | BRBV uses no optimal-transport plan. |
| UniVer, arXiv:2605.04543 | Conditional optimal transport composed across tree levels | BRBV composes whole path-level BV kernels, not level-local transport plans. |
| OPT-Tree (TACL 2025) | Optimizes an adaptive draft-tree structure | Draft-tree adaptation is prior work and is not claimed as a BRBV contribution. |
| TALON, arXiv:2601.07353 | Confidence-aware, budget-driven adaptive token trees | Entropy/confidence-based width/depth allocation is explicitly excluded from BRBV novelty claims. |

This comparison records a bounded literature search, not proof of patent or
publication novelty. The implementation was written from the composition rule
above rather than copied from these works. A paper must cite all related methods
and must not use “first” or “novel” until a broader expert search is complete.
In particular, better adaptive tree construction may be used as an engineering
component in experiments, but it must be attributed to prior work and separated
from the forward conditional-kernel composition being evaluated here.

## 6. Validation gates

Before performance results are admissible:

1. Exhaustively enumerate every selected-leaf draw, every batched BV row draw,
   every dynamic recycle, and every final Target draw on a two-token,
   four-leaf tree for both `K=1` and `K=3`.
2. Check every length-three output sequence against its exact Target probability.
3. Run the full CPU/Tiny-Qwen suite in the pinned remote environment.
4. Run same-process, interleaved `T_target=T_draft=1.0` A/B against
   `tree_gbv_full`, DDTree, and autoregressive Target.
5. Report both committed tokens per round and the added selection cost. A longer
   block alone is not a throughput win.

## 7. Exploratory architecture experiments (GH200; not formal evidence)

These runs held the model pair, prompts, seeds, generation length,
`T_target=T_draft=1.0`, BF16 model weights, and FP64 sampling probabilities
fixed.  They did **not** all hold length, tree budget, and attention backend
fixed, so none is admissible evidence for an architecture-only claim.

| Architecture | Best throughput / DDTree | Decision |
|---|---:|---|
| Masked prefix BRBV, mid-budget | 0.922x | Parameter/budget scan; inadmissible for the formal architecture-only gate. |
| Pack every finite-support leaf | 0.745x | Reject: duplicated shared-prefix compute. |
| Pack only the sampled GBV paths (SDPA) | 0.765x | Reject: regular batching does not repay low acceptance. |
| Pack sampled paths (FlashAttention-2) | 0.894x | Backend differs from DDTree; inadmissible for the formal gate. |

The next architecture must improve the proposal/Target match rather than only
rearranging verification.  In particular, DFlash's position-wise marginals need
a branch-conditional proposal component if GBV is to close the accepted-length
gap.  Any such component is an engineering hypothesis until separately checked
against prior branch-conditioned drafting work; it is not yet a novelty claim.

The formal rerun is fail-closed: both methods use the same loaded Target and
Draft objects in one interleaved process, SDPA for both, L=15, B=45, identical
prompts/order/seeds/output length, BF16 weights, FP64 probabilities, cache and
feature policies, and `T_target=T_draft=1.0`.  Only the declared architecture
fields may differ, and any new module's latency and memory are included.

## 8. Terminal-Mass Tree Block Verification (TM-TBV)

Status: implemented engineering baseline; losslessness tests pass locally;
no GPU throughput result and no novelty claim. Its unchanged fixed-tree law
is not being advanced as the paper's main algorithmic contribution. The next
research direction and its explicit limitations are recorded in
[TREE_COUPLING_RESEARCH.md](TREE_COUPLING_RESEARCH.md).

The formal follow-up is specified in
[TERMINAL_MASS_FORMAL_PROTOCOL.md](TERMINAL_MASS_FORMAL_PROTOCOL.md) and executed
with `python -m gbv_experiments.terminal_formal`. It includes DFlash, DDTree, AR,
two law-preserving ablations, repeated unprofiled timings, same-state replay,
objective scoring, and a completion/decision gate. V24 remains a pilot only.

### 8.1 Motivation and scope

For a fixed verified tree, the DDTree probability law can be viewed as sampling
one Target token at the root and repeating at every matched child.  Its official
implementation is already more parallel than that description: it samples the
Target posterior at **every** verified node in one two-dimensional multinomial,
copies that vector to the host once, and follows the root-to-exit path through
the pre-sampled rows.  Unused independent row draws are discarded.  The local
baseline now mirrors that implementation; an earlier sequential-loop baseline
was not a fair representation of DDTree and must not be used for speed claims.

The accepted tree path is already optimal for this fixed support: no lossless
verifier can emit a depth-`d` tree prefix more often than the Target probability
mass covered by depth `d`.  Consequently, an architecture-only verifier cannot
honestly promise a higher accepted length than DDTree on the same tree.

This bound assumes the output remains Target-distributed **conditional on this
fixed tree**, and that cached nodes correspond to contiguous output prefixes.
It is not a bound on a coupling averaged over randomly sampled proposal trees:
BV and related methods can improve acceptance in that different setting.

TM-TBV changes the execution graph instead.  It treats the entire verified tree
as one variable-length block, analytically marginalizes DDTree's unused node
draws, samples the node at which a Target continuation first exits the tree, and
then samples the exit token.  It keeps DDTree's exact tree, Target/Draft models,
temperatures, budget, attention backend, caches and probability precision.

### 8.2 Terminal-mass construction

Let `T` be a finite prefix tree with root `epsilon`.  A node `v` denotes its
root-to-node token prefix, `C(v)` is the set of its child-token labels, and

```text
pi(v) = product over edges (u --x--> child) on v of p(x | u).
```

Define the exit mass

```text
e(v) = 1 - sum_{x in C(v)} p(x | v)
```

and terminal mass `m(v) = pi(v) e(v)`.  A leaf has `C(v)=empty`, hence
`e(v)=1`.  TM-TBV performs two device-side categorical draws:

1. sample terminal node `V` with probability `m(v)`;
2. sample `X` from `p(x | V) / e(V)` restricted to `x not in C(V)`;
3. emit `prefix(V)` followed by `X`.

Only the final `(V, X)` pair is explicitly copied to the host on the trusted
`validate=False` execution path used by the engine. The public function's
default validation adds three scalar checks and their host synchronizations.
Official DDTree also uses one explicit posterior-vector transfer, so transfer
count is not a candidate advantage. Framework-internal synchronizations and
host-to-device topology copies still need GPU profiling. The implementation is
`tree_block_verify_terminal_mass` and the engine method is
`ddtree_terminal_block`.

### 8.3 Exactness theorem

**Theorem 1 (partition).** `sum_v m(v)=1`.

**Proof.** Every infinite Target continuation has a unique first edge that is
not in `T`, unless it reaches a leaf, which is terminal by definition.  These
events are disjoint.  The probability of reaching `v` is `pi(v)` and the
conditional probability of leaving at `v` is `e(v)`.  Their event masses are
therefore exactly `m(v)` and exhaust the Target sample space.  Equivalently,
the identity follows by telescoping parent mass into child-prefix masses plus
parent exit mass.  QED.

**Theorem 2 (losslessness and coupling to DDTree).** TM-TBV and DDTree's
sequential ancestral Target sampler have the same distribution over every
emitted variable-length block.  Extending either block with ordinary Target
sampling therefore yields the exact Target autoregressive law.

**Proof.** For any terminal node `v` and exit token `x not in C(v)`, TM-TBV
with `e(v)>0` assigns probability

```text
m(v) * p(x | v) / e(v) = pi(v) p(x | v).
```

This is precisely the probability that sequential Target sampling follows all
edges of `v` and then samples `x`.  The events cover all outputs by Theorem 1.
QED.

Nodes with `e(v)=0` have zero terminal mass and are never selected; their
conditional exit distribution need not be defined. The theorem covers root-only
trees, unequal leaf depths, unreachable nodes, and full child-vocabulary coverage.

**Corollary (accepted length).** If `A` is the number of accepted draft nodes,

```text
E[A] = sum_{v in T, v != root} pi(v),
```

which is identical for TM-TBV and DDTree.  Any measured accepted-length change
under a frozen tree is an implementation error or sampling noise, not a claimed
algorithmic gain.

### 8.4 Architecture complexity and the speed claim that is allowed

After the Target rows already exist, official DDTree performs one batched
categorical over all `B+1` verified rows and copies the resulting `B+1` token
ids to the host once.  TM-TBV instead performs one categorical over `B+1`
terminal-node masses, one categorical over the selected exit row, and copies
two ids to the host once.  Its possible advantage is therefore not fewer host
synchronizations.  It avoids categorical sampling from unused leaf rows and
replaces the all-row posterior draw with uncovered-mass reductions on the `I`
internal rows plus one selected vocabulary draw.  The audited implementation
batches prefix products using an ancestor index table of size `O(B L)`,
replacing the original per-edge tensor loop.

Numerical robustness changes the memory and compute cost: with `I` internal
nodes, the implementation directly sums their uncovered token weights in an
`I x |V|` masked copy. It uses `O(I |V| + B L + |V|)` temporary storage and
arithmetic, rather than the original `O(B+|V|)` claim. For FP64 and vocabulary
151,936, the masked copy costs about 1.16 MiB per internal node (at most about
52.2 MiB for `I=45`). This extra bandwidth can erase the reduced sampling work.
The V24 report explicitly records each run's peak allocated/reserved memory,
stage timings, token count, and elapsed decode time.

Let `F` be the common draft/tree/Target-forward/commit time and let `H_D` and
`H_T` be the two verifier overheads.  Since both methods have the same expected
committed tokens, a stationary pooled-throughput comparison improves when
`E[H_T] < E[H_D]`, provided common costs and the distribution of visited contexts
are the same.  Whether reducing posterior rows from `B+1` to roughly `I+1`
outweighs the extra masks, reductions, prefix products, and second categorical
launch is hardware- and tree-shape-dependent. This ratio-of-expectations
statement is not an identity for the expectation of each finite run's
tokens/time ratio. The strict V24 benchmark must report failure if the condition
is not observed on the requested GPU.

V24 interleaves end-to-end runs with a shared constructor and fixed controls.
The realized generated prefixes, and thus subsequent trees, can differ between
methods despite identical seed integers. Its 24 prompt/seed pairs are a screen;
a positive point estimate is not a statistical guarantee. A follow-up should
replay the same captured `(parents, tokens, all_p)` states for a direct verifier
latency comparison and repeat unprofiled end-to-end runs with uncertainty
estimates. No GPU results or captured-state replay are claimed by the local audit.

### 8.4.1 Numerical audit and remaining limitations

The original `1 - sum(child probabilities)` formula has a concrete FP64
counterexample: for a row `[1., 1e-20]` and a sole child with token 0, it computes
zero root-exit mass, although token 1 retains positive weight. Clamping and
renormalizing terminal masses do not repair the missing event. The revised
implementation masks child tokens and **sums the complement directly**. It
normalizes internal edge and exit weights by their combined row total, and
reuses the masked row for the exit draw. It does not mutate input probabilities.

Exactness is a real-arithmetic distributional theorem. FP64 multiplication can
still underflow on extremely improbable deep prefixes, reductions can round,
and finite pseudorandom categorical sampling is not an arbitrary-precision
oracle. The implementation must not be described as bitwise identical to
DDTree or exact for every representable nonzero probability. Likewise,
equivalence to sequential model evaluation assumes the tree-attention rows
represent those same Target conditionals; BF16 attention arithmetic can differ
with execution shape even when masks and caches are correct. The sampler
theorem does not establish bitwise equality of those model forwards.

The dedicated tests exhaustively check complete terminal-node/exit-token event
laws on branching, uneven, zero-probability and fully covered trees, compare
`E[A]` with the prefix-mass sum, preserve tails down to `1e-200`, exercise
root-only and noncontiguous inputs, and check invalid-input errors. Passing
these tests is evidence for the implementation, not a universal floating-point
exactness certificate. Same RNG seeds need not produce the same text under
different samplers because the number and order of random draws differ.

### 8.5 Bounded related-work screen (2026-09-07)

- [DDTree](https://arxiv.org/abs/2604.12989) constructs the same probability-mass
  tree and uses ancestor-masked Target verification; TM-TBV changes only the
  post-forward sampler.
- [Block Verification](https://arxiv.org/abs/2403.10444) is an optimal
  proposal/Target correction kernel for one sampled chain.  TM-TBV does not
  compute a Draft residual or change accepted-prefix mass.
- [Traversal Verification](https://arxiv.org/abs/2505.12398) performs bottom-up
  trials with rejection residual updates.  TM-TBV performs no traversal or
  sibling rejection loop.
- [UniVer](https://arxiv.org/abs/2605.04543) uses top-down conditional optimal
  transport and a post-order decision pass.  TM-TBV uses neither OT nor a
  post-order pass.
- [GBV](https://arxiv.org/abs/2602.16961) and
  [SpecTr-GBV](https://arxiv.org/abs/2604.25925) alter multi-path selection and
  proposal correction.  TM-TBV retains DDTree's Target continuation law.
- [Saguaro](https://arxiv.org/abs/2603.03251) pre-speculates across rounds and
  possible verification outcomes.  TM-TBV performs a within-round analytic
  collapse and creates no speculative next-round cache.
- [DARTree](https://arxiv.org/abs/2608.13524),
  [SpecBlock](https://arxiv.org/abs/2605.07243), and
  [CaDDTree](https://arxiv.org/abs/2606.01813) change drafting or tree
  construction.  Those axes are frozen here.

This is a bounded search, not proof that no unpublished, unindexed, or differently
named implementation exists.  The defensible claim is the exact architectural
difference above, subject to a broader expert and code search before publication.

An independent audit checked the linked papers' full HTML and algorithm sections.
At the level of the output law, TM-TBV **is isomorphic to DDTree's ancestral
Target walk**: it marginalizes the intermediate draws and samples the first-exit
distribution directly. The terminal partition and accepted-length identity are
standard probability identities, not a new coupling theorem. The inspected
BV, Traversal, UniVer, GBV, and SpecTr-GBV algorithms use proposal-dependent
couplings or residuals, so their general algorithms are different from this
fixed-tree marginalization; this does not exclude special-case equivalences.
Saguaro changes cross-round speculation, while DARTree, SpecBlock, and CaDDTree
change drafting, adaptation, or tree construction. Those observations support
only an implementation-level comparison to the inspected algorithms, not an
absolute originality claim or a claim that none of their code contains a
similar sampling optimization.
