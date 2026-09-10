# Prefix-conditional tree and block-verification development record

Status: development gate only; the formal benchmark matrix remains stopped.

## What changed

The implementation adds a frozen-DFlash, low-rank recurrent proposal head. A
single masked Draft forward produces all slot features. At depth `d`, the head
conditions its correction on the tokens sampled at depths `< d`, scores only
the original DFlash top-R support, and returns the actual conditional proposal
law used by verification. Target input embeddings condition the prefix state;
Target output embeddings score candidate tokens.

Four tree integrations are retained as explicit ablations:

- `prefix_core_spur_bv`: a sampled conditional spur, a DDTree core, and exact
  single-path block correction followed by exact Target-only continuation.
- `prefix_sampled_spur_tree`: the same random conditional spur guides a fixed
  tree, which is checked by ordinary exact ancestral Target verification.
- `prefix_core_spur_tree`: a deterministic conditional spur guides the tree.
- `prefix_rescored_tree` and `prefix_beam_tree`: complete conditional-tree
  alternatives. The latter directly expands conditional prefix distributions
  instead of relying on an unconditional DDTree candidate pool.

No path uses CUDA Graph. Each decode round uses one Draft forward and one Target
forward, `bfloat16` model execution with SDPA, FP64 sampling probabilities,
temperature 1, length 15, and a strict 45-node tree budget.

## Exactness contract

For a sampled prefix `z_<d`, the head defines and records
`q_d(. | z_<d, x)` on its top-R support. The block verifier receives these exact
rows, not an unconditional or post-hoc surrogate. Its endpoint/residual
decomposition therefore returns the Target autoregressive law for every
positive temperature. The non-BV tree variants are also exact: their random or
deterministic tree is fixed without observing Target probabilities, after which
ordinary Target ancestral verification is exact conditional on that tree.

The tests include exhaustive enumeration of a finite prefix-dependent proposal
and verify its complete output law against the autoregressive Target to
`1e-10`, plus causality, checkpoint-contract, tree-budget, T=0.3/T=1 preflight,
cache, EOS, and one-forward-per-round checks.

## Official timing scope

The measured numerator is unchanged from the upstream-compatible benchmark:
Draft decode, tree construction, tree compilation, Target verification,
selection/correction, stop checking, and commit are included. Prompt prefill
and the first Draft boundary synchronization are reported separately and
excluded. The block path was corrected to avoid eagerly constructing FP64
full-vocabulary probabilities for all 45 Target rows; it now normalizes the
labelled block rows and reached continuation only.

## Final development gate

H20, Qwen3-4B Target, Qwen3-4B-DFlash-b16 Draft, 12 synthetic development
prompts, 3 independent generation seeds per prompt, 64 output tokens, paired
method ordering. Intervals are prompt-cluster bootstrap intervals. These are
not formal benchmark results.

| Method | TPOT (ms) | Committed/round | vs DDTree | 95% CI | vs DFlash |
|---|---:|---:|---:|---:|---:|
| DFlash | 9.362 | 4.109 | 0.748x | — | 1.000x |
| DDTree | 6.864 | 5.769 | 1.000x | — | 1.337x |
| Prefix conditional block, spur 6 | 7.501 | 5.910 | 0.915x | [0.880, 0.959] | 1.224x |
| Prefix sampled-spur tree, spur 6 | 7.209 | 5.856 | 0.952x | [0.888, 1.016] | 1.273x |
| Prefix greedy-spur tree, spur 6 | 7.016 | 5.963 | 0.981x | [0.935, 1.026] | 1.312x |
| Prefix conditional Beam tree | 8.548 | 5.601 | 0.796x | [0.754, 0.829] | 1.065x |

The architecture does beat DFlash, but it does **not** beat DDTree under the
fair development gate. Consequently no formal matrix was launched. The main
bottleneck remains Target tree verification, followed by Draft execution. The
sparse block path avoids an eager 45-row FP64 softmax by construction, but its
acceptance gain is not large enough to pay for the conditional head and tree
construction. Method-specific stage percentages must be read from each
method's own `diagnostic_profiles` entry rather than transferred across methods.

A dedicated single-prompt diagnostic profile attributes the conditional block
time as follows: Target verification 65.74%, Draft 13.57%, selection/correction
11.32%, tree construction 5.43%, commit 2.98%, and tree compilation 0.93%.
Selection/correction took 51.8 ms versus about 54.1 ms in the earlier eager
full-row profile: removing the redundant materialization helped only slightly
because several small sparse normalization and control kernels remain. This
profile is diagnostic, not a throughput result.

## Training ablations

- v2 rank 128 reduced ordinary holdout KL from 2.448 to 1.920 (21.6%) and is the
  strongest actual decoding checkpoint.
- v2 rank 64 reduced holdout KL to 1.966 but performed worse in decoding.
- v3 used 50% on-policy prefixes and remaining-suffix-weighted KL. Its weighted
  holdout objective fell from 2.145 to 1.846 (14.0%), but its speed screen was
  worse; it is retained as a negative result rather than selected.
- Inverse-CDF sampling, conditional-score strength scans, support scans, and
  direct conditional Beam construction are retained as failure ablations.

## Artifacts

- Code: `src/gbv_experiments/prefix_conditional.py`,
  `src/gbv_experiments/train_prefix_conditional.py`, and
  `scripts/benchmark_prefix_conditional_tree.py`.
- Checkpoints: `checkpoints/prefix_conditional/`.
- Complete per-run rows, reports, hashes, timing profiles, and telemetry:
  `results/pilots/20260910/prefix_conditional/`.
- The final comparison is
  `prefix_v2_sparse_softmax_confirm_20260910/report.json`; the immediately prior
  all-row-softmax comparison is
  `prefix_v2_sampled_tree_confirm_20260910/report.json`.
- Method-specific timing decomposition is in
  `prefix_v2_method_profiles_20260910/report.json`.
