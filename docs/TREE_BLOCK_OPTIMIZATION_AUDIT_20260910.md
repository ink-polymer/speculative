# Tree-block optimization audit (2026-09-10)

## Decision

The formal Qwen3-4B/T=1 matrix was stopped at the user's request. Its 6,232
completed JSONL records remain intact. None of the candidates in this audit is
eligible to replace the registered method or enter a formal matrix.

It is not technically defensible to guarantee the ordering
`candidate > DDTree > DFlash` before measurement. The development gates enforce
that ordering as a success condition; they do not manufacture it by selecting
favorable prompts, seeds, timing scopes, or precision settings.

## Official timing contract

The vendored DDTree and DFlash implementations:

1. exclude Target prompt prefill;
2. reset the timer after the first Draft forward, so that forward is excluded;
3. include subsequent Draft, tree construction, tree compilation, Target
   verification, posterior selection, cache commit, and synchronization; and
4. divide by every returned token, including the anchor sampled during Target
   prefill.

The development gates use `official_scope_time_per_output_token_ms` for their
primary comparisons. Diagnostic CUDA-event stages are reported separately and
are never substituted for synchronized wall time.

## Where time is spent

One diagnostic DDTree profile on the H20 attributed the measured CUDA stages as
follows. These are diagnostic stage shares, not independently additive wall
timers.

| Stage | CUDA time (ms) | Share of measured stages |
|---|---:|---:|
| Target tree verification | 213.747 | 76.33% |
| Draft | 40.509 | 14.47% |
| posterior selection | 10.622 | 3.79% |
| cache/feature commit | 9.401 | 3.36% |
| tree compilation | 3.003 | 1.07% |
| tree construction | 2.683 | 0.96% |
| stop check | 0.082 | 0.03% |

This explains the low ceiling of posterior-only optimization: even eliminating
posterior selection completely has a single-digit upper bound, while a real
implementation still pays launches and synchronization.

## Measured candidates

All runs used Qwen3-4B, the official DFlash draft checkpoint, BF16 model
execution, SDPA attention, FP64 sampling probabilities, T=1, L=15, and tree
budget 45. Development prompts were synthetic and are not formal data.

| Candidate | What changed | Result versus DDTree | Gate |
|---|---|---:|---|
| internal-only Target + reached-leaf forward | first Target pass verifies only internal nodes | 0.802x, 95% CI [0.731, 0.887] on 3 prompts | reject |
| one-leaf prefetch | prefetch first Draft-ranked leaf | 1.007x on 1 prompt; worse than no prefetch | reject |
| two-leaf prefetch | prefetch first two Draft-ranked leaves | 1.000x on 1 prompt; no observed prefetch hit | reject |
| deferred leaf | commit a reached leaf and sample its continuation as next round's root | 0.924x on 1 prompt | reject |
| same-draw CUDA tree walk | identical `torch.multinomial` draw and RNG state; GPU traversal only | 0.977x, 95% CI [0.976, 0.978] on 6 captured real trees | reject |
| PyTorch `flex_attention` micro-prototype | compiled arbitrary tree-mask attention | 0.317x SDPA (0.613 vs 0.194 ms) | reject |

An additional node-budget scan did not find a latency/coverage sweet spot:
B=16, 24, 32, and 45 reached 0.752x, 0.835x, 0.954x, and 0.776x
DDTree respectively. B=32 was closest but its interval crossed 1 only because
the development set had three prompt clusters; the point estimate failed the
gate.

The required baseline sanity check did pass in the three-prompt confirmation:
DDTree/DFlash was 1.645x with 95% CI [1.464, 2.002].

## Architecture-level tree redesign

The second round of work changed the candidate tree itself rather than wrapping
the same DDTree execution. No CUDA Graph was used.

The resulting **core--spur tree** has two parts:

1. a DDTree best-first prefix core, built from the full FP64 Draft position
   probabilities; and
2. one short random spur sampled from the actual support-truncated diffusion
   proposal and inserted before Target evaluation.

The spur is block-verified exactly. Its correction token is then treated as a
fresh Target draw: if that edge exists in the already-verified core, ordinary
ancestral Target verification continues inside the core without another model
forward. The tree is fixed before any Target row is observed, has at most 45
non-root nodes, and uses one Draft and one Target forward per round.

This composition was checked by exhaustive finite-state enumeration at
T=0.3, 0.6, and 1.0. For every enumerated output sequence, the completed law
matched the autoregressive Target law to absolute tolerance 1e-10. The complete
101-test diffusion-scaffold file also passed in the server environment.

### Development throughput

The final focused screen used three synthetic prompts, 64 output tokens, two
interleaved repeats, BF16 SDPA, FP64 probabilities, T=1, L=15, and B<=45.

| Architecture | Mean committed tokens/round | vs DDTree | vs DFlash | Decision |
|---|---:|---:|---:|---|
| DDTree | 7.000 | 1.000x | 1.677x [1.478, 2.041] | baseline |
| core--spur, s=1 | 5.333 | 0.731x [0.618, 0.970] | 1.225x | reject |
| core--spur, s=2 | 5.701 | 0.750x [0.673, 0.831] | 1.258x | reject |
| core--spur, s=4 | 5.583 | 0.775x [0.611, 0.975] | 1.299x | reject |
| core--spur, s=6 | 6.809 | **0.906x [0.843, 0.946]** | **1.520x [1.247, 1.930]** | reject |
| core--spur, s=8 | 5.917 | 0.784x [0.773, 0.804] | 1.314x | reject |

The support-width and budget probes were also negative. For s=6, support
widths 8, 16, and 64 produced 0.740x, 0.776x, and 0.812x DDTree; node budgets
32, 36, and 40 produced 0.820x, 0.859x, and 0.881x. These were one-repeat
exploratory gates, not confirmatory results. Separately, packing two or three
full paths into ordinary causal SDPA achieved only 0.498x and 0.465x DDTree,
because the drop in accepted tokens outweighed the regular attention layout.

The core--spur design is therefore a valid exact architecture and materially
better than the earlier diffusion scaffold (0.645x DDTree), but it does not
beat DDTree and cannot be promoted to the formal matrix.

### Why the remaining 9.4% gap is structural

In a 96-token diagnostic witness, DDTree used 20 rounds and committed 4.75
tokens/round; core--spur s=6 used 22 rounds and committed 4.32. Target
verification was similar per round (28.43 ms versus 26.54 ms), but selection
and correction rose from 1.34 to 4.42 ms/round and tree construction rose from
0.33 to 1.23 ms/round. The random spur also replaces some best-first core nodes.

When Draft and Target distributions are close, DDTree's best-first fixed prefix
set is already the highest-mass use of a fixed node budget. A randomized spur
can help only where its coupling to the eventual Target output recovers more
mass than the core nodes it displaces. On these witnesses that gain did not pay
for the displaced coverage and exact correction machinery. Thus “block
verification on top of a tree” does not imply an automatic speedup over
DDTree.

## Numerical audit of internal-only Target verification

Removing leaf rows changes the BF16 batch shape, so equality to the 46-row
DDTree tensor must not be asserted. On the captured tree, all audited top-1
tokens agreed. Against cached, one-token-at-a-time Target execution:

| Metric over 27 internal nodes | DDTree 46-row batch | internal-only batch |
|---|---:|---:|
| mean total variation | 0.006820 | 0.001266 |
| maximum total variation | 0.056729 | 0.014289 |

Thus the compact computation was numerically closer to the autoregressive BF16
reference on this witness, but numerical non-inferiority did not rescue its
throughput. A method must pass both numerical and speed gates.

## Why fewer Target rows did not make the system faster

The H20 is launch- and occupancy-limited for these small query matrices. Cutting
the verified rows from 46 to roughly 19--28 did not reduce latency in proportion
to FLOPs. When a reached leaf required a separate one-token Target call, the
second model launch dominated the saved work. Deferring that leaf removed the
second launch but also removed the current round's free bonus, increasing the
round count from 9 to 10 in the quick witness. Both effects outweighed row
pruning.

The same-draw experiment isolates the post-processing limit cleanly. It matched
the baseline output and final RNG state for 16 seeds on each of six real trees,
yet the extra CUDA launch made it 2.3% slower than copying 46 integer samples to
the host and walking the tree in Python.

## Next implementation boundary

A material architecture-level speedup now requires changing the proposal quality,
not merely regrouping the same Draft rows: for example, training the Draft to
produce prefix-conditional block proposals or a tree-aware objective whose
accepted-token gain can exceed DDTree's best-first coverage. With the registered
weights frozen, the other credible boundary is a dedicated tree-aware Target
execution kernel. Generic CUDA Graph capture is explicitly outside this method's
scope. Either route must be followed by:

1. per-layer BF16 comparison against cached autoregressive Target execution;
2. same-tree and distribution-law gates;
3. captured-state kernel timing;
4. disjoint synthetic end-to-end confirmation; and only then
5. a newly registered formal matrix.

The stopped 6,232-record matrix must not be resumed under modified source or
combined with any new candidate's results.
