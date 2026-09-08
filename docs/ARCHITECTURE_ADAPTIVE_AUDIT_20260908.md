# Architecture and AdaptiveTree audit — 2026-09-08

This audit supersedes the earlier interpretation that the terminal-mass
end-to-end scan demonstrated a verifier-architecture speedup. It did not. The
end-to-end observations remain valid diagnostic measurements, but their
speedup cannot be attributed to the verifier alone.

No prompt text, generated token ids, model weights, caches, or captured
probability tensors are included in the committed artifacts.

## Same-real-tree verifier replay

The replay freezes the complete verifier input: probability-tree parents,
edge tokens, and every FP64 Target probability row. It also freezes L=15,
B=45, BF16 model weights, Target/Draft SDPA, TF32 off, the model revisions,
and temperatures 0.3/0.6/1.0. Only the verifier implementation changes.

Each model result contains 252 paired timing observations per implementation:
three temperatures, three diagnostic prompts, four captured real tree states
per prompt, seven randomized repeats, and 100 calls per observation. The
metric below is `DDTree verifier latency / candidate verifier latency`; values
above 1 would be faster than DDTree. All runs used an NVIDIA GH200 120GB with
PyTorch 2.9.1 and CUDA 13.0.

| Verifier execution | Qwen3-4B | Qwen3-8B | Architecture gate |
|---|---:|---:|---|
| Original terminal mass, internal rows | 0.290x | 0.308x | FAIL |
| Original terminal mass, dense rows | 0.289x | 0.305x | FAIL |
| Sparse child-complement terminal mass | 0.361x | 0.376x | FAIL |
| One joint `(terminal node, correction token)` draw | 0.487x | 0.496x | FAIL |
| One-launch fused CUDA ancestral traversal | **0.614x** | **0.667x** | FAIL |

The fastest attempted architecture, the fused CUDA traversal, also fails at
every temperature:

| Model | T=0.3 | T=0.6 | T=1.0 | Three-T geometric mean |
|---|---:|---:|---:|---:|
| Qwen3-4B | 0.621x | 0.610x | 0.610x | 0.614x |
| Qwen3-8B | 0.709x | 0.665x | 0.628x | 0.667x |

The exact terminal-event enumeration, rare-positive-tail tests, fairness
tests, and deterministic inverse-CDF path test for the CUDA kernel passed
before timing. The result is therefore not a correctness failure: PyTorch's
batched DDTree `multinomial` remains faster on this shape than the tested
exact alternatives.

Complete sanitized observations:

- [Qwen3-4B verifier replay](../results/audits/20260908/qwen3_4b_terminal_architecture_replay.json)
- [Qwen3-8B verifier replay](../results/audits/20260908/qwen3_8b_terminal_architecture_replay.json)

### Why the end-to-end scan looked faster

The two exact-law samplers consume random numbers differently. Equal seeds
therefore reproduce each method independently, but do not force the methods to
take the same realized acceptance path. A different accepted prefix changes
both tokens committed in that round and every later model state. For example,
the earlier Qwen3-4B T=0.3 tuned terminal result committed about 6.66 tokens per
round while its DDTree control committed about 5.89. That difference is a
sampling-trajectory effect, not evidence that the terminal verifier executed
faster. The Qwen3-4B B=60 result also changed the tree budget, so it was not an
architecture-only pair in the first place.

Consequently, the earlier 1.208x/1.077x end-to-end geometric means are retained
as diagnostic observations only. They are not architecture-speedup claims.

## Why DDTree is slower than DFlash in the partial AdaptiveTree run

Across the seven completed Qwen3-8B dataset pairs, the geometric-mean DDTree
speed ratio versus DFlash is 0.9026x, meaning DDTree throughput is about 9.7%
lower (or its TPOT is about 10.8% higher). This is not only a differing-output
artifact. On the five datasets with at least 28 exact-output pairs, the
post-hoc exact-output geometric mean is 0.8947x:

| Dataset | Exact responses | DDTree speed vs DFlash |
|---|---:|---:|
| GSM8K | 60 | 0.934x |
| HumanEval | 28 | 0.937x |
| LiveCodeBench | 63 | 0.867x |
| MATH-500 | 28 | 0.782x |
| MBPP | 41 | 0.966x |

The mechanism is visible in both the protocol and stage metrics:

1. The selected DDTree is B=512 on six datasets and B=256 on AIME25. It asks
   the Target to verify 257 or 513 tree positions with an explicit 4-D branch
   mask. DFlash verifies one 15-token chain.
2. DDTree must use Target SDPA for that explicit tree mask. The independently
   selected DFlash backend is FlashAttention2 on five of seven datasets and
   SDPA on GSM8K and HumanEval.
3. DDTree gains roughly 2.7--3.4 accepted draft tokens per round, but that gain
   does not amortize the larger Target forward, tree construction, mask
   compilation, posterior sampling, and KV-cache compaction.
4. Representative per-output-token Target verification time is 7.247 ms for
   DDTree versus 5.741 ms for DFlash on MATH-500, and 7.984 versus 6.921 ms on
   LiveCodeBench. DDTree additionally spends about 0.14--0.16 ms/token on tree
   construction on those datasets.

This backend asymmetry is part of the frozen official comparison rather than
a hidden implementation change, but it explains part of the measured gap.

## Why `no_exploration` is the best measured ablation

The name is easy to overinterpret. `no_exploration` only sets the forced
exploration interval to zero. It still runs one warmup per candidate budget,
keeps the latency and acceptance EWMAs, and selects a budget adaptively.

Three effects explain the current result:

1. The workload is stationary enough that repeated exploration is harmful.
   Scheduled exploration has much lower observed throughput than ordinary
   decisions: 0.086 versus 0.155 tokens/ms on GSM8K, 0.103 versus 0.132 on
   MATH-500, 0.072 versus 0.106 on MBPP, and 0.107 versus 0.154 on
   LiveCodeBench.
2. Forced rounds are only about 1--2% of decisions, so they do not explain the
   whole gap. The full controller also drifts toward smaller budgets, whereas
   `no_exploration` settles almost entirely on B=128. For example, the full
   controller uses B=128 for 1,747 of 5,977 AIME25 rounds and 6,268 of 10,090
   MATH-500 rounds; `no_exploration` uses B=128 for 5,642 of 5,652 and 9,667
   of 9,751 rounds respectively.
3. The latency model puts `draft + tree_build` into one global `_fixed_ms`,
   even though tree-build cost depends on the selected budget. Only
   `tree_compile + verify + commit` is budget-specific. Combined with stale
   per-budget EWMAs as context length grows and a fixed method execution order,
   this can create noisy or self-reinforcing budget rankings.

The large all-trajectory result (`no_exploration` 1.488x versus best DDTree)
is confounded by recorded BF16 output mismatches. A stricter comparison of
`no_exploration` directly against full AdaptiveTree on identical-output
responses still favors `no_exploration`, but by a much smaller 1.099x geometric
mean across the five adequately populated datasets. This supports a real
controller effect while rejecting the inflated causal interpretation.

The complete partial audit is
[available here](../results/audits/20260908/qwen3_8b_adaptivetree_partial_audit.json).
It remains post-hoc and the underlying run is not eligible for a strict
lossless claim. A clean follow-up should add a cost-attributed controller
variant, randomize method order, and require strict numerically matched outputs;
the currently running frozen jobs should not be mutated in place.
