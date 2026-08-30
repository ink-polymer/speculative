# Temperature-0 RL speculative-decoding report

Date: 2026-08-30

## Final temperature-specific design

Temperature 0 is optimized independently from temperature 1. The failed PPO
budget policy was replaced by a Bayesian linear contextual bandit whose reward
is end-to-end output-token throughput. Prompt-only features are used; dataset
labels and test outcomes are never policy inputs.

The temperature-0 action set keeps a fixed 128-node verification shape and
selects among the official DDTree topology and three calibrated best-first
topologies. A reusable `StaticCache` CUDA Graph runs target verification. The
target model, ancestor-only tree mask, accepted-path KV commit, temperature,
and output-token accounting remain DDTree-compatible.

The DFlash draft was domain-adapted on target continuations from the disjoint
192-row policy-training split. It uses two epochs, four assistant-side anchors
per row, a 15-token future loss, and learning rate `2e-6`. The original
unmodified DFlash checkpoint and 60-node eager DDTree remain the fixed
comparator.

## Data integrity

The final test file has exactly 2,000 rows:

- GSM8K: 1,169
- MATH-500: 450
- HumanEval: 149
- MBPP: 232

The separate 240-row optimization pool is split into 192 training rows and 48
validation rows. Source-id overlap and formatted-prompt overlap between train,
validation, and official test are all zero.

## Frozen-policy validation

The checkpoint was trained on 192 prompts with 128 generated tokens, then
frozen before a 48-prompt, 512-token validation. Both methods are divided by
the same target-only throughput for each prompt.

| Metric | Original DDTree | T=0 RL policy | Gain |
|---|---:|---:|---:|
| Mean speedup | 5.953x | 8.377x | +2.424x |
| Aggregate speedup | 5.711x | 8.132x | +2.421x |
| Mean committed tokens | 8.979 | 9.384 | +0.406 |

Per-domain policy mean speedup:

| GSM8K | MATH-500 | HumanEval | MBPP |
|---:|---:|---:|---:|
| 8.303x | 9.565x | 7.821x | 6.775x |

The frozen action histogram is 29 depth-reward trees, 18 standard 128-node
trees, and one temperature-calibrated tree. Target-only exact matches were
21/48 for original DDTree and 20/48 for the policy; the one-count difference
is reported because BF16 batched tree verification can take a different greedy
branch from sequential target-only decoding.

## Formal evaluation

The untouched official 2K evaluation uses 128 generated tokens per prompt,
the frozen checkpoint, the original 60-node DDTree comparator, and resume-safe
JSONL output. All 2,000 prompts completed successfully.

| Metric | Original DDTree | T=0 RL policy | Gain |
|---|---:|---:|---:|
| Mean speedup | 5.225x | 7.446x | +2.221x |
| Aggregate speedup | 4.978x | 7.108x | +2.130x |
| Mean committed tokens | 8.086 | 8.515 | +0.428 |

Formal per-domain mean speedup:

| Dataset | Prompts | Original DDTree | T=0 RL policy | Gain |
|---|---:|---:|---:|---:|
| GSM8K | 1,169 | 4.863x | 6.916x | +2.053x |
| MATH-500 | 450 | 6.094x | 8.654x | +2.560x |
| HumanEval | 149 | 5.354x | 7.763x | +2.409x |
| MBPP | 232 | 5.281x | 7.569x | +2.288x |

The frozen action histogram is 1,330 selections of
`calibrated:128:1:0.15`, 659 selections of `ddtree:128`, and 11 selections of
`calibrated:128:1.15:0`. The downloaded 2,000-row JSONL has SHA-256
`3311490183e6fc394e9d9b36fc3039ff01822adf8c809ac339db1a4db0c23adf`.

## Main artifacts

- `scripts/finetune_dflash_draft.py`
- `scripts/benchmark_topology_bandit.py`
- `scripts/run_t0_graph_bandit_pipeline.sh`
- `scripts/run_formal_t0_graph_bandit.sh`
- `src/dflash_specblock/topology_bandit.py`
- `third_party/ddtree_official/ddtree.py`
- `checkpoints/topology_bandit_t0_graph_v1.json`
- `outputs/topology_bandit_t0_graph_v1/validation48x512.jsonl`
- `outputs/topology_bandit_t0_graph_v1/validation48x512_summary.json`
- `outputs/topology_bandit_t0_graph_v1/formal2k_x128.jsonl`
- `outputs/topology_bandit_t0_graph_v1/formal2k_x128_summary.json`
