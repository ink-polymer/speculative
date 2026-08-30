# Temperature-1 RL speculative-decoding report

Date: 2026-08-30

## Final temperature-specific design

Temperature 1 is optimized independently from temperature 0. The failed PPO
budget policy was replaced by a Bayesian linear contextual bandit whose reward
is end-to-end output-token throughput. Prompt-only features are used; dataset
labels and test outcomes are never policy inputs.

The action selects 2, 3, or 5 exact GBV proposal paths once per prompt. GBV
uses rejection-sampling verification for nonzero temperature, so the method
preserves the requested sampling temperature rather than applying greedy
temperature-0 verification. Target model, DFlash draft, sampling seed, token
limits, prompt format, and throughput accounting are shared with the original
DDTree comparator.

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

| Metric | Original DDTree | T=1 RL policy | Gain |
|---|---:|---:|---:|
| Mean speedup | 5.555x | 7.733x | +2.178x |
| Aggregate speedup | 5.198x | 7.568x | +2.371x |
| Mean committed tokens | 8.335 | 14.650 | +6.315 |

Per-domain policy mean speedup:

| GSM8K | MATH-500 | HumanEval | MBPP |
|---:|---:|---:|---:|
| 7.731x | 7.829x | 8.077x | 7.346x |

The frozen action histogram is 40 selections of `gbv:2`, seven of `gbv:3`,
and one of `gbv:5`. Exact token equality is not expected at temperature 1
because comparator and policy consume random draws through different exact
sampling schedules; throughput is therefore computed from each method's own
generated-token count and decode time.

## Formal evaluation

The untouched official 2K evaluation uses 128 generated tokens per prompt,
the frozen checkpoint, the original DDTree comparator, and resume-safe JSONL
output.

| Metric | Original DDTree | T=1 RL policy | Gain |
|---|---:|---:|---:|
| Mean speedup | 4.936x | 7.226x | +2.290x |
| Aggregate speedup | 4.693x | 6.836x | +2.143x |
| Mean committed tokens | 7.576 | 13.521 | +5.944 |

Formal per-domain policy mean speedup:

| GSM8K | MATH-500 | HumanEval | MBPP |
|---:|---:|---:|---:|
| 7.335x | 7.316x | 7.146x | 6.553x |

The formal action histogram is 1,726 selections of `gbv:2`, 259 of `gbv:3`,
and 15 of `gbv:5`. The downloaded 2,000-row JSONL has SHA-256
`33cd867db30c1f6640bcd3822fdf25970c415041c9d183f8b7057f46f99e37cf`.

## Main artifacts

- `scripts/benchmark_topology_bandit.py`
- `scripts/run_formal_t1_bandit.sh`
- `src/dflash_specblock/topology_bandit.py`
- `third_party/ddtree_official/gbv.py`
- `checkpoints/topology_bandit_t1_v1.json`
- `outputs/topology_bandit_t1_v1/validation48x512.jsonl`
- `outputs/topology_bandit_t1_v1/validation48x512_summary.json`
- `outputs/topology_bandit_t1_v1/formal2k_x128.jsonl`
- `outputs/topology_bandit_t1_v1/formal2k_x128_summary.json`
