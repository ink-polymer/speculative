# Live experiment status — 2026-09-08

This page contains sanitized metrics and scheduler identifiers only. Model
weights, prompts, caches, mismatch token dumps, and large tensor captures are not
uploaded.

## Running jobs

| Job | Model | Experiment | Temperatures | State at submission |
|---:|---|---|---|---|
| 2142136 | Qwen3-4B | diffusion scaffold tree block-verification pilot | 0.3, 0.6, 1.0 | PENDING (Priority) |
| 2142137 | Qwen3-8B | diffusion scaffold tree block-verification pilot | 0.3, 0.6, 1.0 | PENDING (Priority) |

The pilots run the full `tests/gbv_paper` gate before loading checkpoints.
Pilot metrics are diagnostic and use three fixed development prompts; they are
not formal benchmark claims.

## AdaptiveTree results available now

Only the Qwen3-8B T=0 run has completed paired SDPA/FlashAttention artifacts,
for seven of ten datasets. Speedup is baseline decode TPOT divided by method
decode TPOT, so values above 1 are faster.

| Dataset | Adaptive vs DFlash | Adaptive vs best DDTree | no_exploration vs DFlash | no_exploration vs best DDTree |
|---|---:|---:|---:|---:|
| GSM8K | 1.324x | 1.378x | 1.408x | 1.467x |
| MATH-500 | 0.910x | 1.138x | 1.185x | 1.482x |
| AIME 2024 | 0.984x | 1.068x | 1.403x | 1.523x |
| AIME 2025 | 0.792x | 0.913x | 1.313x | 1.513x |
| HumanEval | 1.107x | 1.166x | 1.481x | 1.561x |
| MBPP | 0.959x | 1.004x | 1.272x | 1.332x |
| LiveCodeBench | 1.251x | 1.425x | 1.361x | 1.551x |
| **7-dataset geometric mean** | **1.032x** | **1.143x** | **1.343x** | **1.488x** |

AdaptiveTree wins 3/7 datasets against DFlash and 6/7 against tuned DDTree.
The `no_exploration` ablation wins all seven stored aggregate comparisons and is
therefore retained in both the resumed 8B run and the new 4B run.

These speed rows are diagnostic only. The run used
`record-bf16-mismatches`: 354/1,472 backend responses were exact across every
stored method, 1,118 had at least one method mismatch, and 522/736 paired prompts
had different SDPA and FlashAttention target-baseline tokens. Accordingly,
`publication_gate_passed=false` and `strict_lossless_claim_eligible=false`.

Missing AdaptiveTree work: Qwen3-8B SWE-bench, MT-Bench, and Alpaca; all Qwen3-4B
datasets. The page will be updated when new validated artifacts are available.
