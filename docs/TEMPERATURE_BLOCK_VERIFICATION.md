# Temperature sampling and Block Verification

The temperature comparison uses the fixed 2,000-prompt suite at
`datasets/processed/dflash_official/prompts_benchmark_2k.jsonl`:

- GSM8K: 1,169
- HumanEval: 149
- MATH500: 450
- MBPP: 232

Run on the CUDA benchmark host:

```bash
TEMPERATURE=1.0 MAX_NEW_TOKENS=128 \
  bash scripts/run_temperature_block_verification_2k.sh
```

The output contains four methods evaluated with the same model, prompt order,
per-prompt seed, temperature, token limit, and BF16 target backend:

1. `baseline`: ordinary target-model sampling;
2. `dflash`: sampled DFlash block with standard token rejection sampling;
3. `ddtree`: the official DDTree target-sampling tree walk;
4. `block_verification`: sampled DFlash block with Sun et al. (2024) joint Block
   Verification and its block residual distribution.

`block_verification` is intentionally the paper's strict single-draft algorithm.
Applying the same verifier independently to every DDTree leaf is not distribution
preserving because leaf events overlap and share prefixes. A multi-path verifier
requires a separately specified path-selection distribution and residual coupling;
it must not be reported as the 2024 lossless algorithm.

Sampled outputs are not expected to be pairwise token-identical. Compare throughput,
time per token, committed tokens per verification, and statistical output quality or
distributional tests rather than greedy exact match.
