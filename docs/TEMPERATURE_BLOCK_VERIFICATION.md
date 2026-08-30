# Temperature-1 RL-DDTree with traditional verification

The frozen evaluation uses the fixed 2,000-prompt suite at
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

The runner has two phases.  It first trains a clipped discrete-action PPO policy on the
disjoint rank-training prompt set, saves a PyTorch checkpoint, then reloads that
checkpoint with learning disabled for the 2,000-prompt evaluation.

Before training, the runner regenerates
`prompts_ppo_train_tree15_disjoint_2k.jsonl` from the source rank-training set.  It
removes both matching dataset/source IDs and normalized prompt-content matches against
the frozen 2K evaluation set.  With the bundled inputs this leaves 408 training prompts
and zero source-ID or prompt-content overlaps with evaluation.

Training mode times only `ddtree_ppo`; target, DFlash, and fixed-DDTree controls are
run only in frozen evaluation.  This keeps PPO feedback unchanged while avoiding
three unrelated decoding passes for every policy-training prompt.

The runner defaults the draft attention backend to `flash_attention_2`.  The target
continues to use SDPA because DDTree verification requires a custom 4D ancestor mask.

The evaluation output contains four methods with the same model, prompt order,
per-prompt seed, temperature, token limit, and BF16 target backend:

1. `baseline`: ordinary target-model sampling;
2. `dflash`: sampled DFlash block with standard token rejection sampling;
3. `ddtree`: the 60-node fixed-budget official DDTree control using the original
   target-sampling tree walk;
4. `ddtree_ppo`: a best-first DDTree whose nested node budget is selected by
   clipped PPO, verified by the original target-sampling tree walk.

The policy has 15 actions: `30,40,50,60,70,80,90,100,112,128,144,160,192,224,256`
nodes.  Dense actions around the previous 60/100-node operating region let PPO learn
smaller latency-sensitive changes, while the 144-256 actions keep larger trees
available when their acceptance gain justifies the cost.  Context features capture
the candidate budget, proposal probability mass, marginal mass, top-prefix confidence,
and current KV length.

The immediate reward explicitly penalizes tree generation:

`committed_tokens / (draft_ms + verify_ms + lambda * tree_build_ms)`

`tree_build_ms` includes best-first tree construction and target-mask compilation;
`lambda` defaults to `2.0` (`PPO_TREE_BUILD_COST_WEIGHT`) so PPO does not optimize
acceptance length while ignoring the overhead observed in the prior LinUCB run.  The
policy selects only a truncation point from the same nested best-first tree; it never
changes target sampling, the ancestor mask, or cache compaction.

GBV remains vendored only for historical reproducibility.  It is not imported,
warmed up, timed, or summarized by the default T=1 runner.

Sampled outputs are not expected to be pairwise token-identical. Compare throughput,
time per token, committed tokens per verification, and statistical output quality or
distributional tests rather than greedy exact match.
