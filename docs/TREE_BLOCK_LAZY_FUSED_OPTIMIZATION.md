# T=1 lazy-fused tree block optimization

## Status and claim boundary

This is a development candidate.  It is not the registered formal method and
must not replace results produced by commit `e36da7d1`.  It applies only to the
T=1 DDTree probability-tree protocol; no T=0 experiment is part of this work.

The frozen `ddtree_fused_scan` candidate changes only posterior sampling.  It
still constructs all 46 FP64 vocabulary distributions for a B=45 tree before a
single-block CUDA walk reads the few rows on the realized root-to-exit path.
The captured H20 preflight tree had 25 internal nodes and 21 leaves, so close to
half of those FP64 leaf rows could not affect which leaf was reached.

## Development candidates

`ddtree_lazy_softmax_fused_scan` keeps the Target transformer and complete BF16
LM-head batch identical to DDTree.  It normalizes all internal logits in one
FP64 batch, traverses those compact rows in one persistent CUDA kernel, and
normalizes only the reached leaf when a leaf is reached.  For `I` internal
nodes, it constructs `I` or `I+1` FP64 rows instead of `B+1`.

`ddtree_lazy_projection_fused_scan` additionally defers the BF16 LM head for
all leaf nodes.  It projects and normalizes `I` internal rows, then at most one
reached leaf.  This has the larger work reduction, but changing the GEMM batch
shape can change BF16 rounding.  It therefore needs an explicit numerical
audit against the complete-batch logits before any speed result can be used.

Both variants use the same DDTree proposal distribution, `probability_tree`,
L=15, B=45, BF16 weights, SDPA/SDPA backends, TF32 disabled, and FP64 posterior
probabilities.  They implement the same ancestral categorical output law.  A
same seed need not produce the same output as batched multinomial because the
random stream is mapped to categorical draws differently.

## Required gates

Run the development benchmark only on an otherwise idle GPU:

```bash
python scripts/benchmark_lazy_softmax_fused_scan.py \
  --config configs/adaptive_block_qwen3_4b.json \
  --output /fresh/path/qwen3-4b-lazy-fused-dev
```

The script uses synthetic development prompts, balanced five-method order, and
within-method timing repeats.  It compares official DFlash, DDTree, the frozen
fused scan, and both lazy-fused candidates.  An eligible candidate must satisfy:

- candidate/DDTree 95% CI lower bound greater than 1.03;
- candidate/frozen-fused-scan lower bound greater than 1.01;
- candidate/DFlash lower bound greater than 1;
- DDTree/DFlash lower bound greater than 1; and
- measured posterior rows strictly below complete-tree rows.

If both candidates pass, the script preselects the one with the larger
candidate/DDTree lower confidence bound.  That implementation must then be
frozen before a fresh, disjoint confirmation run.  Development prompts and the
currently running formal dataset rows cannot be relabelled as confirmation
data.  A CUDA compile/kernel test, inverse-CDF path test, finite-law test,
official precision audit, numerical audit, and full regression suite are all
mandatory before registration changes.

If neither candidate passes, the result is retained as a negative development
result.  The next optimization should target the common Target tree forward or
tree/cache layout; changing precision, weakening DDTree or DFlash, selecting
only favorable datasets, or modifying the registered run after seeing results
is not allowed.
