# Tree-block optimization development artifacts

These files are synthetic-prompt development measurements, not formal or
held-out results. All primary comparisons use the official synchronized timing
scope, BF16 SDPA model execution, FP64 sampling probabilities, T=1, L=15, at
most B=45 non-root tree nodes, one Draft forward, and one Target forward per
round. `core-spur-stage-profile-20260910-c.json` is diagnostic-only CUDA-event
instrumentation and is not a primary throughput result.

The experiment decisions, timing contract, exactness checks, and summarized
results are documented in
[`docs/TREE_BLOCK_OPTIMIZATION_AUDIT_20260910.md`](../../../../docs/TREE_BLOCK_OPTIMIZATION_AUDIT_20260910.md).
