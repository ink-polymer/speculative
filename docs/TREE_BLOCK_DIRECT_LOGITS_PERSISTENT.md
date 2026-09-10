# Direct-logits persistent tree verifier

This is a development candidate, not formal evidence.

The candidate keeps the registered DDTree proposal, Qwen model revisions,
BF16 model weights/logits, SDPA backends, disabled TF32, temperature 1, L15,
B45, seeds, and cache policy unchanged.  Its only experimental delta is the
posterior verifier.

Instead of materializing an FP64 probability matrix for every node and then
sampling it, one persistent CUDA block consumes the Target logits directly.
For each visited node it computes a stable maximum, FP64 exponential weights,
an FP64 inverse CDF, and the next tree edge in the same launch.  Only the final
packed path is synchronized back to the host.  The full Target vocabulary head
is intentionally unchanged so the comparison does not alter model execution.

The development gate uses synthetic prompts disjoint from the registered
datasets, balanced method rotation, three prompts, 64 generated tokens, and
five repeats.  It compares official DFlash, official DDTree, the earlier
probability-input fused scan, and this candidate in one process.  The candidate
may proceed only if its equal-prompt geometric-mean speedup has a 95% bootstrap
lower bound above 1.03 versus DDTree, above 1.01 versus the earlier fused scan,
and above 1 versus DFlash, while DDTree remains above DFlash.

The CUDA equivalence test compares BF16, FP32, and FP64 logits against the FP64
PyTorch softmax inverse-CDF path under identical generator states.  A failed
compile, equivalence test, or speed gate is retained as a negative result and
does not enter the formal matrix.
