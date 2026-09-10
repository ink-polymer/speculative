# H20 tree/block architecture and model adaptation pilots

These are development-only T=1 pilots on Qwen3-4B with the official BF16,
SDPA, and TF32-disabled precision controls.  No CUDA Graph is used.  Every
candidate preserves exact Target sampling and is compared against B=45
DDTree and DFlash with the same length and output-token commitment.

The synthetic development prompts and the disjoint calibration corpus are
selection data.  Nothing in this directory is a complete formal result.

| Candidate | Change | DDTree speedup | DFlash speedup | Decision |
|---|---|---:|---:|---|
| token-NLL depth calibration | 15 learned proposal temperatures | 0.9615 | 1.5950 | reject |
| Target-KL depth calibration | 15 Target-distilled temperatures | 0.9342 | 1.5699 | reject |
| Target-KL vocabulary calibration | temperatures + shared vocabulary bias | 0.8575 | 1.4126 | reject |
| confidence budget B32/c0.10 | 32 nodes for high-confidence blocks | 1.0008 | 1.6523 | exploratory only; CI crosses 1 |
| confidence budget B36/c0.20 | cost-aware dynamic tree | 0.7890 | 1.3135 | reject |
| confidence budget B40/c0.10 | cost-aware dynamic tree | 0.9000 | 1.4983 | reject |
| parallel rank calibrator | one-shot hidden-state top-R reranker | 0.8709 | 1.4503 | reject |
| block-aligned multi-spine K4 | complete high-mass paths packed into B45 | 0.8405 | 1.3956 | reject |
| causal slot mixer | position-wise branch preparation after one Draft | 0.8250 | 1.3622 | reject |
| Markov branch head, initial | parent-token-conditioned B45 tree | 0.9211 | 1.5307 | reject |
| Markov branch head, vectorized | one-shot depth/parent/child score tensor | 0.9509 | 1.5691 | reject |
| Markov branch head, BF16 final | depth-2 start, B45, official model precision | 0.9945 | 1.6263 | reject; CI crosses 1 |

The token-NLL depth calibrator improved held-out NLL from 5.4225 to 5.1078.
Target-KL temperatures improved their held-out objective from 5.2041 to
4.9369, and the vocabulary-bias variant improved it to 4.8149.  The parallel
rank model improved held-out remaining-prefix KL from 2.0607 to 1.9567.  None
of these surrogate improvements passed the end-to-end DDTree timing gate.

The causal slot mixer improved held-out full-vocabulary Target KL from 3.9446
to 3.8040, but reduced mean committed tokens from 7.00 to 6.17.  The Markov
branch head improved held-out top-support KL from 1.7627 to 1.4219.  Its first
implementation spent 104.2 ms in tree construction versus 29.9 ms for DDTree.
Vectorizing all depth/parent/child scores and reusing the existing proposal
probabilities reduced this to 37.6 ms; Target verification time stayed equal
within profiling noise.  Freezing the unconditioned root and first speculative
depth recovered the same number of decoding rounds as DDTree.

The final BF16 auxiliary-head pilot reached 0.9945x DDTree with a 95% interval
of [0.9818, 1.0165], while remaining 1.6263x faster than DFlash.  Budget and
runtime-support ablations did not cross the DDTree gate.  This is a negative
development result, not evidence that the method exceeds DDTree.  The full
row records and diagnostic profiles are under `architecture_runs/`; trained
artifacts and hashes are under `checkpoints/tree_architecture_20260910/`.

The B32/c0.10 point reduced mean verified nodes from 45 to 32, but mean rounds
rose from 10.33 to 10.67.  Its 1.0008 point estimate had a 95% interval of
[0.8792, 1.1854], and neighboring B36/B40 points failed.  It therefore cannot
be reported as a speedup or promoted to the formal matrix.

The retained formal candidate remains `ddtree_fused_scan`; all mechanisms
added here are opt-in development controls and do not change the formal
configuration.
