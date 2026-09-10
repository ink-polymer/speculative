# H20 reduced-row, latency-shape, and tail-aware branch pilots

These are development-only Qwen3-4B experiments at `T_target=T_draft=1`,
using the official BF16 SDPA configuration, TF32 disabled, and FP64 sampling
probabilities. CUDA Graph was not used. Synthetic prompts were used only for
candidate selection; none of these results belongs in the formal matrix.

| Candidate | Architectural change | vs DDTree | 95% CI | vs DFlash | Decision |
|---|---|---:|---:|---:|---|
| Deferred leaf | verify only ancestor-closed internal nodes | 0.9166x | [0.8025, 0.9928] | 1.5115x | reject |
| Aligned 32 | internal nodes plus high-mass leaves to an aligned row target | 1.0075x | [0.8952, 1.1277] | 1.6595x | exploratory only; reject |
| Aligned 40 | larger aligned leaf prefetch | 0.8485x | [0.6754, 1.0040] | 1.3976x | reject |
| Best static shape, L12/B45 | joint length/budget search over 18 shapes | 0.9281x | [0.8297, 0.9931] | 1.5489x | reject |
| Tail-aware Markov B45 | parent-conditioned head trained with top-R plus exact tail KL | 0.8997x | [0.8170, 0.9784] | 1.4933x | reject |

The deferred-leaf candidate reduced mean Target rows per round from 46 to
26.94, but increased mean decoding rounds from 10.33 to 11.67. On the profiled
prompt, its Target verification stage took 578.39 CUDA ms versus 442.34 ms for
DDTree: the smaller irregular masked-SDPA shape was slower. Aligned-32 recovered
the DDTree point estimate, using 33.52 Target rows per round on average, but its
confidence interval crossed one and it therefore failed the predeclared gate.

The static architecture scan varied draft length in `{6,8,10,12,15}` and tree
budget from 24 through 64. No shape beat the fixed official L15/B45 DDTree.
Budgets above 45 were consistently slower, so a larger fixed tree is not the
missing architectural gain on this H20 configuration.

The tail-aware branch objective fixes a concrete flaw in the earlier local
top-R KL: it retains the Target/Draft probability of exiting top-R as one exact
tail event rather than renormalizing it away. It improved holdout coarse KL from
1.2758 to 1.0599 without adding a Target or Draft forward at runtime. However,
mean committed tokens fell from 7.00 to 6.56 and mean rounds rose from 10.33 to
11.33, so the end-to-end candidate was rejected.

The evidence implies that a large gain cannot honestly be claimed by changing
only the verifier, deleting leaf rows, or statically retuning the existing
position-wise DFlash tree. The next high-upside research step requires a true
full-prefix-conditioned drafter trained against tree coverage or expected
committed length, with all of its runtime cost included. That is a new model
training project, not an optimization result already demonstrated here.
