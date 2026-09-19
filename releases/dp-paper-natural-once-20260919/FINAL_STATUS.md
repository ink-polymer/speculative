# Final dual-H20 status — 2026-09-19 10:29 Beijing

Both queues finished successfully: 4B 2026-09-19T05:46:08+08:00, 8B 2026-09-19T05:50:11+08:00. No FAILED_PHASES records. Each model completed 10,788 groups / 53,280 method executions; long_output was intentionally excluded.

## Key findings

- 4B main DP/AR: 1.971–4.734x across 16 cells; versus adapted DFlash 12/12 wins (1.130–1.286x); versus adapted DDTree 2/8 wins (0.952–1.006x).
- 4B native C=1: DP/native DFlash 1.095–1.182x; DP/native DDTree 0.914–0.929x (0/4 wins).
- 4B ablation: DP was the raw throughput leader in 0/8 cells; best alternatives were 0.29–4.96% faster.
- 8B main DP/AR: 1.737–4.643x across 16 cells; versus adapted DFlash 12/12 wins (1.150–1.321x); versus adapted DDTree 5/8 wins (0.994–1.144x).
- 8B native C=1: DP/native DFlash 1.078–1.210x; DP/native DDTree 0.900–0.923x (0/4 wins).
- 8B ablation: DP was the raw throughput leader in 0/8 cells; best alternatives were 0.29–2.05% faster.

- Heterogeneous B=193, C=4: DP versus DDTree = -3.78% (4B), +7.07% (8B); DP versus DFlash = +24.84%, +25.74%.
- Heterogeneous B=193, C=8: DP versus DFlash = +20.68% (4B), +17.37% (8B).
- Heterogeneous B=384: DP versus DFlash is +14.51% to +25.66% for C=4–16; versus DDTree, 4B is -4.05% at C=4 and +9.24% at C=8, while 8B is +6.42% and +29.52%.
- Quality noninferiority versus AR is supported in 5/8 model-dataset cells. 4B MBPP DP=65.00%, AR=67.03%, difference -2.03 pp, paired 95% interval [-3.75,-0.31] pp.
- Strict sequence-law/BF16 losslessness remains uncertified.

Machine-readable evidence: FINAL_STATUS_20260919_1029.json; raw aggregate tables and QUALITY.json are preserved in the two model subdirectories.
