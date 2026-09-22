# SpecGrove scheduler and open-loop supplement

Medians are across the completed prespecified seed-level replicates; raw seed values remain in the JSON audit record.

## Matched scheduler baseline

| R | C | SpecGrove tok/s | TETRIS-style tok/s | ECHO-style tok/s | vs TETRIS | vs ECHO | DP plan p95 ms |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 193 | 8 | 650.7 | 595.7 | 595.9 | +9.2% | +9.2% | 0.074 |
| 193 | 16 | 772.6 | 751.8 | 751.4 | +2.8% | +2.8% | 0.070 |
| 384 | 16 | 764.8 | 717.6 | 719.7 | +6.6% | +6.3% | 0.139 |
| 384 | 32 | 826.1 | 809.8 | 810.3 | +2.0% | +1.9% | 0.097 |

## Real-time Poisson serving

| Offered req/s | Method | tok/s | completed req/s | queue p95 ms | TTFT p95 ms | E2E p95 ms | Jain |
|---:|:---|---:|---:|---:|---:|---:|---:|
| 1 | dp | 232.0 | 1.17 | 40.0 | 67.9 | 2013.8 | 0.941 |
| 1 | tetris_style | 231.4 | 1.17 | 42.8 | 71.5 | 2115.0 | 0.933 |
| 1 | echo_style | 229.4 | 1.17 | 41.0 | 71.0 | 2152.1 | 0.932 |
| 2 | dp | 365.4 | 1.84 | 41.0 | 68.8 | 2304.4 | 0.932 |
| 2 | tetris_style | 358.9 | 1.81 | 70.0 | 105.0 | 3517.4 | 0.886 |
| 2 | echo_style | 358.9 | 1.81 | 67.0 | 95.7 | 3505.9 | 0.880 |
| 4 | dp | 689.7 | 3.49 | 102.1 | 142.8 | 5116.6 | 0.885 |
| 4 | tetris_style | 674.3 | 3.42 | 121.6 | 173.6 | 5489.7 | 0.929 |
| 4 | echo_style | 668.8 | 3.40 | 109.8 | 168.7 | 5951.3 | 0.940 |
