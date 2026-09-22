# SpecGrove scheduler and open-loop supplement

Medians are across the completed prespecified seed-level replicates; raw seed values remain in the JSON audit record.

## Matched scheduler baseline

| R | C | SpecGrove tok/s | TETRIS-style tok/s | ECHO-style tok/s | vs TETRIS | vs ECHO | DP plan p95 ms |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 193 | 8 | 394.9 | 377.9 | 373.7 | +4.5% | +5.7% | 0.080 |

## Real-time Poisson serving

| Offered req/s | Method | tok/s | completed req/s | queue p95 ms | TTFT p95 ms | E2E p95 ms | Jain |
|---:|:---|---:|---:|---:|---:|---:|---:|
| 4 | dp | 127.0 | 4.15 | 46.5 | 102.1 | 460.8 | 0.899 |
| 4 | tetris_style | 126.8 | 4.15 | 46.9 | 95.7 | 514.3 | 0.870 |
| 4 | echo_style | 126.9 | 4.15 | 54.8 | 97.7 | 496.8 | 0.889 |
