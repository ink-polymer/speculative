# SpecGrove scheduler and open-loop supplement

Medians are across the completed prespecified seed-level replicates; raw seed values remain in the JSON audit record.


## Real-time Poisson serving

| Offered req/s | Method | tok/s | completed req/s | queue p95 ms | TTFT p95 ms | E2E p95 ms | Jain |
|---:|:---|---:|---:|---:|---:|---:|---:|
| 1 | dp | 192.5 | 0.98 | 37.3 | 67.4 | 2048.8 | 0.940 |
| 1 | tetris_style | 193.2 | 0.98 | 39.6 | 70.2 | 2221.7 | 0.931 |
| 1 | echo_style | 192.1 | 0.98 | 40.6 | 70.0 | 2197.8 | 0.926 |
| 2 | dp | 370.4 | 1.87 | 52.3 | 84.0 | 2632.6 | 0.917 |
| 2 | tetris_style | 366.7 | 1.87 | 75.0 | 111.1 | 3795.1 | 0.860 |
| 2 | echo_style | 365.2 | 1.87 | 70.1 | 110.0 | 3979.1 | 0.869 |
| 4 | dp | 682.3 | 3.45 | 4708.3 | 4777.8 | 16701.7 | 0.660 |
| 4 | tetris_style | 671.1 | 3.42 | 5706.5 | 5738.0 | 18746.2 | 0.649 |
| 4 | echo_style | 669.9 | 3.41 | 5234.2 | 5315.6 | 18029.3 | 0.713 |
