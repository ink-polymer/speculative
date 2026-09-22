# SpecGrove scheduler and open-loop supplement

This bundle adds the two serving experiments requested for the ICLR manuscript:

1. matched TETRIS-style and ECHO-style schedulers on the same DFlash/DDTree backend;
2. real wall-clock open-loop serving with Poisson arrivals.

The style suffix is deliberate. These controls transfer each scheduler's allocation
principle to nested DDTree tiers; they are not reproductions of the original vLLM or
SGLang systems.

## Matched controls

- **SpecGrove (`dp`)** selects one nested tree tier per admitted request with the
  measured target-cost curve and causal service weights.
- **TETRIS-style** greedily ranks ancestor-closed tier increments by added draft
  covered mass per verification row. It uses neither service weights nor the
  nonlinear measured cost curve.
- **ECHO-style** applies sparse confidence gates at nested tier boundaries, then
  spends residual capacity on opportunistic coverage expansion. Gate thresholds
  are medians calibrated from draft-only statistics on a disjoint GSM8K slice;
  no target labels are used.

All three controls use identical prompts, request seeds, DFlash proposals, nested
DDTree candidates, packed eager BF16/SDPA target execution, FP32 proposal/posterior
bookkeeping, row capacity, and timing boundaries. Method order is randomized within
each matched request group.

## Protocol

The formal Qwen3-8B run uses seeds 17, 29, and 43, an output cap of 256, and 128
balanced mixed-natural requests for each closed-batch cell:

- `(row budget, concurrency) = (193, 8), (193, 16), (384, 16), (384, 32)`.

The final open-loop study uses 256 balanced mixed-natural requests, Poisson rates
of 1, 2, and 4 requests/s, row budget 384, and at most 32 active requests. Arrivals
occur in real elapsed time. Each record includes queue delay, TTFT, TPOT,
end-to-end latency, throughput, completed request rate, Jain service-rate fairness,
allocator time, per-request metrics, outputs, and the allocator trace. Dynamic
admission uses request-local target caches and repacks them each wave for every
compared method. The earlier 64-request traces are retained as a preliminary audit
record; manuscript claims use the 256-request traces.

## Files

- `experiment/scripts/run_openloop_scheduler_supplement.py`: experiment driver.
- `experiment/scripts/paper_dp_allocators.py`: matched scheduler controls.
- `experiment/src/gbv_experiments/continuous_tree_block_decode.py`: open-loop
  admission and request-level timing support.
- `experiment/tests/test_scheduler_style_baselines.py`: deterministic allocator
  and capacity tests.
- `aggregate_results.py`: seed-level aggregation into JSON and Markdown.
- `results/qwen3_8b/`: formal scheduler records and preliminary 64-request traces.
- `results/qwen3_8b_long256/`: final 256-request Poisson traces and aggregates.

## Validation

The scheduler tests and the existing shared-budget sequence-law suite pass together:

```text
40 passed
```

The end-to-end smoke run completed the matched scheduler and Poisson serving paths
before the formal run was launched. Raw JSON files are written atomically; rerunning
the driver skips completed records.

`audit_results.py` also verifies 108/108 closed-batch group files, 27/27 open-loop
traces, matched arrivals and prompt identities, per-request timestamp ordering,
output-token counts, finite metrics, and row-cap compliance.

```bash
python3 audit_results.py results/qwen3_8b
python3 audit_results.py results/qwen3_8b_long256 \
  --open-loop-only --open-loop-requests 256
```

## Formal Qwen3-8B results

The formal run completed all 108 matched closed-batch groups and all 27 Poisson
traces. SpecGrove exceeded both style controls in every closed-batch seed/capacity
comparison. Median throughput gains were 9.2%/9.2% at `(193,8)`, 1.7%/1.9% at
`(193,16)`, 6.8%/6.6% at `(384,16)`, and 2.0%/2.1% at `(384,32)` versus
TETRIS-style/ECHO-style allocation.

The 256-request traces give a sharper serving result. At the stable 2-request/s
load, paired throughput changes are +0.5%/+0.0% versus TETRIS/ECHO-style, while
p95 TTFT falls by 24.4%/26.5% and p95 end-to-end latency falls by 30.6%/33.8%.
At 4 requests/s, all methods complete only 3.41--3.45 requests/s, so this point is
an overload stress test. SpecGrove's paired median throughput gains are 2.2%/1.7%;
median tail reductions are positive, but their seed-level ranges include negative
values. The 1-request/s traces remain arrival-limited.

The downloaded formal-result archive has SHA-256
`92e01ac3896cf2fb4dbff5b04eee7e415dafc2cd83d68fef2e00f854aa08932a`.
The downloaded 256-request archive has SHA-256
`9f500516bcdfa39232f1a1f085d4a86725148f59fe9faab4519f3f7ce1a6d13a`.

## Aggregate

```bash
python3 aggregate_results.py results/qwen3_8b \
  --output results/qwen3_8b/aggregate.json

python3 aggregate_results.py results/qwen3_8b_long256 \
  --open-loop-only --output results/qwen3_8b_long256/aggregate.json
```

The aggregator treats the random seed as the replicate. It pools closed-batch
request groups within each seed, then reports medians and paired changes across
seeds. Open-loop values are medians of the three trace-level measurements. Raw
seed-level values remain in `aggregate.json`.
