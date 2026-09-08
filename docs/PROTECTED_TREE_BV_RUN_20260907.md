# Protected tree BV: 2026-09-07 development validation

This is a development run, not a completed formal benchmark or novelty claim.

**Final decision:** pilot 2089362 completed with exit code 0. The shared-suffix
protected candidate did not beat DDTree or DFlash on the development prompts.
It is retained as a failed candidate/control, not accepted as the main research
architecture. The user explicitly rejected continued kernel/compiler work;
the subsequent local fusion probe was **not submitted or run on CUDA**.

## Verified before submission

- Local interpreter: `.artifacts/gbv-test-venv/bin/python` (Python 3.13,
  torch 2.7.1, transformers 4.57.1, huggingface-hub 0.36.0).
- 683 tests passed, zero failures/errors/skips. Evidence:
  `.artifacts/protected-integration-tests.xml`.
- Default system Python has an incompatible huggingface-hub version; it was
  not repaired or used to certify the implementation.
- The two recorded toy RM counterexamples now yield expected committed
  lengths 3.24 and 3.24934, at least their early-root BV controls.
- Eleven-method formal orchestration, complete-group resume, report decisions
  and tamper detection are covered by tests. These use synthetic timings and
  do not establish GPU speedup.

## Remote snapshot and job

- SSH alias: `arh`; existing authenticated multiplexed connection reused.
- Base: `/nobackup/proj/disk/naiss2026-3-658/personal/jiage91/fuyile`.
- Isolated source: `src/protected-pilot-20260907-Lg6QyuH0`.
- Uploaded `source.tar.gz` SHA256:
  `a32f95d2dce02aa15a9fa672d3ebf4df8b3f961e34be396aec23de228c1cb2e3`.
- Slurm job **2089362**, `fuyile-protected-pilot`, started
  2026-09-07 09:42:29 CEST (15:42:29 Beijing), on **n417**.
- Allocation: one NVIDIA GH200 120GB, eight CPUs, 100GB host memory;
  wall-time cap one hour. This is not an H200 measurement.
- Reuses existing `envs/gbv`: torch 2.9.1+cu130 and transformers 4.57.1.
  No dependency changes, model downloads, or checkpoint changes requested.
- Target Qwen3-4B revision `1cfa9a7208912126459214e8b04321603b3df60c`;
  DFlash draft revision `b74e3a329c4d963783143b1e970d95b002be72bd`.
- Output: `runs/protected-pilot-2089362`; logs:
  `logs/protected-pilot-2089362.out` and `.err`.
- Existing jobs 2072993 (AdaptiveTree, n141) and 2057806 (GBV, n464)
  were left running and their code/data/results were not modified.

## First remote checks

- Remote test XML: 683 collected, **668 passed, 15 skipped**, zero failures or
  errors (117.177 seconds). All skips are SciPy-dependent legacy LP-oracle
  tests in `test_tree_coupling_oracle.py`; SciPy is not installed in the reused
  environment. They were executed locally. Do not describe the remote run as
  683 tests passed or as a no-skip formal unit gate.
- The protected verifier's exact-law and shared-engine tests executed remotely.
- CPU/CUDA FP64 score and residual probe passed; maximum absolute error
  **6.938893903907228e-16**. This is a numerical parity check, not a proof of the
  CUDA random sampler's full output distribution.
- Runtime: Python 3.11.16, aarch64, torch 2.9.1+cu130, CUDA 13.0,
  transformers 4.57.1, datasets 3.6.0, NumPy 2.2.6, driver 580.159.04.
- CUDA reports 102,005,473,280 available device-memory bytes on the allocated
  device named `NVIDIA GH200 120GB`; retain both the actual value and name.

## Pilot scope and decision rule

Use three hand-written diagnostic prompts, 96 output-token cap, eleven methods,
three paired timing repeats. No official evaluation prompts are used for
development. Capture actual target probability rows and proposal Q outside
timing; replay comparable verifiers on identical states. Save stage profiles,
CPU/CUDA FP64 flow comparisons, raw generated tokens, greedy comparisons,
checkpoint/logit/KV audit and within-method repeatability.

Inspect `pilot_report.json` only after completion. Failure or speed regression
is evidence to retain, not a reason to relabel or suppress a run. No multi-day
formal experiment is launched automatically. Formal use still needs the
registered device, strict correctness gates, existing seven frozen datasets,
three seeds, three timing repeats, objective quality scoring and paired CIs.
The complete registered matrix contains 77,814 response records / 85,734
generation turns; the pilot is not that matrix.

## Final development measurements

All 99 short-generation records completed (three diagnostic prompts, eleven
methods, three repeats); each method has nine records. These are pooled decode
time/token, not official mean-response TPOT, and not formal dataset results.

| Method | Decode ms/token |
| --- | ---: |
| Target | 42.5740 |
| DDTree | 9.2433 |
| DFlash | 10.3062 |
| Protected shared suffix | 14.7737 |
| Old RM | 13.8680 |
| Early-root BV | 12.9664 |
| Shared-suffix tree + ancestral verifier | 10.1632 |

Within-method repeated tokens matched. The bounded-numerical checkpoint gate
passed, but **strict greedy equality did not**. Preserve this distinction; the
former does not authorize a strict-lossless claim. The complete primary report
and detailed evidence are copied into `.artifacts/protected-pilot-2089362/`.

Same-state verifier replay median: protected 5.1915ms, old RM 2.0015ms,
early-root BV 1.5979ms, shared-tree ancestral 0.2819ms. These are implementation
costs only, not acceptance or end-to-end results. The first profiled prompt
also has lower average draft acceptance for the protected run than DDTree;
because the methods visit different histories, this is a diagnostic signal,
not a controlled causal attribution or universal ordering.

Slurm owns the running process independently of SSH. This protects execution
against client disconnection, not against scheduler time limits, node failures,
or invalid code. Completed pilot artifacts are preserved if a later stage fails.
