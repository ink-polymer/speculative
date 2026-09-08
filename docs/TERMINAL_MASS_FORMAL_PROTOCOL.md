# TM-TBV formal protocol, version 1

This protocol is an executable experiment design, not a GPU result. The primary
goal is a measured decode-throughput improvement over DDTree. A completed study
may fail that goal. V24's eight-prompt screen is development evidence only.

## Scope and registered comparison

The existing Qwen3-4B / Qwen3-4B-DFlash-b16 checkpoint commits are reused from
`configs/tree_bv_qwen3_4b.json`. This version makes a single-model claim.
No training or architecture selection on formal results is allowed. A method
change after registration requires a new study directory and a declared new
experiment; earlier outcomes remain in the record. Repeated development on
the existing datasets limits any claim that they are an untouched holdout.

All six methods share the same loaded Target/Draft objects, BF16 weights,
FP64 sampling probabilities, SDPA, disabled TF32/thinking, batch size one,
L=15, B=45, and Target/Draft temperature 1.0. Their names are frozen:

| Method | Purpose |
| --- | --- |
| target_t1 | Target AR throughput and quality reference |
| dflash_match | DFlash block draft with greedy draft tokens and Target matching |
| ddtree | Probability tree, all-node batched posterior sampling and one host transfer |
| tm_full | Terminal mass, batched ancestor products, internal-node exit rows |
| tm_serial_prefix | Same terminal law, serial prefix products; isolates batching |
| tm_dense_exit | Same terminal law, all-node exit rows; isolates internal-row selection |

The DDTree-vs-TM comparison holds tree construction fixed. DFlash has its usual
single-chain support and is a separate method comparison. The official
DDTree posterior execution is reproduced inside the shared engine. This study
does not reproduce every official benchmark detail: data formatting, timing
boundaries and numerics follow the previously used controlled GBV protocol.
Never label its results as a byte-for-byte reproduction of the original paper.
The local vendored source and its model implementation are hashed.

The variant with subtraction `1 - child_mass` is not included as a valid
method: it can erase positive tails. Its counterexample is a correctness test.
Both registered ablations preserve the same mathematical output law. They do
not justify claiming a new proposal–Target BV rejection kernel. Current TM-TBV
is a block decision formed by marginalizing the first-exit distribution.

## Data and run matrix

Use the **existing** prepared seven-dataset directory and its manifest. No
new revision is fetched by this workflow. Source commits, exact selected IDs,
prompt hashes, order, formatter hash and evaluation sidecars are checked and
frozen before generation. If the previous directory is unavailable, recover
it from the existing server study; do not silently replace it with current HF
data. The older 8 handcrafted V24 prompts are not this formal dataset.

| Dataset | Questions/conversations |
| --- | ---: |
| GSM8K | 128 |
| MATH-500 | 128 |
| AIME25 | 30 |
| HumanEval | 164 |
| MBPP full/test | 128 |
| LiveCodeBench release_v6 | 128 |
| MT-Bench | 80, two turns each |

Total: 786 source records / 866 generation turns per method, seed and repeat.
Use seeds 17, 29 and 43, three timing repeats, and up to 2048 new tokens per
turn with the checkpoint's EOS policy. Six methods produce 42,444 conversation
records / 46,764 generation calls. These are planned counts, not completed runs
or a runtime estimate. MT-Bench's next prompt contains that method's actual
first answer. Context overflow fails; no prompt truncation or record filtering.
The protocol does not assert independence from unknown pretrained corpora.

## Evidence stages

1. `plan`: show counts, variants and the decision rule; works on CPU.
2. `freeze`: validate existing prepared data, run data audit, then fix code,
   dataset/model revisions, scripts, tests and the protocol in registration.json.
3. `test`: run the declared CPU tests with JUnit evidence. No skipped/failed
   tests are accepted. Tests cover exact finite-event laws, ablations, official
   DDTree posterior traversal, cache behavior, data selection and resume logic.
4. `diagnostics`: on the registered GPU family, check frozen runtime settings,
   actual checkpoint greedy/cache/tree correctness; capture baseline tree
   states; replay DDTree/TM/ablations on the same tensors; profile each method
   separately on diagnostic prompts. All state files receive SHA-256 hashes.
   The existing preflight's BF16 near-tie diagnostics are preserved, but this
   formal gate requires exact greedy output equality and does not automatically
   approve exceptions. Equality is a finite diagnostic, not a universal proof.
5. `run --repeat 0`, then repeats 1 and 2: independent launches, same
   environment identity, warm up all methods every time. Group order is seeded;
   within a question/seed group a fixed permutation rotates by repeat. All
   methods run in one process on the same allocated GPU for each complete group.
6. `score`: run objective quality scoring on repeat 0 only. Docker isolates
   generated code. The fixed math/code references must pass evaluator checks.
   A scoring-capable host can consume copied artifacts; lack of Docker on the
   GPU cluster does not authorize a fallback to unsandboxed generated code.
7. `report`: verify all registered records, groups, timing repetitions, scores,
   captures and evidence gates before publishing the decision and artifact index.

Primary timers have profiling/capture **disabled**. Engine decode time starts
after Target prefill and initial token sampling, includes the first Draft
forward, tree building, Target verification, sampling and cache commit, and
ends after CUDA synchronization. Its numerator is generated tokens minus one
per turn. Also report model-call end-to-end time (including Target prefill),
prefill/TTFT, token lengths, EOS/cap fractions, calls, acceptance and peak memory.
Tokenization, text decoding and disk I/O are outside the model-call timers;
the term end-to-end must not be expanded to full client/service latency.

The replay measures host wall time with device synchronization and includes
the verifier's Python/topology work. It does not include the Target forward.
Every method uses the same loaded tensor state; seed equality does not imply
the same sampled path. Five warmups, fifty iterations and three repeats are
registered. Stage profiles and replay times cannot replace unprofiled results.

## Statistics and success rule

For each dataset, throughput is total generated decode tokens divided by total
decode time; speedup is candidate throughput / baseline throughput. The primary
endpoint is the geometric mean of the seven per-dataset DDTree speedups with
equal dataset weight, fixed before results. Pooled throughput is secondary.

Use 10,000 dataset-stratified paired bootstrap resamples of question IDs. All
seeds and timing repeats of a question stay in its cluster. They are not extra
independent questions. Report a two-sided 95% percentile interval. The study's
speed target is achieved only if all evidence is complete and the primary CI
lower bound exceeds 1. No best-of-repeats, failed-row exclusion, dataset removal,
or parameter retuning based on formal results. This is a question-sampling CI
conditional on the measured hardware sessions, not a hardware-variance model.

Report the DFlash and AR comparisons, both ablations, each dataset, all timing
repeats, and model-call e2e speedups regardless of their sign. Secondary CIs are
descriptive; there is no uncorrected family of significance claims. Quality is
math accuracy / code pass@1 over three seeds from repeat 0. Repeated timing
seeds do not increase quality sample size. No MT-Bench judge is commissioned;
its throughput is included, but conversational quality is explicitly unscored.
Quality scores do not prove distributional equality or noninferiority.

`formal_complete` means all registered evidence exists and passes integrity
checks. `target_achieved` is a separate result and can be false. Novelty and
publication readiness are not established by either flag.

## Recovery and server execution

The supplied Slurm script runs diagnostics or one timing repeat per job. It
is independent of the login SSH session once accepted by Slurm. Keepalive
reduces idle SSH failures; it cannot guarantee permanent network connectivity.
Walltime, node failures and scheduler cancellation still apply.

Each complete six-method question/seed group is atomically stored. No partial
group is counted. A caught interruption is retained under attempts/; an abrupt
kill can leave an uncommitted group which is rerun in full. Completed groups
are never recomputed merely to improve timings. Results export only after a
repeat has complete coverage. The directory lock prevents concurrent writers.
Re-submit the **same** repeat/output directory after walltime or interruption;
inspect Slurm job status before retrying an uncertain submission to avoid
duplicate GPU jobs. Do not use `afterok` to conceal a partial previous repeat.

On the server, after verifying paths and the actual GPU allocation:

```bash
export GBV_FORMAL_PYTHON=/path/to/compatible/python
bash scripts/run_terminal_mass_formal.sh plan
bash scripts/run_terminal_mass_formal.sh freeze --data-dir /existing/data --output /study/v1
bash scripts/run_terminal_mass_formal.sh test --data-dir /existing/data --output /study/v1
export GBV_FORMAL_DATA=/existing/data
export GBV_FORMAL_OUTPUT=/study/v1
sbatch scripts/terminal-mass-formal.sbatch diagnostics
# Only after successful diagnostics:
sbatch scripts/terminal-mass-formal.sbatch run 0
# Complete repeat 0, then submit repeats 1 and 2; resume any incomplete repeat.
bash scripts/run_terminal_mass_formal.sh status --output /study/v1
# On a host with the registered scoring dependencies and Docker image:
bash scripts/run_terminal_mass_formal.sh score --data-dir /existing/data --output /study/v1
bash scripts/run_terminal_mass_formal.sh report --data-dir /existing/data --output /study/v1
```

The batch defaults refer to the existing Arrhenius project and must be verified
after login. The study defaults to the requested **H200**; matching is by model
token, so GH200 is distinct. If the allocated machine is GH200, resolve that
hardware choice before registration. Device type, memory, driver, versions and
CPU threads are fixed across resumes; hostname, GPU UUID, job ID, utilization,
clock/power readings and attempts are retained per group. Request a dedicated
GPU allocation. An allocation gate rejects other detected compute processes
on the assigned GPU before and after each timing group; it does not terminate
those processes. Review telemetry as well; undetected interference undermines
performance interpretation. No GPU work has been performed locally.
