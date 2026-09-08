# AdaptiveTree + diffusion tree block verification

## Scope

This branch combines two independently auditable lines of work without rewriting
their experimental identities:

- `adaptivetree_paper/` preserves the original non-RL AdaptiveTree artifact based
  on commit `184952f679f303db150a7f582c4d5bb51099be85`.
- The repository root contains the later NVIDIA/CUDA implementation, official
  DDTree and DFlash comparisons, and the positive-temperature diffusion tree
  block-verification prototypes in `src/gbv_experiments/`.

The diffusion work is not presented as a completed AdaptiveTree experiment. Its
registered configurations, tests, proofs, counterexamples, and run outputs remain
separate from the frozen T=0 AdaptiveTree protocol.

## AdaptiveTree GH200 checkpoint

The interrupted Qwen3-8B run under
`/nobackup/proj/disk/naiss2026-3-658/personal/jiage91/fuyile/runs/adaptive-record-8b-unpinned`
completed paired SDPA/FlashAttention artifacts for seven of ten datasets. The
SWE-bench SDPA worker reached case 108/128 before Slurm job `2072993` was killed
for host-memory OOM; MT-Bench and Alpaca did not run.

The following rows use the pinned upstream mean-per-response decode-TPOT metric.
Lower TPOT is better; the two rightmost columns are baseline TPOT divided by
AdaptiveTree TPOT.

| Dataset | DFlash TPOT (ms) | best DDTree TPOT (ms) | AdaptiveTree TPOT (ms) | vs DFlash | vs DDTree |
|---|---:|---:|---:|---:|---:|
| GSM8K | 8.662 | 9.021 | 6.545 | 1.324x | 1.378x |
| MATH-500 | 6.743 | 8.434 | 7.409 | 0.910x | 1.138x |
| AIME 2024 | 8.325 | 9.038 | 8.460 | 0.984x | 1.068x |
| AIME 2025 | 7.754 | 8.936 | 9.792 | 0.792x | 0.913x |
| HumanEval | 8.341 | 8.791 | 7.537 | 1.107x | 1.166x |
| MBPP | 8.883 | 9.300 | 9.260 | 0.959x | 1.004x |
| LiveCodeBench | 8.090 | 9.219 | 6.467 | 1.251x | 1.425x |

Across these seven completed datasets, AdaptiveTree beats the tuned DDTree row
on six datasets and DFlash on three. The unweighted geometric-mean factors are
1.143x versus DDTree and 1.032x versus DFlash. The `no_exploration` ablation wins
all seven stored comparisons, with geometric-mean factors of 1.488x and 1.343x;
this is evidence that the online budget controller needs correction, not evidence
that the main AdaptiveTree method is uniformly best.

The run deliberately used `record-bf16-mismatches`. Only 354 of 1,472 stored
backend responses had every method match that backend's AR tokens, and 522 of 736
paired prompts had different SDPA and FlashAttention AR tokens. Consequently the
stored speed rows are diagnostic only: `publication_gate_passed=false` and
`strict_lossless_claim_eligible=false`.

## Diffusion tree block verification

The current candidate is the scaffolded positive-temperature method documented
in `docs/DIFFUSION_SCAFFOLD_BV.md`. It retains a complete DFlash guard path,
correlated diffusion paths, high-probability fill, prefix merging, tree-internal
continuation, and the exact correction kernel. The proof and tests establish
losslessness only under their stated exact-probability assumptions. They do not
establish universal speed dominance or novelty over all prior work; the strict
DDTree counterexample and overlap audit remain part of the branch.

Local, non-GPU inspection:

```bash
python -m pytest tests/gbv_paper -q
bash scripts/run_diffusion_tree.sh plan --study configs/diffusion_scaffold_t10.json
```

GH200 pilot (new output directory; does not overwrite AdaptiveTree results):

```bash
export DIFFUSION_PILOT_SOURCE=/path/to/this/frozen/checkout
export DIFFUSION_PILOT_FAMILY=diffusion_scaffold
sbatch --export=ALL scripts/diffusion-tree-pilot.sbatch
```

The pilot runs the complete GBV test directory first, then isolated T=0.3, 0.6,
and 1.0 probes. A failed test prevents GPU probes from starting. Formal runs must
use separately frozen registrations and must not relabel GH200 measurements as
H200 evidence.
