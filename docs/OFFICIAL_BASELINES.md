# DFlash and DDTree official baselines

## What is included

- `third_party/dflash_official`: the official `z-lab/dflash` implementation.
- `third_party/ddtree_official`: the repository linked by the DDTree project page.
- `scripts/benchmark_official_dflash.py`: official DFlash generation on this project's JSONL prompts.
- `scripts/benchmark_official_ddtree.py`: official baseline, DFlash, and DDTree generation on the same prompts.

The upstream source directories are not patched. Their exact commits are recorded in
`third_party/OFFICIAL_SOURCES.md`.

## Dataset alignment

The default comparison file is:

```text
datasets/processed/specblock_official/prompts_benchmark_tree15.jsonl
```

It contains the fixed 200-prompt sample already used by this project. Every runner reads the
same JSONL rows in the same order and uses the `prompt` field verbatim. The source suite is
MT-Bench, HumanEval, MATH-500, Alpaca, NQ-Open, and translation. This is intentionally different
from DDTree's native paper suite; the native loaders remain available inside its official checkout.

The current MT-Bench records have already been flattened into one prompt containing both user
turns. This matches the existing DFlash-SpecBlock benchmark's input convention and therefore keeps
the comparison exact, although it is not DDTree's native conversational multi-turn convention.

## CUDA setup

Use a Linux host with an NVIDIA GPU supporting BF16, a compatible CUDA toolkit/driver, and
Python 3.11. The two implementations use isolated environments because their official dependency
versions differ from this project's environment.

```bash
bash scripts/setup_official_references.sh
```

DDTree follows the official dependency file and requires FlashAttention. If FlashAttention cannot
be built for the installed PyTorch/CUDA combination, setup must stop; do not silently replace its
draft attention backend in a performance comparison.

## Run the same 200 prompts

```bash
bash scripts/run_official_comparison.sh
```

For a smoke run:

```bash
MAX_SAMPLES=2 MAX_NEW_TOKENS=32 bash scripts/run_official_comparison.sh
```

Outputs are written to `outputs/official_references/`. For greedy decoding, every record includes
an exact token comparison with the target-only baseline. A mismatch must be investigated before
using the associated timing in a paper table.

Both official implementations use BF16. DDTree uses SDPA for target tree verification and
FlashAttention-2 for the DFlash draft, following its official benchmark.
