# 2026-09-09 tree-block pilot archive

This directory preserves engineering and qualification runs. Every pilot is
`formal_complete=false`; none of these timings may be imported into, pooled
with, or presented as the future 4B/8B full-matrix result.

## Evidence status

- `h20_same_tree_fast_verifier/h20-same-tree-block-heldout-20260909-r1` is a
  failed qualification gate: the candidate/DDTree confidence interval crosses
  1. Its exact fused-verifier source, runner script, and prompt bodies are not
  archived, and its DFlash metadata is obsolete. It is retained only as
  incomplete negative evidence.
- `h20_same_tree_fast_verifier/h20-same-tree-block-official-pilot-20260909-r1`
  passed its speed gate, but its manifest used obsolete DFlash metadata
  (`draft_temperature=1.0`) even though the implementation used the official
  greedy-argmax Draft proposal. Its exact runner script and prepared-input
  manifest are also unavailable. It is superseded and must not be cited.
- `h20_same_tree_fast_verifier/h20-same-tree-block-official-pilot-20260910-r2`
  is the corrected qualification pilot. It contains 504 complete paired raw
  rows and passes independent recalculation. Its run-time `engine.py` is
  preserved at
  `h20_same_tree_fast_verifier/reproduction/r2_source/engine.py` with SHA-256
  `1874a873ab3940ca4d1c58a3cc0f9fb222a3c8599fc5a0efeed9305e1011ed86`.
  This differs intentionally from the later integrated engine and must be
  supplied explicitly when validating the archived pilot.
- Fused-scan unit, vocabulary, and speed-gate files are implementation
  diagnostics. They are not end-to-end model results.

Recalculate the corrected r2 artifact from its raw rows and bound sources with:

```bash
PYTHONPATH=src python scripts/validate_same_tree_pilot.py \
  results/pilots/20260909/h20_same_tree_fast_verifier/h20-same-tree-block-official-pilot-20260910-r2 \
  --engine-source results/pilots/20260909/h20_same_tree_fast_verifier/reproduction/r2_source/engine.py
```

The r2 pilot still covers only Qwen3-4B, seven datasets, eight samples per
dataset, three seeds, and a 256-token cap. Formal reporting requires a fresh
future-server run of the registered Qwen3-4B/Qwen3-8B, eight-dataset,
three-seed, 2,048-token matrix, followed by all integrity and fairness gates.
A superiority claim additionally requires the relevant paired source-cluster
95% confidence-interval lower bound to exceed 1.

The r2 raw rows, reports, and five bound source files are independently
verifiable, but the original prepared-data directory/manifest is not present
in this archive. Therefore this archive is not a self-contained end-to-end
rerun package. The future formal run must prepare and hash a new complete input
snapshot instead of importing pilot inputs or timings.
