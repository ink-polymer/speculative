# ICLR 2027 AdaptiveTree--TBV manuscript

This directory contains an anonymous ICLR 2027 submission draft based on the
repository's frozen architecture and experiment contract.

- `main.tex`: complete paper and mathematical appendix.
- `results_placeholders.tex`: central blank-result macros.
- `result_tables.tex`: main-paper result tables, intentionally blank.
- `result_tables_appendix.tex`: detailed blank tables for datasets, ablations,
  quality, memory, and verifier replay.
- `figures/make_figures.py`: deterministic vector-figure generator.
- `iclr2027_conference.*`, `natbib.sty`, `fancyhdr.sty`: official ICLR 2027
  template assets.

Build with:

```bash
make
```

The probability proof applies to any finite fixed temperature `T > 0`.  The
registered stochastic experiment uses `T = 1`; the deterministic AdaptiveTree
experiment uses a separate `T = 0` greedy-equivalence argument.  Do not pool the
two protocols or populate result cells from pilots and incomplete submatrices.
