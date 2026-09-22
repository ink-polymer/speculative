# SpecGrove ICLR 2027 final bundle

This self-contained bundle contains the revised manuscript, all figure assets,
the final PDF, the scheduler/open-loop experiment code, and complete raw records.

## Manuscript

- `SpecGrove_ICLR2027.pdf`: final rendered paper.
- `paper/paper.tex`: LaTeX source.
- `paper/references.bib`: bibliography.
- `paper/iclr2027_conference.{sty,bst}`: conference style files.
- `paper/*.pdf` and `paper/*.svg`: included figure assets and editable sources.
- `paper/citation_validation.json`: citation-key audit.

Compile from `paper/` with:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error paper.tex
```

The main paper ends on page 9; references and appendices follow. The final build
has no unresolved citations, references, or overfull boxes.

## Scheduler and open-loop experiment

`experiments/scheduler_openloop_supplement/` contains:

- the exact experiment source snapshot;
- all 108 matched closed-batch groups and 324 method executions;
- 27 final 256-request Poisson traces, plus the preliminary 64-request audit traces;
- manifests, completion markers, aggregates, and downloaded result archives;
- deterministic scheduler tests and result-integrity auditors.

Run the integrity audits from that directory with:

```bash
python3 audit_results.py results/qwen3_8b
python3 audit_results.py results/qwen3_8b_long256 \
  --open-loop-only --open-loop-requests 256
```

The audits verify expected cells/rates/seeds, matched request identities and
arrivals, timestamp ordering, token counts, finite metrics, and row-capacity
compliance. See the supplement README for the full protocol and the scope of the
TETRIS-style and ECHO-style controls.

## Headline added results

Across four fixed-concurrency scheduler cells, SpecGrove improves median
throughput by 1.7--9.2% over both matched style controls. In the 256-request
Poisson study, the stable 2-request/s load shows comparable throughput and
24.4--26.5% lower p95 TTFT plus 30.6--33.8% lower p95 end-to-end latency. At
4 requests/s, all methods are overloaded; SpecGrove improves paired median
throughput by 1.7--2.2%, while tail changes vary across arrival traces.

`experiments/scheduler_openloop_supplement/ADDITIONAL_EXPERIMENT_AUDIT.md`
records the remaining lower-priority experimental gaps. `FILE_MANIFEST.sha256`
records the final bundle contents.
