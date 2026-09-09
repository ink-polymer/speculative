#!/usr/bin/env python3
"""Audit the completed registered Qwen3-4B/T=1/H20 experiment submatrix."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from gbv_experiments.t1_submatrix_audit import audit_t1_submatrix


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", "--run-dir", dest="run_dir", type=Path, required=True)
    parser.add_argument("--data", "--data-dir", dest="data_dir", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument(
        "--audit-evidence", "--audit-evidence-dir",
        dest="audit_evidence_dir", type=Path, required=True,
        help=(
            "Directory containing plan.json, formal_matrix_audit.json, "
            "fairness_audit_qwen3_4b.json, distribution_law_audit.json, "
            "sampling_data_audit.json, sampling_gold_audit.json, and "
            "process_evaluator_self_test.json, and "
            "gpu_allocation_before_timing.json"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.epilog = (
        "The preflight is the T=1 environment identity source. See "
        "docs/QWEN3_4B_T1_H20_RUNBOOK.md for the exact independent evidence commands."
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = audit_t1_submatrix(
        args.run_dir,
        args.data_dir,
        args.preflight,
        args.audit_evidence_dir,
        args.output,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
