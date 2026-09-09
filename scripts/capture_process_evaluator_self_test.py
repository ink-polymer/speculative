#!/usr/bin/env python3
"""Capture the registered T=1 production process-evaluator UID/identity proof."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from gbv_experiments.t1_submatrix_audit import capture_process_evaluator_self_test


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evidence = capture_process_evaluator_self_test(args.output)
    print(json.dumps(evidence, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
