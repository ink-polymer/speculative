#!/usr/bin/env python3
"""Summarize target, fixed-DDTree, and frozen topology-bandit JSONL output."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "prompts": len(rows),
        "temperature": rows[0]["temperature"],
        "action_histogram": dict(Counter(row["action"] for row in rows)),
    }
    for method in ("fixed_ddtree", "policy"):
        ratios = []
        method_tokens = method_ms = baseline_tokens = baseline_ms = 0.0
        committed = []
        exact = 0
        per_dataset: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            baseline, measured = row["baseline"], row[method]
            ratio = measured["tokens_per_second"] / baseline["tokens_per_second"]
            ratios.append(ratio)
            per_dataset[str(row.get("dataset"))].append(ratio)
            method_tokens += measured["generated_tokens"]
            method_ms += measured["decode_ms"]
            baseline_tokens += baseline["generated_tokens"]
            baseline_ms += baseline["decode_ms"]
            committed.append(measured["mean_committed"])
            exact += (
                measured["generated_token_ids"] == baseline["generated_token_ids"]
            )
        result[method] = {
            "mean_speedup": statistics.mean(ratios),
            "median_speedup": statistics.median(ratios),
            "aggregate_speedup": (
                (method_tokens / method_ms) / (baseline_tokens / baseline_ms)
            ),
            "mean_committed": statistics.mean(committed),
            "exact_matches": exact,
            "per_dataset_mean_speedup": {
                dataset: statistics.mean(values)
                for dataset, values in sorted(per_dataset.items())
            },
        }
    result["gain_over_fixed"] = {
        "mean_speedup": (
            result["policy"]["mean_speedup"]
            - result["fixed_ddtree"]["mean_speedup"]
        ),
        "aggregate_speedup": (
            result["policy"]["aggregate_speedup"]
            - result["fixed_ddtree"]["aggregate_speedup"]
        ),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    rows = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError("input JSONL is empty")
    summary = summarize(rows)
    rendered = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    print(rendered, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
