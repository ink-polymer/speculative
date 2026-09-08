#!/usr/bin/env python3
"""Summarize temperature>0 target/DFlash/DDTree/BlockV JSONL results."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


METHODS = ("baseline", "dflash", "ddtree", "tree_block_verification")


def _load(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def _method(rows: list[dict[str, Any]], name: str) -> dict[str, Any]:
    elapsed_ms = sum(float(row[name]["decode_ms"]) for row in rows)
    tokens = sum(int(row[name]["generated_tokens"]) for row in rows)
    rounds = [int(x) for row in rows for x in row[name].get("acceptance_lengths", [])]
    baseline_tpt = [
        float(row["baseline"]["decode_ms"])
        / max(int(row["baseline"]["generated_tokens"]), 1)
        for row in rows
    ]
    method_tpt = [
        float(row[name]["decode_ms"]) / max(int(row[name]["generated_tokens"]), 1)
        for row in rows
    ]
    mean_baseline_tpt = sum(baseline_tpt) / len(baseline_tpt)
    mean_method_tpt = sum(method_tpt) / len(method_tpt)
    return {
        "prompts": len(rows),
        "generated_tokens": tokens,
        "mean_time_per_token_ms": mean_method_tpt,
        "tokens_per_second": 1000.0 / mean_method_tpt,
        "speedup_vs_target": mean_baseline_tpt / mean_method_tpt,
        "mean_committed_per_verify": sum(rounds) / len(rounds) if rounds else None,
        "decode_rounds": len(rounds),
        "aggregate_decode_ms": elapsed_ms,
    }


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {name: _method(rows, name) for name in METHODS}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = _load(args.input)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["dataset"])].append(row)
    summary = {
        "policy": {
            "temperature": rows[0]["temperature"] if rows else None,
            "prompts": len(rows),
            "datasets": {name: len(group) for name, group in sorted(groups.items())},
            "sampling_note": (
                "Distributional correctness must be tested statistically; sampled token IDs "
                "are not expected to match pairwise even with a common seed."
            ),
        },
        "overall": _summarize(rows),
        "per_dataset": {
            name: _summarize(group) for name, group in sorted(groups.items())
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
