#!/usr/bin/env python3
"""Summarize the 2K temperature=1 GBV performance benchmark."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"prompts": 0}
    baseline_tpt = [
        float(row["baseline_ms"]) / max(int(row["baseline_tokens"]), 1)
        for row in rows
    ]
    hybrid_tpt = [
        float(row["hybrid_ms"]) / max(int(row["hybrid_tokens"]), 1)
        for row in rows
    ]
    rounds = sum(int(row["verify_iterations"]) for row in rows)
    committed = sum(
        float(row["average_committed_per_verify"]) * int(row["verify_iterations"])
        for row in rows
    )
    baseline_ms = sum(float(row["baseline_ms"]) for row in rows)
    hybrid_ms = sum(float(row["hybrid_ms"]) for row in rows)
    hybrid_tokens = sum(int(row["hybrid_tokens"]) for row in rows)
    mean_baseline_tpt = sum(baseline_tpt) / len(baseline_tpt)
    mean_hybrid_tpt = sum(hybrid_tpt) / len(hybrid_tpt)
    return {
        "prompts": len(rows),
        "generated_tokens": hybrid_tokens,
        "mean_time_per_token_ms": mean_hybrid_tpt,
        "tokens_per_second": 1000.0 / mean_hybrid_tpt,
        "speedup_vs_target": mean_baseline_tpt / mean_hybrid_tpt,
        "aggregate_speedup_vs_target": baseline_ms / hybrid_ms,
        "mean_per_prompt_speedup": sum(
            float(row["wall_clock_speedup"]) for row in rows
        )
        / len(rows),
        "mean_committed_per_verify": committed / rounds if rounds else None,
        "mean_verify_iterations": rounds / len(rows),
        "mean_tree_nodes": sum(float(row["mean_tree_nodes"]) for row in rows)
        / len(rows),
        "baseline_decode_ms_total": baseline_ms,
        "hybrid_decode_ms_total": hybrid_ms,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-prompts", type=int, default=2000)
    args = parser.parse_args()

    rows = load_jsonl(args.input)
    prompts = load_jsonl(args.prompts)
    if len(rows) != args.expected_prompts:
        raise ValueError(
            f"benchmark is incomplete: {len(rows)}/{args.expected_prompts} rows"
        )
    if len(prompts) < len(rows):
        raise ValueError(f"prompt rows {len(prompts)} < benchmark rows {len(rows)}")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[str(prompts[index]["dataset"])].append(row)

    summary = {
        "policy": {
            "temperature": 1.0,
            "gbv_paths": 3,
            "max_new_tokens": 128,
            "expected_prompts": args.expected_prompts,
            "dataset": str(args.prompts),
            "quality_metrics_note": (
                "The fixed prompt JSONL has no references/tests; this summary reports "
                "generation performance and verification efficiency, not task accuracy."
            ),
        },
        "overall": summarize(rows),
        "per_dataset": {
            name: summarize(group) for name, group in sorted(grouped.items())
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
