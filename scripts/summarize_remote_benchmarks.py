#!/usr/bin/env python3
"""Aggregate the strict BF16 RTX 4090 benchmark into JSON and Markdown."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def distribution(values: list[float]) -> dict[str, float | None]:
    return {
        "mean": sum(values) / len(values) if values else None,
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def official_method(rows: list[dict[str, Any]], name: str) -> dict[str, Any]:
    baseline_ms = sum(float(row["baseline"]["decode_ms"]) for row in rows)
    method_ms = sum(float(row[name]["decode_ms"]) for row in rows)
    tokens = sum(int(row[name]["generated_tokens"]) for row in rows)
    lengths = [int(n) for row in rows for n in row[name].get("acceptance_lengths", [])]
    exact_key = f"{name}_greedy_exact_match"
    stage_totals: dict[str, float] = {}
    for row in rows:
        for stage, seconds in row[name].get("stage_times_s", {}).items():
            stage_totals[stage] = stage_totals.get(stage, 0.0) + float(seconds)
    # DDTree's official make_latex_table.py first computes TPT / acceptance per
    # response and then averages responses. Keep this as the primary paper metric.
    baseline_tpt_ms = [
        float(row["baseline"]["decode_ms"]) / max(int(row["baseline"]["generated_tokens"]), 1)
        for row in rows
    ]
    method_tpt_ms = [
        float(row[name]["decode_ms"]) / max(int(row[name]["generated_tokens"]), 1)
        for row in rows
    ]
    acceptance_per_prompt = [
        sum(float(n) for n in row[name].get("acceptance_lengths", []))
        / max(len(row[name].get("acceptance_lengths", [])), 1)
        for row in rows
    ]
    mean_baseline_tpt = sum(baseline_tpt_ms) / len(baseline_tpt_ms)
    mean_method_tpt = sum(method_tpt_ms) / len(method_tpt_ms)
    mean_committed = sum(acceptance_per_prompt) / len(acceptance_per_prompt)
    weighted_committed = sum(lengths) / len(lengths) if lengths else None
    latencies = [float(row[name]["decode_ms"]) for row in rows]
    speedups = [float(row["baseline"]["decode_ms"]) / float(row[name]["decode_ms"]) for row in rows]
    exact = sum(row.get(exact_key) is True for row in rows)
    prefix_lengths: list[float] = []
    prefix_rates: list[float] = []
    for row in rows:
        baseline_ids = row["baseline"]["generated_token_ids"]
        method_ids = row[name]["generated_token_ids"]
        prefix = next(
            (i for i, (left, right) in enumerate(zip(baseline_ids, method_ids)) if left != right),
            min(len(baseline_ids), len(method_ids)),
        )
        prefix_lengths.append(float(prefix))
        prefix_rates.append(prefix / max(len(baseline_ids), 1))
    return {
        "prompts": len(rows),
        "generated_tokens": tokens,
        "decode_ms_total": method_ms,
        "baseline_decode_ms_total": baseline_ms,
        "metric_convention": "DDTree official: ratio of mean per-prompt time-per-token; mean of per-prompt acceptance means",
        "mean_time_per_token_ms": mean_method_tpt,
        "tokens_per_second": 1000.0 / mean_method_tpt,
        "speedup_vs_target": mean_baseline_tpt / mean_method_tpt,
        "per_prompt_speedup": distribution(speedups),
        "decode_latency_ms": distribution(latencies),
        "mean_committed_per_verify": mean_committed,
        "mean_accepted_draft_tokens": mean_committed - 1.0 if mean_committed is not None else None,
        "aggregate_weighted_secondary": {
            "tokens_per_second": tokens / (method_ms / 1000.0),
            "speedup_vs_target": baseline_ms / method_ms,
            "mean_committed_per_verify": weighted_committed,
        },
        "verify_rounds": len(lengths),
        "greedy_exact_matches": exact,
        "greedy_exact_match_rate": exact / len(rows),
        "matching_prefix_tokens": distribution(prefix_lengths),
        "matching_prefix_fraction": distribution(prefix_rates),
        "acceptance_length_distribution": distribution([float(n) for n in lengths]),
        "acceptance_length_histogram": {str(n): lengths.count(n) for n in sorted(set(lengths))},
        "stage_times_s": stage_totals,
    }


def own_method(rows: list[dict[str, Any]]) -> dict[str, Any]:
    baseline_ms = sum(float(row["baseline_ms"]) for row in rows)
    method_ms = sum(float(row["hybrid_ms"]) for row in rows)
    tokens = sum(int(row["hybrid_tokens"]) for row in rows)
    rounds = sum(int(row["verify_iterations"]) for row in rows)
    committed = sum(float(row["average_committed_per_verify"]) * int(row["verify_iterations"]) for row in rows)
    baseline_tpt_ms = [float(row["baseline_ms"]) / max(int(row["baseline_tokens"]), 1) for row in rows]
    method_tpt_ms = [float(row["hybrid_ms"]) / max(int(row["hybrid_tokens"]), 1) for row in rows]
    mean_baseline_tpt = sum(baseline_tpt_ms) / len(baseline_tpt_ms)
    mean_method_tpt = sum(method_tpt_ms) / len(method_tpt_ms)
    mean_committed = sum(float(row["average_committed_per_verify"]) for row in rows) / len(rows)
    weighted_committed = committed / rounds if rounds else None
    speedups = [float(row["baseline_ms"]) / float(row["hybrid_ms"]) for row in rows]
    exact = sum(row.get("greedy_exact_match") is True for row in rows)
    prefix_lengths = [
        float(row["baseline_tokens"] if row.get("first_mismatch_index") is None else row["first_mismatch_index"])
        for row in rows
    ]
    prefix_rates = [
        prefix / max(int(row["baseline_tokens"]), 1)
        for prefix, row in zip(prefix_lengths, rows)
    ]
    return {
        "prompts": len(rows),
        "generated_tokens": tokens,
        "decode_ms_total": method_ms,
        "baseline_decode_ms_total": baseline_ms,
        "metric_convention": "DDTree official-compatible: ratio of mean per-prompt time-per-token; mean of per-prompt acceptance means",
        "mean_time_per_token_ms": mean_method_tpt,
        "tokens_per_second": 1000.0 / mean_method_tpt,
        "speedup_vs_target": mean_baseline_tpt / mean_method_tpt,
        "per_prompt_speedup": distribution(speedups),
        "decode_latency_ms": distribution([float(row["hybrid_ms"]) for row in rows]),
        "mean_committed_per_verify": mean_committed,
        "mean_accepted_draft_tokens": mean_committed - 1.0 if mean_committed is not None else None,
        "aggregate_weighted_secondary": {
            "tokens_per_second": tokens / (method_ms / 1000.0),
            "speedup_vs_target": baseline_ms / method_ms,
            "mean_committed_per_verify": weighted_committed,
        },
        "verify_rounds": rounds,
        "greedy_exact_matches": exact,
        "greedy_exact_match_rate": exact / len(rows),
        "matching_prefix_tokens": distribution(prefix_lengths),
        "matching_prefix_fraction": distribution(prefix_rates),
    }


def group(rows: list[dict[str, Any]], label: Callable[[dict[str, Any]], str]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[label(row)].append(row)
    return dict(sorted(grouped.items()))


def fmt(value: Any, digits: int = 3) -> str:
    return "n/a" if value is None else f"{float(value):.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    args = parser.parse_args()

    official = load_jsonl(args.input_dir / "qwen3_4b_official_dflash_ddtree_bf16.jsonl")
    own = load_jsonl(args.input_dir / "qwen3_4b_own_bf16.jsonl")
    prompts = load_jsonl(args.prompts)
    if len(official) != len(own) or len(official) != len(prompts):
        raise ValueError(f"row count mismatch: official={len(official)}, own={len(own)}, prompts={len(prompts)}")
    dataset_by_index = {int(row.get("index", i)): str(row["dataset"]) for i, row in enumerate(prompts)}
    own_groups = group(own, lambda row: dataset_by_index[int(row["index"])])
    official_groups = group(official, lambda row: str(row["dataset"]))

    methods = {
        "dflash": official_method(official, "dflash"),
        "ddtree": official_method(official, "ddtree"),
        "own_architecture": own_method(own),
    }
    per_dataset: dict[str, Any] = {}
    for dataset in official_groups:
        per_dataset[dataset] = {
            "dflash": official_method(official_groups[dataset], "dflash"),
            "ddtree": official_method(official_groups[dataset], "ddtree"),
            "own_architecture": own_method(own_groups[dataset]),
        }
    summary = {
        "benchmark_policy": {
            "device": "NVIDIA GeForce RTX 4090 24GB",
            "dtype": "bfloat16",
            "prompts": len(official),
            "max_new_tokens": 128,
            "temperature": 0.0,
            "tree_budget": 60,
            "target_attention_backend": "torch.sdpa (common controlled backend)",
            "primary_metric_convention": "DDTree official make_latex_table.py",
            "dataset_path": str(args.prompts),
        },
        "overall": methods,
        "per_dataset": per_dataset,
        "dflash2": {
            "status": "not_run_hardware_insufficient_for_strict_bf16",
            "reason": "Smallest official target/draft pair is Qwen3.8-27B BF16 + 2B BF16 draft; it does not fit a 24GB RTX 4090.",
        },
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    table = []
    for name, metrics in methods.items():
        table.append(
            f"| {name} | {metrics['prompts']} | {fmt(metrics['speedup_vs_target'])}x | {fmt(metrics['tokens_per_second'], 2)} | "
            f"{fmt(metrics['mean_committed_per_verify'])} | {fmt(metrics['mean_accepted_draft_tokens'])} | "
            f"{metrics['greedy_exact_matches']}/{metrics['prompts']} | {fmt(metrics['decode_latency_ms']['p50'], 2)} | {fmt(metrics['decode_latency_ms']['p95'], 2)} |"
        )
    markdown = "\n".join([
        "# RTX 4090 strict BF16 unified benchmark",
        "",
        f"Fixed {len(official)} prompts, greedy decoding, max_new_tokens=128, tree budget=60, common target backend=torch.sdpa.",
        "Primary speedup and acceptance metrics follow DDTree's official `make_latex_table.py`: average per-prompt time-per-token and average per-prompt acceptance length.",
        "",
        "| Method | Prompts | Speedup | tok/s | Mean committed/verify | Mean accepted draft | BF16 bit-exact diagnostic | P50 ms | P95 ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        *table,
        "",
        "The bit-exact column is a numerical diagnostic, not the paper's losslessness metric: BF16 SDPA can choose a different greedy token when single-token and block verification shapes produce near-tied logits.",
        "Aggregate token-weighted metrics are retained only under `aggregate_weighted_secondary` in the JSON.",
        "",
        "DFlash2 was not run in the strict table: the smallest official BF16 pair requires Qwen3.8-27B plus its 2B draft and does not fit the 24GB RTX 4090.",
        "",
        "Full per-dataset metrics, distributions, histograms, stage timings, and totals are in `summary_bf16.json`.",
        "",
    ])
    args.markdown_output.write_text(markdown, encoding="utf-8")
    print(json.dumps(summary["overall"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
