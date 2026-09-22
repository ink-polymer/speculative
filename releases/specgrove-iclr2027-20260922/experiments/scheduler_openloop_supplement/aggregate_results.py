#!/usr/bin/env python3
"""Aggregate matched scheduler and real Poisson serving records.

The script keeps seed-level aggregates so repeated request groups do not become
pseudo-replicates.  Paper summaries use medians across the three prespecified
seeds; `per_seed` retains every measured value for audit.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics


METHODS = ("dp", "tetris_style", "echo_style")


def percentile(values, q):
    values = sorted(float(value) for value in values)
    if not values:
        return None
    position = (len(values) - 1) * q
    lo, hi = math.floor(position), math.ceil(position)
    if lo == hi:
        return values[lo]
    fraction = position - lo
    return values[lo] * (1 - fraction) + values[hi] * fraction


def median(values):
    return statistics.median(float(value) for value in values)


def aggregate_scheduler(root):
    buckets = defaultdict(lambda: defaultdict(lambda: {
        "tokens": 0, "wall_ms": 0.0, "planning_ms": [], "groups": 0,
    }))
    files = sorted((root / "scheduler_baselines" / "groups").glob("*.json"))
    for path in files:
        record = json.loads(path.read_text())
        identity = record["identity"]
        cell_seed = (
            int(identity["row_budget"]), int(identity["concurrency"]),
            int(identity["seed"]),
        )
        for run in record["runs"]:
            item = buckets[cell_seed][run["method"]]
            item["tokens"] += int(run["summary"]["output_tokens"])
            item["wall_ms"] += float(run["summary"]["wall_ms"])
            item["planning_ms"].extend(
                float(row["planning_ms"]) for row in run["allocator_trace"]
            )
            item["groups"] += 1

    per_seed = []
    for (row_budget, concurrency, seed), methods in sorted(buckets.items()):
        if set(methods) != set(METHODS):
            raise RuntimeError(f"Incomplete methods for {(row_budget, concurrency, seed)}")
        row = {"row_budget": row_budget, "concurrency": concurrency, "seed": seed}
        for method in METHODS:
            item = methods[method]
            row[method] = {
                "tokens_per_second": 1000 * item["tokens"] / item["wall_ms"],
                "output_tokens": item["tokens"],
                "wall_ms": item["wall_ms"],
                "groups": item["groups"],
                "planning_p95_ms": percentile(item["planning_ms"], 0.95),
                "planning_mean_ms": statistics.mean(item["planning_ms"]),
            }
        per_seed.append(row)

    cells = []
    by_cell = defaultdict(list)
    for row in per_seed:
        by_cell[(row["row_budget"], row["concurrency"])].append(row)
    for (row_budget, concurrency), rows in sorted(by_cell.items()):
        cell = {"row_budget": row_budget, "concurrency": concurrency,
                "seeds": [row["seed"] for row in rows]}
        for method in METHODS:
            cell[method] = {
                "tokens_per_second_median": median(
                    row[method]["tokens_per_second"] for row in rows
                ),
                "tokens_per_second_by_seed": [
                    row[method]["tokens_per_second"] for row in rows
                ],
                "planning_p95_ms_median": median(
                    row[method]["planning_p95_ms"] for row in rows
                ),
            }
        dp = cell["dp"]["tokens_per_second_median"]
        cell["dp_gain_percent"] = {
            method: 100 * (dp / cell[method]["tokens_per_second_median"] - 1)
            for method in METHODS[1:]
        }
        cell["paired_dp_gain_percent"] = {}
        for method in METHODS[1:]:
            gains = [100 * (
                row["dp"]["tokens_per_second"]
                / row[method]["tokens_per_second"] - 1
            ) for row in rows]
            cell["paired_dp_gain_percent"][method] = {
                "median": median(gains), "min": min(gains), "max": max(gains),
                "by_seed": gains,
            }
        cells.append(cell)
    return {"files": len(files), "per_seed": per_seed, "cells": cells}


def aggregate_open_loop(root):
    per_run = []
    files = sorted((root / "open_loop").glob("*.json"))
    for path in files:
        record = json.loads(path.read_text())
        summary = record["summary"]
        per_run.append({
            "method": record["method"], "seed": int(record["seed"]),
            "offered_rate_rps": float(record["offered_rate_rps"]),
            "tokens_per_second": float(summary["tokens_per_second"]),
            "completed_rps": float(summary["completed_rps"]),
            "queue_p95_ms": float(summary["queue_ms"]["p95"]),
            "ttft_p95_ms": float(summary["ttft_ms"]["p95"]),
            "tpot_p95_ms": float(summary["tpot_ms"]["p95"]),
            "e2e_p95_ms": float(summary["e2e_ms"]["p95"]),
            "jain_request_service_rate": float(summary["jain_request_service_rate"]),
            "planning_p95_ms": float(summary["planning"]["p95_ms"]),
        })
    buckets = defaultdict(list)
    for row in per_run:
        buckets[(row["offered_rate_rps"], row["method"])].append(row)
    rates = []
    for rate in sorted({key[0] for key in buckets}):
        result = {"offered_rate_rps": rate}
        for method in METHODS:
            rows = buckets[(rate, method)]
            if not rows:
                raise RuntimeError(f"Missing open-loop records for {(rate, method)}")
            result[method] = {
                field + "_median": median(row[field] for row in rows)
                for field in (
                    "tokens_per_second", "completed_rps", "queue_p95_ms",
                    "ttft_p95_ms", "tpot_p95_ms", "e2e_p95_ms",
                    "jain_request_service_rate", "planning_p95_ms",
                )
            }
            result[method]["seeds"] = [row["seed"] for row in rows]
        result["dp_gain_percent"] = {}
        for baseline in METHODS[1:]:
            result["dp_gain_percent"][baseline] = {
                "tokens_per_second": 100 * (
                    result["dp"]["tokens_per_second_median"]
                    / result[baseline]["tokens_per_second_median"] - 1
                ),
                "ttft_p95_reduction": 100 * (
                    1 - result["dp"]["ttft_p95_ms_median"]
                    / result[baseline]["ttft_p95_ms_median"]
                ),
                "e2e_p95_reduction": 100 * (
                    1 - result["dp"]["e2e_p95_ms_median"]
                    / result[baseline]["e2e_p95_ms_median"]
                ),
            }
            dp_by_seed = {row["seed"]: row for row in buckets[(rate, "dp")]}
            baseline_by_seed = {
                row["seed"]: row for row in buckets[(rate, baseline)]
            }
            paired = {}
            for field, direction in (
                ("tokens_per_second", "higher"),
                ("ttft_p95_ms", "lower"),
                ("e2e_p95_ms", "lower"),
            ):
                values = []
                for seed in sorted(dp_by_seed):
                    dp_value = dp_by_seed[seed][field]
                    base_value = baseline_by_seed[seed][field]
                    values.append(100 * (
                        dp_value / base_value - 1
                        if direction == "higher" else 1 - dp_value / base_value
                    ))
                paired[field] = {
                    "median": median(values), "min": min(values),
                    "max": max(values), "by_seed": values,
                }
            result.setdefault("paired_dp_change_percent", {})[baseline] = paired
        rates.append(result)
    return {"files": len(files), "per_run": per_run, "rates": rates}


def markdown(report):
    lines = [
        "# SpecGrove scheduler and open-loop supplement", "",
        "Medians are across the completed prespecified seed-level replicates; raw seed values remain in the JSON audit record.", "",
    ]
    if report["scheduler"] is not None:
        lines.extend([
            "## Matched scheduler baseline", "",
            "| R | C | SpecGrove tok/s | TETRIS-style tok/s | ECHO-style tok/s | vs TETRIS | vs ECHO | DP plan p95 ms |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for row in report["scheduler"]["cells"]:
            lines.append(
                f'| {row["row_budget"]} | {row["concurrency"]} | '
                f'{row["dp"]["tokens_per_second_median"]:.1f} | '
                f'{row["tetris_style"]["tokens_per_second_median"]:.1f} | '
                f'{row["echo_style"]["tokens_per_second_median"]:.1f} | '
                f'{row["dp_gain_percent"]["tetris_style"]:+.1f}% | '
                f'{row["dp_gain_percent"]["echo_style"]:+.1f}% | '
                f'{row["dp"]["planning_p95_ms_median"]:.3f} |'
            )
    lines.extend([
        "", "## Real-time Poisson serving", "",
        "| Offered req/s | Method | tok/s | completed req/s | queue p95 ms | TTFT p95 ms | E2E p95 ms | Jain |",
        "|---:|:---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in report["open_loop"]["rates"]:
        for method in METHODS:
            value = row[method]
            lines.append(
                f'| {row["offered_rate_rps"]:.0f} | {method} | '
                f'{value["tokens_per_second_median"]:.1f} | '
                f'{value["completed_rps_median"]:.2f} | '
                f'{value["queue_p95_ms_median"]:.1f} | '
                f'{value["ttft_p95_ms_median"]:.1f} | '
                f'{value["e2e_p95_ms_median"]:.1f} | '
                f'{value["jain_request_service_rate_median"]:.3f} |'
            )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--open-loop-only", action="store_true")
    args = parser.parse_args()
    report = {
        "aggregation": (
            "Seed is the replicate. Scheduler throughput pools request groups "
            "within each seed, then takes the median across seeds. Open-loop "
            "metrics take the median of the three trace-level values."
        ),
        "scheduler": None if args.open_loop_only else aggregate_scheduler(args.root),
        "open_loop": aggregate_open_loop(args.root),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    args.output.with_suffix(".md").write_text(markdown(report))
    print(markdown(report))


if __name__ == "__main__":
    main()
