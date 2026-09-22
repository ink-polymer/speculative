#!/usr/bin/env python3
"""Fail-fast structural audit for the scheduler/Poisson result bundle."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path


METHODS = {"dp", "tetris_style", "echo_style"}
SEEDS = {17, 29, 43}
CELLS = {(193, 8), (193, 16), (384, 16), (384, 32)}
RATES = {1.0, 2.0, 4.0}


def finite_tree(value):
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tree(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--open-loop-requests", type=int, default=64)
    parser.add_argument("--open-loop-only", action="store_true")
    args = parser.parse_args()
    root = args.root
    complete = json.loads((root / "complete.json").read_text())
    expected_phase = "open_loop" if args.open_loop_only else "all"
    assert complete == {
        "completed_at": complete["completed_at"],
        "phase": expected_phase, "quick": False,
    }
    assert not list(root.rglob("*.tmp"))

    group_counts = Counter()
    scheduler_files = []
    if not args.open_loop_only:
        scheduler_files = sorted(
            (root / "scheduler_baselines" / "groups").glob("*.json")
        )
        assert len(scheduler_files) == 108
        for path in scheduler_files:
            record = json.loads(path.read_text())
            assert finite_tree(record)
            identity = record["identity"]
            cell = (int(identity["row_budget"]), int(identity["concurrency"]))
            seed = int(identity["seed"])
            assert cell in CELLS and seed in SEEDS
            assert len(identity["source_ids"]) == cell[1]
            assert set(record["method_order"]) == METHODS
            assert {run["method"] for run in record["runs"]} == METHODS
            for run in record["runs"]:
                assert len(run["outputs"]) == cell[1]
                assert run["summary"]["output_tokens"] == sum(
                    len(output) for output in run["outputs"]
                )
                assert all(
                    int(wave["rows"]) <= cell[0]
                    for wave in run["allocator_trace"]
                )
            group_counts[(cell, seed)] += 1
        assert group_counts == Counter({
            (cell, seed): 128 // cell[1] for cell in CELLS for seed in SEEDS
        })

    open_files = sorted((root / "open_loop").glob("*.json"))
    assert len(open_files) == 27
    traces = defaultdict(dict)
    for path in open_files:
        record = json.loads(path.read_text())
        assert finite_tree(record)
        key = (float(record["offered_rate_rps"]), int(record["seed"]))
        method = record["method"]
        assert key[0] in RATES and key[1] in SEEDS and method in METHODS
        assert method not in traces[key]
        traces[key][method] = record
        arrivals = record["arrival_times_s"]
        assert len(arrivals) == args.open_loop_requests and arrivals[0] == 0
        assert all(a <= b for a, b in zip(arrivals, arrivals[1:]))
        assert (
            len(record["request_metrics"])
            == len(record["outputs"])
            == args.open_loop_requests
        )
        for metric, output in zip(record["request_metrics"], record["outputs"]):
            assert metric["output_tokens"] == len(output)
            assert metric["arrival_ms"] <= metric["admitted_ms"]
            assert metric["admitted_ms"] <= metric["first_token_ms"]
            assert metric["first_token_ms"] <= metric["completed_ms"]
        assert all(
            int(wave["rows"]) <= 384 for wave in record["allocator_trace"]
        )
    assert set(traces) == {(rate, seed) for rate in RATES for seed in SEEDS}
    for records in traces.values():
        assert set(records) == METHODS
        reference = next(iter(records.values()))
        for record in records.values():
            assert record["arrival_times_s"] == reference["arrival_times_s"]
            assert record["source_ids"] == reference["source_ids"]

    print(json.dumps({
        "complete": True, "scheduler_group_files": len(scheduler_files),
        "open_loop_trace_files": len(open_files),
        "cells": sorted([list(cell) for cell in CELLS]),
        "rates": sorted(RATES), "seeds": sorted(SEEDS),
    }, indent=2))


if __name__ == "__main__":
    main()
