"""Fixed diagnostic search space and aggregation for diffusion scaffold tuning.

This module deliberately keeps tuning results separate from preregistered formal
results.  The grid is constructed before any model timing is observed, and every
candidate is measured against the same canonical DFlash and DDTree controls.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
import math

from .config import Variant


BASELINE_NAMES = ("dflash_l15_b45", "ddtree_l15_b45")
TEMPERATURES = (0.3, 0.6, 1.0)
SCAFFOLD_GRID = (
    (1, 8, 16),
    (1, 8, 24),
    (1, 12, 24),
    (1, 12, 36),
    (1, 15, 30),
    (1, 15, 45),
    (1, 15, 60),
    (2, 8, 24),
    (2, 12, 36),
    (2, 15, 45),
)


def optimization_variants(temperature: float) -> list[Variant]:
    """Return the frozen coarse grid for one positive target temperature."""
    if temperature not in TEMPERATURES:
        raise ValueError("Optimization temperature must be 0.3, 0.6 or 1.0")
    base = Variant(
        name="base", method="ddtree", paths=1, length=15,
        temperature=temperature, draft_temperature=temperature,
        tree_budget=45, probability_dtype="float64",
    )
    variants = [
        replace(base, name=BASELINE_NAMES[0], method="dflash"),
        replace(base, name=BASELINE_NAMES[1], method="ddtree"),
    ]
    for paths, length, budget in SCAFFOLD_GRID:
        stem = f"k{paths}_l{length}_b{budget}"
        variants.extend((
            replace(base, name=f"{stem}_terminal", method="diffusion_scaffold_bv",
                    paths=paths, length=length, tree_budget=budget),
            replace(base, name=f"{stem}_ancestral", method="diffusion_scaffold_ancestral",
                    paths=paths, length=length, tree_budget=budget),
        ))
    if len({variant.name for variant in variants}) != len(variants):
        raise AssertionError("Optimization variants must have unique names")
    for variant in variants:
        variant.validate()
    return variants


def _aggregate(records: list[dict]) -> dict:
    decode_ms = sum(record["decode_ms"] for record in records)
    decode_tokens = sum(record["decode_tokens"] for record in records)
    rounds = [round_ for record in records for round_ in record["rounds"]]
    if decode_tokens < 1 or not rounds:
        raise ValueError("Optimization records require decode tokens and rounds")
    return {
        "records": len(records),
        "decode_tokens": decode_tokens,
        "decode_ms_per_token": decode_ms / decode_tokens,
        "e2e_ms_per_token": (
            sum(record["e2e_ms"] for record in records)
            / sum(record["generated_tokens"] for record in records)
        ),
        "mean_accepted_draft_tokens": (
            sum(round_["accepted_draft_tokens"] for round_ in rounds) / len(rounds)
        ),
        "mean_committed_tokens": (
            sum(round_["committed_tokens"] for round_ in rounds) / len(rounds)
        ),
        "mean_tree_nodes": sum(round_["tree_nodes"] for round_ in rounds) / len(rounds),
        "repeat_equal": len({
            (record["prompt"], record["generated_sha256"])
            for record in records
        }) == len({record["prompt"] for record in records}),
    }


def summarize(rows: list[dict]) -> dict:
    """Aggregate weighted TPOT and rank only candidates measured at all T values."""
    grouped: dict[tuple[float, str], list[dict]] = defaultdict(list)
    metadata = {}
    for row in rows:
        key = (row["temperature"], row["variant"])
        grouped[key].append(row)
        metadata[key] = {field: row[field] for field in
                         ("method", "paths", "length", "tree_budget")}
    per_temperature = []
    for temperature in sorted({key[0] for key in grouped}):
        baseline = {}
        for name in BASELINE_NAMES:
            if (temperature, name) not in grouped:
                raise ValueError(f"Missing {name} baseline at T={temperature}")
            baseline[name] = _aggregate(grouped[temperature, name])["decode_ms_per_token"]
        for (current_temperature, name), records in grouped.items():
            if current_temperature != temperature:
                continue
            metrics = _aggregate(records)
            metrics.update(metadata[temperature, name])
            metrics.update({
                "temperature": temperature,
                "variant": name,
                "speedup_vs_dflash": baseline[BASELINE_NAMES[0]] / metrics["decode_ms_per_token"],
                "speedup_vs_ddtree": baseline[BASELINE_NAMES[1]] / metrics["decode_ms_per_token"],
            })
            per_temperature.append(metrics)
    per_temperature.sort(key=lambda row: (row["temperature"], row["decode_ms_per_token"], row["variant"]))

    by_candidate: dict[str, list[dict]] = defaultdict(list)
    for row in per_temperature:
        if row["variant"] not in BASELINE_NAMES:
            by_candidate[row["variant"]].append(row)
    expected_temperatures = sorted({row["temperature"] for row in per_temperature})
    overall = []
    for name, records in by_candidate.items():
        if sorted(record["temperature"] for record in records) != expected_temperatures:
            continue
        overall.append({
            "variant": name,
            **{field: records[0][field] for field in ("method", "paths", "length", "tree_budget")},
            "temperatures": expected_temperatures,
            "geomean_speedup_vs_dflash": math.exp(sum(
                math.log(record["speedup_vs_dflash"]) for record in records
            ) / len(records)),
            "geomean_speedup_vs_ddtree": math.exp(sum(
                math.log(record["speedup_vs_ddtree"]) for record in records
            ) / len(records)),
            "all_repeat_equal": all(record["repeat_equal"] for record in records),
        })
    overall.sort(key=lambda row: (
        -min(row["geomean_speedup_vs_dflash"], row["geomean_speedup_vs_ddtree"]),
        row["variant"],
    ))
    return {
        "selection_scope": "diagnostic fixed-grid optimization; not a formal or held-out claim",
        "baseline_names": list(BASELINE_NAMES),
        "per_temperature": per_temperature,
        "overall_ranking": overall,
    }


def profile_names(summary: dict, temperature: float, count: int = 2) -> list[str]:
    """Choose the fastest candidates plus both controls for stage profiling."""
    candidates = [row for row in summary["per_temperature"]
                  if row["temperature"] == temperature and row["variant"] not in BASELINE_NAMES]
    candidates.sort(key=lambda row: (row["decode_ms_per_token"], row["variant"]))
    return [*BASELINE_NAMES, *(row["variant"] for row in candidates[:count])]
