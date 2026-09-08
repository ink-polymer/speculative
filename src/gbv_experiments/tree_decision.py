from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .common import read_jsonl, write_json


def _ratio(numerator, denominator):
    return float(numerator / denominator) if denominator > 0 else None


def _expected_pairs(manifest: dict) -> set[tuple[str, str, int]]:
    return {
        (dataset, source_id, seed)
        for dataset, source_id, _ in manifest["prompt_ids"]
        for seed in manifest["seeds"]
    }


def _index_rows(rows: list[dict], variant: str) -> dict[tuple[str, str, int], dict]:
    index = {}
    for row in rows:
        if row["variant"] != variant:
            continue
        key = (row["dataset"], row["source_id"], row["seed"])
        if key in index:
            raise ValueError(f"Duplicate row key for {variant}: {key}")
        index[key] = row
    return index


def _pair_rows(rows: list[dict], variant_a: str, variant_b: str,
               expected: set[tuple[str, str, int]]) -> list[tuple[dict, dict]]:
    index_a = _index_rows(rows, variant_a)
    index_b = _index_rows(rows, variant_b)
    keys_a, keys_b = set(index_a), set(index_b)
    if keys_a != expected:
        missing = sorted(expected - keys_a)
        extra = sorted(keys_a - expected)
        if missing:
            example = ", ".join(f"{d}/{s}/{seed}" for d, s, seed in missing[:6])
            raise ValueError(f"Missing {len(missing)} rows for {variant_a}: {example}")
        if extra:
            example = ", ".join(f"{d}/{s}/{seed}" for d, s, seed in extra[:6])
            raise ValueError(f"Unexpected {len(extra)} rows for {variant_a}: {example}")
    if keys_b != expected:
        missing = sorted(expected - keys_b)
        extra = sorted(keys_b - expected)
        if missing:
            example = ", ".join(f"{d}/{s}/{seed}" for d, s, seed in missing[:6])
            raise ValueError(f"Missing {len(missing)} rows for {variant_b}: {example}")
        if extra:
            example = ", ".join(f"{d}/{s}/{seed}" for d, s, seed in extra[:6])
            raise ValueError(f"Unexpected {len(extra)} rows for {variant_b}: {example}")

    missing_pairs = sorted(expected - (keys_a & keys_b))
    if missing_pairs:
        example = ", ".join(f"{d}/{s}/{seed}" for d, s, seed in missing_pairs[:6])
        raise ValueError(f"Unpaired rows: missing {len(missing_pairs)} {variant_a}/{variant_b} pairs: {example}")

    return [(index_a[k], index_b[k]) for k in sorted(expected)]


def _bootstrap_cluster_speedup(pairs: list[tuple[dict, dict]], bootstrap: int = 1000,
                              seed: int = 0):
    clusters: dict[tuple[str, str], list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])
    for recycle_row, ddtree_row in pairs:
        key = (recycle_row["dataset"], recycle_row["source_id"])
        clusters[key][0] += float(recycle_row["decode_ms"])
        clusters[key][1] += float(recycle_row["decode_tokens"])
        clusters[key][2] += float(ddtree_row["decode_ms"])
        clusters[key][3] += float(ddtree_row["decode_tokens"])

    if not clusters:
        return {
            "speedup_vs_ddtree": None,
            "speedup_vs_ddtree_ci_low": None,
            "speedup_vs_ddtree_ci_high": None,
            "clusters": 0,
            "matched_pairs": 0,
        }

    data = np.asarray(list(clusters.values()), dtype=np.float64)
    recycle_time = data[:, 0].sum()
    recycle_tokens = data[:, 1].sum()
    ddtree_time = data[:, 2].sum()
    ddtree_tokens = data[:, 3].sum()
    speedup = _ratio(ddtree_time * recycle_tokens, ddtree_tokens * recycle_time)

    if bootstrap < 2 or len(data) < 2:
        return {
            "speedup_vs_ddtree": speedup,
            "speedup_vs_ddtree_ci_low": None,
            "speedup_vs_ddtree_ci_high": None,
            "clusters": len(data),
            "matched_pairs": len(pairs),
        }

    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(bootstrap):
        sample = data[rng.integers(0, len(data), len(data))].sum(axis=0)
        value = _ratio(sample[2] * sample[1], sample[3] * sample[0])
        if value is not None:
            estimates.append(value)

    if estimates:
        low, high = np.quantile(estimates, [0.025, 0.975]).tolist()
    else:
        low, high = None, None

    return {
        "speedup_vs_ddtree": speedup,
        "speedup_vs_ddtree_ci_low": low,
        "speedup_vs_ddtree_ci_high": high,
        "clusters": len(data),
        "matched_pairs": len(pairs),
    }


def _stage_ms(row: dict, stage: str, *, prefer_cuda: bool) -> float | None:
    stages = row.get("stages")
    if not isinstance(stages, dict):
        return None
    if prefer_cuda and isinstance(stages.get("cuda_event_ms"), dict) and stage in stages["cuda_event_ms"]:
        value = stages["cuda_event_ms"][stage]
        return None if value is None else float(value)
    if isinstance(stages.get("host_ms"), dict) and stage in stages["host_ms"]:
        value = stages["host_ms"][stage]
        return None if value is None else float(value)
    return None


def _round_values(rows: list[dict], name: str):
    values = []
    for row in rows:
        for round_row in row.get("rounds", []) or []:
            if round_row is None:
                continue
            value = round_row.get(name)
            if value is not None:
                values.append(float(value))
    return values


def _recycle_summary(rows: list[dict]):
    segments = _round_values(rows, "tree_bv_segments")
    recycled = _round_values(rows, "recycled_corrections")
    if not segments:
        return {
            "mean_tree_bv_segments": None,
            "mean_recycled_corrections": None,
            "recycle_hit_fraction": None,
            "target_tokens_per_decode": None,
            "select_and_correct_ms_per_round": None,
            "select_and_correct_cuda_ms_per_round": None,
            "accepted_per_round": None,
            "committed_per_round": None,
            "tree_nodes_per_round": None,
        }

    total_rounds = len(segments)
    total_select_host = 0.0
    total_select_cuda = 0.0
    select_host_rounds = 0
    select_cuda_rounds = 0
    for row in rows:
        rounds = row.get("rounds") or []
        if not rounds:
            continue
        value = _stage_ms(row, "select_and_correct", prefer_cuda=False)
        if value is not None:
            total_select_host += value
            select_host_rounds += len(rounds)
        value = _stage_ms(row, "select_and_correct", prefer_cuda=True)
        if value is not None:
            total_select_cuda += value
            select_cuda_rounds += len(rounds)

    accepted = _round_values(rows, "accepted_draft_tokens")
    committed = _round_values(rows, "committed_tokens")
    nodes = _round_values(rows, "tree_nodes")
    target_tokens = sum(float(row["target_tokens_processed"]) for row in rows)
    decode_tokens = sum(float(row["decode_tokens"]) for row in rows)

    return {
        "mean_tree_bv_segments": _ratio(sum(segments), total_rounds),
        "mean_recycled_corrections": _ratio(sum(recycled), len(recycled)),
        "recycle_hit_fraction": _ratio(sum(1 for x in recycled if x > 0), len(recycled)),
        "target_tokens_per_decode": _ratio(target_tokens, decode_tokens),
        "select_and_correct_ms_per_round": _ratio(total_select_host, select_host_rounds),
        "select_and_correct_cuda_ms_per_round": _ratio(total_select_cuda, select_cuda_rounds),
        "accepted_per_round": _ratio(sum(accepted), len(accepted)),
        "committed_per_round": _ratio(sum(committed), len(committed)),
        "tree_nodes_per_round": _ratio(sum(nodes), len(nodes)),
    }


def _dataset_expected_pairs(manifest: dict, dataset: str) -> int:
    source_counts = {
        (ds, source) for ds, source, _ in manifest["prompt_ids"]
    }
    return len([_ for ds, source in source_counts if ds == dataset]) * len(manifest["seeds"])


def _decision_for_model(run_dir: Path, bootstrap_samples: int):
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.exists():
        raise ValueError(f"Missing run manifest for model directory {run_dir}")

    manifest = json.loads(manifest_path.read_text())
    rows = read_jsonl(run_dir / "results.jsonl")
    expected = _expected_pairs(manifest)
    pairs = _pair_rows(rows, "tree_bv_recycle", "ddtree", expected)

    by_dataset: dict[str, list[tuple[dict, dict]]] = defaultdict(list)
    for recycle_row, ddtree_row in pairs:
        by_dataset[recycle_row["dataset"]].append((recycle_row, ddtree_row))

    overall = _bootstrap_cluster_speedup(pairs, bootstrap_samples)
    result = {
        "run_id": manifest["run_id"],
        "model": manifest["model"]["target"],
        "expected_pairs": len(expected),
        "matched_pairs": overall["matched_pairs"],
        "clusters": overall["clusters"],
        "overall": {
            **overall,
            "recycle_summary": _recycle_summary([recycle_row for recycle_row, _ in pairs]),
        },
        "per_dataset": {},
    }

    for dataset in manifest["dataset_names"]:
        dataset_pairs = by_dataset.get(dataset, [])
        dataset_overall = _bootstrap_cluster_speedup(dataset_pairs, bootstrap_samples)
        result["per_dataset"][dataset] = {
            "dataset": dataset,
            "expected_pairs": _dataset_expected_pairs(manifest, dataset),
            "matched_pairs": dataset_overall["matched_pairs"],
            **dataset_overall,
            "recycle_summary": _recycle_summary([recycle_row for recycle_row, _ in dataset_pairs]),
        }

    result["passed"] = (
        result["overall"]["matched_pairs"] == result["overall"]["expected_pairs"]
        and result["overall"]["clusters"] > 0
        and result["overall"]["speedup_vs_ddtree_ci_low"] is not None
        and result["overall"]["speedup_vs_ddtree_ci_low"] > 1.0
    )
    return result


def _fmt(value):
    return "--" if value is None else f"{value:.4f}"


def _fmt_ci(low, high):
    return "--" if low is None or high is None else f"[{low:.4f}, {high:.4f}]"


def write_decision(output: Path, model_ids: list[str], bootstrap_samples: int = 1000):
    records = [_decision_for_model(output / model_id, bootstrap_samples) for model_id in model_ids]
    all_models_pass = all(record["passed"] for record in records) if records else False
    payload = {
        "study": "branch_recycling_tree_bv",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "decision_rule": (
            "Tree-BV (tree_bv_recycle) paired with DDTree, with per-(dataset,source_id) "
            "bootstrap clustering that keeps all seeds together. "
            "Pass if 95% paired CI lower bound > 1.0 and all expected pairs are present."
        ),
        "bootstrap_samples": bootstrap_samples,
        "model_count": len(records),
        "models": records,
        "all_models_pass": all_models_pass,
        "pass_fraction": float(sum(record["passed"] for record in records) / max(1, len(records))),
        "note": (
            "This is a statistical gate, not a guaranteed dominance guarantee. "
            "Results may move within confidence intervals across reruns."
        ),
    }
    write_json(output / "tree_bv_decision.json", payload)

    md_lines = [
        "# BRBV-vs-DDTree Decision",
        "",
        f"Generated at: {payload['generated_at']}",
        "Pass rule: paired bootstrap CI lower bound > 1.0 with per-(dataset, source_id) clusters",
        f"All models pass: {'yes' if all_models_pass else 'no'}",
        "",
        "## Per-model results",
        "| model | expected pairs | matched pairs | overall speedup vs DDTree | 95% CI | recycle hit rate | recycled corrections / round | segments / round | passed |",
        "|---|---:|---:|---:|---|---:|---:|---:|---|",
    ]
    for record in records:
        overall = record["overall"]
        recycle_summary = overall["recycle_summary"]
        md_lines.append(
            f"| {record['model']} | {record['expected_pairs']} | {record['matched_pairs']} | "
            f"{_fmt(overall['speedup_vs_ddtree'])} | {_fmt_ci(overall['speedup_vs_ddtree_ci_low'], overall['speedup_vs_ddtree_ci_high'])} | "
            f"{_fmt(recycle_summary['recycle_hit_fraction'])} | {_fmt(recycle_summary['mean_recycled_corrections'])} | {_fmt(recycle_summary['mean_tree_bv_segments'])} | "
            f"{'yes' if record['passed'] else 'no'} |"
        )

    md_lines.extend([
        "",
        "## JSON output",
        "- decision: `tree_bv_decision.json`",
    ])
    (output / "tree_bv_decision.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return payload
