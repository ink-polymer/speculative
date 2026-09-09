"""Independently validate a same-tree block-verification pilot artifact.

The validator reads raw rows and recomputes every reported paired speedup and
cluster-bootstrap confidence interval.  It deliberately does not import the
benchmark runner so that a broken or changed runtime implementation cannot
silently validate its own output.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np


EXPECTED_METHODS = {"dflash", "ddtree", "tree_block_verification"}
EXPECTED_PRECISION = {
    "dtype": "bfloat16",
    "target_attention": "sdpa",
    "draft_attention": "sdpa",
    "allow_tf32": False,
}
SOURCE_PATHS = {
    "engine": "src/gbv_experiments/engine.py",
    "fused_verifier": "src/gbv_experiments/fused_tree_sampling.py",
    "official_verifier": "src/gbv_experiments/sampling.py",
    "script": "scripts/run_same_tree_official_pilot.py",
    "config": "configs/same_tree_block_qwen3_4b_pilot.json",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value: object) -> str:
    canonical = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def close(left: float, right: float, tolerance: float = 1e-12) -> bool:
    return abs(left - right) <= tolerance


def display_path(path: Path, repo_root: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(repo_root.resolve()))
    except ValueError:
        return str(resolved)


def compare(rows: list[dict], manifest: dict, candidate: str,
            baseline: str, bootstrap_samples: int = 10_000) -> dict:
    lookup = {
        (row["variant"], row["dataset"], str(row["source_id"]), row["seed"]): row
        for row in rows
    }
    source_logs: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        if row["variant"] != candidate:
            continue
        reference = lookup[
            (baseline, row["dataset"], str(row["source_id"]), row["seed"])
        ]
        candidate_tpot = row["decode_ms"] / row["decode_tokens"]
        baseline_tpot = reference["decode_ms"] / reference["decode_tokens"]
        source_logs[row["dataset"]][str(row["source_id"])].append(
            math.log(baseline_tpot / candidate_tpot)
        )

    clustered = {
        dataset: np.asarray([
            np.mean(seed_logs) for seed_logs in sources.values()
        ], dtype=np.float64)
        for dataset, sources in source_logs.items()
    }
    datasets = manifest["datasets"]
    if set(clustered) != set(datasets):
        raise AssertionError("comparison is missing a registered dataset")
    dataset_logs = {dataset: float(clustered[dataset].mean())
                    for dataset in datasets}
    rng = np.random.default_rng(manifest["order_policy"]["seed"])
    draws = np.zeros(bootstrap_samples, dtype=np.float64)
    for dataset in datasets:
        values = clustered[dataset]
        draws += rng.choice(
            values, size=(bootstrap_samples, len(values)), replace=True,
        ).mean(1) / len(datasets)
    low, high = np.exp(np.quantile(draws, [.025, .975]))
    return {
        "candidate": candidate,
        "baseline": baseline,
        "speedup": math.exp(sum(dataset_logs.values()) / len(dataset_logs)),
        "ci95": [float(low), float(high)],
        "per_dataset": {
            dataset: math.exp(dataset_logs[dataset]) for dataset in datasets
        },
        "source_clusters_per_dataset": {
            dataset: len(clustered[dataset]) for dataset in datasets
        },
        "seeds_per_source": len(next(iter(source_logs[datasets[0]].values()))),
        "bootstrap_samples": bootstrap_samples,
    }


def validate(result_dir: Path, repo_root: Path,
             engine_source: Path | None = None) -> dict:
    rows_document = json.loads((result_dir / "rows.json").read_text())
    manifest = json.loads((result_dir / "manifest.json").read_text())
    report = json.loads((result_dir / "report.json").read_text())
    rows = rows_document["rows"]

    if manifest["formal_complete"] or report["formal_complete"]:
        raise AssertionError("qualification pilot must not claim formal completion")
    if not rows_document["complete"]:
        raise AssertionError("raw row document is incomplete")
    if len(rows) != manifest["expected_records"]:
        raise AssertionError("raw record count differs from the manifest")

    expected_prompt_ids = [
        (str(dataset), str(source_id), str(prompt_hash))
        for dataset, source_id, prompt_hash
        in manifest["data_selection"]["prompt_ids"]
    ]
    if len(expected_prompt_ids) != len(set(expected_prompt_ids)):
        raise AssertionError("manifest contains duplicate prompt identities")
    if len(expected_prompt_ids) != manifest["per_dataset"] * len(manifest["datasets"]):
        raise AssertionError("manifest prompt count differs from the pilot design")
    expected_source_ids = [
        [dataset, source_id] for dataset, source_id, _ in expected_prompt_ids
    ]
    if (digest(expected_source_ids)
            != manifest["data_selection"]["source_ids_sha256"]):
        raise AssertionError("manifest source selection digest changed")
    actual_prompt_ids = {
        (str(row["dataset"]), str(row["source_id"]), row["prompt_sha256"])
        for row in rows
    }
    if actual_prompt_ids != set(expected_prompt_ids):
        raise AssertionError("raw source/prompt set differs from the manifest")
    if {row["seed"] for row in rows} != set(manifest["seeds"]):
        raise AssertionError("raw seed set differs from the manifest")

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], str(row["source_id"]), row["seed"])].append(row)
    if len(groups) != manifest["expected_paired_groups"]:
        raise AssertionError("paired-group count differs from the manifest")
    for group in groups.values():
        if len(group) != 3 or {row["variant"] for row in group} != EXPECTED_METHODS:
            raise AssertionError("paired group does not contain exactly three methods")
        if len({row["sampling_seed"] for row in group}) != 1:
            raise AssertionError("paired methods used different sampling seeds")
        if len({row["prompt_sha256"] for row in group}) != 1:
            raise AssertionError("paired methods used different prompts")
        if sorted(row["execution_position"] for row in group) != [0, 1, 2]:
            raise AssertionError("paired method order is malformed")

    if len(manifest["seeds"]) != 3:
        raise AssertionError("validator expects the registered three-seed design")
    prompt_seed_coverage: dict[tuple[str, str], set[int]] = defaultdict(set)
    for dataset, source_id, seed in groups:
        prompt_seed_coverage[(dataset, source_id)].add(seed)
    if (set(prompt_seed_coverage)
            != {(dataset, source_id) for dataset, source_id, _
                in expected_prompt_ids}
            or any(seeds != set(manifest["seeds"])
                   for seeds in prompt_seed_coverage.values())):
        raise AssertionError("a prompt does not contain every registered seed")
    position_counts = {}
    for dataset in manifest["datasets"]:
        position_counts[dataset] = {}
        for method in sorted(EXPECTED_METHODS):
            counts = Counter(
                row["execution_position"] for row in rows
                if row["dataset"] == dataset and row["variant"] == method
            )
            expected = Counter({position: manifest["per_dataset"]
                                for position in (0, 1, 2)})
            if counts != expected:
                raise AssertionError("method positions are not exactly balanced")
            position_counts[dataset][method] = {
                str(position): counts[position] for position in (0, 1, 2)
            }
    recalculated = {}
    comparisons = (
        ("candidate_vs_ddtree", "tree_block_verification", "ddtree",
         report["comparisons"]["ddtree"]),
        ("candidate_vs_dflash", "tree_block_verification", "dflash",
         report["comparisons"]["dflash"]),
        ("ddtree_vs_dflash", "ddtree", "dflash", report["baseline_sanity"]),
    )
    for label, candidate, baseline, reported in comparisons:
        value = compare(rows, manifest, candidate, baseline)
        if not close(value["speedup"], reported["speedup"]):
            raise AssertionError(f"{label} point estimate changed")
        if any(not close(left, right)
               for left, right in zip(value["ci95"], reported["ci95"])):
            raise AssertionError(f"{label} confidence interval changed")
        if any(not close(value["per_dataset"][dataset],
                         reported["per_dataset"][dataset])
               for dataset in manifest["datasets"]):
            raise AssertionError(f"{label} per-dataset estimates changed")
        recalculated[label] = value

    if (report["records"] != len(rows)
            or report["paired_groups"] != len(groups)
            or not report["gate_passed"]):
        raise AssertionError("reported completion or gate decision changed")

    variants = {variant["name"]: variant for variant in manifest["variants"]}
    if set(variants) != EXPECTED_METHODS:
        raise AssertionError("manifest methods changed")
    if variants["dflash"]["draft_temperature"] is not None:
        raise AssertionError("official DFlash must record greedy Draft control")
    if not (variants["ddtree"]["draft_temperature"] == 1.0
            and variants["tree_block_verification"]["draft_temperature"] == 1.0):
        raise AssertionError("DDTree same-tree pair must use T=1 Draft sampling")
    dflash = manifest["dflash_control_match"]
    if (dflash["comparison"] != "official_dflash_control"
            or dflash["method_specific_proposals"]["dflash"]
            != "greedy_argmax masked block"):
        raise AssertionError("DFlash official-control witness changed")
    if manifest["official_precision"] != EXPECTED_PRECISION:
        raise AssertionError("official precision/backend controls changed")
    witness = manifest["same_tree_runtime_witness"]
    if not (witness["passed"]
            and witness["probability_shape"] == [46, 151936]
            and witness["probability_dtype"] == "torch.float64"):
        raise AssertionError("same-tree runtime witness failed")

    source_files = {}
    for key, relative in SOURCE_PATHS.items():
        path = ((engine_source if key == "engine" and engine_source is not None
                 else repo_root / relative)).resolve()
        actual = sha256(path)
        expected = manifest["source_sha256"][key]
        if actual != expected:
            raise AssertionError(
                f"source hash mismatch for {key}: expected {expected}, got {actual}"
            )
        source_files[key] = {
            "path": display_path(path, repo_root), "sha256": actual,
        }

    artifacts = {
        name: sha256(result_dir / name)
        for name in ("manifest.json", "report.json", "rows.json")
    }
    return {
        "kind": "independent_same_tree_pilot_validation",
        "integrity": "PASS",
        "formal_complete": False,
        "result_dir": display_path(result_dir, repo_root),
        "records": len(rows),
        "paired_groups": len(groups),
        "paired_group_integrity": "PASS_SAME_PROMPT_SAME_SEED_THREE_METHODS",
        "manifest_prompt_and_seed_coverage": (
            "PASS_EXACT_SOURCE_PROMPT_SET_AND_ALL_REGISTERED_SEEDS"
        ),
        "balanced_positions": (
            f"PASS_{manifest['per_dataset']}_PER_POSITION_CELL"
        ),
        "position_counts": position_counts,
        "independent_recalculation": recalculated,
        "strict_speed_gate": {
            "passed": all(value["ci95"][0] > 1.0
                          for value in recalculated.values()),
            "rule": "all three independently recalculated 95% CI lower bounds > 1",
        },
        "dflash_metadata": "PASS_NULL_DRAFT_TEMPERATURE_AND_GREEDY_ARGMAX",
        "official_precision": "PASS_BF16_SDPA_SDPA_TF32_FALSE",
        "same_tree_runtime_witness": "PASS_46_BY_151936_FP64",
        "source_hashes": {"status": "PASS", "files": source_files},
        "artifact_sha256": artifacts,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--repo-root", type=Path,
                        default=Path(__file__).resolve().parents[1])
    parser.add_argument("--engine-source", type=Path)
    parser.add_argument("--output", type=Path,
                        help="optionally save the validation JSON")
    args = parser.parse_args()
    rendered = json.dumps(validate(
        args.result_dir, args.repo_root, args.engine_source,
    ), indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
