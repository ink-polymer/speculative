"""Paired official-data pilot for same-DDTree fast block verification."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import importlib.metadata
import math
from pathlib import Path
import platform

import numpy as np
import torch

from gbv_experiments.common import digest, file_hash, prompt_seed, write_json
from gbv_experiments.config import Variant, build_variants, load_config
from gbv_experiments.conversation import encode_messages, generate_conversation
from gbv_experiments.data import load_prepared, user_turns
from gbv_experiments.engine import load_models
from gbv_experiments.fairness import assert_architecture_only_pair
from gbv_experiments.runner import output_lock, scheduled_variants, stop_token_ids
from gbv_experiments.terminal_formal import allocation_gate, model_gate, telemetry


def select_pilot_rows(rows: list[dict], dataset_names: list[str],
                      per_dataset: int) -> list[dict]:
    selected = []
    for dataset in dataset_names:
        part = [row for row in rows if row["dataset"] == dataset]
        if len(part) < per_dataset:
            raise ValueError(f"{dataset} has fewer than {per_dataset} prepared rows")
        selected.extend(part[:per_dataset])
    return selected


def dataset_local_schedule(rows: list[dict]) -> tuple[list[int], dict[str, int]]:
    """Number rows independently inside each dataset for balanced rotation."""
    counts: dict[str, int] = {}
    ordinals = []
    for row in rows:
        dataset = row["dataset"]
        ordinals.append(counts.get(dataset, 0))
        counts[dataset] = counts.get(dataset, 0) + 1
    return ordinals, counts


def compact(result: dict, variant: str, row: dict, seed: int,
            sampling_seed: int, position: int, ordinal: int) -> dict:
    return {
        "variant": variant,
        "dataset": row["dataset"],
        "source_id": row["source_id"],
        "prompt_sha256": row["prompt_sha256"],
        "seed": seed,
        "sampling_seed": sampling_seed,
        "execution_position": position,
        "method_order_ordinal": ordinal,
        "generated_sha256": digest(result["generated_token_ids"]),
        "generated_tokens": result["generated_tokens"],
        "decode_tokens": result["decode_tokens"],
        "prefill_ms": result["prefill_ms"],
        "decode_ms": result["decode_ms"],
        "e2e_ms": result["e2e_ms"],
        "target_forward_calls": result["target_forward_calls"],
        "draft_forward_calls": result["draft_forward_calls"],
        "target_tokens_processed": result["target_tokens_processed"],
        "turn_count": result["turn_count"],
        "finish_reason": result["finish_reason"],
        "peak_allocated_bytes": result["peak_allocated_bytes"],
        "rounds": result["rounds"],
    }


def compare(rows: list[dict], candidate: str, baseline: str,
            dataset_names: list[str], bootstrap_samples: int) -> dict:
    lookup = {
        (row["variant"], row["dataset"], row["source_id"], row["seed"]): row
        for row in rows
    }
    source_logs: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        if row["variant"] != candidate:
            continue
        key = (baseline, row["dataset"], row["source_id"], row["seed"])
        reference = lookup[key]
        candidate_tpot = row["decode_ms"] / row["decode_tokens"]
        baseline_tpot = reference["decode_ms"] / reference["decode_tokens"]
        source_logs[row["dataset"]][row["source_id"]].append(
            math.log(baseline_tpot / candidate_tpot)
        )

    clustered = {
        dataset: np.asarray([
            np.mean(seed_logs) for seed_logs in sources.values()
        ], dtype=np.float64)
        for dataset, sources in source_logs.items()
    }
    if set(clustered) != set(dataset_names):
        raise RuntimeError("Comparison is missing a dataset")
    dataset_logs = {
        dataset: float(clustered[dataset].mean()) for dataset in dataset_names
    }
    rng = np.random.default_rng(20260914)
    draws = np.zeros(bootstrap_samples, dtype=np.float64)
    for dataset in dataset_names:
        values = clustered[dataset]
        draws += rng.choice(
            values, size=(bootstrap_samples, len(values)), replace=True,
        ).mean(1) / len(dataset_names)
    low, high = np.exp(np.quantile(draws, [.025, .975]))
    return {
        "candidate": candidate,
        "baseline": baseline,
        "metric": "equal_dataset_geomean_decode_speedup",
        "speedup": math.exp(sum(dataset_logs.values()) / len(dataset_logs)),
        "ci95": [float(low), float(high)],
        "per_dataset": {
            dataset: math.exp(dataset_logs[dataset]) for dataset in dataset_names
        },
        "source_clusters_per_dataset": {
            dataset: len(clustered[dataset]) for dataset in dataset_names
        },
        "seeds_per_source": len(next(iter(source_logs[dataset_names[0]].values()))),
        "bootstrap_samples": bootstrap_samples,
    }


def aggregate(rows: list[dict], variant: str) -> dict:
    selected = [row for row in rows if row["variant"] == variant]
    rounds = [round_ for row in selected for round_ in row["rounds"]]
    tokens = sum(row["decode_tokens"] for row in selected)
    elapsed = sum(row["decode_ms"] for row in selected)
    return {
        "records": len(selected),
        "decode_tokens": tokens,
        "decode_ms": elapsed,
        "tokens_per_second": 1000 * tokens / elapsed,
        "mean_accepted_draft_tokens_per_round": sum(
            round_["accepted_draft_tokens"] for round_ in rounds
        ) / len(rounds),
        "mean_committed_tokens_per_round": sum(
            round_["committed_tokens"] for round_ in rounds
        ) / len(rounds),
        "rounds": len(rounds),
    }


@torch.inference_mode()
def run(config_path: Path, data_dir: Path, output: Path, device: str,
        per_dataset: int) -> dict:
    if not 4 <= per_dataset <= 32:
        raise ValueError("per_dataset must be in 4..32")
    cfg = load_config(config_path)
    precision = {
        "dtype": cfg["model"].get("dtype"),
        "target_attention": cfg["model"].get("target_attention"),
        "draft_attention": cfg["model"].get("draft_attention"),
        "allow_tf32": cfg["model"].get("allow_tf32"),
    }
    if precision != {
        "dtype": "bfloat16", "target_attention": "sdpa",
        "draft_attention": "sdpa", "allow_tf32": False,
    }:
        raise ValueError(f"Official execution controls changed: {precision}")
    entries = build_variants(cfg)
    variants = {
        entry["variant"]["name"]: Variant(**entry["variant"])
        for entry in entries
    }
    expected_names = {"dflash", "ddtree", "tree_block_verification"}
    if set(variants) != expected_names:
        raise ValueError(f"Unexpected methods: {sorted(variants)}")
    expected_methods = {
        "dflash": "dflash",
        "ddtree": "ddtree",
        "tree_block_verification": "ddtree_fused_scan",
    }
    actual_methods = {name: variant.method for name, variant in variants.items()}
    if actual_methods != expected_methods:
        raise ValueError(
            "Official pilot variant names must retain their registered methods: "
            f"expected={expected_methods}, actual={actual_methods}"
        )
    fairness = assert_architecture_only_pair(
        variants["ddtree"], variants["tree_block_verification"], cfg["model"],
    )
    dflash_match = assert_architecture_only_pair(
        variants["ddtree"], variants["dflash"], cfg["model"],
    )
    data_manifest, all_rows = load_prepared(
        data_dir, cfg["datasets"], cfg["evaluation"],
    )
    data = select_pilot_rows(all_rows, cfg["datasets"], per_dataset)
    ordinals, dataset_counts = dataset_local_schedule(data)
    if set(dataset_counts.values()) != {per_dataset}:
        raise AssertionError("Pilot dataset balance failed")

    with output_lock(output):
        if (output / "report.json").exists() or (output / "rows.json").exists():
            raise ValueError("Output exists; use a fresh path")
        allocation_gate(device)
        torch.set_num_threads(8)
        manifest = {
            "kind": "paired_official_dataset_same_tree_block_pilot",
            "formal_complete": False,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "candidate_frozen_before_official_pilot": True,
            "official_precision": precision,
            "temperature": 1.0,
            "datasets": cfg["datasets"],
            "per_dataset": per_dataset,
            "seeds": cfg["seeds"],
            "max_new_tokens": cfg["max_new_tokens"],
            "expected_paired_groups": len(data) * len(cfg["seeds"]),
            "expected_records": len(data) * len(cfg["seeds"]) * len(entries),
            "variants": [entry["variant"] for entry in entries],
            "fairness": fairness,
            "dflash_control_match": dflash_match,
            "data_selection": {
                "policy": "first N rows of the pinned DDTree-count selection in each dataset",
                "source_ids_sha256": digest([
                    [row["dataset"], row["source_id"]] for row in data
                ]),
                "prompt_ids": [
                    [row["dataset"], row["source_id"], row["prompt_sha256"]]
                    for row in data
                ],
                "prepared_manifest_sha256": file_hash(data_dir / "manifest.json"),
                "prepared_coverage": data_manifest["coverage"],
            },
            "order_policy": cfg["method_order"],
            "success_rule": "candidate beats DDTree and DFlash and DDTree beats DFlash; every equal-dataset source-cluster bootstrap 95% CI lower bound > 1",
            "same_tree_contract": {
                "constructor": "probability_tree",
                "draft_nodes": 45,
                "target_rows": 46,
                "candidate_only_delta": "official all-row batched multinomial -> one persistent CUDA block scanning reached FP64 rows",
                "sampling_law": "identical ancestral categorical law with independently mapped random uniforms",
            },
            "limitations": [
                f"Performance pilot uses {per_dataset} pinned samples per dataset "
                "rather than every registered sample",
                f"Generation is capped at {cfg['max_new_tokens']} new tokens, so "
                "this is not the 2048-token full study",
                "Quality scoring is not claimed by this speed pilot",
                "H20 measurements must not be relabelled as H200 measurements",
            ],
            "source_sha256": {
                "engine": file_hash(Path(__file__).resolve().parents[1] / "src/gbv_experiments/engine.py"),
                "fused_verifier": file_hash(Path(__file__).resolve().parents[1] / "src/gbv_experiments/fused_tree_sampling.py"),
                "official_verifier": file_hash(Path(__file__).resolve().parents[1] / "src/gbv_experiments/sampling.py"),
                "script": file_hash(Path(__file__).resolve()),
                "config": file_hash(config_path),
            },
        }
        write_json(output / "manifest.json", manifest)
        engine, tokenizer = load_models(cfg["model"], device)
        model_gate(engine)
        properties = torch.cuda.get_device_properties(device)
        manifest["runtime"] = {
            "gpu": torch.cuda.get_device_name(device),
            "gpu_memory_bytes": properties.total_memory,
            "compute_capability": [properties.major, properties.minor],
            "driver_and_clocks_start": telemetry(device),
            "python": platform.python_version(),
            "versions": {name: importlib.metadata.version(name) for name in
                         ("torch", "transformers", "datasets",
                          "huggingface-hub", "numpy")},
        }
        stops = stop_token_ids(engine, tokenizer)
        warmup = encode_messages(
            tokenizer, [{"role": "user", "content": "Compute 1 + 1."}],
            cfg["model"], device,
        )
        for variant in variants.values():
            engine.generate(warmup, variant, cfg["warmup_tokens"], stops, seed=0)

        witness = {}
        witness_ids = encode_messages(
            tokenizer, [{"role": "user", "content": user_turns(data[0])[0]}],
            cfg["model"], device,
        )
        for name in ("ddtree", "tree_block_verification"):
            def observe(parents, tree_tokens, all_p, witness_name=name):
                if witness_name not in witness:
                    witness[witness_name] = (
                        list(parents), list(tree_tokens), all_p.detach().cpu().clone(),
                    )
            engine.generate(
                witness_ids, variants[name], 32, stops,
                seed=prompt_seed(cfg["seeds"][0], data[0]["dataset"],
                                 data[0]["source_id"]),
                tree_observer=observe,
            )
        left, right = witness["ddtree"], witness["tree_block_verification"]
        if left[:2] != right[:2] or not torch.equal(left[2], right[2]):
            raise RuntimeError("Same-tree runtime witness failed")
        manifest["same_tree_runtime_witness"] = {
            "passed": True,
            "tree_sha256": digest([left[0], left[1]]),
            "probability_shape": list(left[2].shape),
            "probability_dtype": str(left[2].dtype),
        }
        write_json(output / "manifest.json", manifest)

        rows = []
        for seed_index, seed in enumerate(cfg["seeds"]):
            for row_index, row in enumerate(data):
                sampling_seed = prompt_seed(seed, row["dataset"], row["source_id"])
                ordinal = seed_index * dataset_counts[row["dataset"]] + ordinals[row_index]
                order = scheduled_variants(
                    entries, cfg["method_order"], ordinal, sampling_seed,
                )
                allocation_gate(device)
                for position, entry in enumerate(order):
                    variant = Variant(**entry["variant"])
                    result = generate_conversation(
                        engine, tokenizer, row, variant,
                        cfg["max_new_tokens"], stops, sampling_seed,
                        cfg["model"], profile=False,
                    )
                    if variant.method in {"ddtree", "ddtree_fused_scan"} and any(
                            round_["tree_nodes"] > 45 for round_ in result["rounds"]):
                        raise RuntimeError(f"{variant.name} exceeded B=45")
                    rows.append(compact(
                        result, variant.name, row, seed, sampling_seed,
                        position, ordinal,
                    ))
                write_json(output / "rows.json", {
                    "primary_timing": False,
                    "complete": False,
                    "completed_groups": len(rows) // len(entries),
                    "rows": rows,
                })
                print(
                    f"official-pilot group={len(rows) // len(entries)}/"
                    f"{manifest['expected_paired_groups']} dataset={row['dataset']} "
                    f"source={row['source_id']} seed={seed}",
                    flush=True,
                )

        comparisons = {
            baseline: compare(
                rows, "tree_block_verification", baseline,
                cfg["datasets"], cfg["bootstrap_samples"],
            ) for baseline in ("ddtree", "dflash")
        }
        baseline_sanity = compare(
            rows, "ddtree", "dflash", cfg["datasets"],
            cfg["bootstrap_samples"],
        )
        passed = (
            all(value["ci95"][0] > 1 for value in comparisons.values())
            and baseline_sanity["ci95"][0] > 1
        )
        report = {
            "gate_passed": passed,
            "formal_complete": False,
            "comparisons": comparisons,
            "baseline_sanity": baseline_sanity,
            "aggregate": {
                name: aggregate(rows, name) for name in variants
            },
            "records": len(rows),
            "paired_groups": len(rows) // len(entries),
            "telemetry_end": telemetry(device),
            "decision": (
                "proceed_to_full_registered_suite" if passed
                else "stop_and_review_power_or_implementation"
            ),
        }
        write_json(output / "rows.json", {
            "primary_timing": False,
            "complete": True,
            "completed_groups": len(rows) // len(entries),
            "rows": rows,
        })
        write_json(output / "report.json", report)
        print(report, flush=True)
        return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=Path("configs/same_tree_block_qwen3_4b_pilot.json"),
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--per-dataset", type=int, default=8)
    args = parser.parse_args()
    run(
        args.config.resolve(), args.data_dir.resolve(), args.output.resolve(),
        args.device, args.per_dataset,
    )


if __name__ == "__main__":
    main()
