"""Official mean-decode-TPOT tables, with Adaptive/ablation columns added."""
from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import torch

from .common import atomic_json, digest, file_hash, load_json
from .official_data import check_manifest
from .official_spec import BUDGETS, LIMITS, MODELS, UPSTREAM, load_source, verify_sources
from .official_worker import method_names, response_tokens
from .official_audit import validate_hardware


def run_stem(dataset, model_index, backend):
    suffix = "sdpa" if backend == "sdpa" else "flash_attn"
    return f"{dataset}__model{model_index}__temp0.0__{suffix}"


def load_completed(path, identity):
    completion = load_json(path.with_suffix(".complete.json"))
    if completion["identity"] != identity or completion["sha256"] != file_hash(path):
        raise ValueError(f"Run hash/identity changed: {path}")
    # Only load artifacts written by this runner, after matching their completion
    # hash and immutable run contract. Never load downloaded datasets as pickle.
    run = torch.load(path, weights_only=False, map_location="cpu")
    if (run["protocol_identity"] != identity or len(run["responses"]) != completion["turns"]
            or run["methods"] != completion["methods"] or run["smoke"] != completion["smoke"]
            or len({r["_audit"]["index"] for r in run["responses"]}) != completion["cases"]):
        raise ValueError("Run response identity/count mismatch")
    return run


def validate_run_contract(run, source_lock, nproc, smoke_count, environment):
    maximum = 32 if smoke_count else 2048
    args = run["args"]
    if (run["source_lock"] != source_lock or run["world_size"] != nproc
            or run["block_size"] != 16 or run["smoke"] != bool(smoke_count)
            or args["max_new_tokens"] != maximum or args["max_samples"] != LIMITS[args["dataset"]]
            or args["tree_budget"] != ",".join(map(str, BUDGETS))
            or args["flash_attn"] != (run["target_attn_implementation"] == "flash_attention_2")
            or len(run["hardware"]) != nproc
            or {h["rank"] for h in run["hardware"]} != set(range(nproc))):
        raise ValueError("Run source, hardware or generation settings differ from contract")
    validate_hardware(run["hardware"], environment, nproc)


def validate_pair(sdpa, flash, dataset, model_index, variants, expected_rows):
    expected_keys = {(r["index"], t) for r in expected_rows for t in range(len(r["turns"]))}
    indexed = []
    for backend, run in (("sdpa", sdpa), ("flash_attention_2", flash)):
        if (run["target_attn_implementation"] != backend
                or run["draft_attn_implementation"] != "flash_attention_2"
                or run["args"]["model_name_or_path"] != MODELS[model_index][0]
                or run["args"]["draft_name_or_path"] != MODELS[model_index][1]
                or run["args"]["dataset"] != dataset or run["args"]["temperature"] != 0):
            raise ValueError("Wrong official model/dataset/backend run")
        methods = method_names(backend, variants)
        if run["methods"] != methods:
            raise ValueError("Missing official budget or ablation")
        rows = {}
        for response in run["responses"]:
            audit = response["_audit"]
            key = audit["index"], audit["turn"]
            if key in rows or audit["exact_match"] is not True or set(response) != {*methods, "_audit"}:
                raise ValueError("Duplicate, failed or incomplete response")
            reference = response_tokens(response["baseline"])
            for method in methods:
                value = response[method]
                if (response_tokens(value) != reference or not reference
                        or len(reference) != value.num_output_tokens
                        or not math.isfinite(value.time_per_output_token) or value.time_per_output_token <= 0
                        or not value.acceptance_lengths):
                    raise ValueError("Invalid official tokens/timing/acceptance")
            rows[key] = response
        if set(rows) != expected_keys:
            raise ValueError("Incomplete official sampled dataset or missing MT-Bench turn")
        indexed.append(rows)
    for key in expected_keys:
        left, right = indexed[0][key], indexed[1][key]
        if (left["_audit"]["input_sha256"] != right["_audit"]["input_sha256"]
                or response_tokens(left["baseline"]) != response_tokens(right["baseline"])):
            raise ValueError("SDPA/FA2 inputs or greedy outputs differ; no lossless comparative table")


def official_rows(sdpa, flash, variants):
    verify_sources()
    # Reuse the exact mean and backend-selection functions used by the authors.
    table = load_source("_ddtree_pinned_table", UPSTREAM / "make_latex_table.py")
    baseline = table.best_run_data(sdpa, flash, "baseline")
    base_tpot = table.mean_time_per_token(baseline, "baseline")
    dflash = table.best_run_data(sdpa, flash, "dflash")
    keys = [f"ddtree_tb{b}" for b in BUDGETS]
    best_ddtree = max(keys, key=lambda k: base_tpot / table.mean_time_per_token(sdpa, k))
    dd_tpot = table.mean_time_per_token(sdpa, best_ddtree)
    results = []
    for label, run, key in [("DFlash", dflash, "dflash"), ("DDTree-best", sdpa, best_ddtree)] + [
            (key, sdpa, key) for key in keys + list(variants)]:
        tpot = table.mean_time_per_token(run, key)
        results.append({"method":label, "selected_key":key,
            "mean_decode_tpot_seconds":tpot, "speedup_vs_target":base_tpot/tpot,
            "speedup_vs_best_ddtree":dd_tpot/tpot,
            "mean_acceptance_length":table.mean_acceptance_length(run, key),
            "target_baseline_backend":baseline["target_attn_implementation"],
            "method_backend":run["target_attn_implementation"]})
    return results


def summarize(directory, data_dir, config, identity, model_indices, datasets, smoke_count=0):
    recorded = load_json(directory / "contract.json")
    metadata = recorded["metadata"]
    if (recorded["identity"] != identity or digest(metadata) != identity
            or metadata["config"] != config or metadata["model_indices"] != model_indices
            or metadata["datasets"] != datasets or metadata["smoke_count"] != smoke_count
            or metadata["dataset_manifest"] != check_manifest(data_dir)
            or metadata["source_manifest"] != verify_sources()):
        raise ValueError("Summary inputs do not match the immutable run contract")
    environment = load_json(directory / "environment.json")
    if (not environment["cuda"] or not environment["gpu"]
            or environment["nproc_per_node"] != metadata["nproc_per_node"]):
        raise ValueError("Missing or inconsistent GPU environment record")
    source_lock = load_json(data_dir / "source_revisions.json")
    rows = []
    for dataset in datasets:
        expected = load_json(data_dir / f"{dataset}.json")
        if smoke_count:
            expected = expected[:smoke_count]
        for model_index in model_indices:
            runs = [load_completed(directory / (run_stem(dataset, model_index, backend)+".pt"), identity)
                    for backend in ("sdpa", "flash_attention_2")]
            for run in runs:
                validate_run_contract(run, source_lock, metadata["nproc_per_node"], smoke_count, environment)
            validate_pair(*runs, dataset, model_index, config["variants"], expected)
            for row in official_rows(*runs, config["variants"]):
                rows.append({"dataset":dataset, "model":MODELS[model_index][0],
                             "cases":len(expected), "turns":sum(len(r["turns"]) for r in expected), **row})
    report = {"protocol":"ddtree_official_t0", "training":False, "full_split":False,
              "protocol_identity":identity, "environment_sha256":file_hash(directory / "environment.json"),
              "official_samples":not bool(smoke_count), "publication_gate_passed":not bool(smoke_count),
              "full_official_t0_model_dataset_matrix":model_indices==list(range(3)) and datasets==list(LIMITS),
              "metric":"mean(per-response decode time/output tokens) ratio; excludes target prefill and first speculative draft",
              "baseline":"best mean-TPOT AR/DFlash backend independently; best DDTree budget, as upstream",
              "accuracy_scope":"exact agreement with official target-only baseline, not task grading or a BF16 mathematical guarantee",
              "rows":rows}
    atomic_json(directory / "tables.json", report)
    with (directory / "tables.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# DDTree 官方口径：T=0 AdaptiveTree", "",
             "解码 TPOT 均值之比；非全量数据，按官方 seed=0 抽样。无训练。", "",
             "| 模型 | 数据集 | 方法 | 相对 Target | 相对最佳 DDTree | 接受长度 |",
             "|---|---|---|---:|---:|---:|"]
    if smoke_count:
        lines[0] = "# SMOKE ONLY：不可用于论文"
    lines += [f"| {r['model']} | {r['dataset']} | {r['method']} | {r['speedup_vs_target']:.4f}× | {r['speedup_vs_best_ddtree']:.4f}× | {r['mean_acceptance_length']:.3f} |" for r in rows]
    (directory / "tables.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
