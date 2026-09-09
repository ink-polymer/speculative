"""Official mean-decode-TPOT tables, with Adaptive/ablation columns added."""
from __future__ import annotations

import csv
import math

import torch

from .common import atomic_json, digest, file_hash, load_json
from .official_data import check_manifest
from .official_spec import BUDGETS, LIMITS, MODELS, UPSTREAM, load_source, verify_sources
from .official_worker import method_names, response_tokens
from .official_audit import validate_hardware
from .controller import COST_ATTRIBUTED_VARIANT


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
            or run.get("greedy_audit_policy", "strict")
               != completion.get("greedy_audit_policy", "strict")
            or len({r["_audit"]["index"] for r in run["responses"]}) != completion["cases"]):
        raise ValueError("Run response identity/count mismatch")
    return run


def validate_run_contract(run, source_lock, nproc, smoke_count, environment,
                          greedy_audit_policy="strict", diagnostic_variants=(),
                          wandb_settings=None, method_order_policy="official-fixed"):
    maximum = 32 if smoke_count else 2048
    args = run["args"]
    if (run["source_lock"] != source_lock or run["world_size"] != nproc
            or run["block_size"] != 16 or run["smoke"] != bool(smoke_count)
            or args["max_new_tokens"] != maximum or args["max_samples"] != LIMITS[args["dataset"]]
            or args["tree_budget"] != ",".join(map(str, BUDGETS))
            or args["flash_attn"] != (run["target_attn_implementation"] == "flash_attention_2")
            or run.get("greedy_audit_policy", "strict") != greedy_audit_policy
            or run.get("method_order_policy", "official-fixed") != method_order_policy
            or run.get("diagnostic_variants", []) != list(diagnostic_variants)
            or run.get("wandb") != wandb_settings
            or len(run["hardware"]) != nproc
            or {h["rank"] for h in run["hardware"]} != set(range(nproc))):
        raise ValueError("Run source, hardware or generation settings differ from contract")
    validate_hardware(run["hardware"], environment, nproc)


def validate_pair(sdpa, flash, dataset, model_index, variants, expected_rows,
                  greedy_audit_policy="strict"):
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
        method_positions = {method:[] for method in methods}
        for response in run["responses"]:
            audit = response["_audit"]
            key = audit["index"], audit["turn"]
            if (key in rows or type(audit.get("exact_match")) is not bool
                    or set(response) != {*methods, "_audit"}):
                raise ValueError("Duplicate, failed or incomplete response")
            reference = response_tokens(response["baseline"])
            order = audit.get("method_order", methods)
            if (not isinstance(order, list) or len(order) != len(methods)
                    or set(order) != set(methods)):
                raise ValueError("Invalid or incomplete method execution order")
            if run.get("method_order_policy", "official-fixed") == "official-fixed":
                if order != methods:
                    raise ValueError("Official fixed method order changed")
            elif run.get("method_order_policy") == "balanced-rotation":
                for position, method in enumerate(order):
                    method_positions[method].append(position)
            else:
                raise ValueError("Unknown stored method order policy")
            mismatching_methods = set()
            for method in methods:
                value = response[method]
                tokens = response_tokens(value)
                if tokens != reference:
                    mismatching_methods.add(method)
                if (not reference or len(tokens) != value.num_output_tokens
                        or not math.isfinite(value.time_per_output_token) or value.time_per_output_token <= 0
                        or not value.acceptance_lengths):
                    raise ValueError("Invalid official timing/acceptance artifact")
            if greedy_audit_policy == "strict":
                if mismatching_methods:
                    raise ValueError("Invalid official tokens/timing/acceptance")
                if audit["exact_match"] is not True:
                    raise ValueError("Greedy audit does not match stored response tokens")
            elif (greedy_audit_policy != "record-bf16-mismatches"
                    or audit["exact_match"] != (not mismatching_methods)
                    or set(audit.get("mismatching_methods", [])) != mismatching_methods
                    or audit.get("greedy_audit_policy") != greedy_audit_policy):
                raise ValueError("Invalid BF16 mismatch-recording audit")
            rows[key] = response
        if set(rows) != expected_keys:
            raise ValueError("Incomplete official sampled dataset or missing MT-Bench turn")
        if run.get("method_order_policy") == "balanced-rotation":
            for positions in method_positions.values():
                counts = [positions.count(position) for position in range(len(methods))]
                if max(counts) - min(counts) > 1:
                    raise ValueError("Method execution positions are not balanced")
        indexed.append(rows)
    for key in expected_keys:
        left, right = indexed[0][key], indexed[1][key]
        if left["_audit"]["input_sha256"] != right["_audit"]["input_sha256"]:
            raise ValueError("SDPA/FA2 inputs or greedy outputs differ; no lossless comparative table")
        if (greedy_audit_policy == "strict"
                and response_tokens(left["baseline"]) != response_tokens(right["baseline"])):
            raise ValueError("SDPA/FA2 inputs or greedy outputs differ; no lossless comparative table")
    total = sum(len(values) for values in indexed)
    exact = sum(response["_audit"]["exact_match"] for values in indexed for response in values.values())
    cross_backend = sum(response_tokens(indexed[0][key]["baseline"])
                        != response_tokens(indexed[1][key]["baseline"])
                        for key in expected_keys)
    return {"responses":total, "exact_responses":exact,
            "mismatching_responses":total-exact,
            "cross_backend_baseline_mismatches":cross_backend}


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


def controlled_sdpa_rows(sdpa, variants):
    """Fair architecture table with one target backend for every method."""
    verify_sources()
    table = load_source("_ddtree_pinned_controlled_table", UPSTREAM / "make_latex_table.py")
    base_tpot = table.mean_time_per_token(sdpa, "baseline")
    keys = [f"ddtree_tb{budget}" for budget in BUDGETS]
    best_ddtree = max(keys, key=lambda key: base_tpot / table.mean_time_per_token(sdpa, key))
    dd_tpot = table.mean_time_per_token(sdpa, best_ddtree)
    rows = []
    for label, key in ([('DFlash', 'dflash'), ('DDTree-best', best_ddtree)]
                       + [(key, key) for key in keys + list(variants)]):
        tpot = table.mean_time_per_token(sdpa, key)
        rows.append({"method":label, "selected_key":key,
            "mean_decode_tpot_seconds":tpot, "speedup_vs_target":base_tpot/tpot,
            "speedup_vs_best_ddtree":dd_tpot/tpot,
            "mean_acceptance_length":table.mean_acceptance_length(sdpa, key),
            "target_baseline_backend":"sdpa", "method_backend":"sdpa"})
    return rows


def diagnostic_budget_usage(run, method):
    """Summarize the per-response traces reset by ``adaptive_generate``."""
    candidates = run["diagnostic_controllers"][method]["budget_candidates"]
    budgets = []
    for response in run["responses"]:
        result = response[method]
        trace = getattr(result, "adaptive_decisions", None)
        if not isinstance(trace, list):
            raise ValueError(f"Missing AdaptiveTree decision trace for {method}")
        for observation in trace:
            decision = observation.get("decision") if isinstance(observation, dict) else None
            budget = decision.get("budget") if isinstance(decision, dict) else None
            if (type(budget) is not int or budget not in candidates
                    or observation.get("tree_nodes") not in (None, budget)):
                raise ValueError(f"Invalid AdaptiveTree budget decision for {method}")
            budgets.append(budget)
    if not budgets:
        raise ValueError(f"Empty AdaptiveTree decision trace for {method}")
    counts = {str(budget):budgets.count(budget) for budget in sorted(set(budgets))}
    maximum = run["diagnostic_controllers"][method]["maximum_draft_nodes"]
    above_128 = sum(budget > 128 for budget in budgets)
    at_cap = sum(budget == maximum for budget in budgets)
    return {"method":method,
            "candidate_budgets":candidates,
            "rounds":len(budgets), "selected_budget_counts":counts,
            "max_selected_budget":max(budgets),
            "rounds_above_128":above_128,
            "share_rounds_above_128":above_128/len(budgets),
            "rounds_at_candidate_cap":at_cap,
            "share_rounds_at_candidate_cap":at_cap/len(budgets)}


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
    audit_policy = metadata.get("greedy_audit_policy", "strict")
    method_order_policy = metadata.get("method_order_policy", "official-fixed")
    diagnostic_variants = tuple(metadata.get("diagnostic_variants", ()))
    variants = (*config["variants"], *diagnostic_variants)
    audit_stats = []
    budget_usage = []
    controlled_rows = []
    for dataset in datasets:
        expected = load_json(data_dir / f"{dataset}.json")
        if smoke_count:
            expected = expected[:smoke_count]
        for model_index in model_indices:
            runs = [load_completed(directory / (run_stem(dataset, model_index, backend)+".pt"), identity)
                    for backend in ("sdpa", "flash_attention_2")]
            for run in runs:
                validate_run_contract(run, source_lock, metadata["nproc_per_node"], smoke_count,
                                      environment, audit_policy, diagnostic_variants,
                                      metadata.get("wandb"), method_order_policy)
            audit_stats.append(validate_pair(*runs, dataset, model_index, variants,
                                             expected, audit_policy))
            for method in diagnostic_variants:
                budget_usage.append({"dataset":dataset, "model":MODELS[model_index][0],
                                     **diagnostic_budget_usage(runs[0], method)})
            for row in official_rows(*runs, variants):
                rows.append({"dataset":dataset, "model":MODELS[model_index][0],
                             "cases":len(expected), "turns":sum(len(r["turns"]) for r in expected), **row})
            for row in controlled_sdpa_rows(runs[0], variants):
                controlled_rows.append({"dataset":dataset, "model":MODELS[model_index][0],
                    "cases":len(expected), "turns":sum(len(r["turns"]) for r in expected), **row})
    diagnostic_protocol = ("ddtree_official_t0_diagnostic_cost_attribution"
        if diagnostic_variants == (COST_ATTRIBUTED_VARIANT,)
        else "ddtree_official_t0_diagnostic_extended_budget")
    report = {"protocol":(diagnostic_protocol if diagnostic_variants else "ddtree_official_t0"),
              "training":False, "full_split":False,
              "protocol_identity":identity, "environment_sha256":file_hash(directory / "environment.json"),
              "official_samples":not bool(smoke_count),
              "greedy_audit_policy":audit_policy,
              "method_order_policy":method_order_policy,
              "publication_gate_passed":(not diagnostic_variants and not bool(smoke_count)
                                           and audit_policy == "strict"
                                           and method_order_policy == "official-fixed"),
              "strict_lossless_claim_eligible":audit_policy == "strict",
              "controlled_comparison_gate_passed":(
                  not bool(smoke_count) and audit_policy == "strict"
                  and method_order_policy == "balanced-rotation"
                  and all(stat["mismatching_responses"] == 0 for stat in audit_stats)),
              "primary_fairness_table":"controlled_same_backend_rows",
              "numerical_audit":{"responses":sum(s["responses"] for s in audit_stats),
                  "exact_responses":sum(s["exact_responses"] for s in audit_stats),
                  "mismatching_responses":sum(s["mismatching_responses"] for s in audit_stats),
                  "cross_backend_baseline_mismatches":sum(s["cross_backend_baseline_mismatches"] for s in audit_stats)},
              "full_official_t0_model_dataset_matrix":(not diagnostic_variants
                  and model_indices==list(range(3)) and datasets==list(LIMITS)),
              "metric":"mean(per-response decode time/output tokens) ratio; excludes target prefill and first speculative draft",
              "baseline":"best mean-TPOT AR/DFlash backend independently; best DDTree budget, as upstream",
              "accuracy_scope":("exact agreement with official target-only baseline, not task grading or a BF16 mathematical guarantee"
                  if audit_policy == "strict" else
                  "BF16 token mismatches are retained and counted; speed rows are not a strict lossless claim"),
              "rows":rows, "controlled_same_backend_rows":controlled_rows}
    if diagnostic_variants:
        report["diagnostic_budget_usage"] = budget_usage
    atomic_json(directory / "tables.json", report)
    with (directory / "tables.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (directory / "tables_controlled_sdpa.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(controlled_rows[0]))
        writer.writeheader()
        writer.writerows(controlled_rows)
    lines = ["# DDTree 官方口径：T=0 AdaptiveTree", "",
             "解码 TPOT 均值之比；非全量数据，按官方 seed=0 抽样。无训练。", "",
             "| 模型 | 数据集 | 方法 | 相对 Target | 相对最佳 DDTree | 接受长度 |",
             "|---|---|---|---:|---:|---:|"]
    if smoke_count:
        lines[0] = "# SMOKE ONLY：不可用于论文"
    elif diagnostic_variants:
        lines[0] = "# DIAGNOSTIC：成本归因/扩展预算实验，不属于冻结官方矩阵"
    elif audit_policy != "strict":
        lines[0] = "# BF16 mismatch-recording benchmark：不可声称严格无损"
    lines += [f"| {r['model']} | {r['dataset']} | {r['method']} | {r['speedup_vs_target']:.4f}× | {r['speedup_vs_best_ddtree']:.4f}× | {r['mean_acceptance_length']:.3f} |" for r in rows]
    lines += ["", "## 公平主表：统一 Target SDPA 后端", "",
              "此表才用于 AdaptiveTree、DFlash 与 DDTree 的架构对比；最佳后端结果仅为辅助表。", "",
              "| 模型 | 数据集 | 方法 | 相对 SDPA Target | 相对最佳 DDTree | 接受长度 |",
              "|---|---|---|---:|---:|---:|"]
    lines += [f"| {r['model']} | {r['dataset']} | {r['method']} | {r['speedup_vs_target']:.4f}× | {r['speedup_vs_best_ddtree']:.4f}× | {r['mean_acceptance_length']:.3f} |" for r in controlled_rows]
    if budget_usage:
        lines += ["", "## 诊断预算使用", "",
                  "| 模型 | 数据集 | 方法 | 轮数 | >128 占比 | 上限占比 | 选择计数 |",
                  "|---|---|---|---:|---:|---:|---|"]
        lines += [f"| {item['model']} | {item['dataset']} | {item['method']} | "
                  f"{item['rounds']} | {item['share_rounds_above_128']:.2%} | "
                  f"{item['share_rounds_at_candidate_cap']:.2%} | "
                  f"{item['selected_budget_counts']} |" for item in budget_usage]
    (directory / "tables.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
