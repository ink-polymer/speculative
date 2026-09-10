"""Official mean-decode-TPOT tables, with Adaptive/ablation columns added."""
from __future__ import annotations

import csv
import math

import torch

from .common import (METHOD_SCHEMA_VERSION, PRIMARY_ADAPTIVE_METHOD, atomic_json,
                     code_identity, digest, file_hash, load_json)
from .official_data import check_manifest
from .official_spec import BUDGETS, LIMITS, MODELS, UPSTREAM, load_source, verify_sources
from .official_worker import method_names, response_tokens
from .official_audit import validate_hardware
from .controller import expected_official_controller_configs


def method_role(key):
    if key == PRIMARY_ADAPTIVE_METHOD:
        return "primary"
    if key == "adaptive_legacy":
        return "historical_control"
    if key.startswith("adaptive_"):
        return "ablation"
    return "baseline"


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
                          wandb_settings=None, method_order_policy="official-fixed",
                          deprecated_cli_aliases=()):
    maximum = 32 if smoke_count else 2048
    args = run["args"]
    if (run["source_lock"] != source_lock or run["world_size"] != nproc
            or run["block_size"] != 16 or run["smoke"] != bool(smoke_count)
            or args["max_new_tokens"] != maximum or args["max_samples"] != LIMITS[args["dataset"]]
            or args["tree_budget"] != ",".join(map(str, BUDGETS))
            or args["flash_attn"] != (run["target_attn_implementation"] == "flash_attention_2")
            or run.get("greedy_audit_policy", "strict") != greedy_audit_policy
            or run.get("method_order_policy", "official-fixed") != method_order_policy
            or diagnostic_variants
            or "diagnostic_variants" in run
            or "diagnostic_controllers" in run
            or run.get("method_schema_version") != METHOD_SCHEMA_VERSION
            or run.get("primary_adaptive_method") != PRIMARY_ADAPTIVE_METHOD
            or run.get("deprecated_cli_aliases", []) != list(deprecated_cli_aliases)
            or run.get("wandb") != wandb_settings
            or len(run["hardware"]) != nproc
            or {h["rank"] for h in run["hardware"]} != set(range(nproc))):
        raise ValueError("Run source, hardware or generation settings differ from contract")
    controller_configs = run.get("controller_configs")
    expected_controllers = (expected_official_controller_configs()
        if run["target_attn_implementation"] == "sdpa" else {})
    if controller_configs != expected_controllers:
        raise ValueError("Run AdaptiveTree method schema differs from contract")
    validate_hardware(run["hardware"], environment, nproc)


def validate_pair(sdpa, flash, dataset, model_index, variants, expected_rows,
                  greedy_audit_policy="strict"):
    expected_keys = {(r["index"], t) for r in expected_rows for t in range(len(r["turns"]))}
    expected_prompt_hashes = {
        (row["index"], turn): {
            "source_turns_sha256": digest(row["turns"]),
            "user_turn_sha256": digest(user_content),
        }
        for row in expected_rows
        for turn, user_content in enumerate(row["turns"])
    }
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
            if (key in rows or key not in expected_keys
                    or type(audit.get("exact_match")) is not bool
                    or set(response) != {*methods, "_audit"}):
                raise ValueError("Duplicate, failed or incomplete response")
            if ({field:audit.get(field) for field in expected_prompt_hashes[key]}
                    != expected_prompt_hashes[key]):
                raise ValueError("Official source prompt identity changed")
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
        history_method = f"ddtree_tb{BUDGETS[-1]}" if backend == "sdpa" else "dflash"
        for index, turn in expected_keys:
            expected_history = [
                response_tokens(rows[(index, previous)][history_method])
                for previous in range(turn)
            ]
            if rows[(index, turn)]["_audit"].get("conditioning_history_sha256") != digest(expected_history):
                raise ValueError("Official multi-turn conditioning history changed")
        if run.get("method_order_policy") == "balanced-rotation":
            for positions in method_positions.values():
                counts = [positions.count(position) for position in range(len(methods))]
                if max(counts) - min(counts) > 1:
                    raise ValueError("Method execution positions are not balanced")
        indexed.append(rows)
    cross_backend_inputs = 0
    for key in expected_keys:
        left, right = indexed[0][key], indexed[1][key]
        if left["_audit"]["input_sha256"] != right["_audit"]["input_sha256"]:
            cross_backend_inputs += 1
            explained_mt_continuation = (
                greedy_audit_policy == "record-bf16-mismatches"
                and dataset == "mt-bench"
                and key[1] > 0
                and left["_audit"]["conditioning_history_sha256"]
                    != right["_audit"]["conditioning_history_sha256"]
            )
            if not explained_mt_continuation:
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
            "cross_backend_input_mismatches":cross_backend_inputs,
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
            "method_role":method_role(key),
            "mean_decode_tpot_seconds":tpot, "speedup_vs_target":base_tpot/tpot,
            "speedup_vs_best_ddtree":dd_tpot/tpot,
            "mean_acceptance_length":table.mean_acceptance_length(run, key),
            **pairwise_exact_output_stats(run, key),
            "target_baseline_backend":baseline["target_attn_implementation"],
            "method_backend":run["target_attn_implementation"],
            "exact_output_reference_backend":run["target_attn_implementation"]})
    return results


def pairwise_exact_output_stats(run, key):
    total = len(run["responses"])
    exact = sum(
        response_tokens(response[key]) == response_tokens(response["baseline"])
        for response in run["responses"]
    )
    return {"responses":total, "exact_output_responses":exact,
            "output_divergence_responses":total - exact,
            "exact_output_rate":exact / total}


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
            "method_role":method_role(key),
            "mean_decode_tpot_seconds":tpot, "speedup_vs_target":base_tpot/tpot,
            "speedup_vs_best_ddtree":dd_tpot/tpot,
            "mean_acceptance_length":table.mean_acceptance_length(sdpa, key),
            **pairwise_exact_output_stats(sdpa, key),
            "target_baseline_backend":"sdpa", "method_backend":"sdpa"})
    return rows


def controlled_exact_output_subset_rows(sdpa, variants):
    """Pairwise content-matched timing diagnostic; never a population estimate."""
    keys = [f"ddtree_tb{budget}" for budget in BUDGETS]
    rows = []
    for label, key in ([('DFlash', 'dflash')]
                       + [(key, key) for key in keys + list(variants)]):
        matched = [response for response in sdpa["responses"]
                   if response_tokens(response[key]) == response_tokens(response["baseline"])]
        baseline_mean = (sum(response["baseline"].time_per_output_token
                             for response in matched) / len(matched) if matched else None)
        method_mean = (sum(response[key].time_per_output_token
                           for response in matched) / len(matched) if matched else None)
        rows.append({"method":label, "selected_key":key,
            "method_role":method_role(key),
            **pairwise_exact_output_stats(sdpa, key),
            "baseline_mean_decode_tpot_seconds":baseline_mean,
            "method_mean_decode_tpot_seconds":method_mean,
            "exact_subset_speedup_vs_target":(
                baseline_mean / method_mean if matched else None
            ),
            "target_baseline_backend":"sdpa", "method_backend":"sdpa",
            "estimand":"post_hoc_pairwise_exact_output_subset",
            "selection_bias_warning":True})
    return rows


def adaptive_budget_usage(run, method):
    """Summarize the per-response traces reset by ``adaptive_generate``."""
    # The fallback keeps pre-migration artifacts inspectable with an explicit
    # historical field; new contracts never emit diagnostic_controllers.
    controllers = run.get("controller_configs", run.get("diagnostic_controllers", {}))
    candidates = controllers[method]["budget_candidates"]
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
    maximum = controllers[method]["maximum_draft_nodes"]
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


# Public compatibility name for scripts that only inspect old diagnostic runs.
diagnostic_budget_usage = adaptive_budget_usage


def summarize(directory, data_dir, config, identity, model_indices, datasets, smoke_count=0):
    recorded = load_json(directory / "contract.json")
    metadata = recorded["metadata"]
    if (recorded["identity"] != identity or digest(metadata) != identity
            or metadata["config"] != config or metadata["model_indices"] != model_indices
            or metadata["datasets"] != datasets or metadata["smoke_count"] != smoke_count
            or metadata.get("code_identity") != code_identity()
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
    if (metadata.get("method_schema_version") != METHOD_SCHEMA_VERSION
            or metadata.get("primary_adaptive_method") != PRIMARY_ADAPTIVE_METHOD
            or "diagnostic_variants" in metadata):
        raise ValueError("Summary requires the canonical AdaptiveTree method schema")
    deprecated_cli_aliases = tuple(metadata.get("deprecated_cli_aliases", ()))
    variants = tuple(config["variants"])
    audit_stats = []
    budget_usage = []
    input_artifacts = []
    controlled_rows = []
    controlled_exact_rows = []
    for dataset in datasets:
        expected = load_json(data_dir / f"{dataset}.json")
        if smoke_count:
            expected = expected[:smoke_count]
        for model_index in model_indices:
            runs = []
            for backend in ("sdpa", "flash_attention_2"):
                path = directory / (run_stem(dataset, model_index, backend) + ".pt")
                marker_path = path.with_suffix(".complete.json")
                run = load_completed(path, identity)
                marker = load_json(marker_path)
                input_artifacts.append({
                    "dataset":dataset,
                    "model_index":model_index,
                    "model":MODELS[model_index][0],
                    "backend":backend,
                    "artifact":path.name,
                    "artifact_sha256":marker["sha256"],
                    "completion_marker":marker_path.name,
                    "completion_marker_sha256":file_hash(marker_path),
                    "protocol_identity":marker["identity"],
                })
                runs.append(run)
            for run in runs:
                validate_run_contract(run, source_lock, metadata["nproc_per_node"], smoke_count,
                                      environment, audit_policy, (),
                                      metadata.get("wandb"), method_order_policy,
                                      deprecated_cli_aliases)
            pair_stats = validate_pair(*runs, dataset, model_index, variants,
                                       expected, audit_policy)
            audit_stats.append(pair_stats)
            for method in variants:
                budget_usage.append({"dataset":dataset, "model":MODELS[model_index][0],
                                     **adaptive_budget_usage(runs[0], method)})
            for row in official_rows(*runs, variants):
                rows.append({"dataset":dataset, "model":MODELS[model_index][0],
                             "cases":len(expected), "turns":sum(len(r["turns"]) for r in expected),
                             "comparison_scope":"upstream_best_backend_auxiliary",
                             "cross_backend_inputs_identical":(
                                 pair_stats["cross_backend_input_mismatches"] == 0),
                             **row})
            for row in controlled_sdpa_rows(runs[0], variants):
                controlled_rows.append({"dataset":dataset, "model":MODELS[model_index][0],
                    "cases":len(expected), "turns":sum(len(r["turns"]) for r in expected), **row})
            for row in controlled_exact_output_subset_rows(runs[0], variants):
                controlled_exact_rows.append({"dataset":dataset,
                    "model":MODELS[model_index][0], "cases":len(expected),
                    "turns":sum(len(r["turns"]) for r in expected), **row})
    report = {"protocol":"ddtree_official_t0",
              "training":False, "full_split":False,
              "method_schema_version":METHOD_SCHEMA_VERSION,
              "primary_adaptive_method":PRIMARY_ADAPTIVE_METHOD,
              "protocol_identity":identity, "environment_sha256":file_hash(directory / "environment.json"),
              "dataset_manifest":metadata["dataset_manifest"],
              "source_lock":source_lock,
              "input_artifacts":input_artifacts,
              "official_samples":not bool(smoke_count),
              "greedy_audit_policy":audit_policy,
              "method_order_policy":method_order_policy,
              "publication_gate_passed":(not bool(smoke_count)
                                           and audit_policy == "strict"
                                           and method_order_policy == "official-fixed"),
              "strict_lossless_claim_eligible":audit_policy == "strict",
              "controlled_protocol_gate_passed":(
                  not bool(smoke_count)
                  and method_order_policy == "balanced-rotation"
              ),
              "controlled_comparison_gate_passed":(
                  not bool(smoke_count) and audit_policy == "strict"
                  and method_order_policy == "balanced-rotation"
                  and all(stat["mismatching_responses"] == 0 for stat in audit_stats)),
              "primary_fairness_table":"controlled_same_backend_rows",
              "cross_backend_auxiliary_table_warning":(
                  "MT-Bench continuation rows with different recorded SDPA/FA2 histories are not "
                  "same-context pairs; the best-backend table is auxiliary. The controlled SDPA "
                  "table remains the architecture-comparison table."
                  if any(s["cross_backend_input_mismatches"] for s in audit_stats) else None
              ),
              "numerical_audit":{"responses":sum(s["responses"] for s in audit_stats),
                  "exact_responses":sum(s["exact_responses"] for s in audit_stats),
                  "mismatching_responses":sum(s["mismatching_responses"] for s in audit_stats),
                  "cross_backend_input_mismatches":sum(
                      s["cross_backend_input_mismatches"] for s in audit_stats),
                  "cross_backend_baseline_mismatches":sum(s["cross_backend_baseline_mismatches"] for s in audit_stats)},
              "full_official_t0_model_dataset_matrix":(
                  model_indices==list(range(3)) and datasets==list(LIMITS)),
              "metric":"mean(per-response decode time/output tokens) ratio; excludes target prefill and first speculative draft",
              "baseline":"best mean-TPOT AR/DFlash backend independently; best DDTree budget, as upstream",
              "accuracy_scope":("exact agreement with official target-only baseline, not task grading or a BF16 mathematical guarantee"
                  if audit_policy == "strict" else
                  "token divergences observed during BF16 execution are retained and counted; "
                  "their cause is not inferred, and speed rows are not a strict lossless claim"),
              "rows":rows, "controlled_same_backend_rows":controlled_rows,
              "primary_rows":[row for row in controlled_rows
                              if row["method_role"] == "primary"],
              "ablation_rows":[row for row in controlled_rows
                               if row["method_role"] in {"ablation", "historical_control"}],
              "controlled_exact_output_subset_rows":controlled_exact_rows,
              "exact_output_subset_warning":(
                  "Post-hoc content-matched rows are selection-biased diagnostics, not "
                  "population speedup estimates."
              )}
    report["adaptive_budget_usage"] = budget_usage
    atomic_json(directory / "tables.json", report)
    with (directory / "tables.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (directory / "tables_controlled_sdpa.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(controlled_rows[0]))
        writer.writeheader()
        writer.writerows(controlled_rows)
    with (directory / "tables_exact_output_subset_diagnostic.csv").open(
            "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(controlled_exact_rows[0]))
        writer.writeheader()
        writer.writerows(controlled_exact_rows)
    lines = ["# DDTree 官方口径：T=0 AdaptiveTree", "",
             "解码 TPOT 均值之比；非全量数据，按官方 seed=0 抽样。无训练。", "",
             "| 模型 | 数据集 | 方法 | 相对 Target | 相对最佳 DDTree | 接受长度 |",
             "|---|---|---|---:|---:|---:|"]
    if smoke_count:
        lines[0] = "# SMOKE ONLY：不可用于论文"
    elif audit_policy != "strict":
        lines[0] = "# BF16 执行中观察到输出差异的记录型 benchmark：不可声称严格无损"
    if report["cross_backend_auxiliary_table_warning"]:
        lines[3:3] = [
            "警告：MT-Bench 后续轮存在已记录的跨后端上下文差异；下方最佳后端表不是同上下文配对。",
            "公平架构比较请使用统一 SDPA 主表。",
            "",
        ]
    lines += [f"| {r['model']} | {r['dataset']} | {r['method']} | {r['speedup_vs_target']:.4f}× | {r['speedup_vs_best_ddtree']:.4f}× | {r['mean_acceptance_length']:.3f} |" for r in rows]
    lines += ["", "## 公平主表：统一 Target SDPA 后端", "",
              "此表才用于 AdaptiveTree、DFlash 与 DDTree 的架构对比；最佳后端结果仅为辅助表。", "",
              "| 模型 | 数据集 | 方法 | 相对 SDPA Target | 相对最佳 DDTree | 接受长度 |",
              "|---|---|---|---:|---:|---:|"]
    lines += [f"| {r['model']} | {r['dataset']} | {r['method']} | {r['speedup_vs_target']:.4f}× | {r['speedup_vs_best_ddtree']:.4f}× | {r['mean_acceptance_length']:.3f} |" for r in controlled_rows]
    lines += ["", "## 完全相同输出配对子集（选择偏差诊断）", "",
              "这不是总体加速比估计，只用于隔离输出内容差异。", "",
              "| 模型 | 数据集 | 方法 | 相同输出数/总数 | 子集相对 Target |",
              "|---|---|---|---:|---:|"]
    def fmt(value):
        return "--" if value is None else f"{value:.4f}×"
    lines += [f"| {r['model']} | {r['dataset']} | {r['method']} | "
              f"{r['exact_output_responses']}/{r['responses']} | "
              f"{fmt(r['exact_subset_speedup_vs_target'])} |" for r in controlled_exact_rows]
    if budget_usage:
        lines += ["", "## AdaptiveTree 预算使用", "",
                  "| 模型 | 数据集 | 方法 | 轮数 | >128 占比 | 上限占比 | 选择计数 |",
                  "|---|---|---|---:|---:|---:|---|"]
        lines += [f"| {item['model']} | {item['dataset']} | {item['method']} | "
                  f"{item['rounds']} | {item['share_rounds_above_128']:.2%} | "
                  f"{item['share_rounds_at_candidate_cap']:.2%} | "
                  f"{item['selected_budget_counts']} |" for item in budget_usage]
    (directory / "tables.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
