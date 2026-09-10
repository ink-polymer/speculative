"""Development gate for internal-only Target verification of a DDTree.

The candidate keeps the official DDTree proposal and ancestral sampling law.
It runs the first Target verification only on the ancestor-closed internal
subtree, then runs one cached single-token Target step iff the realized walk
reaches a leaf.  Synthetic prompts are used only for candidate selection.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import math
from pathlib import Path
import random

import numpy as np
import torch

from gbv_experiments.common import digest, file_hash, write_json
from gbv_experiments.config import Variant, load_config
from gbv_experiments.conversation import encode_messages
from gbv_experiments.engine import load_models
from gbv_experiments.fairness import (
    assert_architecture_only_pair,
    assert_official_dflash_control,
)
from gbv_experiments.runner import output_lock, stop_token_ids
from gbv_experiments.sampling import probabilities
from gbv_experiments.terminal_formal import allocation_gate, model_gate, telemetry


DEVELOPMENT_PROMPTS = (
    "Explain why binary search needs a monotone predicate and give a short example.",
    "A tank is three fifths full. After adding 48 liters it is nine tenths full. Find its capacity.",
    "Write Python code for stable deduplication of a list while preserving order.",
    "Prove that the product of three consecutive integers is divisible by six.",
    "Compare optimistic and pessimistic concurrency control in one paragraph.",
    "Solve x squared minus 11x plus 24 equals zero and verify both roots.",
)


def variants() -> list[Variant]:
    base = Variant(
        name="base", method="ddtree", paths=1, length=15,
        temperature=1.0, draft_temperature=1.0, tree_budget=45,
        probability_dtype="float64",
    )
    values = [
        replace(base, name="dflash", method="dflash", draft_temperature=None),
        replace(base, name="ddtree", method="ddtree"),
        replace(base, name="fused_scan", method="ddtree_fused_scan"),
        replace(base, name="deferred_leaf",
                method="ddtree_lazy_target_deferred_leaf"),
    ]
    for value in values:
        value.validate()
    return values


def balanced_order(prompt: int, repeat: int, names: list[str]) -> list[str]:
    order = list(names)
    random.Random(20260910 + prompt).shuffle(order)
    shift = repeat % len(order)
    return order[shift:] + order[:shift]


def compact(result: dict, prompt: int, repeat: int, position: int,
            variant: Variant) -> dict:
    rounds = result["rounds"]
    target_rows = [row.get("target_verified_rows") for row in rounds]
    full_rows = [row.get("full_target_tree_rows") for row in rounds]
    return {
        "prompt":prompt,
        "repeat":repeat,
        "position":position,
        "variant":variant.name,
        "generated_sha256":digest(result["generated_token_ids"]),
        "decode_tokens":result["decode_tokens"],
        "decode_ms":result["decode_ms"],
        "official_scope_time_per_output_token_ms":result[
            "official_scope_time_per_output_token_ms"
        ],
        "e2e_ms":result["e2e_ms"],
        "round_count":len(rounds),
        "mean_target_rows":(
            sum(target_rows) / len(target_rows)
            if target_rows and all(value is not None for value in target_rows)
            else None
        ),
        "mean_full_rows":(
            sum(full_rows) / len(full_rows)
            if full_rows and all(value is not None for value in full_rows)
            else None
        ),
        "leaf_forward_fraction":(
            sum(bool(row.get("lazy_leaf_target_forward")) for row in rounds)
            / len(rounds)
            if variant.method.startswith("ddtree_lazy_target") and rounds else None
        ),
        "prefetched_leaf_hit_fraction":(
            sum(bool(row.get("prefetched_leaf_hit")) for row in rounds)
            / len(rounds)
            if variant.method.startswith("ddtree_lazy_target") and rounds else None
        ),
    }


def compare(rows: list[dict], candidate: str, baseline: str, field: str,
            bootstrap: int = 20_000) -> dict:
    logs = []
    for prompt in sorted({row["prompt"] for row in rows}):
        selected = [row for row in rows if row["prompt"] == prompt]
        means = {
            name:sum(row[field] for row in selected if row["variant"] == name)
            / sum(row["variant"] == name for row in selected)
            for name in (candidate, baseline)
        }
        logs.append(math.log(means[baseline] / means[candidate]))
    values = np.asarray(logs, dtype=np.float64)
    rng = np.random.default_rng(20260910)
    draws = rng.choice(
        values, size=(bootstrap, len(values)), replace=True,
    ).mean(1)
    low, high = np.exp(np.quantile(draws, (0.025, 0.975)))
    return {
        "candidate":candidate,
        "baseline":baseline,
        "field":field,
        "speedup":math.exp(float(values.mean())),
        "ci95":[float(low), float(high)],
        "prompt_clusters":len(values),
        "bootstrap_samples":bootstrap,
    }


def aggregate(rows: list[dict], name: str) -> dict:
    selected = [row for row in rows if row["variant"] == name]
    tokens = sum(row["decode_tokens"] for row in selected)
    milliseconds = sum(row["decode_ms"] for row in selected)
    target_rows = [row["mean_target_rows"] for row in selected
                   if row["mean_target_rows"] is not None]
    full_rows = [row["mean_full_rows"] for row in selected
                 if row["mean_full_rows"] is not None]
    leaf_fractions = [row["leaf_forward_fraction"] for row in selected
                      if row["leaf_forward_fraction"] is not None]
    prefetch_fractions = [row["prefetched_leaf_hit_fraction"] for row in selected
                          if row["prefetched_leaf_hit_fraction"] is not None]
    return {
        "records":len(selected),
        "decode_tokens":tokens,
        "decode_tokens_per_second":1000 * tokens / milliseconds,
        "official_scope_mean_tpot_ms":sum(
            row["official_scope_time_per_output_token_ms"] for row in selected
        ) / len(selected),
        "mean_target_rows_per_round":sum(target_rows) / len(target_rows)
        if target_rows else None,
        "mean_full_rows_per_round":sum(full_rows) / len(full_rows)
        if full_rows else None,
        "mean_leaf_forward_fraction":sum(leaf_fractions) / len(leaf_fractions)
        if leaf_fractions else None,
        "mean_prefetched_leaf_hit_fraction":sum(prefetch_fractions)
        / len(prefetch_fractions) if prefetch_fractions else None,
    }


@torch.inference_mode()
def numerical_audit(engine, ids, full_variant, lazy_variant,
                    attempts: int = 8) -> dict:
    audits = []
    leaf_observed = False
    for offset in range(attempts):
        full = []
        lazy = []

        def observe_full(parents, tokens, p):
            if not full:
                full.append((
                    list(parents), list(tokens), p.detach().cpu().clone(),
                ))

        def observe_lazy(parents, tokens, internal_nodes, p, leaf, leaf_p):
            if not lazy:
                lazy.append((
                    list(parents), list(tokens), list(internal_nodes),
                    p.detach().cpu().clone(), leaf,
                    None if leaf_p is None else leaf_p.detach().cpu().clone(),
                ))

        seed = 20261000 + offset
        full_result = engine.generate(
            ids, full_variant, 2, [], seed=seed,
            tree_observer=observe_full,
        )
        engine.generate(
            ids, lazy_variant, 2, [], seed=seed,
            lazy_target_observer=observe_lazy,
        )
        parents, tokens, full_p = full[0]
        (lazy_parents, lazy_tokens, internal_nodes, internal_p,
         leaf, leaf_p) = lazy[0]
        topology_equal = (parents, tokens) == (lazy_parents, lazy_tokens)
        expected_internal = full_p.index_select(
            0, torch.tensor(internal_nodes)
        )
        internal_difference = (internal_p - expected_internal).abs()
        internal_tv = 0.5 * internal_difference.sum(-1)
        depths = [0] * len(parents)
        for node in range(1, len(parents)):
            depths[node] = depths[parents[node]] + 1
        anchor = full_result["generated_token_ids"][0]

        def ar_probability(node: int) -> torch.Tensor:
            path = []
            cursor = node
            while cursor:
                path.append(tokens[cursor - 1])
                cursor = parents[cursor]
            path.reverse()
            cache = engine.cache_factory()
            engine.target_forward(
                ids, cache, hidden=False, last_only=True,
            )
            ar_output = None
            for token in [anchor] + path:
                ar_output = engine.target_forward(
                    torch.tensor([[token]], device=engine.device),
                    cache, hidden=False, last_only=True,
                )
            return probabilities(
                ar_output.logits[0, -1], full_variant.temperature,
                torch.float64,
            ).detach().cpu()

        row_audit = []
        for row, node in enumerate(internal_nodes):
            row_audit.append({
                "node":node,
                "depth":depths[node],
                "total_variation":float(internal_tv[row]),
                "max_absolute_probability_error":float(
                    internal_difference[row].max()
                ),
                "full_top1_token":int(expected_internal[row].argmax()),
                "lazy_top1_token":int(internal_p[row].argmax()),
                "full_top1_probability":float(expected_internal[row].max()),
                "lazy_top1_probability":float(internal_p[row].max()),
            })
        row_audit.sort(key=lambda value: value["total_variation"], reverse=True)
        item = {
            "seed":seed,
            "topology_equal":topology_equal,
            "full_rows":len(parents),
            "internal_rows":len(internal_nodes),
            "internal_max_absolute_probability_error":float(
                internal_difference.max()
            ),
            "internal_max_total_variation":float(internal_tv.max()),
            "internal_top1_equal":bool(
                internal_p.argmax(-1).eq(expected_internal.argmax(-1)).all()
            ),
            "reached_leaf":leaf,
            "leaf_max_absolute_probability_error":None,
            "leaf_total_variation":None,
            "leaf_top1_equal":None,
            "leaf_vs_ar_total_variation":None,
            "full_leaf_vs_ar_total_variation":None,
            "leaf_vs_ar_top1_equal":None,
            "deferred_leaf":False,
            "largest_internal_row_differences":row_audit[:5],
            "ar_reference":[],
        }
        if offset == 0:
            audit_nodes = internal_nodes
            for node in audit_nodes:
                ar_p = ar_probability(node)
                internal_row = internal_nodes.index(node)
                full_row = expected_internal[internal_row]
                lazy_row = internal_p[internal_row]
                item["ar_reference"].append({
                    "node":node,
                    "depth":depths[node],
                    "full_vs_ar_total_variation":float(
                        0.5 * (full_row - ar_p).abs().sum()
                    ),
                    "lazy_vs_ar_total_variation":float(
                        0.5 * (lazy_row - ar_p).abs().sum()
                    ),
                    "full_vs_ar_top1_equal":bool(
                        full_row.argmax() == ar_p.argmax()
                    ),
                    "lazy_vs_ar_top1_equal":bool(
                        lazy_row.argmax() == ar_p.argmax()
                    ),
                })
        if leaf >= 0:
            leaf_observed = True
            if leaf_p is None:
                item["deferred_leaf"] = True
            else:
                leaf_difference = (leaf_p - full_p[leaf]).abs()
                leaf_ar_p = ar_probability(leaf)
                item.update({
                    "leaf_max_absolute_probability_error":float(
                        leaf_difference.max()
                    ),
                    "leaf_total_variation":float(0.5 * leaf_difference.sum()),
                    "leaf_top1_equal":bool(
                        leaf_p.argmax() == full_p[leaf].argmax()
                    ),
                    "leaf_vs_ar_total_variation":float(
                        0.5 * (leaf_p - leaf_ar_p).abs().sum()
                    ),
                    "full_leaf_vs_ar_total_variation":float(
                        0.5 * (full_p[leaf] - leaf_ar_p).abs().sum()
                    ),
                    "leaf_vs_ar_top1_equal":bool(
                        leaf_p.argmax() == leaf_ar_p.argmax()
                    ),
                })
        audits.append(item)
        if leaf_observed and offset >= 2:
            break
    max_internal_tv = max(row["internal_max_total_variation"] for row in audits)
    leaf_rows = [row for row in audits
                 if row["leaf_vs_ar_total_variation"] is not None]
    deferred_leaf_observed = any(row["deferred_leaf"] for row in audits)
    max_leaf_tv = max(
        (row["leaf_total_variation"] for row in leaf_rows), default=None,
    )
    max_leaf_ar_tv = max(
        (row["leaf_vs_ar_total_variation"] for row in leaf_rows),
        default=None,
    )
    ar_rows = audits[0]["ar_reference"]
    max_full_ar_tv = max(
        row["full_vs_ar_total_variation"] for row in ar_rows
    )
    max_lazy_ar_tv = max(
        row["lazy_vs_ar_total_variation"] for row in ar_rows
    )
    mean_full_ar_tv = sum(
        row["full_vs_ar_total_variation"] for row in ar_rows
    ) / len(ar_rows)
    mean_lazy_ar_tv = sum(
        row["lazy_vs_ar_total_variation"] for row in ar_rows
    ) / len(ar_rows)
    numerical_margin = 5e-3
    passed = bool(
        leaf_observed
        and (deferred_leaf_observed or bool(leaf_rows))
        and all(row["topology_equal"] for row in audits)
        and all(row["internal_top1_equal"] for row in audits)
        and len(ar_rows) == audits[0]["internal_rows"]
        and all(row["lazy_vs_ar_top1_equal"] for row in ar_rows)
        and all(row["leaf_vs_ar_top1_equal"] for row in leaf_rows)
        and all(
            row["lazy_vs_ar_total_variation"]
            <= row["full_vs_ar_total_variation"] + numerical_margin
            for row in ar_rows
        )
        and all(
            row["leaf_vs_ar_total_variation"]
            <= row["full_leaf_vs_ar_total_variation"] + numerical_margin
            for row in leaf_rows
        )
        and max_lazy_ar_tv <= max_full_ar_tv
        and mean_lazy_ar_tv <= mean_full_ar_tv
    )
    return {
        "passed":passed,
        "official_precision_preserved":"BF16 model/SDPA; FP64 probabilities",
        "predeclared_ar_noninferiority_margin":numerical_margin,
        "leaf_observed":leaf_observed,
        "deferred_leaf_observed":deferred_leaf_observed,
        "max_internal_total_variation":max_internal_tv,
        "max_leaf_total_variation":max_leaf_tv,
        "max_leaf_vs_ar_total_variation":max_leaf_ar_tv,
        "max_full_ddtree_vs_ar_total_variation":max_full_ar_tv,
        "max_lazy_target_vs_ar_total_variation":max_lazy_ar_tv,
        "mean_full_ddtree_vs_ar_total_variation":mean_full_ar_tv,
        "mean_lazy_target_vs_ar_total_variation":mean_lazy_ar_tv,
        "attempts":audits,
    }


@torch.inference_mode()
def run(config: Path, output: Path, device: str, tokens: int,
        repeats: int, prompt_count: int) -> dict:
    if not 32 <= tokens <= 256 or repeats < 3:
        raise ValueError("tokens must be 32..256 and repeats must be at least 3")
    if not 1 <= prompt_count <= len(DEVELOPMENT_PROMPTS):
        raise ValueError("prompt-count is outside the development prompt bank")
    cfg = load_config(config)
    expected_precision = {
        "dtype":"bfloat16", "target_attention":"sdpa",
        "draft_attention":"sdpa", "allow_tf32":False,
    }
    actual_precision = {
        key:cfg["model"].get(key) for key in expected_precision
    }
    if actual_precision != expected_precision:
        raise ValueError(f"Official precision controls changed: {actual_precision}")
    declared = variants()
    by_name = {variant.name:variant for variant in declared}
    fairness = {
        name:assert_architecture_only_pair(
            by_name["ddtree"], by_name[name], cfg["model"],
        )
        for name in ("deferred_leaf",)
    }
    dflash_fairness = assert_official_dflash_control(
        by_name["ddtree"], by_name["dflash"], cfg["model"],
    )

    with output_lock(output):
        if any(path.name != ".writer.lock" for path in output.iterdir()):
            raise ValueError("Output exists; use a fresh development directory")
        allocation_gate(device)
        engine, tokenizer = load_models(cfg["model"], device)
        model_gate(engine)
        stops = stop_token_ids(engine, tokenizer)
        prompts = DEVELOPMENT_PROMPTS[:prompt_count]
        encoded = [
            encode_messages(
                tokenizer, [{"role":"user", "content":prompt}],
                cfg["model"], device,
            )
            for prompt in prompts
        ]
        manifest = {
            "kind":"development_lazy_target_tree_gate",
            "formal_complete":False,
            "created_at_utc":datetime.now(timezone.utc).isoformat(),
            "official_precision":actual_precision,
            "temperature":1.0,
            "tokens":tokens,
            "repeats":repeats,
            "prompt_count":prompt_count,
            "prompt_policy":"synthetic development prompts; never formal data",
            "prompt_sha256":[digest(prompt) for prompt in prompts],
            "variants":[variant.to_dict() for variant in declared],
            "fairness":fairness,
            "dflash_fairness":dflash_fairness,
            "timing_primary":"vendored official unweighted response TPOT",
            "runtime":{
                "gpu":torch.cuda.get_device_name(device),
                "telemetry_start":telemetry(device),
            },
            "source_sha256":{
                "engine":file_hash(Path(__file__).resolve().parents[1]
                                   / "src/gbv_experiments/engine.py"),
                "sampling":file_hash(Path(__file__).resolve().parents[1]
                                     / "src/gbv_experiments/sampling.py"),
                "script":file_hash(Path(__file__).resolve()),
                "config":file_hash(config),
            },
        }
        write_json(output / "manifest.json", manifest)
        audits = {
            name:numerical_audit(
                engine, encoded[0], by_name["ddtree"], by_name[name],
            )
            for name in ("deferred_leaf",)
        }
        write_json(output / "numerical_audit.json", audits)

        for variant in declared:
            engine.generate(encoded[0], variant, 32, stops, seed=20260910)

        rows = []
        names = list(by_name)
        for prompt, ids in enumerate(encoded):
            for repeat in range(repeats):
                allocation_gate(device)
                for position, name in enumerate(
                        balanced_order(prompt, repeat, names)):
                    result = engine.generate(
                        ids, by_name[name], tokens, stops,
                        seed=20260910 + prompt,
                    )
                    row = compact(
                        result, prompt, repeat, position, by_name[name],
                    )
                    rows.append(row)
                    write_json(output / "rows.json", {
                        "primary_timing":True, "rows":rows,
                    })
                    print(
                        f"prompt={prompt + 1}/{len(encoded)} "
                        f"repeat={repeat + 1}/{repeats} method={name} "
                        f"official_tpot="
                        f"{row['official_scope_time_per_output_token_ms']:.6f}",
                        flush=True,
                    )

        official_field = "official_scope_time_per_output_token_ms"
        comparisons = {
            "deferred_leaf_vs_ddtree":compare(
                rows, "deferred_leaf", "ddtree", official_field,
            ),
            "deferred_leaf_vs_fused_scan":compare(
                rows, "deferred_leaf", "fused_scan", official_field,
            ),
            "deferred_leaf_vs_dflash":compare(
                rows, "deferred_leaf", "dflash", official_field,
            ),
            "ddtree_vs_dflash":compare(
                rows, "ddtree", "dflash", official_field,
            ),
        }
        aggregates = {name:aggregate(rows, name) for name in names}
        candidate_names = ("deferred_leaf",)
        row_savings = {
            name:aggregates[name]["mean_target_rows_per_round"]
            < aggregates[name]["mean_full_rows_per_round"]
            for name in candidate_names
        }
        candidate_gates = {
            name:bool(
                audits[name]["passed"] and row_savings[name]
                and comparisons[f"{name}_vs_ddtree"]["ci95"][0] > 1
                and comparisons[f"{name}_vs_fused_scan"]["ci95"][0] > 1
            )
            for name in candidate_names
        }
        eligible = [name for name in candidate_names if candidate_gates[name]]
        selected_candidate = max(
            eligible,
            key=lambda name:comparisons[f"{name}_vs_ddtree"]["speedup"],
            default=None,
        )
        gate_passed = bool(
            selected_candidate is not None
            and comparisons["ddtree_vs_dflash"]["ci95"][0] > 1
        )

        stage_profiles = {}
        for name in names:
            allocation_gate(device)
            result = engine.generate(
                encoded[0], by_name[name], min(tokens, 64), stops,
                seed=20260911, profile=True,
            )
            stage_profiles[name] = {
                "stages":result["stages"],
                "stage_profile":result["stage_profile"],
                "timing_contract":result["timing_contract"],
            }
        write_json(output / "stage_profiles.json", {
            "primary_timing":False,
            "diagnostic_only":True,
            "profiles":stage_profiles,
        })
        report = {
            "gate_passed":gate_passed,
            "formal_complete":False,
            "numerical_audit":audits,
            "comparisons":comparisons,
            "aggregate":aggregates,
            "row_savings":row_savings,
            "candidate_gates":candidate_gates,
            "selected_candidate":selected_candidate,
            "stage_profiles_file":"stage_profiles.json",
            "telemetry_end":telemetry(device),
            "decision":(
                "eligible for a fresh disjoint confirmation run"
                if gate_passed else "reject or redesign; do not enter formal matrix"
            ),
        }
        write_json(output / "report.json", report)
        print(report, flush=True)
        return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--prompt-count", type=int, default=3)
    args = parser.parse_args()
    run(
        args.config.resolve(), args.output.resolve(), args.device,
        args.tokens, args.repeats, args.prompt_count,
    )


if __name__ == "__main__":
    main()
