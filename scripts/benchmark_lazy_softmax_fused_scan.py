"""Development-only gate for lazy-softmax fused DDTree block decoding.

This benchmark deliberately uses a synthetic prompt bank rather than any row in
the registered formal datasets.  It compares the new candidate, the frozen
fused-scan candidate, official DDTree, and official greedy-Draft DFlash inside
one process with balanced method order.  Passing this gate authorizes a fresh
confirmation run; it never upgrades these development prompts to formal data.
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
from gbv_experiments.fairness import (assert_architecture_only_pair,
                                      assert_official_dflash_control)
from gbv_experiments.runner import output_lock, stop_token_ids
from gbv_experiments.terminal_formal import allocation_gate, model_gate, telemetry


DEVELOPMENT_PROMPTS = (
    "Explain why binary search needs a monotone predicate and give a short example.",
    "A tank is three fifths full. After adding 48 liters it is nine tenths full. Find its capacity.",
    "Write Python code for stable deduplication of a list while preserving order.",
    "Prove that the product of three consecutive integers is divisible by six.",
    "Compare optimistic and pessimistic concurrency control in one paragraph.",
    "Solve x squared minus 11x plus 24 equals zero and verify both roots.",
    "Give pseudocode for topological sorting and state how a cycle is detected.",
    "Rewrite this sentence concisely: Due to the fact that the cache was empty, recomputation was required.",
    "Design three edge cases for a function that parses signed decimal integers.",
    "A train covers 210 km at one speed and 180 km at a speed 15 km/h slower in equal times. Find both speeds.",
    "Explain the loop invariant of insertion sort without using more than five sentences.",
    "Write SQL to return customers who placed orders in every month of 2025.",
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
        replace(base, name="lazy_softmax_fused_scan",
                method="ddtree_lazy_softmax_fused_scan"),
        replace(base, name="lazy_projection_fused_scan",
                method="ddtree_lazy_projection_fused_scan"),
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
    projected = [row.get("posterior_probability_rows") for row in rounds]
    return {
        "prompt": prompt,
        "repeat": repeat,
        "position": position,
        "variant": variant.name,
        "generated_sha256": digest(result["generated_token_ids"]),
        "decode_tokens": result["decode_tokens"],
        "decode_ms": result["decode_ms"],
        "e2e_ms": result["e2e_ms"],
        "round_count": len(rounds),
        "mean_projected_rows": (
            sum(projected) / len(projected)
            if projected and all(value is not None for value in projected)
            else None
        ),
        "mean_full_rows": (
            sum(row["full_vocabulary_projection_rows"] for row in rounds)
            / len(rounds)
            if projected and all(value is not None for value in projected)
            else None
        ),
    }


def compare(rows: list[dict], candidate: str, baseline: str,
            bootstrap: int = 20_000) -> dict:
    logs = []
    for prompt in sorted({row["prompt"] for row in rows}):
        prompt_rows = [row for row in rows if row["prompt"] == prompt]
        tpot = {}
        for name in (candidate, baseline):
            selected = [row for row in prompt_rows if row["variant"] == name]
            tpot[name] = sum(row["decode_ms"] for row in selected) / sum(
                row["decode_tokens"] for row in selected
            )
        logs.append(math.log(tpot[baseline] / tpot[candidate]))
    values = np.asarray(logs, dtype=np.float64)
    rng = np.random.default_rng(20260910)
    draws = rng.choice(
        values, size=(bootstrap, len(values)), replace=True,
    ).mean(axis=1)
    low, high = np.exp(np.quantile(draws, (0.025, 0.975)))
    return {
        "candidate": candidate,
        "baseline": baseline,
        "metric": "equal-prompt geometric-mean decode speedup",
        "speedup": math.exp(float(values.mean())),
        "ci95": [float(low), float(high)],
        "prompt_clusters": len(values),
        "bootstrap_samples": bootstrap,
    }


def aggregate(rows: list[dict], name: str) -> dict:
    selected = [row for row in rows if row["variant"] == name]
    tokens = sum(row["decode_tokens"] for row in selected)
    milliseconds = sum(row["decode_ms"] for row in selected)
    projected = [row["mean_projected_rows"] for row in selected
                 if row["mean_projected_rows"] is not None]
    full = [row["mean_full_rows"] for row in selected
            if row["mean_full_rows"] is not None]
    return {
        "records": len(selected),
        "decode_tokens": tokens,
        "decode_ms": milliseconds,
        "tokens_per_second": 1000 * tokens / milliseconds,
        "mean_projected_rows_per_round": (
            sum(projected) / len(projected) if projected else None
        ),
        "mean_full_rows_per_round": sum(full) / len(full) if full else None,
    }


@torch.inference_mode()
def run(config: Path, output: Path, device: str, tokens: int,
        repeats: int, prompt_count: int | None = None) -> dict:
    if not 64 <= tokens <= 256 or repeats < 5 or repeats % 5:
        raise ValueError("tokens must be 64..256; repeats must be a multiple of five")
    prompt_count = len(DEVELOPMENT_PROMPTS) if prompt_count is None else prompt_count
    if not 1 <= prompt_count <= len(DEVELOPMENT_PROMPTS):
        raise ValueError(
            f"prompt-count must be in 1..{len(DEVELOPMENT_PROMPTS)}"
        )
    selected_prompts = DEVELOPMENT_PROMPTS[:prompt_count]
    cfg = load_config(config)
    expected_precision = {
        "dtype": "bfloat16", "target_attention": "sdpa",
        "draft_attention": "sdpa", "allow_tf32": False,
    }
    actual_precision = {
        key: cfg["model"].get(key) for key in expected_precision
    }
    if actual_precision != expected_precision:
        raise ValueError(
            f"Official precision/backend controls changed: {actual_precision}"
        )
    declared = variants()
    by_name = {value.name: value for value in declared}
    fairness = {
        name: assert_architecture_only_pair(
            by_name["ddtree"], by_name[name], cfg["model"],
        )
        for name in ("fused_scan", "lazy_softmax_fused_scan",
                     "lazy_projection_fused_scan")
    }
    dflash_fairness = assert_official_dflash_control(
        by_name["ddtree"], by_name["dflash"], cfg["model"],
    )

    with output_lock(output):
        if ((output / "manifest.json").exists()
                or (output / "rows.json").exists()
                or (output / "report.json").exists()):
            raise ValueError("Output exists; choose a fresh development directory")
        allocation_gate(device)
        manifest = {
            "kind": "development_lazy_softmax_fused_scan_gate",
            "formal_complete": False,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "temperature": 1.0,
            "tokens": tokens,
            "repeats": repeats,
            "prompt_count": prompt_count,
            "variants": [value.to_dict() for value in declared],
            "official_precision": actual_precision,
            "fairness": fairness,
            "dflash_fairness": dflash_fairness,
            "prompt_policy": "synthetic development prompts; never formal or held out",
            "prompt_sha256": [digest(prompt) for prompt in selected_prompts],
            "success_rule": (
                "an eligible candidate must have DDTree CI low > 1.03, fused-scan "
                "CI low > 1.01, DFlash CI low > 1, and fewer probability rows; "
                "DDTree/DFlash CI low must exceed 1. Freeze the eligible candidate "
                "with the larger DDTree CI lower bound for fresh confirmation"
            ),
            "source_sha256": {
                "engine": file_hash(Path(__file__).resolve().parents[1]
                                    / "src/gbv_experiments/engine.py"),
                "candidate": file_hash(Path(__file__).resolve().parents[1]
                                       / "src/gbv_experiments/fused_tree_sampling.py"),
                "script": file_hash(Path(__file__).resolve()),
                "config": file_hash(config),
            },
        }
        write_json(output / "manifest.json", manifest)
        engine, tokenizer = load_models(cfg["model"], device)
        model_gate(engine)
        stops = stop_token_ids(engine, tokenizer)
        encoded = [encode_messages(
            tokenizer, [{"role": "user", "content": prompt}],
            cfg["model"], device,
        ) for prompt in selected_prompts]
        manifest["runtime"] = {
            "gpu": torch.cuda.get_device_name(device),
            "telemetry_start": telemetry(device),
        }
        write_json(output / "manifest.json", manifest)

        for value in declared:
            engine.generate(encoded[0], value, 32, stops, seed=20260910)

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
                        "primary_timing": False, "rows": rows,
                    })
                    print(
                        f"dev prompt={prompt + 1}/{len(encoded)} "
                        f"repeat={repeat + 1}/{repeats} method={name} "
                        f"tpot={row['decode_ms'] / row['decode_tokens']:.6f}",
                        flush=True,
                    )

        comparisons = {
            "candidate_vs_ddtree": compare(
                rows, "lazy_softmax_fused_scan", "ddtree",
            ),
            "candidate_vs_fused_scan": compare(
                rows, "lazy_softmax_fused_scan", "fused_scan",
            ),
            "candidate_vs_dflash": compare(
                rows, "lazy_softmax_fused_scan", "dflash",
            ),
            "projection_candidate_vs_ddtree": compare(
                rows, "lazy_projection_fused_scan", "ddtree",
            ),
            "projection_candidate_vs_fused_scan": compare(
                rows, "lazy_projection_fused_scan", "fused_scan",
            ),
            "projection_candidate_vs_dflash": compare(
                rows, "lazy_projection_fused_scan", "dflash",
            ),
            "ddtree_vs_dflash": compare(rows, "ddtree", "dflash"),
        }
        first = {
            (row["prompt"], row["variant"]): row["generated_sha256"]
            for row in rows if row["repeat"] == 0
        }
        repeat_equal = all(
            row["generated_sha256"] == first[row["prompt"], row["variant"]]
            for row in rows
        )
        aggregates = {name: aggregate(rows, name) for name in names}
        candidate_specs = {
            "lazy_softmax_fused_scan": (
                "candidate_vs_ddtree", "candidate_vs_fused_scan",
                "candidate_vs_dflash",
            ),
            "lazy_projection_fused_scan": (
                "projection_candidate_vs_ddtree",
                "projection_candidate_vs_fused_scan",
                "projection_candidate_vs_dflash",
            ),
        }
        candidate_gates = {}
        for name, comparison_names in candidate_specs.items():
            candidate_gates[name] = bool(
                comparisons[comparison_names[0]]["ci95"][0] > 1.03
                and comparisons[comparison_names[1]]["ci95"][0] > 1.01
                and comparisons[comparison_names[2]]["ci95"][0] > 1
                and aggregates[name]["mean_projected_rows_per_round"]
                < aggregates[name]["mean_full_rows_per_round"]
            )
        eligible = [name for name, passed in candidate_gates.items() if passed]
        selected_candidate = max(
            eligible,
            key=lambda name: comparisons[candidate_specs[name][0]]["ci95"][0],
            default=None,
        )
        quick_ranking = sorted(
            candidate_specs,
            key=lambda name: comparisons[candidate_specs[name][0]]["speedup"],
            reverse=True,
        )
        report = {
            "gate_passed": bool(
                repeat_equal
                and comparisons["ddtree_vs_dflash"]["ci95"][0] > 1
                and selected_candidate is not None
            ),
            "formal_complete": False,
            "within_method_repeat_equal": repeat_equal,
            "comparisons": comparisons,
            "candidate_gates": candidate_gates,
            "selected_candidate_for_fresh_confirmation": selected_candidate,
            "quick_selection_metric": "point estimate versus DDTree",
            "quick_candidate_ranking": quick_ranking,
            "quick_selected_candidate": quick_ranking[0],
            "quick_selection_is_formal_evidence": False,
            "aggregate": aggregates,
            "telemetry_end": telemetry(device),
            "next_step": "fresh disjoint confirmation run only if this gate passes",
        }
        write_json(output / "report.json", report)
        print(report, flush=True)
        return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--prompt-count", type=int, default=len(DEVELOPMENT_PROMPTS),
    )
    args = parser.parse_args()
    run(args.config.resolve(), args.output.resolve(), args.device,
        args.tokens, args.repeats, args.prompt_count)


if __name__ == "__main__":
    main()
