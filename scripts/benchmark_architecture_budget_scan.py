"""Development scan for architecture-level DDTree/TBV node allocation.

The candidate keeps the positive-temperature probability tree, Target/Draft
models, BF16 SDPA execution, and FP64 sampling law.  Unlike a verifier-only
gate, it varies the number of Target-verified tree nodes, so any improvement
comes from the decoding architecture rather than CUDA Graphs or a backend swap.
Synthetic prompts are selection data and can never become formal results.
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
from gbv_experiments.runner import output_lock, stop_token_ids
from gbv_experiments.terminal_formal import allocation_gate, model_gate, telemetry


PROMPTS = (
    "Explain why binary search needs a monotone predicate and give a short example.",
    "A tank is three fifths full. After adding 48 liters it is nine tenths full. Find its capacity.",
    "Write Python code for stable deduplication of a list while preserving order.",
)


def variants(budgets: list[int]) -> list[Variant]:
    base = Variant(
        name="base", method="ddtree", paths=1, length=15,
        temperature=1.0, draft_temperature=1.0, tree_budget=45,
        probability_dtype="float64",
    )
    values = [
        replace(base, name="dflash", method="dflash", draft_temperature=None),
        replace(base, name="ddtree_b45", method="ddtree"),
        *[
            replace(
                base, name=f"adaptive_tbv_b{budget}",
                method="ddtree_fused_scan", tree_budget=budget,
            )
            for budget in budgets
        ],
    ]
    for value in values:
        value.validate()
    return values


def compare(rows: list[dict], candidate: str, baseline: str,
            bootstrap: int = 20_000) -> dict:
    logs = []
    for prompt in range(len(PROMPTS)):
        selected = [row for row in rows if row["prompt"] == prompt]
        means = {
            name: sum(
                row["official_tpot_ms"] for row in selected
                if row["variant"] == name
            ) / sum(row["variant"] == name for row in selected)
            for name in (candidate, baseline)
        }
        logs.append(math.log(means[baseline] / means[candidate]))
    values = np.asarray(logs)
    rng = np.random.default_rng(20260910)
    draws = rng.choice(
        values, size=(bootstrap, len(values)), replace=True,
    ).mean(1)
    low, high = np.exp(np.quantile(draws, (0.025, 0.975)))
    return {
        "candidate": candidate,
        "baseline": baseline,
        "speedup": math.exp(float(values.mean())),
        "ci95": [float(low), float(high)],
        "prompt_clusters": len(values),
    }


@torch.inference_mode()
def run(config: Path, output: Path, budgets: list[int], tokens: int,
        repeats: int, device: str) -> dict:
    if (not budgets or len(set(budgets)) != len(budgets)
            or any(not 1 <= budget <= 45 for budget in budgets)):
        raise ValueError("Budgets must be unique integers in 1..45")
    if not 32 <= tokens <= 128 or repeats not in (1, 2, 3, 4, 5):
        raise ValueError("Use 32..128 tokens and 1..5 repeats")
    cfg = load_config(config)
    precision = {
        key: cfg["model"].get(key) for key in (
            "dtype", "target_attention", "draft_attention", "allow_tf32",
        )
    }
    if precision != {
            "dtype": "bfloat16", "target_attention": "sdpa",
            "draft_attention": "sdpa", "allow_tf32": False}:
        raise ValueError(f"Official precision controls changed: {precision}")
    declared = variants(budgets)
    by_name = {value.name: value for value in declared}
    with output_lock(output):
        if (output / "report.json").exists():
            raise ValueError("Output exists; choose a fresh directory")
        allocation_gate(device)
        engine, tokenizer = load_models(cfg["model"], device)
        model_gate(engine)
        stops = stop_token_ids(engine, tokenizer)
        encoded = [
            encode_messages(
                tokenizer, [{"role": "user", "content": prompt}],
                cfg["model"], device,
            )
            for prompt in PROMPTS
        ]
        for value in declared:
            engine.generate(encoded[0], value, 24, stops, seed=20260910)
        rows = []
        names = list(by_name)
        for repeat in range(repeats):
            for prompt, ids in enumerate(encoded):
                order = list(names)
                random.Random(20260910 + 100 * prompt + repeat).shuffle(order)
                shift = repeat % len(order)
                order = order[shift:] + order[:shift]
                for position, name in enumerate(order):
                    result = engine.generate(
                        ids, by_name[name], tokens, stops,
                        seed=20260910 + prompt,
                    )
                    rounds = result["rounds"]
                    row = {
                        "prompt": prompt,
                        "repeat": repeat,
                        "position": position,
                        "variant": name,
                        "generated_sha256": digest(result["generated_token_ids"]),
                        "official_tpot_ms": result[
                            "official_scope_time_per_output_token_ms"
                        ],
                        "decode_tokens": result["decode_tokens"],
                        "round_count": len(rounds),
                        "mean_tree_nodes": sum(
                            round_["tree_nodes"] for round_ in rounds
                        ) / len(rounds),
                        "mean_committed_tokens": sum(
                            round_["committed_tokens"] for round_ in rounds
                        ) / len(rounds),
                    }
                    rows.append(row)
                    write_json(output / "rows.json", {
                        "primary_timing": False, "rows": rows,
                    })
                    print(
                        f"budget-scan repeat={repeat + 1}/{repeats} "
                        f"prompt={prompt + 1}/{len(encoded)} method={name} "
                        f"official_tpot={row['official_tpot_ms']:.6f}",
                        flush=True,
                    )
        comparisons = {
            name: {
                "versus_ddtree": compare(rows, name, "ddtree_b45"),
                "versus_dflash": compare(rows, name, "dflash"),
            }
            for name in names if name.startswith("adaptive_tbv_")
        }
        best = max(
            comparisons,
            key=lambda name: comparisons[name]["versus_ddtree"]["speedup"],
        )
        report = {
            "kind": "architecture_level_ddtree_tbv_budget_scan",
            "formal_complete": False,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "uses_cuda_graph": False,
            "official_precision": precision,
            "temperature": 1.0,
            "length": 15,
            "baseline_budget": 45,
            "candidate_budgets": budgets,
            "tokens": tokens,
            "repeats": repeats,
            "prompt_policy": "synthetic development prompts only",
            "comparisons": comparisons,
            "best_point_estimate": best,
            "ddtree_vs_dflash": compare(rows, "ddtree_b45", "dflash"),
            "decision": (
                "continue_to_dynamic_budget_policy"
                if comparisons[best]["versus_ddtree"]["speedup"] > 1
                else "reject_budget_only_architecture"
            ),
            "source_sha256": {
                "engine": file_hash(
                    Path(__file__).resolve().parents[1]
                    / "src/gbv_experiments/engine.py"
                ),
                "script": file_hash(Path(__file__).resolve()),
                "config": file_hash(config),
            },
            "telemetry_end": telemetry(device),
        }
        write_json(output / "report.json", report)
        print(report, flush=True)
        return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budgets", type=int, nargs="+",
                        default=[16, 24, 32, 45])
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    run(
        args.config.resolve(), args.output.resolve(), args.budgets,
        args.tokens, args.repeats, args.device,
    )


if __name__ == "__main__":
    main()
