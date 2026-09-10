"""Screen block-verification-friendly tree architectures on synthetic prompts.

Every candidate keeps L=15, maximum B=45, T=1, one Draft forward, one Target
tree forward, BF16 SDPA, and FP64 probabilities.  Trees and verifier laws may
differ, which is the intended architecture delta.  This is development data,
not a held-out or formal comparison.
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


def variants(*, focused_core_spur: bool = False) -> list[Variant]:
    base = Variant(
        name="base", method="ddtree", paths=1, length=15,
        temperature=1.0, draft_temperature=1.0, tree_budget=45,
        probability_dtype="float64", diffusion_support_size=8,
    )
    values = [
        replace(base, name="dflash", method="dflash", draft_temperature=None),
        replace(base, name="ddtree", method="ddtree"),
        replace(
            base, name="shared_suffix_protected",
            method="root_protected_bv", paths=3,
        ),
        replace(base, name="atom_coupled", method="atom_tree_bv", paths=3),
        replace(
            base, name="packed_blocks_k2",
            method="tree_gbv_packed", paths=2,
        ),
        replace(
            base, name="packed_blocks_k3",
            method="tree_gbv_packed", paths=3,
        ),
        replace(
            base, name="diffusion_scaffold",
            method="diffusion_scaffold_bv", paths=1,
            diffusion_support_size=32,
        ),
        *[
            replace(
                base, name=f"core_spur_{spur}",
                method="diffusion_core_spur_bv", paths=1,
                diffusion_support_size=32,
                diffusion_spur_length=spur,
            )
            for spur in (1, 2, 4, 6, 8)
        ],
        *[
            replace(
                base, name=f"core_spur_6_s{support}",
                method="diffusion_core_spur_bv", paths=1,
                diffusion_support_size=support,
                diffusion_spur_length=6,
            )
            for support in (8, 16, 64)
        ],
        *[
            replace(
                base, name=f"core_spur_6_b{budget}",
                method="diffusion_core_spur_bv", paths=1,
                diffusion_support_size=32,
                diffusion_spur_length=6, tree_budget=budget,
            )
            for budget in (32, 36, 40)
        ],
        replace(
            base, name="budgeted_prefix_recycle",
            method="tree_gbv_budgeted_prefix_recycle_sparse_lazy", paths=3,
        ),
    ]
    if focused_core_spur:
        values = [
            value for value in values
            if value.name in {"dflash", "ddtree"}
            or value.name.startswith("core_spur_")
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


def aggregate(rows: list[dict], name: str) -> dict:
    selected = [row for row in rows if row["variant"] == name]
    return {
        "records": len(selected),
        "mean_official_tpot_ms": sum(
            row["official_tpot_ms"] for row in selected
        ) / len(selected),
        "mean_tree_nodes": sum(
            row["mean_tree_nodes"] for row in selected
        ) / len(selected),
        "mean_committed_tokens_per_round": sum(
            row["mean_committed_tokens"] for row in selected
        ) / len(selected),
        "mean_round_count": sum(
            row["round_count"] for row in selected
        ) / len(selected),
    }


@torch.inference_mode()
def run(config: Path, output: Path, tokens: int, repeats: int,
        device: str, *, focused_core_spur: bool = False) -> dict:
    if not 32 <= tokens <= 128 or repeats not in (1, 2, 3):
        raise ValueError("Use 32..128 tokens and 1..3 repeats")
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
    declared = variants(focused_core_spur=focused_core_spur)
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
                        "round_count": len(rounds),
                        "mean_tree_nodes": sum(
                            round_["tree_nodes"] for round_ in rounds
                        ) / len(rounds),
                        "mean_committed_tokens": sum(
                            round_["committed_tokens"] for round_ in rounds
                        ) / len(rounds),
                        "target_forward_calls": result["target_forward_calls"],
                        "draft_forward_calls": result["draft_forward_calls"],
                    }
                    if name not in {"dflash", "ddtree"} and any(
                            round_["tree_nodes"] > 45 for round_ in rounds):
                        raise RuntimeError(f"{name} exceeded B=45")
                    rows.append(row)
                    write_json(output / "rows.json", {
                        "primary_timing": False, "rows": rows,
                    })
                    print(
                        f"architecture-scan repeat={repeat + 1}/{repeats} "
                        f"prompt={prompt + 1}/{len(encoded)} method={name} "
                        f"official_tpot={row['official_tpot_ms']:.6f} "
                        f"commit={row['mean_committed_tokens']:.3f}",
                        flush=True,
                    )
        candidates = [name for name in names if name not in {"dflash", "ddtree"}]
        comparisons = {
            name: {
                "versus_ddtree": compare(rows, name, "ddtree"),
                "versus_dflash": compare(rows, name, "dflash"),
            }
            for name in candidates
        }
        best = max(
            candidates,
            key=lambda name: comparisons[name]["versus_ddtree"]["speedup"],
        )
        report = {
            "kind": "block_friendly_tree_architecture_scan",
            "formal_complete": False,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "uses_cuda_graph": False,
            "official_precision": precision,
            "frozen_common_controls": {
                "temperature": 1.0,
                "length": 15,
                "maximum_tree_budget": 45,
                "maximum_draft_forwards_per_round": 1,
                "maximum_target_forwards_per_round": 1,
            },
            "tokens": tokens,
            "repeats": repeats,
            "prompt_policy": "synthetic development prompts only",
            "focused_core_spur": focused_core_spur,
            "aggregate": {name: aggregate(rows, name) for name in names},
            "comparisons": comparisons,
            "best_point_estimate": best,
            "ddtree_vs_dflash": compare(rows, "ddtree", "dflash"),
            "decision": (
                "optimize_best_tree_architecture"
                if comparisons[best]["versus_ddtree"]["speedup"] > 1
                else "none_of_the_existing_tree_architectures_beats_ddtree"
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
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--focused-core-spur", action="store_true")
    args = parser.parse_args()
    run(
        args.config.resolve(), args.output.resolve(), args.tokens,
        args.repeats, args.device,
        focused_core_spur=args.focused_core_spur,
    )


if __name__ == "__main__":
    main()
