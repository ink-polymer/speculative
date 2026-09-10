"""Development timing gate for a prefix-conditional core--spur tree.

This is intentionally not a formal benchmark.  It interleaves the canonical
DFlash and DDTree controls with the unconditioned and trained conditional spur
architectures under the same T=1, L=15, B=45, BF16-SDPA protocol.  Primary
timing uses synchronized upstream-compatible TPOT; a separate profiled pass is
diagnostic only.
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
from gbv_experiments.prefix_conditional import load_prefix_conditional_head
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


def declared_variants(spurs: tuple[int, ...],
                      supports: tuple[int, ...],
                      strengths: tuple[float, ...]) -> list[Variant]:
    base = Variant(
        name="base", method="ddtree", paths=1, length=15,
        temperature=1., draft_temperature=1., tree_budget=45,
        probability_dtype="float64", diffusion_support_size=32,
    )
    values = [
        replace(base, name="dflash", method="dflash", draft_temperature=None),
        replace(base, name="ddtree", method="ddtree"),
        replace(
            base, name="unconditioned_spur_6",
            method="diffusion_core_spur_bv", diffusion_spur_length=6,
        ),
        replace(
            base, name="prefix_rescored_tree",
            method="prefix_rescored_tree",
        ),
        replace(
            base, name="prefix_beam_tree",
            method="prefix_beam_tree",
        ),
        *[
            replace(
                base, name=(
                    f"prefix_spur_{spur}_r{support}_x{strength:g}"
                ),
                method="prefix_core_spur_bv", diffusion_spur_length=spur,
                diffusion_support_size=support,
                prefix_strength=strength,
            )
            for spur in spurs for support in supports for strength in strengths
        ],
        *[
            replace(
                base, name=f"prefix_sampled_tree_spur_{spur}",
                method="prefix_sampled_spur_tree", diffusion_spur_length=spur,
            )
            for spur in spurs
        ],
        *[
            replace(
                base, name=f"prefix_tree_spur_{spur}",
                method="prefix_core_spur_tree", diffusion_spur_length=spur,
            )
            for spur in spurs
        ],
    ]
    for value in values:
        value.validate()
    return values


def aggregate(rows: list[dict], name: str) -> dict:
    selected = [row for row in rows if row["variant"] == name]
    return {
        "records": len(selected),
        "mean_official_tpot_ms": sum(
            row["official_tpot_ms"] for row in selected
        ) / len(selected),
        "mean_committed_tokens_per_round": sum(
            row["mean_committed_tokens"] for row in selected
        ) / len(selected),
        "mean_round_count": sum(
            row["round_count"] for row in selected
        ) / len(selected),
        "mean_tree_nodes": sum(
            row["mean_tree_nodes"] for row in selected
        ) / len(selected),
    }


def compare(rows: list[dict], candidate: str, baseline: str,
            prompt_count: int,
            bootstrap: int = 20_000) -> dict:
    values = []
    for prompt in range(prompt_count):
        selected = [row for row in rows if row["prompt"] == prompt]
        mean = {
            name: sum(
                row["official_tpot_ms"] for row in selected
                if row["variant"] == name
            ) / sum(row["variant"] == name for row in selected)
            for name in (candidate, baseline)
        }
        values.append(math.log(mean[baseline] / mean[candidate]))
    logs = np.asarray(values)
    rng = np.random.default_rng(20260910)
    samples = rng.choice(
        logs, size=(bootstrap, len(logs)), replace=True,
    ).mean(1)
    low, high = np.exp(np.quantile(samples, (0.025, 0.975)))
    return {
        "candidate": candidate,
        "baseline": baseline,
        "speedup": math.exp(float(logs.mean())),
        "ci95": [float(low), float(high)],
        "prompt_clusters": len(logs),
    }


@torch.inference_mode()
def run(config: Path, checkpoint: Path, output: Path, tokens: int,
        repeats: int, device: str, spurs: tuple[int, ...],
        supports: tuple[int, ...], strengths: tuple[float, ...],
        prompt_count: int) -> dict:
    if not 32 <= tokens <= 128 or not 1 <= repeats <= 5:
        raise ValueError("Use 32..128 tokens and 1..5 repeats")
    if not 1 <= prompt_count <= len(DEVELOPMENT_PROMPTS):
        raise ValueError(
            f"prompt-count must be in 1..{len(DEVELOPMENT_PROMPTS)}"
        )
    selected_prompts = DEVELOPMENT_PROMPTS[:prompt_count]
    cfg = load_config(config)
    model_cfg = cfg["model"]
    precision = {
        key: model_cfg.get(key) for key in (
            "dtype", "target_attention", "draft_attention", "allow_tf32",
        )
    }
    if precision != {
            "dtype": "bfloat16", "target_attention": "sdpa",
            "draft_attention": "sdpa", "allow_tf32": False}:
        raise ValueError(f"Official precision controls changed: {precision}")
    methods = declared_variants(spurs, supports, strengths)
    by_name = {value.name: value for value in methods}
    with output_lock(output):
        if (output / "report.json").exists():
            raise ValueError("Output exists; choose a fresh directory")
        allocation_gate(device)
        engine, tokenizer = load_models(model_cfg, device)
        model_gate(engine)
        engine.proposal_adapter = load_prefix_conditional_head(
            checkpoint, int(engine.draft.config.hidden_size),
            torch.device(device), {
                "target": model_cfg["target"],
                "target_revision": model_cfg["target_revision"],
                "draft": model_cfg["draft"],
                "draft_revision": model_cfg["draft_revision"],
            },
        )
        stops = stop_token_ids(engine, tokenizer)
        encoded = [
            encode_messages(
                tokenizer, [{"role": "user", "content": prompt}],
                model_cfg, device,
            )
            for prompt in selected_prompts
        ]
        for value in methods:
            engine.generate(encoded[0], value, 24, stops, seed=20260910)
        rows = []
        names = list(by_name)
        for repeat in range(repeats):
            for prompt, ids in enumerate(encoded):
                order = list(names)
                random.Random(20260910 + 100 * prompt).shuffle(order)
                shift = repeat % len(order)
                order = order[shift:] + order[:shift]
                for position, name in enumerate(order):
                    result = engine.generate(
                        ids, by_name[name], tokens, stops,
                        seed=20260910 + 1000 * repeat + prompt,
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
                            item["tree_nodes"] for item in rounds
                        ) / len(rounds),
                        "mean_committed_tokens": sum(
                            item["committed_tokens"] for item in rounds
                        ) / len(rounds),
                        "target_forward_calls": result["target_forward_calls"],
                        "draft_forward_calls": result["draft_forward_calls"],
                    }
                    if (name.startswith(("prefix_", "unconditioned_"))
                            and any(item["tree_nodes"] > 45 for item in rounds)):
                        raise RuntimeError(f"{name} exceeded B=45")
                    if result["draft_forward_calls"] != len(rounds):
                        raise RuntimeError(f"{name} used more than one Draft per round")
                    rows.append(row)
                    write_json(output / "rows.json", {
                        "primary_timing": False, "rows": rows,
                    })
                    print(
                        f"prefix-screen repeat={repeat + 1}/{repeats} "
                        f"prompt={prompt + 1}/{len(encoded)} method={name} "
                        f"tpot={row['official_tpot_ms']:.6f} "
                        f"commit={row['mean_committed_tokens']:.3f}",
                        flush=True,
                    )
        candidates = [name for name in names if name.startswith("prefix_")]
        comparisons = {
            name: {
                "versus_ddtree": compare(
                    rows, name, "ddtree", prompt_count,
                ),
                "versus_dflash": compare(
                    rows, name, "dflash", prompt_count,
                ),
                "versus_unconditioned": compare(
                    rows, name, "unconditioned_spur_6", prompt_count,
                ),
            }
            for name in candidates
        }
        best = max(
            candidates,
            key=lambda name: comparisons[name]["versus_ddtree"]["speedup"],
        )
        profiles = {}
        block_candidate = next(
            name for name in candidates if name.startswith("prefix_spur_")
        )
        profile_names = tuple(dict.fromkeys((
            "dflash", "ddtree", "unconditioned_spur_6",
            block_candidate, best,
        )))
        for name in profile_names:
            value = engine.generate(
                encoded[0], by_name[name], tokens, stops,
                seed=20260910, profile=True,
            )
            profiles[name] = {
                "official_tpot_ms": value[
                    "official_scope_time_per_output_token_ms"
                ],
                "mean_committed_tokens": sum(
                    item["committed_tokens"] for item in value["rounds"]
                ) / len(value["rounds"]),
                "stage_profile": value["stage_profile"],
            }
        report = {
            "kind": "prefix_conditional_core_spur_development_gate",
            "formal_complete": False,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "uses_cuda_graph": False,
            "official_precision": precision,
            "frozen_common_controls": {
                "temperature": 1., "length": 15, "tree_budget": 45,
                "maximum_draft_forwards_per_round": 1,
                "maximum_target_forwards_per_round": 1,
            },
            "tokens": tokens,
            "repeats": repeats,
            "prompt_count": prompt_count,
            "prompt_policy": "synthetic development prompts only",
            "aggregate": {name: aggregate(rows, name) for name in names},
            "comparisons": comparisons,
            "ddtree_vs_dflash": compare(
                rows, "ddtree", "dflash", prompt_count,
            ),
            "best_point_estimate": best,
            "decision": (
                "continue_training_and_heldout_gate"
                if comparisons[best]["versus_ddtree"]["speedup"] > 1.
                else "optimize_or_reject_prefix_head"
            ),
            "diagnostic_profiles": profiles,
            "source_sha256": {
                "checkpoint": file_hash(checkpoint),
                "engine": file_hash(
                    Path(__file__).resolve().parents[1]
                    / "src/gbv_experiments/engine.py"
                ),
                "head": file_hash(
                    Path(__file__).resolve().parents[1]
                    / "src/gbv_experiments/prefix_conditional.py"
                ),
                "script": file_hash(Path(__file__).resolve()),
                "config": file_hash(config),
            },
            "telemetry_end": telemetry(device),
        }
        write_json(output / "report.json", report)
        print(report, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=48)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--spurs", default="2,4,6,8")
    parser.add_argument("--supports", default="32")
    parser.add_argument("--strengths", default="1")
    parser.add_argument("--prompt-count", type=int, default=3)
    args = parser.parse_args()
    spurs = tuple(int(value) for value in args.spurs.split(","))
    supports = tuple(int(value) for value in args.supports.split(","))
    strengths = tuple(float(value) for value in args.strengths.split(","))
    if not spurs or any(not 1 <= value < 15 for value in spurs):
        raise ValueError("Every spur must be in [1, 14]")
    if not supports or any(not 1 <= value <= 256 for value in supports):
        raise ValueError("Every support must be in [1, 256]")
    if not strengths or any(not 0 <= value <= 2 for value in strengths):
        raise ValueError("Every strength must be in [0, 2]")
    run(
        args.config.resolve(), args.checkpoint.resolve(),
        args.output.resolve(), args.tokens, args.repeats,
        args.device, spurs, supports, strengths, args.prompt_count,
    )


if __name__ == "__main__":
    main()
