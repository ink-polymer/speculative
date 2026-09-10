"""Development scan for architecture-level DDTree/TBV node allocation.

The candidate keeps the positive-temperature probability tree, Target/Draft
models, BF16 SDPA execution, and FP64 sampling law.  Unlike a verifier-only
gate, it varies the number of Target-verified tree nodes, so any improvement
comes from the decoding architecture rather than CUDA Graphs or a backend swap.
It can also scan a proposal-only scoring temperature at fixed B=45.  This
changes the tree shape but not Target T=1 sampling.  Synthetic prompts are
selection data and can never become formal results.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random

import numpy as np
import torch

from gbv_experiments.common import digest, file_hash, write_json
from gbv_experiments.config import Variant, load_config
from gbv_experiments.conversation import encode_messages
from gbv_experiments.engine import load_models
from gbv_experiments.prefix_conditional import load_rank_calibrated_head
from gbv_experiments.slot_mixer import load_markov_branch_head, load_slot_mixer
from gbv_experiments.runner import output_lock, stop_token_ids
from gbv_experiments.terminal_formal import allocation_gate, model_gate, telemetry


PROMPTS = (
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


def temperature_name(value: float) -> str:
    return f"adaptive_tbv_pt{value:.3f}".replace(".", "p")


def depth_reward_name(value: float) -> str:
    return f"adaptive_utility_r{value:+.3f}".replace("+", "p").replace(
        "-", "m"
    ).replace(".", "p")


def schedule_name(start: float, end: float) -> str:
    return f"adaptive_schedule_{start:.2f}_{end:.2f}".replace(".", "p")


def adaptive_budget_name(minimum: int, threshold: float) -> str:
    return f"adaptive_cost_b{minimum}_c{threshold:.3f}".replace(".", "p")


def learned_calibration(path: Path) -> tuple[tuple[float, ...], str | None]:
    payload = json.loads(path.read_text())
    values = tuple(float(value) for value in payload["temperatures"])
    if (payload.get("holdout", {}).get("improved") is not True
            or payload.get("adds_model_forward_at_runtime") is not False
            or payload.get("changes_target_distribution") is not False):
        raise ValueError("Calibration artifact did not pass its holdout/runtime gates")
    bias_path = str(path) if isinstance(payload.get("vocab_bias"), list) else None
    return values, bias_path


def variants(budgets: list[int], tree_temperatures: list[float],
             sparse_temperatures: list[float],
             temperature_budget_grid: bool = False,
             depth_rewards: list[float] | None = None,
             temperature_schedules: list[tuple[float, float]] | None = None,
             learned_temperature_schedule: tuple[float, ...] | None = None,
             learned_bias_path: str | None = None,
             adaptive_budgets: list[tuple[int, float]] | None = None,
             rank_calibrated: bool = False,
             block_spines: list[int] | None = None,
             slot_mixer: bool = False,
             markov_branch: bool = False,
             markov_budgets: list[int] | None = None,
             latency_shapes: list[tuple[int, int]] | None = None,
             ) -> list[Variant]:
    depth_rewards = depth_rewards or []
    temperature_schedules = temperature_schedules or []
    adaptive_budgets = adaptive_budgets or []
    block_spines = block_spines or []
    markov_budgets = markov_budgets or [45]
    latency_shapes = latency_shapes or []
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
                base, name=f"adaptive_shape_l{length}_b{budget}",
                method="ddtree", length=length, tree_budget=budget,
            )
            for length, budget in latency_shapes
        ],
        *([] if temperature_budget_grid else [
            replace(
                base, name=f"adaptive_tbv_b{budget}",
                method="ddtree_fused_scan", tree_budget=budget,
            )
            for budget in budgets
        ]),
        *([] if temperature_budget_grid else [
            replace(
                base, name=temperature_name(value),
                method="ddtree_fused_scan",
                tree_proposal_temperature=value,
            )
            for value in tree_temperatures
        ]),
        *([
            replace(
                base,
                name=f"{temperature_name(value)}_b{budget}",
                method="ddtree_fused_scan", tree_budget=budget,
                tree_proposal_temperature=value,
            )
            for value in tree_temperatures for budget in budgets
        ] if temperature_budget_grid else []),
        *[
            replace(
                base, name=depth_reward_name(value),
                method="ddtree_fused_scan", tree_depth_reward=value,
            )
            for value in depth_rewards
        ],
        *[
            replace(
                base, name=schedule_name(start, end),
                method="ddtree_fused_scan",
                tree_proposal_temperature=start,
                tree_proposal_temperature_end=end,
            )
            for start, end in temperature_schedules
        ],
        *([] if learned_temperature_schedule is None else [
            replace(
                base, name="adaptive_learned_depth_calibration",
                method="ddtree_fused_scan",
                tree_proposal_temperature_schedule=learned_temperature_schedule,
                tree_proposal_bias_path=learned_bias_path,
            )
        ]),
        *[
            replace(
                base, name=adaptive_budget_name(minimum, threshold),
                method="ddtree_fused_scan",
                tree_adaptive_min_budget=minimum,
                tree_adaptive_confidence_threshold=threshold,
            )
            for minimum, threshold in adaptive_budgets
        ],
        *([] if not rank_calibrated else [
            replace(
                base, name="adaptive_rank_calibrated_tree",
                method="rank_calibrated_tree",
            )
        ]),
        *[
            replace(
                base, name=f"adaptive_block_spines_k{count}",
                method="block_aligned_tree", paths=count,
            )
            for count in block_spines
        ],
        *([] if not slot_mixer else [
            replace(
                base, name="adaptive_branch_slot_mixer",
                method="ddtree_slot_mixer",
            )
        ]),
        *([] if not markov_branch else [
            replace(
                base, name=f"adaptive_markov_branch_b{budget}",
                method="ddtree_markov_branch", tree_budget=budget,
            )
            for budget in markov_budgets
        ]),
        *[
            replace(
                base,
                name=temperature_name(value).replace(
                    "adaptive_tbv_", "adaptive_sparse_"
                ),
                method="ddtree_sparse_exit_fused_scan",
                tree_proposal_temperature=value,
            )
            for value in sparse_temperatures
        ],
    ]
    for value in values:
        value.validate()
    return values


def compare(rows: list[dict], candidate: str, baseline: str,
            bootstrap: int = 20_000) -> dict:
    logs = []
    for prompt in sorted({row["prompt"] for row in rows}):
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
        repeats: int, device: str,
        tree_temperatures: list[float] | None = None,
        sparse_temperatures: list[float] | None = None,
        prompt_count: int = 3,
        temperature_budget_grid: bool = False,
        depth_rewards: list[float] | None = None,
        temperature_schedules: list[tuple[float, float]] | None = None,
        learned_temperature_schedule: tuple[float, ...] | None = None,
        learned_bias_path: str | None = None,
        adaptive_budgets: list[tuple[int, float]] | None = None,
        rank_checkpoint: Path | None = None,
        block_spines: list[int] | None = None,
        slot_mixer_checkpoint: Path | None = None,
        markov_branch_checkpoint: Path | None = None,
        markov_budgets: list[int] | None = None,
        latency_shapes: list[tuple[int, int]] | None = None,
        ) -> dict:
    tree_temperatures = tree_temperatures or []
    sparse_temperatures = sparse_temperatures or []
    depth_rewards = depth_rewards or []
    temperature_schedules = temperature_schedules or []
    adaptive_budgets = adaptive_budgets or []
    block_spines = block_spines or []
    markov_budgets = markov_budgets or [45]
    latency_shapes = latency_shapes or []
    if (len(set(budgets)) != len(budgets)
            or any(not 1 <= budget <= 45 for budget in budgets)):
        raise ValueError("Budgets must be unique integers in 1..45")
    if (len(set(tree_temperatures)) != len(tree_temperatures)
            or any(not 0.1 <= value <= 4.0
                   for value in tree_temperatures)
            or len(set(sparse_temperatures)) != len(sparse_temperatures)
            or any(not 0.1 <= value <= 4.0
                   for value in sparse_temperatures)
            or len(set(depth_rewards)) != len(depth_rewards)
            or any(not -2.0 <= value <= 2.0 for value in depth_rewards)
            or len(set(temperature_schedules)) != len(temperature_schedules)
            or any(not 0.1 <= value <= 4.0
                   for pair in temperature_schedules for value in pair)
            or len(set(block_spines)) != len(block_spines)
            or any(not 1 <= value <= 16 for value in block_spines)
            or len(set(markov_budgets)) != len(markov_budgets)
            or any(not 1 <= value <= 45 for value in markov_budgets)
            or len(set(latency_shapes)) != len(latency_shapes)
            or any(not 2 <= length <= 32 or not 1 <= budget <= 256
                   for length, budget in latency_shapes)
            or (not budgets and not tree_temperatures
                and not sparse_temperatures and not depth_rewards
                and not temperature_schedules
                and learned_temperature_schedule is None
                and not adaptive_budgets and rank_checkpoint is None
                and not block_spines and slot_mixer_checkpoint is None
                and markov_branch_checkpoint is None
                and not latency_shapes)):
        raise ValueError(
            "Tree temperatures must be unique values in 0.1..4.0, and at "
            "least one scan value is required"
        )
    if not 32 <= tokens <= 128 or repeats not in (1, 2, 3, 4, 5):
        raise ValueError("Use 32..128 tokens and 1..5 repeats")
    if not 1 <= prompt_count <= len(PROMPTS):
        raise ValueError(f"prompt-count must be in 1..{len(PROMPTS)}")
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
    declared = variants(
        budgets, tree_temperatures, sparse_temperatures,
        temperature_budget_grid, depth_rewards, temperature_schedules,
        learned_temperature_schedule, learned_bias_path, adaptive_budgets,
        rank_checkpoint is not None,
        block_spines,
        slot_mixer_checkpoint is not None,
        markov_branch_checkpoint is not None,
        markov_budgets,
        latency_shapes,
    )
    by_name = {value.name: value for value in declared}
    with output_lock(output):
        if (output / "report.json").exists():
            raise ValueError("Output exists; choose a fresh directory")
        allocation_gate(device)
        engine, tokenizer = load_models(cfg["model"], device)
        model_gate(engine)
        learned_adapters = sum(value is not None for value in (
            rank_checkpoint, slot_mixer_checkpoint, markov_branch_checkpoint,
        ))
        if learned_adapters > 1:
            raise ValueError("Use one learned proposal adapter per scan")
        if rank_checkpoint is not None:
            engine.proposal_adapter = load_rank_calibrated_head(
                rank_checkpoint, int(engine.draft.config.hidden_size),
                engine.device,
            )
        if slot_mixer_checkpoint is not None:
            engine.proposal_adapter = load_slot_mixer(
                slot_mixer_checkpoint,
                int(engine.draft.config.hidden_size), engine.device,
                {
                    "target_revision": cfg["model"]["target_revision"],
                    "draft_revision": cfg["model"]["draft_revision"],
                }, dtype=next(engine.draft.parameters()).dtype,
            )
        if markov_branch_checkpoint is not None:
            engine.proposal_adapter = load_markov_branch_head(
                markov_branch_checkpoint,
                int(engine.draft.config.hidden_size), engine.device,
                {
                    "target_revision": cfg["model"]["target_revision"],
                    "draft_revision": cfg["model"]["draft_revision"],
                }, dtype=next(engine.draft.parameters()).dtype,
            )
        stops = stop_token_ids(engine, tokenizer)
        encoded = [
            encode_messages(
                tokenizer, [{"role": "user", "content": prompt}],
                cfg["model"], device,
            )
            for prompt in PROMPTS[:prompt_count]
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
        profile_names = ["ddtree_b45"] + [
            name for name in names if name.startswith("adaptive_")
        ]
        stage_profiles = {}
        for name in profile_names:
            result = engine.generate(
                encoded[0], by_name[name], tokens, stops,
                seed=20260910, profile=True,
            )
            stage_profiles[name] = {
                "diagnostic_only": True,
                "official_tpot_ms": result[
                    "official_scope_time_per_output_token_ms"
                ],
                "round_count": len(result["rounds"]),
                "mean_committed_tokens": sum(
                    row["committed_tokens"] for row in result["rounds"]
                ) / len(result["rounds"]),
                "stages": result["stages"],
                "stage_profile": result["stage_profile"],
            }
        write_json(output / "stage_profiles.json", {
            "primary_timing": False,
            "profiles": stage_profiles,
        })
        comparisons = {
            name: {
                "versus_ddtree": compare(rows, name, "ddtree_b45"),
                "versus_dflash": compare(rows, name, "dflash"),
            }
            for name in names if name.startswith("adaptive_")
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
            "candidate_tree_proposal_temperatures": tree_temperatures,
            "candidate_sparse_exit_temperatures": sparse_temperatures,
            "temperature_budget_grid": temperature_budget_grid,
            "candidate_tree_depth_rewards": depth_rewards,
            "candidate_tree_temperature_schedules": temperature_schedules,
            "candidate_learned_temperature_schedule": (
                learned_temperature_schedule
            ),
            "candidate_learned_bias_path": learned_bias_path,
            "candidate_adaptive_budgets": adaptive_budgets,
            "rank_checkpoint": (
                str(rank_checkpoint) if rank_checkpoint is not None else None
            ),
            "candidate_block_spines": block_spines,
            "slot_mixer_checkpoint": (
                str(slot_mixer_checkpoint)
                if slot_mixer_checkpoint is not None else None
            ),
            "markov_branch_checkpoint": (
                str(markov_branch_checkpoint)
                if markov_branch_checkpoint is not None else None
            ),
            "candidate_markov_budgets": markov_budgets,
            "candidate_latency_shapes": [
                {"length": length, "tree_budget": budget}
                for length, budget in latency_shapes
            ],
            "target_sampling_temperature": 1.0,
            "tokens": tokens,
            "repeats": repeats,
            "prompt_count": prompt_count,
            "prompt_policy": "synthetic development prompts only",
            "stage_profiles_file": "stage_profiles.json",
            "comparisons": comparisons,
            "best_point_estimate": best,
            "ddtree_vs_dflash": compare(rows, "ddtree_b45", "dflash"),
            "decision": (
                "continue_to_disjoint_confirmation"
                if comparisons[best]["versus_ddtree"]["ci95"][0] > 1
                else "exploratory_point_estimate_only"
                if comparisons[best]["versus_ddtree"]["speedup"] > 1
                else "reject_architecture_candidate"
            ),
            "source_sha256": {
                "engine": file_hash(
                    Path(__file__).resolve().parents[1]
                    / "src/gbv_experiments/engine.py"
                ),
                "script": file_hash(Path(__file__).resolve()),
                "config": file_hash(config),
                "rank_checkpoint": (
                    file_hash(rank_checkpoint)
                    if rank_checkpoint is not None else None
                ),
                "slot_mixer_checkpoint": (
                    file_hash(slot_mixer_checkpoint)
                    if slot_mixer_checkpoint is not None else None
                ),
                "markov_branch_checkpoint": (
                    file_hash(markov_branch_checkpoint)
                    if markov_branch_checkpoint is not None else None
                ),
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
    parser.add_argument("--budgets", type=int, nargs="*",
                        default=[16, 24, 32, 45])
    parser.add_argument("--tree-temperatures", type=float, nargs="*",
                        default=[])
    parser.add_argument("--sparse-temperatures", type=float, nargs="*",
                        default=[])
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--prompt-count", type=int, default=3)
    parser.add_argument("--temperature-budget-grid", action="store_true")
    parser.add_argument("--depth-rewards", type=float, nargs="*", default=[])
    parser.add_argument(
        "--temperature-schedules", nargs="*", default=[],
        metavar="START:END",
    )
    parser.add_argument("--learned-schedule", type=Path)
    parser.add_argument(
        "--adaptive-budgets", nargs="*", default=[], metavar="MIN:THRESHOLD",
    )
    parser.add_argument("--rank-checkpoint", type=Path)
    parser.add_argument("--block-spines", type=int, nargs="*", default=[])
    parser.add_argument("--slot-mixer-checkpoint", type=Path)
    parser.add_argument("--markov-branch-checkpoint", type=Path)
    parser.add_argument(
        "--markov-budgets", type=int, nargs="*", default=[45],
    )
    parser.add_argument(
        "--latency-shapes", nargs="*", default=[], metavar="L:B",
        help="exact ancestral DDTree shape candidates (length:node-budget)",
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    schedules = []
    for value in args.temperature_schedules:
        try:
            start, end = (float(part) for part in value.split(":"))
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Invalid temperature schedule {value!r}; use START:END"
            ) from error
        schedules.append((start, end))
    learned, learned_bias_path = (
        learned_calibration(args.learned_schedule.resolve())
        if args.learned_schedule is not None else (None, None)
    )
    adaptive_budgets = []
    for value in args.adaptive_budgets:
        try:
            minimum_text, threshold_text = value.split(":")
            pair = (int(minimum_text), float(threshold_text))
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Invalid adaptive budget {value!r}; use MIN:THRESHOLD"
            ) from error
        adaptive_budgets.append(pair)
    latency_shapes = []
    for value in args.latency_shapes:
        try:
            length_text, budget_text = value.split(":")
            pair = (int(length_text), int(budget_text))
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Invalid latency shape {value!r}; use L:B"
            ) from error
        latency_shapes.append(pair)
    run(
        args.config.resolve(), args.output.resolve(), args.budgets,
        args.tokens, args.repeats, args.device,
        tree_temperatures=args.tree_temperatures,
        sparse_temperatures=args.sparse_temperatures,
        prompt_count=args.prompt_count,
        temperature_budget_grid=args.temperature_budget_grid,
        depth_rewards=args.depth_rewards,
        temperature_schedules=schedules,
        learned_temperature_schedule=learned,
        learned_bias_path=learned_bias_path,
        adaptive_budgets=adaptive_budgets,
        rank_checkpoint=(
            args.rank_checkpoint.resolve()
            if args.rank_checkpoint is not None else None
        ),
        block_spines=args.block_spines,
        slot_mixer_checkpoint=(
            args.slot_mixer_checkpoint.resolve()
            if args.slot_mixer_checkpoint is not None else None
        ),
        markov_branch_checkpoint=(
            args.markov_branch_checkpoint.resolve()
            if args.markov_branch_checkpoint is not None else None
        ),
        markov_budgets=args.markov_budgets,
        latency_shapes=latency_shapes,
    )


if __name__ == "__main__":
    main()
