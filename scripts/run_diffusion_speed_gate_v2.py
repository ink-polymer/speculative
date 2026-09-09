"""Run the frozen T=1, B=45 tree-block validation gate on untouched prompts.

The support size (R=32) was selected on the separate three-prompt development
scan before any timing on this V2 prompt bank.  This is an H20 validation pilot,
not the preregistered seven-dataset formal experiment.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import importlib.metadata
import math
from pathlib import Path
import platform
import random

import numpy as np
import torch

from gbv_experiments.common import digest, file_hash, write_json
from gbv_experiments.config import Variant, load_config
from gbv_experiments.conversation import encode_messages
from gbv_experiments.engine import load_models
from gbv_experiments.runner import output_lock, stop_token_ids
from gbv_experiments.terminal_formal import allocation_gate, model_gate, telemetry


# Frozen before the V2 timing run.  None of these prompts appears in the V1
# gate or in terminal_formal.DIAGNOSTIC_PROMPTS used for support selection.
VALIDATION_PROMPTS = (
    "A tank is 3/5 full. After 24 liters are removed it is 1/3 full. Find the tank's capacity and show each equation.",
    "Write a Python function that merges overlapping closed intervals and state its time complexity.",
    "Explain with a minimal example why Dijkstra's algorithm can fail on a graph with a negative edge.",
    "Solve 2^(x+1) = 32 and check the answer by substitution.",
    "Write a SQL query to find customers who ordered in every month of 2025 from orders(customer_id, ordered_at).",
    "A cyclist covers 30 km uphill at 15 km/h and returns downhill at 30 km/h. Compute the round-trip average speed.",
    "Give pseudocode for topological sorting with indegrees and state when it detects a cycle.",
    "Make this sentence direct and concise: It is our recommendation that an investigation of the incident should be conducted.",
    "Prove by induction that 1 + 2 + ... + n equals n(n+1)/2.",
    "Implement a queue using two stacks and explain the amortized cost of enqueue and dequeue.",
    "Distinguish isolation from durability in database transactions using one failure example for each.",
    "Two fair dice are rolled repeatedly until their sum is seven. Derive the expected number of rolls.",
    "Find the derivative of x^2 ln(x) for x greater than zero and explain the product rule step.",
    "Write a JavaScript function that groups objects by a supplied key without mutating the input array.",
    "Explain why a hash table can have expected constant-time lookup but worst-case linear lookup.",
    "A rectangle has perimeter 54 and length three more than twice its width. Find both dimensions.",
    "Give a counterexample showing that pairwise independence does not imply mutual independence.",
    "Design an API pagination response that remains stable when new records are inserted, and justify the cursor fields.",
)


def variants(*, candidate_length: int = 15, candidate_budget: int = 45,
             candidate_support: int = 32) -> list[Variant]:
    base = Variant(
        name="base", method="ddtree", paths=1, length=15,
        temperature=1., draft_temperature=1., tree_budget=45,
        probability_dtype="float64", diffusion_support_size=8,
    )
    result = [
        replace(base, name="dflash", method="dflash"),
        replace(base, name="ddtree", method="ddtree"),
        replace(
            base, name="tree_block", method="diffusion_scaffold_bv",
            length=candidate_length, tree_budget=candidate_budget,
            diffusion_support_size=candidate_support,
        ),
    ]
    for variant in result:
        variant.validate()
    return result


def paired_order(prompt_index: int, repeat: int, names: list[str]) -> list[str]:
    order = list(names)
    random.Random(20260910 + prompt_index).shuffle(order)
    shift = repeat % len(order)
    return order[shift:] + order[:shift]


def compact(result: dict, *, prompt: int, repeat: int, order: int, variant: Variant) -> dict:
    return {
        "prompt": prompt,
        "repeat": repeat,
        "order": order,
        "variant": variant.name,
        "generated_sha256": digest(result["generated_token_ids"]),
        "generated_tokens": result["generated_tokens"],
        "decode_tokens": result["decode_tokens"],
        "prefill_ms": result["prefill_ms"],
        "decode_ms": result["decode_ms"],
        "e2e_ms": result["e2e_ms"],
        "target_forward_calls": result["target_forward_calls"],
        "draft_forward_calls": result["draft_forward_calls"],
        "peak_allocated_bytes": result["peak_allocated_bytes"],
        "rounds": result["rounds"],
    }


def compare(rows: list[dict], candidate: str, baseline: str,
            *, bootstrap_samples: int = 10000) -> dict:
    by_prompt = []
    for prompt in sorted({row["prompt"] for row in rows}):
        selected = [row for row in rows if row["prompt"] == prompt]
        metrics = {}
        for name in (candidate, baseline):
            records = [row for row in selected if row["variant"] == name]
            metrics[name] = sum(row["decode_ms"] for row in records) / sum(
                row["decode_tokens"] for row in records
            )
        by_prompt.append(math.log(metrics[baseline] / metrics[candidate]))
    values = np.asarray(by_prompt, dtype=np.float64)
    rng = np.random.default_rng(20260910)
    draws = rng.choice(values, size=(bootstrap_samples, len(values)), replace=True).mean(1)
    low, high = np.exp(np.quantile(draws, [0.025, 0.975]))
    return {
        "candidate": candidate,
        "baseline": baseline,
        "metric": "equal_prompt_geomean_decode_speedup",
        "speedup": math.exp(float(values.mean())),
        "ci95": [float(low), float(high)],
        "prompt_clusters": len(values),
        "bootstrap_samples": bootstrap_samples,
    }


@torch.inference_mode()
def run(config_path: Path, output: Path, device: str, tokens: int, repeats: int,
        *, post_gate_diagnostic: bool = False, candidate_length: int = 15,
        candidate_budget: int = 45, candidate_support: int = 32) -> dict:
    if not 64 <= tokens <= 256 or repeats < 3 or repeats % 3:
        raise ValueError("tokens must be 64..256 and repeats must be a positive multiple of 3")
    cfg = load_config(config_path)
    official_precision = {
        "dtype": cfg["model"].get("dtype"),
        "target_attention": cfg["model"].get("target_attention"),
        "draft_attention": cfg["model"].get("draft_attention"),
        "allow_tf32": cfg["model"].get("allow_tf32"),
    }
    if official_precision != {
        "dtype": "bfloat16", "target_attention": "sdpa",
        "draft_attention": "sdpa", "allow_tf32": False,
    }:
        raise ValueError(f"Official precision/backend controls changed: {official_precision}")
    if not post_gate_diagnostic and (candidate_length, candidate_budget, candidate_support) != (15, 45, 32):
        raise ValueError("The held-out V2 gate has a frozen L=15/B=45/R=32 candidate")
    if post_gate_diagnostic and (candidate_length > 15 or candidate_budget > 45):
        raise ValueError("A diagnostic candidate cannot exceed the baseline length or budget")
    declared = variants(
        candidate_length=candidate_length,
        candidate_budget=candidate_budget,
        candidate_support=candidate_support,
    )
    by_name = {variant.name: variant for variant in declared}
    if (not post_gate_diagnostic
            and len({(v.length, v.tree_budget, v.temperature, v.draft_temperature,
                     v.probability_dtype) for v in declared}) != 1):
        raise AssertionError("Speed gate variants do not share the registered controls")
    with output_lock(output):
        if (output / "report.json").exists() or (output / "rows.json").exists():
            raise ValueError("Speed-gate output already exists; choose a fresh directory")
        allocation_gate(device)
        torch.set_num_threads(8)
        manifest = {
            "kind": ("paired_h20_post_gate_diagnostic"
                     if post_gate_diagnostic else "paired_h20_validation_speed_gate_v2"),
            "formal_complete": False,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "candidate_fixed_before_validation_timing": not post_gate_diagnostic,
            "candidate_selection_source": "H20 three-prompt development support scan; T=1 B=45 R=32 terminal was the best exact-budget tree-block candidate",
            "official_precision": official_precision,
            "temperature": 1.0,
            "tokens": tokens,
            "repeats": repeats,
            "variants": [variant.to_dict() for variant in declared],
            "candidate_controls": {
                "length": candidate_length,
                "tree_budget": candidate_budget,
                "diffusion_support_size": candidate_support,
            },
            "prompt_sha256": [digest(prompt) for prompt in VALIDATION_PROMPTS],
            "prompt_nonoverlap": "No overlap with V1 validation or development prompts",
            "order_policy": "per-prompt seeded shuffle, cyclic rotation balanced every three repeats",
            "success_rule": "tree block must beat both baselines and DDTree must beat DFlash; every paired prompt-bootstrap 95% CI lower bound must be strictly greater than 1",
            "limitations": [
                "Validation pilot on hand-written prompts, not a formal seven-dataset result",
                "H20 results must not be relabelled as the preregistered H200 study",
                "Speed evidence does not establish novelty or universal distributional dominance",
                *(["V2 prompts are reused after an implementation change; this run is development-only and cannot pass a held-out gate"]
                  if post_gate_diagnostic else []),
            ],
            "source_sha256": {
                "engine": file_hash(Path(__file__).resolve().parents[1] / "src/gbv_experiments/engine.py"),
                "tree_block": file_hash(Path(__file__).resolve().parents[1] / "src/gbv_experiments/diffusion_tree_bv.py"),
                "script": file_hash(Path(__file__).resolve()),
                "config": file_hash(config_path),
            },
        }
        write_json(output / "manifest.json", manifest)
        engine, tokenizer = load_models(cfg["model"], device)
        model_gate(engine)
        props = torch.cuda.get_device_properties(device)
        manifest["runtime"] = {
            "gpu": torch.cuda.get_device_name(device),
            "gpu_memory_bytes": props.total_memory,
            "compute_capability": [props.major, props.minor],
            "driver_and_clocks_start": telemetry(device),
            "python": platform.python_version(),
            "versions": {name: importlib.metadata.version(name) for name in
                         ("torch", "transformers", "huggingface-hub", "numpy")},
        }
        write_json(output / "manifest.json", manifest)
        stops = stop_token_ids(engine, tokenizer)
        encoded = [encode_messages(
            tokenizer, [{"role": "user", "content": prompt}], cfg["model"], device,
        ) for prompt in VALIDATION_PROMPTS]
        for variant in declared:
            engine.generate(encoded[0], variant, 32, stops, seed=20260910)
        rows = []
        names = list(by_name)
        for prompt_index, ids in enumerate(encoded):
            for repeat in range(repeats):
                allocation_gate(device)
                for position, name in enumerate(paired_order(prompt_index, repeat, names)):
                    variant = by_name[name]
                    result = engine.generate(
                        ids, variant, tokens, stops, seed=20260910 + prompt_index,
                    )
                    row = compact(
                        result, prompt=prompt_index, repeat=repeat,
                        order=position, variant=variant,
                    )
                    limit = candidate_budget if name == "tree_block" else 45
                    if name in {"ddtree", "tree_block"} and any(
                            round_["tree_nodes"] > limit for round_ in result["rounds"]):
                        raise RuntimeError(f"{name} tree budget B={limit} was exceeded")
                    rows.append(row)
                    write_json(output / "rows.json", {"primary_timing": False, "rows": rows})
                    print(
                        f"gate-v2 prompt={prompt_index + 1}/{len(encoded)} "
                        f"repeat={repeat + 1}/{repeats} method={name} "
                        f"tpot={row['decode_ms'] / row['decode_tokens']:.6f}",
                        flush=True,
                    )
        comparisons = {
            name: compare(rows, "tree_block", name)
            for name in ("ddtree", "dflash")
        }
        baseline_sanity = compare(rows, "ddtree", "dflash")
        first = {(row["prompt"], row["variant"]): row["generated_sha256"]
                 for row in rows if row["repeat"] == 0}
        repeat_equal = all(
            row["generated_sha256"] == first[row["prompt"], row["variant"]]
            for row in rows
        )
        thresholds_met = (repeat_equal
                          and all(value["ci95"][0] > 1 for value in comparisons.values())
                          and baseline_sanity["ci95"][0] > 1)
        report = {
            "gate_passed": None if post_gate_diagnostic else thresholds_met,
            "diagnostic_thresholds_met": thresholds_met,
            "formal_complete": False,
            "within_method_repeat_equal": repeat_equal,
            "comparisons": comparisons,
            "baseline_sanity": baseline_sanity,
            "records": len(rows),
            "telemetry_end": telemetry(device),
            "decision": (
                "freeze_for_new_heldout_validation" if thresholds_met
                else "stop_and_optimize"
            ),
        }
        write_json(output / "report.json", report)
        print(report, flush=True)
        return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/gbv_paper_ddtree_counts.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tokens", type=int, default=96)
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--post-gate-diagnostic", action="store_true")
    parser.add_argument("--candidate-length", type=int, default=15)
    parser.add_argument("--candidate-budget", type=int, default=45)
    parser.add_argument("--candidate-support", type=int, default=32)
    args = parser.parse_args()
    run(args.config.resolve(), args.output.resolve(), args.device, args.tokens, args.repeats,
        post_gate_diagnostic=args.post_gate_diagnostic,
        candidate_length=args.candidate_length,
        candidate_budget=args.candidate_budget,
        candidate_support=args.candidate_support)


if __name__ == "__main__":
    main()
