"""Run a fail-closed H20 speed gate for the frozen T=1 tree-block candidate.

This is a paired validation pilot, not a formal H200 result.  The candidate was
fixed from the earlier development scan before any timing on the current host:
K=1, L=15, B=45, diffusion support R=64, terminal tree-block continuation.
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


VALIDATION_PROMPTS = (
    "A store discounts an $80 item by 15%, then adds 8% sales tax. Compute the final price and show the arithmetic.",
    "Write a Python function that returns the first non-repeating character in a string, or None if there is none.",
    "Explain why a binary search requires a sorted search space, using one concrete counterexample.",
    "Find all integer solutions to x^2 - 5x + 6 = 0 and verify each solution.",
    "Design a SQL query that returns the three highest-revenue products per category from sales(category, product, revenue).",
    "A train travels 120 km at 80 km/h and then 90 km at 60 km/h. What is its average speed for the whole trip?",
    "Give pseudocode for detecting a cycle in a directed graph and state its time and space complexity.",
    "Rewrite this sentence to be concise while preserving meaning: Due to the fact that the server was unavailable, deployment was postponed.",
    "Prove that the sum of the first n odd positive integers is n squared.",
    "Implement an LRU cache interface with get and put in expected O(1) time and explain the data structures.",
    "Compare optimistic and pessimistic locking for a high-contention database workload in four precise points.",
    "A fair coin is tossed until two consecutive heads appear. Derive the expected number of tosses.",
)


def variants() -> list[Variant]:
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
            diffusion_support_size=64,
        ),
    ]
    for variant in result:
        variant.validate()
    return result


def paired_order(prompt_index: int, repeat: int, names: list[str]) -> list[str]:
    order = list(names)
    random.Random(20260909 + prompt_index).shuffle(order)
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


def compare(rows: list[dict], baseline: str, *, bootstrap_samples: int = 10000) -> dict:
    by_prompt = []
    for prompt in sorted({row["prompt"] for row in rows}):
        selected = [row for row in rows if row["prompt"] == prompt]
        metrics = {}
        for name in ("tree_block", baseline):
            records = [row for row in selected if row["variant"] == name]
            metrics[name] = sum(row["decode_ms"] for row in records) / sum(
                row["decode_tokens"] for row in records
            )
        by_prompt.append(math.log(metrics[baseline] / metrics["tree_block"]))
    values = np.asarray(by_prompt, dtype=np.float64)
    rng = np.random.default_rng(20260909)
    draws = rng.choice(values, size=(bootstrap_samples, len(values)), replace=True).mean(1)
    low, high = np.exp(np.quantile(draws, [0.025, 0.975]))
    return {
        "candidate": "tree_block",
        "baseline": baseline,
        "metric": "equal_prompt_geomean_decode_speedup",
        "speedup": math.exp(float(values.mean())),
        "ci95": [float(low), float(high)],
        "prompt_clusters": len(values),
        "bootstrap_samples": bootstrap_samples,
    }


@torch.inference_mode()
def run(config_path: Path, output: Path, device: str, tokens: int, repeats: int) -> dict:
    if not 64 <= tokens <= 256 or repeats < 3 or repeats % 3:
        raise ValueError("tokens must be 64..256 and repeats must be a positive multiple of 3")
    cfg = load_config(config_path)
    declared = variants()
    by_name = {variant.name: variant for variant in declared}
    if len({(v.length, v.tree_budget, v.temperature, v.draft_temperature,
            v.probability_dtype) for v in declared}) != 1:
        raise AssertionError("Speed gate variants do not share the registered controls")
    with output_lock(output):
        if (output / "report.json").exists() or (output / "rows.json").exists():
            raise ValueError("Speed-gate output already exists; choose a fresh directory")
        allocation_gate(device)
        torch.set_num_threads(8)
        manifest = {
            "kind": "paired_h20_validation_speed_gate",
            "formal_complete": False,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "candidate_fixed_before_current_host_timing": True,
            "candidate_selection_source": "2026-09-08 separate-host Qwen3-4B support scan",
            "temperature": 1.0,
            "tokens": tokens,
            "repeats": repeats,
            "variants": [variant.to_dict() for variant in declared],
            "prompt_sha256": [digest(prompt) for prompt in VALIDATION_PROMPTS],
            "order_policy": "per-prompt seeded shuffle, cyclic rotation balanced every three repeats",
            "success_rule": "both paired prompt-bootstrap 95% CI lower bounds strictly greater than 1",
            "limitations": [
                "Validation pilot on hand-written prompts, not a formal seven-dataset result",
                "H20 results must not be relabelled as the preregistered H200 study",
                "Speed evidence does not establish novelty or universal distributional dominance",
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
            engine.generate(encoded[0], variant, 32, stops, seed=20260909)
        rows = []
        names = list(by_name)
        for prompt_index, ids in enumerate(encoded):
            for repeat in range(repeats):
                allocation_gate(device)
                for position, name in enumerate(paired_order(prompt_index, repeat, names)):
                    variant = by_name[name]
                    result = engine.generate(
                        ids, variant, tokens, stops, seed=20260909 + prompt_index,
                    )
                    row = compact(
                        result, prompt=prompt_index, repeat=repeat,
                        order=position, variant=variant,
                    )
                    if name in {"ddtree", "tree_block"} and any(
                            round_["tree_nodes"] > 45 for round_ in result["rounds"]):
                        raise RuntimeError("Equal B=45 tree budget was exceeded")
                    rows.append(row)
                    write_json(output / "rows.json", {"primary_timing": False, "rows": rows})
                    print(
                        f"gate prompt={prompt_index + 1}/{len(encoded)} "
                        f"repeat={repeat + 1}/{repeats} method={name} "
                        f"tpot={row['decode_ms'] / row['decode_tokens']:.6f}",
                        flush=True,
                    )
        comparisons = {name: compare(rows, name) for name in ("ddtree", "dflash")}
        first = {(row["prompt"], row["variant"]): row["generated_sha256"]
                 for row in rows if row["repeat"] == 0}
        repeat_equal = all(
            row["generated_sha256"] == first[row["prompt"], row["variant"]]
            for row in rows
        )
        passed = repeat_equal and all(value["ci95"][0] > 1 for value in comparisons.values())
        report = {
            "gate_passed": passed,
            "formal_complete": False,
            "within_method_repeat_equal": repeat_equal,
            "comparisons": comparisons,
            "records": len(rows),
            "telemetry_end": telemetry(device),
            "decision": "proceed_to_formal_design" if passed else "stop_and_optimize",
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
    args = parser.parse_args()
    run(args.config.resolve(), args.output.resolve(), args.device, args.tokens, args.repeats)


if __name__ == "__main__":
    main()
