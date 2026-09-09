"""Held-out H20 gate for same-DDTree fast block verification.

The candidate reuses DDTree's probability tree and every Target probability
row.  Its only change is replacing the official all-row ``multinomial`` draw
with a persistent CUDA block that scans only rows reached by the ancestral
walk.  This is a validation pilot, not the seven-dataset formal result.
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
from gbv_experiments.fairness import assert_architecture_only_pair
from gbv_experiments.runner import output_lock, stop_token_ids
from gbv_experiments.terminal_formal import allocation_gate, model_gate, telemetry


# Frozen after the verifier-only gate and before any end-to-end timing of this
# implementation.  These prompts are disjoint from both earlier diffusion
# scaffold gates and terminal_formal.DIAGNOSTIC_PROMPTS.
HELDOUT_PROMPTS = (
    "A reservoir gains 18 liters per minute for 25 minutes, then loses 12 liters per minute for 15 minutes. Find the net change.",
    "Write a Python function that returns the length of the longest substring without repeated characters and give its complexity.",
    "Explain why breadth-first search finds shortest paths in an unweighted graph, using the queue invariant.",
    "Solve 3x - 7 = 2x + 11 and verify the result in the original equation.",
    "Write SQL that returns each department's second-highest distinct salary from employees(department_id, salary).",
    "A boat travels 48 km downstream in 3 hours and the same distance upstream in 4 hours. Find the boat and current speeds.",
    "Give pseudocode for union-find with path compression and union by rank, and state its amortized complexity.",
    "Rewrite concisely: In the event that payment is not received by Friday, access will be suspended.",
    "Prove that the square of every odd integer is congruent to one modulo eight.",
    "Implement binary exponentiation for nonnegative integer exponents and explain why it is logarithmic.",
    "Compare snapshot isolation and serializable isolation using one write-skew example.",
    "A biased coin has probability 0.4 of heads. Find the expected tosses until the first head and derive it.",
    "Differentiate (x^3 + 1)/(x - 1) and state the domain before simplifying.",
    "Write a TypeScript function that deep-freezes a JSON-like object without using external packages.",
    "Explain the ABA problem in lock-free algorithms and describe one standard mitigation.",
    "A right triangle has legs differing by 7 and area 60. Find both positive leg lengths.",
    "Give a concrete example of a relation that is symmetric and transitive but not reflexive.",
    "Design idempotency handling for a payment POST endpoint, including storage fields and retry behavior.",
)


def variants() -> list[Variant]:
    base = Variant(
        name="base", method="ddtree", paths=1, length=15,
        temperature=1.0, draft_temperature=1.0, tree_budget=45,
        probability_dtype="float64",
    )
    declared = [
        replace(base, name="dflash", method="dflash"),
        replace(base, name="ddtree", method="ddtree"),
        replace(base, name="tree_block_verification", method="ddtree_fused_scan"),
    ]
    for variant in declared:
        variant.validate()
    return declared


def paired_order(prompt_index: int, repeat: int, names: list[str]) -> list[str]:
    order = list(names)
    random.Random(20260913 + prompt_index).shuffle(order)
    shift = repeat % len(order)
    return order[shift:] + order[:shift]


def compact(result: dict, *, prompt: int, repeat: int, order: int,
            variant: Variant) -> dict:
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
            bootstrap_samples: int = 20000) -> dict:
    logs = []
    for prompt in sorted({row["prompt"] for row in rows}):
        selected = [row for row in rows if row["prompt"] == prompt]
        tpot = {}
        for name in (candidate, baseline):
            records = [row for row in selected if row["variant"] == name]
            tpot[name] = sum(row["decode_ms"] for row in records) / sum(
                row["decode_tokens"] for row in records
            )
        logs.append(math.log(tpot[baseline] / tpot[candidate]))
    values = np.asarray(logs, dtype=np.float64)
    rng = np.random.default_rng(20260913)
    draws = rng.choice(
        values, size=(bootstrap_samples, len(values)), replace=True,
    ).mean(1)
    low, high = np.exp(np.quantile(draws, [.025, .975]))
    return {
        "candidate": candidate,
        "baseline": baseline,
        "metric": "equal_prompt_geomean_decode_speedup",
        "speedup": math.exp(float(values.mean())),
        "ci95": [float(low), float(high)],
        "prompt_clusters": len(values),
        "bootstrap_samples": bootstrap_samples,
    }


def aggregate(rows: list[dict], name: str) -> dict:
    selected = [row for row in rows if row["variant"] == name]
    rounds = [round_ for row in selected for round_ in row["rounds"]]
    decode_tokens = sum(row["decode_tokens"] for row in selected)
    decode_ms = sum(row["decode_ms"] for row in selected)
    return {
        "decode_tokens": decode_tokens,
        "decode_ms": decode_ms,
        "tokens_per_second": 1000 * decode_tokens / decode_ms,
        "mean_accepted_draft_tokens_per_round": sum(
            round_["accepted_draft_tokens"] for round_ in rounds
        ) / len(rounds),
        "mean_committed_tokens_per_round": sum(
            round_["committed_tokens"] for round_ in rounds
        ) / len(rounds),
        "rounds": len(rounds),
    }


@torch.inference_mode()
def run(config_path: Path, output: Path, device: str,
        tokens: int, repeats: int) -> dict:
    if not 96 <= tokens <= 256 or repeats < 3 or repeats % 3:
        raise ValueError("tokens must be 96..256 and repeats a positive multiple of 3")
    cfg = load_config(config_path)
    precision = {
        "dtype": cfg["model"].get("dtype"),
        "target_attention": cfg["model"].get("target_attention"),
        "draft_attention": cfg["model"].get("draft_attention"),
        "allow_tf32": cfg["model"].get("allow_tf32"),
    }
    expected_precision = {
        "dtype": "bfloat16", "target_attention": "sdpa",
        "draft_attention": "sdpa", "allow_tf32": False,
    }
    if precision != expected_precision:
        raise ValueError(f"Official precision/backend controls changed: {precision}")

    declared = variants()
    by_name = {variant.name: variant for variant in declared}
    ddtree = by_name["ddtree"]
    candidate = by_name["tree_block_verification"]
    fairness = assert_architecture_only_pair(ddtree, candidate, cfg["model"])
    dflash_fairness = assert_architecture_only_pair(
        ddtree, by_name["dflash"], cfg["model"],
    )

    with output_lock(output):
        if (output / "report.json").exists() or (output / "rows.json").exists():
            raise ValueError("Output exists; choose a fresh directory")
        allocation_gate(device)
        torch.set_num_threads(8)
        manifest = {
            "kind": "heldout_same_ddtree_fast_block_validation_gate",
            "formal_complete": False,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "candidate_frozen_before_end_to_end_prompt_timing": True,
            "official_precision": precision,
            "temperature": 1.0,
            "tokens": tokens,
            "repeats": repeats,
            "variants": [variant.to_dict() for variant in declared],
            "fairness": fairness,
            "dflash_control_match": dflash_fairness,
            "same_tree_contract": {
                "constructor": "probability_tree",
                "tree_budget": 45,
                "target_probability_rows": "identical FP64 tensor for either verifier",
                "candidate_only_delta": "official all-row batched multinomial -> one persistent CUDA block with a CUB FP64 scan over reached rows",
                "sampling_law": "same ancestral categorical distribution; realized paths need not match because RNG algorithms consume streams differently",
            },
            "prompt_sha256": [digest(prompt) for prompt in HELDOUT_PROMPTS],
            "prompt_policy": "frozen disjoint bank; prompt text is not serialized into outputs",
            "order_policy": "seeded per-prompt shuffle with balanced cyclic rotation",
            "success_rule": "candidate must beat DDTree and DFlash and DDTree must beat DFlash; all prompt-bootstrap 95% CI lower bounds > 1",
            "limitations": [
                "Held-out hand-written validation pilot, not the formal seven-dataset experiment",
                "H20 result must not be relabelled as an H200 result",
                "Model-level BF16 tree rows can differ numerically from sequential autoregressive rows; all compared methods retain the official BF16+SDPA execution",
            ],
            "source_sha256": {
                "engine": file_hash(Path(__file__).resolve().parents[1] / "src/gbv_experiments/engine.py"),
                "fused_verifier": file_hash(Path(__file__).resolve().parents[1] / "src/gbv_experiments/fused_tree_sampling.py"),
                "official_verifier": file_hash(Path(__file__).resolve().parents[1] / "src/gbv_experiments/sampling.py"),
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
        stops = stop_token_ids(engine, tokenizer)
        encoded = [encode_messages(
            tokenizer, [{"role": "user", "content": prompt}],
            cfg["model"], device,
        ) for prompt in HELDOUT_PROMPTS]

        # Untimed runtime witness: both variants must present the exact same
        # first real tree and probability tensor to their respective verifiers.
        witness = {}
        for name in ("ddtree", "tree_block_verification"):
            def observe(parents, tree_tokens, all_p, witness_name=name):
                if witness_name not in witness:
                    witness[witness_name] = (
                        list(parents), list(tree_tokens), all_p.detach().cpu().clone(),
                    )
            engine.generate(
                encoded[0], by_name[name], 32, stops, seed=20260913,
                tree_observer=observe,
            )
        left, right = witness["ddtree"], witness["tree_block_verification"]
        if left[:2] != right[:2] or not torch.equal(left[2], right[2]):
            raise RuntimeError("Same-tree/runtime Target-row witness failed")
        manifest["same_tree_runtime_witness"] = {
            "passed": True,
            "nodes": len(left[0]),
            "tree_sha256": digest([left[0], left[1]]),
            "probability_shape": list(left[2].shape),
            "probability_dtype": str(left[2].dtype),
        }
        write_json(output / "manifest.json", manifest)

        for variant in declared:
            engine.generate(encoded[0], variant, 32, stops, seed=20260914)

        rows = []
        names = list(by_name)
        for prompt_index, ids in enumerate(encoded):
            for repeat in range(repeats):
                allocation_gate(device)
                for position, name in enumerate(
                        paired_order(prompt_index, repeat, names)):
                    variant = by_name[name]
                    result = engine.generate(
                        ids, variant, tokens, stops,
                        seed=20260913 + prompt_index,
                    )
                    row = compact(
                        result, prompt=prompt_index, repeat=repeat,
                        order=position, variant=variant,
                    )
                    if name in {"ddtree", "tree_block_verification"} and any(
                            round_["tree_nodes"] > 45
                            for round_ in result["rounds"]):
                        raise RuntimeError(f"{name} exceeded B=45")
                    rows.append(row)
                    write_json(output / "rows.json", {
                        "primary_timing": False, "rows": rows,
                    })
                    print(
                        f"same-tree-e2e prompt={prompt_index + 1}/{len(encoded)} "
                        f"repeat={repeat + 1}/{repeats} method={name} "
                        f"tpot={row['decode_ms'] / row['decode_tokens']:.6f}",
                        flush=True,
                    )

        comparisons = {
            baseline: compare(
                rows, "tree_block_verification", baseline,
            ) for baseline in ("ddtree", "dflash")
        }
        baseline_sanity = compare(rows, "ddtree", "dflash")
        first = {
            (row["prompt"], row["variant"]): row["generated_sha256"]
            for row in rows if row["repeat"] == 0
        }
        repeat_equal = all(
            row["generated_sha256"] == first[row["prompt"], row["variant"]]
            for row in rows
        )
        passed = (
            repeat_equal
            and all(value["ci95"][0] > 1 for value in comparisons.values())
            and baseline_sanity["ci95"][0] > 1
        )
        report = {
            "gate_passed": passed,
            "formal_complete": False,
            "within_method_repeat_equal": repeat_equal,
            "comparisons": comparisons,
            "baseline_sanity": baseline_sanity,
            "aggregate": {
                name: aggregate(rows, name) for name in names
            },
            "records": len(rows),
            "telemetry_end": telemetry(device),
            "decision": (
                "freeze_for_formal_dataset_suite" if passed
                else "stop_and_optimize_or_increase_power"
            ),
        }
        write_json(output / "report.json", report)
        print(report, flush=True)
        return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=Path("configs/gbv_paper_ddtree_counts.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=6)
    args = parser.parse_args()
    run(
        args.config.resolve(), args.output.resolve(), args.device,
        args.tokens, args.repeats,
    )


if __name__ == "__main__":
    main()
