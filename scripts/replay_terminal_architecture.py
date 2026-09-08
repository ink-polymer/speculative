"""Replay DDTree verifiers on identical real tree tensors.

This is a diagnostic microbenchmark, not an end-to-end throughput result.  It
captures probability-tree states from the model, then times only the verifier
call.  Prompts, token ids, and probability tensors are never serialized.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import random
import statistics
import time

import torch

from gbv_experiments.common import digest, file_hash
from gbv_experiments.config import Variant, load_config
from gbv_experiments.conversation import encode_messages
from gbv_experiments.engine import load_models
from gbv_experiments.fairness import assert_architecture_only_pair
from gbv_experiments.terminal_formal import (DIAGNOSTIC_PROMPTS,
                                             allocation_gate, model_gate,
                                             replay_functions)


TEMPERATURES = (0.3, 0.6, 1.0)
METHODS = ("ddtree", "tm_full", "tm_dense_exit")


def _geomean(values):
    values = list(values)
    if not values or any(value <= 0 for value in values):
        raise ValueError("Geometric means require positive values")
    return math.exp(statistics.fmean(math.log(value) for value in values))


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _summarize(rows: list[dict]) -> dict:
    paired = {}
    for row in rows:
        key = (row["temperature"], row["prompt_index"], row["state_index"],
               row["repeat"])
        paired.setdefault(key, {})[row["method"]] = row["ms_per_call"]
    if any(set(value) != set(METHODS) for value in paired.values()):
        raise ValueError("Incomplete verifier timing pair")

    result = {}
    for method in METHODS[1:]:
        by_temperature = {}
        all_speedups = []
        for temperature in TEMPERATURES:
            speedups = [
                value["ddtree"] / value[method]
                for key, value in paired.items()
                if key[0] == temperature
            ]
            all_speedups.extend(speedups)
            by_temperature[str(temperature)] = {
                "paired_observations": len(speedups),
                "geomean_speedup_vs_ddtree": _geomean(speedups),
                "median_speedup_vs_ddtree": statistics.median(speedups),
                "min_speedup_vs_ddtree": min(speedups),
                "max_speedup_vs_ddtree": max(speedups),
            }
        result[method] = {
            "paired_observations": len(all_speedups),
            "geomean_speedup_vs_ddtree": _geomean(all_speedups),
            "median_speedup_vs_ddtree": statistics.median(all_speedups),
            "by_temperature": by_temperature,
            "wins_architecture_gate": all(
                value["geomean_speedup_vs_ddtree"] > 1
                and value["median_speedup_vs_ddtree"] > 1
                for value in by_temperature.values()
            ),
        }
    return result


def _variant(name: str, method: str, temperature: float) -> Variant:
    return Variant(
        name=name, method=method, paths=1, length=15,
        temperature=temperature, draft_temperature=temperature,
        tree_budget=45, probability_dtype="float64",
    )


@torch.inference_mode()
def run(args) -> dict:
    if args.output.exists():
        raise ValueError("Output already exists; choose a new path")
    cfg = load_config(args.config)
    baseline = _variant("ddtree", "ddtree", 1.0)
    fairness = {
        method: assert_architecture_only_pair(
            baseline, _variant(method, variant_method, 1.0), cfg["model"]
        )
        for method, variant_method in (
            ("tm_full", "ddtree_terminal_block"),
            ("tm_dense_exit", "ddtree_terminal_dense"),
        )
    }

    allocation_gate(args.device)
    engine, tokenizer = load_models(cfg["model"], args.device)
    model_gate(engine)
    device = engine.device
    report = {
        "schema": 1,
        "kind": "same_real_tree_verifier_replay",
        "complete": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": args.config.name,
        "config_sha256": file_hash(args.config),
        "model": cfg["model"],
        "runtime": {
            "gpu": torch.cuda.get_device_name(device),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "python": platform.python_version(),
        },
        "contract": {
            "same_captured_parents_tokens_and_target_probabilities": True,
            "same_tree_constructor": "probability_tree",
            "same_length": 15,
            "same_tree_budget": 45,
            "same_probability_dtype": "float64",
            "timed_scope": "verifier call including its topology preparation and host result transfer",
            "excluded": ["draft forward", "tree construction", "target forward", "cache compaction"],
            "not_an_end_to_end_speedup": True,
            "prompts_or_tokens_serialized": False,
        },
        "fairness": fairness,
        "settings": {
            "temperatures": list(TEMPERATURES),
            "prompt_sha256": [digest(prompt) for prompt in DIAGNOSTIC_PROMPTS],
            "capture_tokens": args.capture_tokens,
            "states_per_prompt": args.states_per_prompt,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "repeats": args.repeats,
        },
        "rows": [],
    }
    _write(args.output, report)

    for temperature in TEMPERATURES:
        variant = _variant("ddtree", "ddtree", temperature)
        for prompt_index, prompt in enumerate(DIAGNOSTIC_PROMPTS):
            ids = encode_messages(
                tokenizer, [{"role": "user", "content": prompt}],
                cfg["model"], device,
            )
            captured = []

            def observe(parents, proposed, all_p):
                if len(captured) < args.states_per_prompt:
                    captured.append({
                        "kind": "probability_tree",
                        "parents": list(parents),
                        "tokens": list(proposed),
                        "all_p": all_p.detach().cpu().clone(),
                    })

            engine.generate(
                ids, variant, args.capture_tokens, [],
                seed=args.seed + prompt_index, tree_observer=observe,
            )
            if len(captured) != args.states_per_prompt:
                raise RuntimeError(
                    f"Captured {len(captured)} states, expected {args.states_per_prompt}"
                )

            for state_index, state in enumerate(captured):
                gpu_state = {
                    key: value.to(device) if isinstance(value, torch.Tensor) else value
                    for key, value in state.items()
                }
                functions = {
                    name: function
                    for name, function in replay_functions(gpu_state).items()
                    if name in METHODS
                }
                if set(functions) != set(METHODS):
                    raise RuntimeError("Replay function set changed")
                for name, function in functions.items():
                    generator = torch.Generator(device=device).manual_seed(
                        args.seed + state_index
                    )
                    for _ in range(args.warmup):
                        function(generator)

                for repeat in range(args.repeats):
                    order = list(METHODS)
                    random.Random(
                        args.seed + 1000 * prompt_index + 10 * state_index + repeat
                    ).shuffle(order)
                    for name in order:
                        generator = torch.Generator(device=device).manual_seed(
                            args.seed + repeat
                        )
                        torch.cuda.synchronize(device)
                        allocated_before = torch.cuda.memory_allocated(device)
                        torch.cuda.reset_peak_memory_stats(device)
                        started = time.perf_counter()
                        for _ in range(args.iterations):
                            functions[name](generator)
                        torch.cuda.synchronize(device)
                        total_ms = 1000 * (time.perf_counter() - started)
                        report["rows"].append({
                            "temperature": temperature,
                            "prompt_index": prompt_index,
                            "state_index": state_index,
                            "nodes": len(state["parents"]),
                            "method": name,
                            "repeat": repeat,
                            "iterations": args.iterations,
                            "total_ms": total_ms,
                            "ms_per_call": total_ms / args.iterations,
                            "incremental_peak_allocated_bytes": max(
                                0, torch.cuda.max_memory_allocated(device)
                                - allocated_before,
                            ),
                        })
                del functions, gpu_state
            del captured
            _write(args.output, report)
            print(
                f"replay T={temperature} prompt={prompt_index + 1}/"
                f"{len(DIAGNOSTIC_PROMPTS)} rows={len(report['rows'])}",
                flush=True,
            )

    report["summary"] = _summarize(report["rows"])
    report["complete"] = True
    report["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    _write(args.output, report)
    print(json.dumps(report["summary"], indent=2, sort_keys=True), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--capture-tokens", type=int, default=64)
    parser.add_argument("--states-per-prompt", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--seed", type=int, default=20260908)
    args = parser.parse_args()
    if min(args.capture_tokens, args.states_per_prompt, args.warmup,
           args.iterations, args.repeats) < 1:
        parser.error("All count arguments must be positive")
    run(args)


if __name__ == "__main__":
    main()
