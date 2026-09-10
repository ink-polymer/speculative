"""Gate sparse-exit verification on captured real DDTree states."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import math
from pathlib import Path
import random
import time

import numpy as np
import torch

from gbv_experiments.common import digest, file_hash, write_json
from gbv_experiments.config import Variant, load_config
from gbv_experiments.conversation import encode_messages
from gbv_experiments.engine import load_models
from gbv_experiments.fused_tree_sampling import (
    tree_verify_ancestral_fused_scan,
    tree_verify_ancestral_sparse_exit_fused_scan,
)
from gbv_experiments.runner import output_lock
from gbv_experiments.sampling import tree_verify_ancestral_batched
from gbv_experiments.terminal_formal import DIAGNOSTIC_PROMPTS, allocation_gate, model_gate


def inverse_cdf_gate(device: str) -> None:
    parents = [-1, 0, 0, 1]
    tokens = [0, 1, 2]
    p = torch.tensor(
        [[.2, .3, .5], [.4, .1, .5], [.3, .6, .1], [.7, .2, .1]],
        dtype=torch.float64, device=device,
    )
    children = {(parent, tokens[node - 1]): node
                for node, parent in enumerate(parents[1:], 1)}
    for seed in range(64):
        expected_generator = torch.Generator(device=device).manual_seed(seed)
        actual_generator = torch.Generator(device=device).manual_seed(seed)
        uniforms = torch.rand(
            4, dtype=torch.float64, device=device, generator=expected_generator,
        ).cpu().tolist()
        expected_nodes, node, bonus = [], 0, None
        for uniform in uniforms:
            bonus = int(torch.searchsorted(
                p[node].cpu().cumsum(0),
                torch.tensor(uniform, dtype=torch.float64),
            ))
            child = children.get((node, bonus))
            if child is None:
                break
            expected_nodes.append(child)
            node = child
        actual = tree_verify_ancestral_fused_scan(
            parents, tokens, p, actual_generator, validate=True,
        )
        expected = (
            expected_nodes,
            [tokens[index - 1] for index in expected_nodes],
            bonus,
        )
        if actual != expected:
            raise AssertionError(
                f"Scan kernel differs from inverse CDF at seed {seed}: "
                f"{actual} != {expected}"
            )


def compare(rows: list[dict], candidate: str, baseline: str,
            bootstrap_samples: int = 10000) -> dict:
    clusters = {}
    for row in rows:
        clusters.setdefault(row["state"], {})[row["method"], row["repeat"]] = row["ms_per_call"]
    state_logs = []
    for state, values in sorted(clusters.items()):
        ratios = [
            values[baseline, repeat] / values[candidate, repeat]
            for repeat in sorted({key[1] for key in values})
        ]
        state_logs.append(sum(map(math.log, ratios)) / len(ratios))
    logs = np.asarray(state_logs, dtype=np.float64)
    rng = np.random.default_rng(20260912)
    draws = rng.choice(logs, size=(bootstrap_samples, len(logs)), replace=True).mean(1)
    low, high = np.exp(np.quantile(draws, [.025, .975]))
    speedup = math.exp(float(logs.mean()))
    return {
        "candidate": candidate,
        "baseline": baseline,
        "metric": "same_real_tree_verifier_speedup",
        "speedup": speedup,
        "ci95": [float(low), float(high)],
        "state_clusters": len(logs),
        "bootstrap_samples": bootstrap_samples,
        "gate_passed": bool(low > 1),
    }


@torch.inference_mode()
def run(config: Path, output: Path, device: str, states_per_prompt: int,
        warmup: int, iterations: int, repeats: int) -> dict:
    if min(states_per_prompt, warmup, iterations, repeats) < 1:
        raise ValueError("All benchmark counts must be positive")
    cfg = load_config(config)
    precision = {
        "dtype": cfg["model"].get("dtype"),
        "target_attention": cfg["model"].get("target_attention"),
        "draft_attention": cfg["model"].get("draft_attention"),
        "allow_tf32": cfg["model"].get("allow_tf32"),
    }
    if precision != {
        "dtype": "bfloat16", "target_attention": "sdpa",
        "draft_attention": "sdpa", "allow_tf32": False,
    }:
        raise ValueError(f"Official precision/backend controls changed: {precision}")
    with output_lock(output.parent):
        if output.exists():
            raise ValueError("Benchmark output exists; choose a fresh path")
        allocation_gate(device)
        inverse_cdf_gate(device)
        engine, tokenizer = load_models(cfg["model"], device)
        model_gate(engine)
        variant = Variant(
            name="ddtree_capture", method="ddtree", paths=1, length=15,
            temperature=1., draft_temperature=1., tree_budget=45,
            probability_dtype="float64",
        )
        captured = []
        for prompt_index, prompt in enumerate(DIAGNOSTIC_PROMPTS):
            ids = encode_messages(
                tokenizer, [{"role": "user", "content": prompt}],
                cfg["model"], device,
            )
            states = []

            def observe(parents, tokens, all_p):
                if len(states) < states_per_prompt:
                    states.append((list(parents), list(tokens), all_p.detach().cpu().clone()))

            engine.generate(
                ids, variant, 96, [], seed=20260912 + prompt_index,
                tree_observer=observe,
            )
            if len(states) != states_per_prompt:
                raise RuntimeError("Did not capture the requested DDTree states")
            captured.extend((prompt_index, *state) for state in states)

        rows = []
        for state_index, (prompt_index, parents, tokens, cpu_p) in enumerate(captured):
            p = cpu_p.to(device)
            functions = {
                "ddtree": lambda generator: tree_verify_ancestral_batched(
                    parents, tokens, p, generator, validate=False,
                ),
                "fused_scan": lambda generator: tree_verify_ancestral_fused_scan(
                    parents, tokens, p, generator, validate=False,
                ),
                "sparse_exit": lambda generator: (
                    tree_verify_ancestral_sparse_exit_fused_scan(
                        parents, tokens, p, generator, validate=False,
                    )
                ),
            }
            for name, function in functions.items():
                generator = torch.Generator(device=device).manual_seed(20260912)
                for _ in range(warmup):
                    function(generator)
            for repeat in range(repeats):
                order = list(functions)
                random.Random(20260912 + state_index * 100 + repeat).shuffle(order)
                for name in order:
                    generator = torch.Generator(device=device).manual_seed(
                        20260912 + repeat
                    )
                    torch.cuda.synchronize(device)
                    started = time.perf_counter()
                    for _ in range(iterations):
                        functions[name](generator)
                    torch.cuda.synchronize(device)
                    elapsed = (time.perf_counter() - started) * 1000
                    rows.append({
                        "state": state_index,
                        "prompt": prompt_index,
                        "nodes": len(parents),
                        "method": name,
                        "repeat": repeat,
                        "iterations": iterations,
                        "total_ms": elapsed,
                        "ms_per_call": elapsed / iterations,
                    })
                    print(
                        f"same-tree state={state_index + 1}/{len(captured)} "
                        f"repeat={repeat + 1}/{repeats} method={name} "
                        f"ms={elapsed / iterations:.6f}", flush=True,
                    )
        comparisons = {
            "sparse_exit_vs_ddtree": compare(rows, "sparse_exit", "ddtree"),
            "sparse_exit_vs_fused_scan": compare(
                rows, "sparse_exit", "fused_scan",
            ),
            "fused_scan_vs_ddtree": compare(rows, "fused_scan", "ddtree"),
        }
        report = {
            "kind": "same_real_ddtree_sparse_exit_gate",
            "formal_complete": False,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "official_precision": precision,
            "temperature": 1.0,
            "same_tree_and_target_rows": True,
            "candidate_changes_only_verifier": True,
            "registered_inverse_cdf_seeds_passed": 64,
            "sparse_exit_exact_conditional_test": (
                "tests/gbv_paper/test_terminal_mass.py::"
                "test_sparse_exit_fused_scan_matches_exact_conditional_construction"
            ),
            "settings": {
                "states_per_prompt": states_per_prompt,
                "warmup": warmup,
                "iterations": iterations,
                "repeats": repeats,
                "prompt_sha256": [digest(prompt) for prompt in DIAGNOSTIC_PROMPTS],
            },
            "versions": {name: importlib.metadata.version(name) for name in
                         ("torch", "transformers", "numpy")},
            "source_sha256": {
                "fused_sampler": file_hash(Path(__file__).resolve().parents[1] / "src/gbv_experiments/fused_tree_sampling.py"),
                "official_sampler": file_hash(Path(__file__).resolve().parents[1] / "src/gbv_experiments/sampling.py"),
                "script": file_hash(Path(__file__).resolve()),
                "config": file_hash(config),
            },
            "comparisons": comparisons,
            "gate_passed": bool(
                comparisons["sparse_exit_vs_ddtree"]["ci95"][0] > 1
                and comparisons["sparse_exit_vs_fused_scan"]["ci95"][0] > 1
            ),
            "rows": rows,
        }
        write_json(output, report)
        print(comparisons, flush=True)
        return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--states-per-prompt", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    run(args.config.resolve(), args.output.resolve(), args.device,
        args.states_per_prompt, args.warmup, args.iterations, args.repeats)


if __name__ == "__main__":
    main()
