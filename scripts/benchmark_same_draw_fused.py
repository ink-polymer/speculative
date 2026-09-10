"""Micro-gate a same-multinomial CUDA tree walk on captured DDTree states."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
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
    tree_verify_ancestral_same_draw_fused,
)
from gbv_experiments.sampling import tree_verify_ancestral_batched
from gbv_experiments.terminal_formal import (
    DIAGNOSTIC_PROMPTS,
    allocation_gate,
    model_gate,
    telemetry,
)


def summarize(rows: list[dict], bootstrap: int = 20_000) -> dict:
    logs = []
    for state in sorted({row["state"] for row in rows}):
        selected = [row for row in rows if row["state"] == state]
        means = {
            method:sum(row["ms_per_call"] for row in selected
                       if row["method"] == method)
            / sum(row["method"] == method for row in selected)
            for method in ("ddtree", "same_draw_fused")
        }
        logs.append(math.log(means["ddtree"] / means["same_draw_fused"]))
    values = np.asarray(logs, dtype=np.float64)
    rng = np.random.default_rng(20260910)
    draws = rng.choice(
        values, size=(bootstrap, len(values)), replace=True,
    ).mean(1)
    low, high = np.exp(np.quantile(draws, (0.025, 0.975)))
    return {
        "metric":"same_multinomial_real_tree_verifier_speedup",
        "speedup":math.exp(float(values.mean())),
        "ci95":[float(low), float(high)],
        "state_clusters":len(values),
        "gate_passed":bool(low > 1),
    }


@torch.inference_mode()
def run(config: Path, output: Path, device: str, states_per_prompt: int,
        warmup: int, iterations: int, repeats: int) -> dict:
    if min(states_per_prompt, warmup, iterations, repeats) < 1:
        raise ValueError("All benchmark counts must be positive")
    if output.exists():
        raise ValueError("Output exists; choose a fresh file")
    cfg = load_config(config)
    precision = {
        key:cfg["model"].get(key) for key in (
            "dtype", "target_attention", "draft_attention", "allow_tf32",
        )
    }
    if precision != {
            "dtype":"bfloat16", "target_attention":"sdpa",
            "draft_attention":"sdpa", "allow_tf32":False}:
        raise ValueError(f"Official precision controls changed: {precision}")
    allocation_gate(device)
    engine, tokenizer = load_models(cfg["model"], device)
    model_gate(engine)
    variant = Variant(
        name="capture", method="ddtree", paths=1, length=15,
        temperature=1.0, draft_temperature=1.0, tree_budget=45,
        probability_dtype="float64",
    )
    captured = []
    for prompt_index, prompt in enumerate(DIAGNOSTIC_PROMPTS):
        ids = encode_messages(
            tokenizer, [{"role":"user", "content":prompt}],
            cfg["model"], device,
        )
        states = []

        def observe(parents, tokens, all_p):
            if len(states) < states_per_prompt:
                states.append((
                    list(parents), list(tokens), all_p.detach().clone(),
                ))

        engine.generate(
            ids, variant, 96, [], seed=20260910 + prompt_index,
            tree_observer=observe,
        )
        if len(states) != states_per_prompt:
            raise RuntimeError("Did not capture requested DDTree states")
        captured.extend((prompt_index, *state) for state in states)

    correctness = []
    for state, (prompt, parents, tokens, p) in enumerate(captured):
        for seed in range(16):
            expected_generator = torch.Generator(device=device).manual_seed(seed)
            actual_generator = torch.Generator(device=device).manual_seed(seed)
            expected = tree_verify_ancestral_batched(
                parents, tokens, p, expected_generator, validate=False,
            )
            actual = tree_verify_ancestral_same_draw_fused(
                parents, tokens, p, actual_generator, validate=False,
            )
            if expected != actual:
                raise AssertionError(
                    f"State {state}, seed {seed}: {actual} != {expected}"
                )
            if not torch.equal(
                    expected_generator.get_state(),
                    actual_generator.get_state()):
                raise AssertionError("Candidate changed the RNG state")
        correctness.append({
            "state":state, "prompt":prompt,
            "tree_sha256":digest([parents, tokens]),
            "seeds_passed":16,
        })

    rows = []
    for state, (prompt, parents, tokens, p) in enumerate(captured):
        functions = {
            "ddtree":lambda generator:tree_verify_ancestral_batched(
                parents, tokens, p, generator, validate=False,
            ),
            "same_draw_fused":lambda generator:
                tree_verify_ancestral_same_draw_fused(
                    parents, tokens, p, generator, validate=False,
                ),
        }
        for name, function in functions.items():
            generator = torch.Generator(device=device).manual_seed(20260910)
            for _ in range(warmup):
                function(generator)
        for repeat in range(repeats):
            order = list(functions)
            random.Random(20260910 + state * 100 + repeat).shuffle(order)
            for name in order:
                generator = torch.Generator(device=device).manual_seed(
                    20260910 + repeat
                )
                torch.cuda.synchronize(device)
                started = time.perf_counter()
                for _ in range(iterations):
                    functions[name](generator)
                torch.cuda.synchronize(device)
                elapsed_ms = (time.perf_counter() - started) * 1000
                row = {
                    "state":state, "prompt":prompt, "method":name,
                    "repeat":repeat, "nodes":len(parents),
                    "iterations":iterations,
                    "ms_per_call":elapsed_ms / iterations,
                }
                rows.append(row)
                print(
                    f"state={state + 1}/{len(captured)} "
                    f"repeat={repeat + 1}/{repeats} method={name} "
                    f"ms={row['ms_per_call']:.6f}", flush=True,
                )
    summary = summarize(rows)
    report = {
        "kind":"same_multinomial_fused_tree_walk_gate",
        "formal_complete":False,
        "created_at_utc":datetime.now(timezone.utc).isoformat(),
        "official_precision":precision,
        "same_tree":True,
        "same_target_probabilities":True,
        "same_multinomial_shape_and_rng_state":True,
        "correctness":correctness,
        "settings":{
            "states_per_prompt":states_per_prompt,
            "warmup":warmup,
            "iterations":iterations,
            "repeats":repeats,
        },
        "summary":summary,
        "rows":rows,
        "telemetry_end":telemetry(device),
        "source_sha256":{
            "candidate":file_hash(Path(__file__).resolve().parents[1]
                                  / "src/gbv_experiments/fused_tree_sampling.py"),
            "script":file_hash(Path(__file__).resolve()),
            "config":file_hash(config),
        },
    }
    write_json(output, report)
    print(summary, flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--states-per-prompt", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    run(
        args.config.resolve(), args.output.resolve(), args.device,
        args.states_per_prompt, args.warmup, args.iterations, args.repeats,
    )


if __name__ == "__main__":
    main()
