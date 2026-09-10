"""Measure the tree-construction headroom inside a DDTree candidate pool.

This is a diagnostic upper bound, not a runnable decoding method.  A large
DDTree is verified once, then the exact Target prefix masses observed in that
forward are used offline to select the best ancestry-closed B-node subtree.
The comparison is against the first B nodes of the same large DDTree, which is
exactly the ordinary DDTree allocation at that budget.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from gbv_experiments.config import Variant, load_config
from gbv_experiments.conversation import encode_messages
from gbv_experiments.engine import load_models
from gbv_experiments.runner import stop_token_ids
from gbv_experiments.terminal_formal import allocation_gate, model_gate

from benchmark_prefix_conditional_tree import DEVELOPMENT_PROMPTS


def prefix_masses(parents: list[int], tokens: list[int], p: torch.Tensor):
    mass = torch.ones(len(parents), dtype=torch.float64, device=p.device)
    for node, (parent, token) in enumerate(zip(parents[1:], tokens), 1):
        mass[node] = mass[parent] * p[parent, token].double()
    return mass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=45)
    parser.add_argument("--pool-budget", type=int, default=180)
    parser.add_argument("--tokens", type=int, default=48)
    parser.add_argument("--prompt-count", type=int, default=3)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if not 1 <= args.budget <= args.pool_budget or args.prompt_count < 1:
        raise ValueError("Invalid oracle-tree controls")

    cfg = load_config(args.config)["model"]
    allocation_gate(args.device)
    engine, tokenizer = load_models(cfg, args.device)
    model_gate(engine)
    stops = stop_token_ids(engine, tokenizer)
    base = Variant(
        name="oracle_pool", method="ddtree", paths=1, length=15,
        temperature=1., draft_temperature=1., tree_budget=args.pool_budget,
        probability_dtype="float64",
    )
    base.validate()
    rows = []
    for prompt_index, prompt in enumerate(
            DEVELOPMENT_PROMPTS[:args.prompt_count]):
        ids = encode_messages(
            tokenizer, [{"role": "user", "content": prompt}], cfg,
            args.device,
        )
        observations = []

        def observe(parents, tokens, p):
            masses = prefix_masses(parents, tokens, p)
            # Prefix probability never exceeds its parent's probability, so
            # the exact top-B set is ancestry closed.
            oracle = torch.topk(masses[1:], args.budget).indices + 1
            baseline = torch.arange(
                1, args.budget + 1, device=masses.device,
            )
            observations.append({
                "ddtree_expected_accepted": float(masses[baseline].sum()),
                "oracle_expected_accepted": float(masses[oracle].sum()),
                "oracle_nodes_outside_ddtree": int(
                    (~torch.isin(oracle, baseline)).sum()
                ),
            })

        engine.generate(
            ids, base, args.tokens, stops, seed=20260910 + prompt_index,
            tree_observer=observe,
        )
        ddtree = sum(x["ddtree_expected_accepted"] for x in observations)
        oracle = sum(x["oracle_expected_accepted"] for x in observations)
        row = {
            "prompt": prompt_index,
            "rounds": len(observations),
            "ddtree_expected_accepted": ddtree / len(observations),
            "oracle_expected_accepted": oracle / len(observations),
            "oracle_acceptance_ratio": oracle / ddtree,
            "mean_oracle_nodes_outside_ddtree": sum(
                x["oracle_nodes_outside_ddtree"] for x in observations
            ) / len(observations),
            "observations": observations,
        }
        rows.append(row)
        print(json.dumps({k: v for k, v in row.items()
                          if k != "observations"}), flush=True)

    report = {
        "diagnostic_only": True,
        "temperature": 1.,
        "budget": args.budget,
        "pool_budget": args.pool_budget,
        "rows": rows,
        "mean_oracle_acceptance_ratio": sum(
            row["oracle_acceptance_ratio"] for row in rows
        ) / len(rows),
        "mean_ddtree_expected_accepted": sum(
            row["ddtree_expected_accepted"] for row in rows
        ) / len(rows),
        "mean_oracle_expected_accepted": sum(
            row["oracle_expected_accepted"] for row in rows
        ) / len(rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
