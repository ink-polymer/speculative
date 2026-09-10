"""Diagnostic stage profiles for DDTree and core-spur architectures."""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import torch

from gbv_experiments.common import write_json
from gbv_experiments.config import Variant, load_config
from gbv_experiments.conversation import encode_messages
from gbv_experiments.engine import load_models
from gbv_experiments.runner import stop_token_ids
from gbv_experiments.terminal_formal import allocation_gate, model_gate, telemetry


PROMPT = "Explain why binary search needs a monotone predicate and give a short example."


@torch.inference_mode()
def run(config: Path, output: Path, tokens: int, device: str) -> None:
    cfg = load_config(config)
    allocation_gate(device)
    engine, tokenizer = load_models(cfg["model"], device)
    model_gate(engine)
    stops = stop_token_ids(engine, tokenizer)
    ids = encode_messages(
        tokenizer, [{"role": "user", "content": PROMPT}],
        cfg["model"], device,
    )
    base = Variant(
        name="ddtree", method="ddtree", paths=1, length=15,
        temperature=1., draft_temperature=1., tree_budget=45,
        probability_dtype="float64", diffusion_support_size=32,
    )
    variants = [
        base,
        replace(
            base, name="core_spur_2", method="diffusion_core_spur_bv",
            diffusion_spur_length=2,
        ),
        replace(
            base, name="core_spur_6", method="diffusion_core_spur_bv",
            diffusion_spur_length=6,
        ),
    ]
    report = {"uses_cuda_graph": False, "tokens": tokens, "profiles": {}}
    for variant in variants:
        engine.generate(ids, variant, 32, stops, seed=20260910)
        result = engine.generate(
            ids, variant, tokens, stops, seed=20260910, profile=True,
        )
        report["profiles"][variant.name] = {
            "official_tpot_ms": result[
                "official_scope_time_per_output_token_ms"
            ],
            "rounds": len(result["rounds"]),
            "mean_committed_tokens": sum(
                round_["committed_tokens"] for round_ in result["rounds"]
            ) / len(result["rounds"]),
            "stage_profile": result["stage_profile"],
        }
    report["telemetry_end"] = telemetry(device)
    write_json(output, report)
    print(report, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=96)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    run(args.config, args.output, args.tokens, args.device)


if __name__ == "__main__":
    main()
