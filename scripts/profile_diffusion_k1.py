"""Profile official-precision DDTree and K=1 diffusion tree-block stages."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch

from gbv_experiments.common import file_hash, write_json
from gbv_experiments.config import Variant, load_config
from gbv_experiments.conversation import encode_messages
from gbv_experiments.engine import load_models
from gbv_experiments.runner import output_lock, stop_token_ids
from gbv_experiments.terminal_formal import DIAGNOSTIC_PROMPTS, allocation_gate, model_gate


def declared_variants() -> tuple[Variant, ...]:
    return (
        Variant(name="dflash", method="dflash", paths=1, length=15,
                temperature=1., draft_temperature=1., tree_budget=45,
                probability_dtype="float64"),
        Variant(name="ddtree", method="ddtree", paths=1, length=15,
                temperature=1., draft_temperature=1., tree_budget=45,
                probability_dtype="float64"),
        Variant(name="tree_block", method="diffusion_scaffold_bv", paths=1,
                length=15, temperature=1., draft_temperature=1., tree_budget=45,
                probability_dtype="float64", diffusion_support_size=32),
    )


@torch.inference_mode()
def run(config: Path, output: Path, device: str, tokens: int, repeats: int) -> dict:
    if not 32 <= tokens <= 256 or repeats not in (1, 2, 3):
        raise ValueError("tokens must be 32..256 and repeats must be 1..3")
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
    variants = declared_variants()
    for variant in variants:
        variant.validate()

    with output_lock(output.parent):
        if output.exists():
            raise ValueError("Profile output exists; choose a fresh path")
        allocation_gate(device)
        engine, tokenizer = load_models(cfg["model"], device)
        model_gate(engine)
        stops = stop_token_ids(engine, tokenizer)
        encoded = [encode_messages(
            tokenizer, [{"role": "user", "content": prompt}], cfg["model"], device,
        ) for prompt in DIAGNOSTIC_PROMPTS]
        for variant in variants:
            engine.generate(encoded[0], variant, 24, stops, seed=20260911)

        rows = []
        for repeat in range(repeats):
            for prompt_index, ids in enumerate(encoded):
                for variant in variants:
                    allocation_gate(device)
                    result = engine.generate(
                        ids, variant, tokens, stops,
                        seed=20260911 + prompt_index, profile=True,
                    )
                    rows.append({
                        "variant": variant.name,
                        "prompt": prompt_index,
                        "repeat": repeat,
                        "decode_tokens": result["decode_tokens"],
                        "decode_ms": result["decode_ms"],
                        "stages": result["stages"],
                        "rounds": result["rounds"],
                    })
                    print(
                        f"profile prompt={prompt_index + 1}/{len(encoded)} "
                        f"repeat={repeat + 1}/{repeats} method={variant.name}",
                        flush=True,
                    )

        summary = {}
        for variant in variants:
            selected = [row for row in rows if row["variant"] == variant.name]
            decode_tokens = sum(row["decode_tokens"] for row in selected)
            stages = defaultdict(float)
            for row in selected:
                for stage, milliseconds in row["stages"]["cuda_event_ms"].items():
                    stages[stage] += milliseconds
            summary[variant.name] = {
                "decode_ms_per_token": sum(row["decode_ms"] for row in selected) / decode_tokens,
                "cuda_stage_ms_per_token": {
                    stage: milliseconds / decode_tokens
                    for stage, milliseconds in sorted(stages.items())
                },
                "mean_committed_tokens": sum(
                    round_["committed_tokens"]
                    for row in selected for round_ in row["rounds"]
                ) / sum(len(row["rounds"]) for row in selected),
            }
        report = {
            "kind": "development_stage_profile",
            "formal_complete": False,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "official_precision": precision,
            "tokens": tokens,
            "repeats": repeats,
            "variants": [variant.to_dict() for variant in variants],
            "source_sha256": {
                "engine": file_hash(Path(__file__).resolve().parents[1] / "src/gbv_experiments/engine.py"),
                "tree_block": file_hash(Path(__file__).resolve().parents[1] / "src/gbv_experiments/diffusion_tree_bv.py"),
                "script": file_hash(Path(__file__).resolve()),
                "config": file_hash(config),
            },
            "summary": summary,
            "rows": rows,
        }
        write_json(output, report)
        print(summary, flush=True)
        return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tokens", type=int, default=96)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    run(args.config.resolve(), args.output.resolve(), args.device, args.tokens, args.repeats)


if __name__ == "__main__":
    main()
