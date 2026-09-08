"""Run a fixed, diagnostic tree-depth/budget/backend optimization on CUDA."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import random

import torch

from gbv_experiments.common import ROOT, digest, file_hash, write_json
from gbv_experiments.config import load_config
from gbv_experiments.conversation import encode_messages
from gbv_experiments.diffusion_optimization import (
    BASELINE_NAMES, SCAFFOLD_GRID, TEMPERATURES, optimization_variants,
    profile_names, summarize,
)
from gbv_experiments.engine import load_models
from gbv_experiments.runner import output_lock, stop_token_ids
from gbv_experiments.terminal_formal import DIAGNOSTIC_PROMPTS


def compact_result(result: dict, *, prompt: int, repeat: int, variant) -> dict:
    return {
        "temperature": variant.temperature,
        "variant": variant.name,
        "method": variant.method,
        "paths": variant.paths,
        "length": variant.length,
        "tree_budget": variant.tree_budget,
        "prompt": prompt,
        "repeat": repeat,
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


@torch.inference_mode()
def run(config_path: Path, output: Path, device: str, tokens: int, repeats: int) -> dict:
    if not 32 <= tokens <= 256 or repeats not in (1, 2, 3):
        raise ValueError("tokens must be 32..256 and repeats must be 1..3")
    cfg = load_config(config_path)
    with output_lock(output):
        if (output / "summary.json").exists() or (output / "scan.json").exists():
            raise ValueError("Optimization output already exists; choose a new directory")
        manifest = {
            "kind": "diagnostic_diffusion_scaffold_optimization",
            "formal_complete": False,
            "selection_fixed_before_timing": True,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "config": str(config_path.resolve().relative_to(ROOT)),
            "config_sha256": file_hash(config_path),
            "model": cfg["model"],
            "device": device,
            "temperatures": list(TEMPERATURES),
            "baselines": list(BASELINE_NAMES),
            "grid": [{"paths": k, "length": length, "tree_budget": budget}
                     for k, length, budget in SCAFFOLD_GRID],
            "continuation_backends": ["terminal", "ancestral"],
            "prompts_sha256": [digest(prompt) for prompt in DIAGNOSTIC_PROMPTS],
            "tokens": tokens,
            "repeats": repeats,
            "limitations": [
                "The same three development prompts are used for selection and measurement",
                "Checkpoint equivalence is audited separately and is not implied by speed",
                "A winning candidate requires a later frozen held-out validation",
            ],
        }
        write_json(output / "manifest.json", manifest)
        engine, tokenizer = load_models(cfg["model"], device)
        manifest["gpu"] = torch.cuda.get_device_name(torch.device(device))
        write_json(output / "manifest.json", manifest)
        stops = stop_token_ids(engine, tokenizer)
        encoded = [encode_messages(tokenizer, [{"role": "user", "content": prompt}],
                                   cfg["model"], device)
                   for prompt in DIAGNOSTIC_PROMPTS]
        rows = []
        for temperature in TEMPERATURES:
            variants = optimization_variants(temperature)
            by_name = {variant.name: variant for variant in variants}
            for variant in variants:
                engine.generate(encoded[0], variant, 24, stops, seed=20260908)
            for repeat in range(repeats):
                order = list(by_name)
                random.Random(20260908 + repeat + int(10 * temperature)).shuffle(order)
                for prompt_index, ids in enumerate(encoded):
                    for name in order:
                        variant = by_name[name]
                        result = engine.generate(
                            ids, variant, tokens, stops,
                            seed=20260908 + prompt_index,
                        )
                        rows.append(compact_result(
                            result, prompt=prompt_index, repeat=repeat, variant=variant,
                        ))
                        write_json(output / "scan.json", {"primary_timing": False, "rows": rows})
                        print(
                            f"scan T={temperature} repeat={repeat + 1}/{repeats} "
                            f"prompt={prompt_index + 1}/{len(encoded)} variant={name} "
                            f"tpot={result['decode_ms'] / result['decode_tokens']:.6f}",
                            flush=True,
                        )
        summary = summarize(rows)
        profiles = []
        for temperature in TEMPERATURES:
            by_name = {variant.name: variant for variant in optimization_variants(temperature)}
            for name in profile_names(summary, temperature):
                result = engine.generate(
                    encoded[0], by_name[name], min(tokens, 96), stops,
                    seed=20260908, profile=True,
                )
                profiles.append({
                    "temperature": temperature,
                    "variant": name,
                    "decode_ms_per_token": result["decode_ms"] / result["decode_tokens"],
                    "stages": result["stages"],
                    "rounds": result["rounds"],
                })
        summary.update({
            "formal_complete": False,
            "checkpoint_gate": "separate_required",
            "profiles": profiles,
        })
        write_json(output / "summary.json", summary)
        print({"top": summary["overall_ranking"][:5]}, flush=True)
        return summary


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
