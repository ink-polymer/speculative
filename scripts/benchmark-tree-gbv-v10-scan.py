"""Coarse same-process scan for unbiased finite-tree GBV + block verification."""
from __future__ import annotations

import json
import os
from pathlib import Path
import random
import runpy

import torch

from gbv_experiments.config import Variant, load_config
from gbv_experiments.conversation import encode_messages
from gbv_experiments.engine import load_models


ROOT = Path(os.environ["FUYILE_ROOT"])
SOURCE = ROOT / "src/gbv-optimized"
OUTPUT = ROOT / "logs" / f"tree-gbv-v10-scan-{os.environ['SLURM_JOB_ID']}.json"
shared = runpy.run_path(str(ROOT / "scripts/benchmark-tree-gbv-v2.py"))
summary = shared["summary"]
PROMPTS = shared["VALIDATION_PROMPTS"]


def save(value):
    temporary = OUTPUT.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(OUTPUT)


@torch.inference_mode()
def main():
    cfg = load_config(SOURCE / "configs/gbv_paper_full.json")
    engine, tokenizer = load_models(cfg["model"], "cuda:0")
    methods = [
        Variant(
            name="ddtree", method="ddtree", paths=1, length=15,
            temperature=1.0, draft_temperature=1.0, tree_budget=45,
            probability_dtype="float64",
        )
    ]
    for leaves in (2, 3, 4, 6, 8):
        for virtual_k in (1, 2, 4, 8):
            methods.append(Variant(
                name=f"tree_full_lf{leaves}_k{virtual_k}",
                method="tree_gbv_full", paths=virtual_k, length=14,
                temperature=1.0, draft_temperature=1.0,
                tree_budget=leaves, probability_dtype="float64",
            ))
    for leaves in (4, 6, 8):
        for virtual_k in (2, 4):
            methods.append(Variant(
                name=f"tree_prefix_lf{leaves}_k{virtual_k}",
                method="tree_gbv_prefix", paths=virtual_k, length=14,
                temperature=1.0, draft_temperature=1.0,
                tree_budget=leaves, probability_dtype="float64",
            ))

    report = {
        "schema": 10,
        "purpose": "coarse finite-tree GBV block-verification scan",
        "temperature": {"target": 1.0, "draft": 1.0},
        "probability_dtype": "float64",
        "complete": False,
        "validation": {},
    }
    warmup_ids = encode_messages(
        tokenizer, [{"role": "user", "content": "Compute 1 + 1."}],
        cfg["model"], engine.device,
    )
    for index, variant in enumerate(methods):
        engine.generate(warmup_ids, variant, 24, [], seed=index + 1, profile=False)

    collected = {variant.name: [] for variant in methods}
    by_name = {variant.name: variant for variant in methods}
    for prompt_index, prompt in enumerate(PROMPTS[:4]):
        ids = encode_messages(
            tokenizer, [{"role": "user", "content": prompt}],
            cfg["model"], engine.device,
        )
        for seed in (1709, 2903):
            order = methods.copy()
            random.Random(seed + prompt_index * 100_003).shuffle(order)
            for variant in order:
                collected[variant.name].append(
                    engine.generate(ids, variant, 96, [], seed=seed, profile=True)
                )
                print(
                    f"prompt={prompt_index + 1}/4 seed={seed} {variant.name}",
                    flush=True,
                )
        report["validation"] = {
            name: {"variant": by_name[name].to_dict(), "summary": summary(results)}
            for name, results in collected.items() if results
        }
        save(report)

    ddtree_tps = report["validation"]["ddtree"]["summary"]["decode_tokens_per_second"]
    ranking = []
    for name, value in report["validation"].items():
        if name == "ddtree":
            continue
        result = value["summary"]
        ranking.append({
            "name": name,
            "decode_tokens_per_second": result["decode_tokens_per_second"],
            "speedup_vs_ddtree": result["decode_tokens_per_second"] / ddtree_tps,
            "mean_committed_tokens_per_round": result["mean_committed_tokens_per_round"],
            "mean_accepted_draft_tokens_per_round": result["mean_accepted_draft_tokens_per_round"],
            "mean_tree_nodes_per_round": result["mean_tree_nodes_per_round"],
        })
    ranking.sort(key=lambda row: row["decode_tokens_per_second"], reverse=True)
    report["ranking"] = ranking
    report["complete"] = True
    save(report)
    print(json.dumps({"ddtree_tps": ddtree_tps, "top": ranking[:10]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
