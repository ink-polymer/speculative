"""High-budget scan for lossless GBV block verification on a prefix tree."""
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
OUTPUT = ROOT / "logs" / f"tree-gbv-v15-packed-{os.environ['SLURM_JOB_ID']}.json"
shared = runpy.run_path(str(ROOT / "scripts/benchmark-tree-gbv-v2.py"))
summary = shared["summary"]
PROMPTS = shared["VALIDATION_PROMPTS"]


def save(value):
    temp = OUTPUT.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(OUTPUT)


@torch.inference_mode()
def main():
    cfg = load_config(SOURCE / "configs/gbv_paper_full.json")
    engine, tokenizer = load_models(cfg["model"], "cuda:0")
    methods = [Variant(
        name="ddtree", method="ddtree", paths=1, length=15,
        temperature=1.0, draft_temperature=1.0, tree_budget=45,
        probability_dtype="float64",
    )]
    for length in (8, 11, 14):
        for budget in (6, 8):
            for virtual_k in (1, 2):
                methods.append(Variant(
                    name=f"packed_n{length}_b{budget}_k{virtual_k}",
                    method="tree_gbv_prefix_recycle_packed", paths=virtual_k,
                    length=length, temperature=1.0, draft_temperature=1.0,
                    tree_budget=budget, probability_dtype="float64",
                ))
    report = {
        "schema": 15, "complete": False,
        "purpose": "path-packed tree GBV architecture",
        "temperature": {"target": 1.0, "draft": 1.0},
        "probability_dtype": "float64", "validation": {},
    }
    warmup = encode_messages(tokenizer, [{"role": "user", "content": "Compute 1 + 1."}],
                             cfg["model"], engine.device)
    for index, variant in enumerate(methods):
        engine.generate(warmup, variant, 24, [], seed=index + 1, profile=False)
    collected = {v.name: [] for v in methods}
    by_name = {v.name: v for v in methods}
    for prompt_index, prompt in enumerate(PROMPTS[:4]):
        ids = encode_messages(tokenizer, [{"role": "user", "content": prompt}],
                              cfg["model"], engine.device)
        for seed in (1709, 2903):
            order = methods.copy()
            random.Random(seed + prompt_index * 100_003).shuffle(order)
            for variant in order:
                collected[variant.name].append(
                    engine.generate(ids, variant, 96, [], seed=seed, profile=True)
                )
                print(f"prompt={prompt_index + 1}/4 seed={seed} {variant.name}", flush=True)
        report["validation"] = {
            name: {"variant": by_name[name].to_dict(), "summary": summary(rows)}
            for name, rows in collected.items() if rows
        }
        save(report)
    ddtree = report["validation"]["ddtree"]["summary"]["decode_tokens_per_second"]
    report["ranking"] = sorted(({
        "name": name,
        "decode_tokens_per_second": value["summary"]["decode_tokens_per_second"],
        "speedup_vs_ddtree": value["summary"]["decode_tokens_per_second"] / ddtree,
        "mean_committed_tokens_per_round": value["summary"]["mean_committed_tokens_per_round"],
        "mean_tree_nodes_per_round": value["summary"]["mean_tree_nodes_per_round"],
        "stage_cuda_ms": value["summary"]["stage_cuda_ms"],
    } for name, value in report["validation"].items() if name != "ddtree"),
        key=lambda row: row["decode_tokens_per_second"], reverse=True)
    report["complete"] = True
    save(report)
    print(json.dumps({"ddtree_tps": ddtree, "top": report["ranking"][:10]}, indent=2))


if __name__ == "__main__":
    main()
