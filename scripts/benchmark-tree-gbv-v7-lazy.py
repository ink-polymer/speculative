"""Same-process A/B of eager and lazy sparse Block Verification."""
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
OUTPUT = ROOT / "logs" / f"tree-gbv-v7-lazy-{os.environ['SLURM_JOB_ID']}.json"
shared = runpy.run_path(str(ROOT / "scripts/benchmark-tree-gbv-v2.py"))
summary = shared["summary"]
PROMPTS = shared["VALIDATION_PROMPTS"]


def save(value):
    temporary = OUTPUT.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(OUTPUT)


def per_round(summary_value, stage):
    return summary_value["stage_cuda_ms"][stage] / summary_value["rounds"]


@torch.inference_mode()
def main():
    cfg = load_config(SOURCE / "configs/gbv_paper_full.json")
    engine, tokenizer = load_models(cfg["model"], "cuda:0")
    common = dict(
        paths=10,
        length=14,
        temperature=1.0,
        draft_temperature=1.0,
        tree_budget=6,
        probability_dtype="float64",
    )
    methods = [
        Variant(
            name="tree_sparse_eager_reference",
            method="tree_gbv_full_sparse_ref",
            **common,
        ),
        Variant(
            name="tree_sparse_lazy",
            method="tree_gbv_full",
            **common,
        ),
        Variant(
            name="ddtree",
            method="ddtree",
            paths=1,
            length=15,
            temperature=1.0,
            draft_temperature=1.0,
            tree_budget=45,
            probability_dtype="float64",
        ),
        Variant(name="target", method="target", paths=1, temperature=1.0),
    ]
    report = {
        "schema": 7,
        "purpose": "same-process eager-versus-lazy sparse Block Verification A/B",
        "formal_outputs_modified": False,
        "temperature_contract": {
            "target_temperature": 1.0,
            "draft_temperature": 1.0,
            "all_methods_same_target_temperature": True,
        },
        "distribution_contract": {
            "claim": "exact output-law-preserving implementation factorization",
            "algorithm_source": "Block Verification (Sun et al., 2024)",
            "novel_algorithm_claimed": False,
            "validation": "exhaustive small-vocabulary output-law enumeration plus full test suite",
        },
        "gpu": torch.cuda.get_device_name(0),
        "validation": {},
        "complete": False,
    }
    warmup_ids = encode_messages(
        tokenizer,
        [{"role": "user", "content": "Compute 1 + 1."}],
        cfg["model"],
        engine.device,
    )
    for index, variant in enumerate(methods):
        engine.generate(warmup_ids, variant, 32, [], seed=index + 1, profile=False)

    collected = {variant.name: [] for variant in methods}
    by_name = {variant.name: variant for variant in methods}
    for prompt_index, prompt in enumerate(PROMPTS):
        ids = encode_messages(
            tokenizer,
            [{"role": "user", "content": prompt}],
            cfg["model"],
            engine.device,
        )
        for seed in (1709, 2903):
            order = methods.copy()
            random.Random(seed + prompt_index * 100_003).shuffle(order)
            for variant in order:
                collected[variant.name].append(
                    engine.generate(ids, variant, 128, [], seed=seed, profile=True)
                )
                print(
                    f"prompt={prompt_index + 1}/{len(PROMPTS)} "
                    f"seed={seed} {variant.name}",
                    flush=True,
                )
        report["validation"] = {
            name: {"variant": by_name[name].to_dict(), "summary": summary(results)}
            for name, results in collected.items()
            if results
        }
        save(report)

    eager = report["validation"]["tree_sparse_eager_reference"]["summary"]
    lazy = report["validation"]["tree_sparse_lazy"]["summary"]
    ddtree = report["validation"]["ddtree"]["summary"]
    target = report["validation"]["target"]["summary"]
    eager_select_per_round = per_round(eager, "select_and_correct")
    lazy_select_per_round = per_round(lazy, "select_and_correct")
    report["comparison"] = {
        "lazy_over_eager_throughput": (
            lazy["decode_tokens_per_second"] / eager["decode_tokens_per_second"]
        ),
        "lazy_over_ddtree_throughput": (
            lazy["decode_tokens_per_second"] / ddtree["decode_tokens_per_second"]
        ),
        "lazy_speedup_over_target": (
            lazy["decode_tokens_per_second"] / target["decode_tokens_per_second"]
        ),
        "eager_selection_cuda_ms_per_round": eager_select_per_round,
        "lazy_selection_cuda_ms_per_round": lazy_select_per_round,
        "selection_cuda_time_per_round_ratio": (
            lazy_select_per_round / eager_select_per_round
        ),
        "finite_sample_acceptance": {
            name: {
                key: value["summary"][key]
                for key in (
                    "rounds",
                    "mean_accepted_draft_tokens_per_round",
                    "mean_committed_tokens_per_round",
                    "mean_tree_nodes_per_round",
                )
            }
            for name, value in report["validation"].items()
            if name.startswith("tree_sparse_")
        },
    }
    report["complete"] = True
    save(report)
    print(json.dumps(report["comparison"], indent=2), flush=True)


if __name__ == "__main__":
    main()
