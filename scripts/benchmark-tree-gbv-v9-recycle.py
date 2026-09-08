"""Same-process A/B for residual-branch recycling Tree+GBV."""
from __future__ import annotations

import json
import os
from pathlib import Path
import random
import runpy
import statistics

import torch

from gbv_experiments.config import Variant, load_config
from gbv_experiments.conversation import encode_messages
from gbv_experiments.engine import load_models


ROOT = Path(os.environ["FUYILE_ROOT"])
SOURCE = ROOT / "src/gbv-optimized"
OUTPUT = ROOT / "logs" / f"tree-gbv-v9-recycle-{os.environ['SLURM_JOB_ID']}.json"
shared = runpy.run_path(str(ROOT / "scripts/benchmark-tree-gbv-v2.py"))
summary = shared["summary"]
PROMPTS = shared["VALIDATION_PROMPTS"]


def save(value):
    temporary = OUTPUT.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(OUTPUT)


def per_round(summary_value, stage):
    return summary_value["stage_cuda_ms"][stage] / summary_value["rounds"]


def recycling_summary(results):
    rounds = [row for result in results for row in result["rounds"]]
    return {
        "mean_bv_segments_per_round": statistics.mean(
            row["tree_bv_segments"] for row in rounds
        ),
        "mean_recycled_corrections_per_round": statistics.mean(
            row["recycled_corrections"] for row in rounds
        ),
        "fraction_of_rounds_with_recycling": statistics.mean(
            row["recycled_corrections"] > 0 for row in rounds
        ),
        "total_recycled_corrections": sum(
            row["recycled_corrections"] for row in rounds
        ),
    }


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
        Variant(name="tree_gbv_base", method="tree_gbv_full", **common),
        Variant(name="tree_gbv_recycle", method="tree_gbv_recycle", **common),
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
        "schema": 9,
        "purpose": "same-process residual-branch-recycling Tree+GBV A/B",
        "formal_outputs_modified": False,
        "temperature_contract": {
            "target_temperature": 1.0,
            "draft_temperature": 1.0,
            "all_methods_same_target_temperature": True,
        },
        "distribution_contract": {
            "claim": "composition of exact conditional GBV/BV kernels",
            "validation": "exhaustive small-tree enumeration of every dynamic sample branch plus full suite",
            "probability_dtype": "float64",
        },
        "design_boundary": {
            "mechanism": "reuse a BV correction token as the next anchor only when it is already a verified child, then run conditional GBV/BV on that subtree",
            "not_used": [
                "leaf-to-root traversal and sibling rejection updates",
                "local or global optimal transport",
                "copied implementations from related papers"
            ],
            "related_work_checked": [
                "Block Verification, arXiv:2403.10444",
                "Traversal Verification, arXiv:2505.12398",
                "GBV, arXiv:2602.16961",
                "SpecTr-GBV, arXiv:2604.25925",
                "UniVer, arXiv:2605.04543"
            ],
            "novelty_claimed": False,
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
        if collected["tree_gbv_recycle"]:
            report["recycling"] = recycling_summary(
                collected["tree_gbv_recycle"]
            )
        save(report)

    base = report["validation"]["tree_gbv_base"]["summary"]
    recycle = report["validation"]["tree_gbv_recycle"]["summary"]
    ddtree = report["validation"]["ddtree"]["summary"]
    target = report["validation"]["target"]["summary"]
    base_select_per_round = per_round(base, "select_and_correct")
    recycle_select_per_round = per_round(recycle, "select_and_correct")
    report["comparison"] = {
        "recycle_over_base_throughput": (
            recycle["decode_tokens_per_second"] / base["decode_tokens_per_second"]
        ),
        "recycle_over_ddtree_throughput": (
            recycle["decode_tokens_per_second"] / ddtree["decode_tokens_per_second"]
        ),
        "recycle_speedup_over_target": (
            recycle["decode_tokens_per_second"] / target["decode_tokens_per_second"]
        ),
        "base_selection_cuda_ms_per_round": base_select_per_round,
        "recycle_selection_cuda_ms_per_round": recycle_select_per_round,
        "selection_cuda_time_per_round_ratio": (
            recycle_select_per_round / base_select_per_round
        ),
        "base_rounds": base["rounds"],
        "recycle_rounds": recycle["rounds"],
        "base_mean_committed_tokens_per_round": (
            base["mean_committed_tokens_per_round"]
        ),
        "recycle_mean_committed_tokens_per_round": (
            recycle["mean_committed_tokens_per_round"]
        ),
    }
    report["complete"] = True
    save(report)
    print(json.dumps({
        "comparison": report["comparison"],
        "recycling": report["recycling"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
