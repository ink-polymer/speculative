"""Strict architecture-only screen for hard-budget BRBV versus DDTree."""
from __future__ import annotations

import json
import os
from pathlib import Path
import random
import runpy
import hashlib

import torch

from gbv_experiments.config import Variant, load_config
from gbv_experiments.conversation import encode_messages
from gbv_experiments.engine import load_models
from gbv_experiments.fairness import assert_architecture_only_pair
from gbv_experiments.slot_mixer import load_ratio_transport_head, load_slot_mixer


ROOT = Path(os.environ["FUYILE_ROOT"])
SOURCE = ROOT / "src/gbv-optimized"
RUN_SCHEMA = int(os.environ.get("GBV_STRICT_SCHEMA", "18"))
RUN_TAG = os.environ.get("GBV_STRICT_TAG", "tree-gbv-v18-strict-budget")
CANDIDATE_NAME = os.environ.get("GBV_CANDIDATE_NAME", "hard_budget_brbv")
CANDIDATE_METHOD = os.environ.get(
    "GBV_CANDIDATE_METHOD", "tree_gbv_budgeted_prefix_recycle"
)
ADAPTER_CHECKPOINT = os.environ.get("GBV_PROPOSAL_ADAPTER")
ADAPTER_KIND = os.environ.get("GBV_PROPOSAL_ADAPTER_KIND", "slot_mixer")
OUTPUT = ROOT / "logs" / f"{RUN_TAG}-{os.environ['SLURM_JOB_ID']}.json"
shared = runpy.run_path(str(ROOT / "scripts/benchmark-tree-gbv-v2.py"))
summary = shared["summary"]
PROMPTS = shared["VALIDATION_PROMPTS"]


def save(value):
    temporary = OUTPUT.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(OUTPUT)


def assert_runtime_backend(engine):
    if engine.target.config._attn_implementation != "sdpa":
        raise RuntimeError("Target attention backend drifted from SDPA")
    if engine.draft.config._attn_implementation != "sdpa":
        raise RuntimeError("Draft attention backend drifted from SDPA")


@torch.inference_mode()
def main():
    cfg = load_config(SOURCE / "configs/gbv_paper_full.json")
    baseline = Variant(
        name="ddtree", method="ddtree", paths=1, length=15,
        temperature=1.0, draft_temperature=1.0, tree_budget=45,
        probability_dtype="float64",
    )
    candidate = Variant(
        name=CANDIDATE_NAME, method=CANDIDATE_METHOD,
        paths=3, length=15, temperature=1.0, draft_temperature=1.0,
        tree_budget=45, probability_dtype="float64",
    )
    fairness = assert_architecture_only_pair(baseline, candidate, cfg["model"])
    methods = [baseline, candidate]
    engine, tokenizer = load_models(cfg["model"], "cuda:0")
    adapter_record = None
    if ADAPTER_CHECKPOINT:
        checkpoint = Path(ADAPTER_CHECKPOINT)
        loader = (
            load_ratio_transport_head if ADAPTER_KIND == "ratio_transport"
            else load_slot_mixer
        )
        engine.proposal_adapter = loader(
            checkpoint,
            int(engine.draft.config.hidden_size),
            engine.device,
            {
                "target_revision": cfg["model"]["target_revision"],
                "draft_revision": cfg["model"]["draft_revision"],
            },
            dtype=next(engine.draft.parameters()).dtype,
        )
        adapter_record = {
            "path": str(checkpoint),
            "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        }
    assert_runtime_backend(engine)

    report = {
        "schema": RUN_SCHEMA,
        "complete": False,
        "purpose": "strict architecture-only hard-budget BRBV screen",
        "fairness": fairness,
        "model": cfg["model"],
        "proposal_adapter": adapter_record,
        "prompt_count": 4,
        "seeds": [1709, 2903],
        "max_new_tokens": 96,
        "validation": {},
    }
    warmup = encode_messages(
        tokenizer, [{"role": "user", "content": "Compute 1 + 1."}],
        cfg["model"], engine.device,
    )
    for variant in methods:
        assert_runtime_backend(engine)
        engine.generate(warmup, variant, 24, [], seed=1, profile=False)

    collected = {variant.name: [] for variant in methods}
    by_name = {variant.name: variant for variant in methods}
    for prompt_index, prompt in enumerate(PROMPTS[:4]):
        ids = encode_messages(
            tokenizer, [{"role": "user", "content": prompt}],
            cfg["model"], engine.device,
        )
        for seed in report["seeds"]:
            order = methods.copy()
            random.Random(seed + prompt_index * 100_003).shuffle(order)
            for variant in order:
                assert_runtime_backend(engine)
                result = engine.generate(
                    ids, variant, report["max_new_tokens"], [], seed=seed,
                    profile=True,
                )
                if any(round_["tree_nodes"] > 45 for round_ in result["rounds"]):
                    raise AssertionError(f"{variant.name} exceeded B=45")
                collected[variant.name].append(result)
                print(
                    f"prompt={prompt_index + 1}/4 seed={seed} {variant.name}",
                    flush=True,
                )
        report["validation"] = {
            name: {"variant": by_name[name].to_dict(), "summary": summary(rows)}
            for name, rows in collected.items() if rows
        }
        save(report)

    ddtree = report["validation"]["ddtree"]["summary"]
    proposed = report["validation"][CANDIDATE_NAME]["summary"]
    report["result"] = {
        "ddtree_tps": ddtree["decode_tokens_per_second"],
        "candidate_tps": proposed["decode_tokens_per_second"],
        "speedup_vs_ddtree": (
            proposed["decode_tokens_per_second"]
            / ddtree["decode_tokens_per_second"]
        ),
        "ddtree_committed_per_round": ddtree["mean_committed_tokens_per_round"],
        "candidate_committed_per_round": proposed["mean_committed_tokens_per_round"],
        "candidate_mean_tree_nodes": proposed["mean_tree_nodes_per_round"],
    }
    report["complete"] = True
    save(report)
    print(json.dumps(report["result"], indent=2))


if __name__ == "__main__":
    main()
