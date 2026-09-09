"""Strict architecture-only screen of exact probability-tree samplers."""
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
from gbv_experiments.fairness import assert_architecture_only_pair


ROOT = Path(os.environ["FUYILE_ROOT"])
SOURCE = Path(os.environ.get("GBV_STRICT_SOURCE", ROOT / "src/gbv-optimized"))
RUN_SCHEMA = 24
RUN_TAG = os.environ.get("GBV_STRICT_TAG", "tree-gbv-v24-terminal-mass")
CANDIDATE_NAME = os.environ.get(
    "GBV_STRICT_CANDIDATE_NAME", "ddtree_terminal_mass_block"
)
CANDIDATE_METHOD = os.environ.get(
    "GBV_STRICT_CANDIDATE_METHOD", "ddtree_terminal_block"
)
CONFIG = Path(os.environ.get(
    "GBV_STRICT_CONFIG", SOURCE / "configs/gbv_paper_full.json"
))
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


def summarize_runs(rows):
    value = summary(rows)
    # Preserve explicit memory evidence even if the shared summary changes.
    for key in ("peak_allocated_bytes", "peak_reserved_bytes"):
        values = [row[key] for row in rows if row.get(key) is not None]
        value[key] = max(values) if values else None
    return value


@torch.inference_mode()
def main():
    cfg = load_config(CONFIG)
    baseline = Variant(
        name="ddtree", method="ddtree", paths=1, length=15,
        temperature=1.0, draft_temperature=1.0, tree_budget=45,
        probability_dtype="float64",
    )
    candidate = Variant(
        name=CANDIDATE_NAME, method=CANDIDATE_METHOD, paths=1,
        length=15, temperature=1.0, draft_temperature=1.0,
        tree_budget=45, probability_dtype="float64",
    )
    fairness = assert_architecture_only_pair(baseline, candidate, cfg["model"])
    methods = [baseline, candidate]
    engine, tokenizer = load_models(cfg["model"], "cuda:0")
    assert_runtime_backend(engine)
    for name, model in (("Target", engine.target), ("Draft", engine.draft)):
        actual_dtypes = {parameter.dtype for parameter in model.parameters()
                         if parameter.is_floating_point()}
        if actual_dtypes != {torch.bfloat16}:
            raise RuntimeError(f"{name} runtime weights are not BF16: {actual_dtypes}")
    if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
        raise RuntimeError("Runtime TF32 gate failed")

    report = {
        "schema": RUN_SCHEMA,
        "complete": False,
        "purpose": "strict exact tree sampler versus official batched-posterior DDTree",
        "gpu": torch.cuda.get_device_name(engine.device),
        "torch": torch.__version__,
        "fairness": fairness,
        "model": cfg["model"],
        "prompt_count": 8,
        "seeds": [1709, 2903, 3911],
        "max_new_tokens": 128,
        "contract": {
            "same_tree_constructor": "probability_tree",
            "same_expected_accepted_length": True,
            "same_seed_does_not_couple_realized_paths": True,
            "ddtree_posterior": "one batched categorical over every verified Target row",
            "only_delta": (
                "all-node batched posterior -> exact persistent multi-SM ancestral tree walk"
                if CANDIDATE_METHOD == "ddtree_fused_parallel"
                else "all-node batched posterior -> exact persistent single-block ancestral tree walk"
                if CANDIDATE_METHOD == "ddtree_fused"
                else "eager all-row vocabulary projection -> batched internal rows plus one reached leaf"
                if CANDIDATE_METHOD == "ddtree_lazy_projection"
                else "all-node batched posterior -> first-exit terminal-mass block draw"
            ),
            "candidate_method": CANDIDATE_METHOD,
            "terminal_mass_implementation": (
                "direct uncovered-mass sums, batched ancestor products"
                if CANDIDATE_METHOD.startswith("ddtree_terminal") else None
            ),
            "extra_temporary_memory": (
                "O(cooperative_blocks + tree_depth)"
                if CANDIDATE_METHOD == "ddtree_fused_parallel"
                else "O(tree_depth)"
                if CANDIDATE_METHOD == "ddtree_fused"
                else "O(internal_nodes * vocabulary)"
                if CANDIDATE_METHOD == "ddtree_lazy_projection"
                else "O(internal_nodes * vocabulary + B * L)"
            ),
            "probability_validation": "shared FP64 probability rule; candidate kernel validate=False",
        },
        "validation": {},
        "runs": [],
    }
    expected_gpu = os.environ.get("GBV_EXPECTED_GPU_SUBSTRING")
    if expected_gpu and expected_gpu not in report["gpu"]:
        raise RuntimeError(
            f"GPU identity gate failed: expected {expected_gpu!r}, got {report['gpu']!r}"
        )

    warmup = encode_messages(
        tokenizer, [{"role": "user", "content": "Compute 1 + 1."}],
        cfg["model"], engine.device,
    )
    for variant in methods:
        engine.generate(warmup, variant, 32, [], seed=1, profile=False)

    collected = {variant.name: [] for variant in methods}
    by_name = {variant.name: variant for variant in methods}
    for prompt_index, prompt in enumerate(PROMPTS[: report["prompt_count"]]):
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
                report["runs"].append({
                    "method": variant.name, "prompt_index": prompt_index,
                    "seed": seed, "decode_tokens": result["decode_tokens"],
                    "decode_ms": result["decode_ms"], "stages": result["stages"],
                    "round_count": len(result["rounds"]),
                    "peak_allocated_bytes": result["peak_allocated_bytes"],
                    "peak_reserved_bytes": result["peak_reserved_bytes"],
                })
                print(
                    f"prompt={prompt_index + 1}/{report['prompt_count']} "
                    f"seed={seed} {variant.name}", flush=True,
                )
        report["validation"] = {
            name: {"variant": by_name[name].to_dict(), "summary": summarize_runs(rows)}
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
        "passed_screen": (
            proposed["decode_tokens_per_second"]
            > ddtree["decode_tokens_per_second"]
        ),
    }
    report["complete"] = True
    save(report)
    print(json.dumps(report["result"], indent=2))


if __name__ == "__main__":
    main()
