"""Frozen two-model evaluation for Branch-Recycling Tree Block Verification."""
from __future__ import annotations

import gc
import json
from pathlib import Path
import re

from .common import read_jsonl, write_json
from .config import build_variants, load_config
from .data import evaluation_policy
from .runner import key, make_plan


STUDY = "branch_recycling_tree_bv"
PRIMARY_NAMES = {"target_t1", "gbv", "ddtree", "tree_gbv_base", "tree_bv_recycle"}


def _validate_method_contract(cfg: dict) -> None:
    entries = build_variants(cfg)
    by_name = {entry["variant"]["name"]: entry for entry in entries}
    if not PRIMARY_NAMES <= set(by_name):
        raise ValueError(f"Tree-BV suite is missing primary variants: {sorted(PRIMARY_NAMES-set(by_name))}")
    if any(name not in PRIMARY_NAMES and "controls" not in entry["groups"]
           for name, entry in by_name.items()):
        raise ValueError("Non-primary Tree-BV variants must be labeled as controls")
    if any("main" not in by_name[name]["groups"] for name in PRIMARY_NAMES):
        raise ValueError("Every primary Tree-BV variant must be in the main group")

    variants = {name: entry["variant"] for name, entry in by_name.items()}
    if (variants["target_t1"]["method"], variants["target_t1"]["paths"]) != ("target", 1):
        raise ValueError("Tree-BV study requires one T=1 autoregressive Target baseline")
    if (variants["gbv"]["method"], variants["gbv"]["paths"], variants["gbv"]["length"]) != ("gbv", 3, 15):
        raise ValueError("Tree-BV study requires the original K=3, L=15 GBV control")
    ddtree = variants["ddtree"]
    if (ddtree["method"], ddtree["paths"], ddtree["length"], ddtree["tree_budget"]) != ("ddtree", 1, 15, 45):
        raise ValueError("Tree-BV study requires the frozen L=15, B=45 DDTree baseline")

    base, recycle = variants["tree_gbv_base"], variants["tree_bv_recycle"]
    if (base["method"], recycle["method"]) != ("tree_gbv_full", "tree_gbv_recycle"):
        raise ValueError("Invalid Tree-BV base/recycling methods")
    matched = ("paths", "length", "temperature", "draft_temperature", "tree_budget",
               "probability_dtype", "share_prefixes", "reuse_draft_cache",
               "draft_attention", "condition_features")
    if any(base[field] != recycle[field] for field in matched):
        raise ValueError("Tree-BV base and recycling variants must differ only in verification")

    for variant in variants.values():
        if variant["temperature"] != 1.0 or variant["probability_dtype"] != "float64":
            raise ValueError("All Tree-BV formal variants require T=1 and FP64 probabilities")
        if variant["method"] != "target" and variant["draft_temperature"] != 1.0:
            raise ValueError("All speculative variants require the same T_target=T_draft=1")
    model = cfg["model"]
    if (model.get("dtype"), model.get("target_attention"), model.get("draft_attention"),
            model.get("allow_tf32"), model.get("enable_thinking")) != (
            "bfloat16", "sdpa", "sdpa", False, False):
        raise ValueError("Tree-BV model execution settings are not frozen")


def load_tree_suite(path: Path, model_ids=None):
    spec = json.loads(path.read_text())
    if (set(spec) != {"study", "models"} or spec.get("study") != STUDY or
            not isinstance(spec.get("models"), list) or not spec["models"]):
        raise ValueError("A Tree-BV suite requires its study ID and a nonempty models list")
    models, seen, pairs = [], set(), set()
    for entry in spec["models"]:
        if set(entry) != {"id", "config"} or not re.fullmatch(r"[a-z0-9_]+", entry["id"]):
            raise ValueError("Invalid Tree-BV suite model entry")
        if entry["id"] in seen:
            raise ValueError("Duplicate Tree-BV suite model ID")
        seen.add(entry["id"])
        config_path = (path.parent / entry["config"]).resolve()
        cfg = load_config(config_path)
        if "explicit_variants" not in cfg:
            raise ValueError("Tree-BV suite configs must explicitly freeze every variant")
        _validate_method_contract(cfg)
        pair = (cfg["model"]["target"], cfg["model"]["draft"])
        if pair in pairs:
            raise ValueError("Duplicate Tree-BV suite model pair")
        pairs.add(pair)
        models.append({"id": entry["id"], "config_path": config_path, "config": cfg})
    reference = models[0]["config"]
    for model in models[1:]:
        cfg = model["config"]
        common = ("datasets", "seeds", "max_new_tokens", "scoring")
        if any(cfg.get(field) != reference.get(field) for field in common):
            raise ValueError("Tree-BV models must use the same data, seeds, limits, and scoring")
        if evaluation_policy(cfg["datasets"], cfg.get("evaluation")) != evaluation_policy(
                reference["datasets"], reference.get("evaluation")):
            raise ValueError("Tree-BV models must share the same sample selection")
    if model_ids is not None:
        if not model_ids or len(set(model_ids)) != len(model_ids) or set(model_ids) - seen:
            raise ValueError("Unknown or duplicate Tree-BV suite model selection")
        models = [model for model in models if model["id"] in model_ids]
    return models


def tree_phase_variants(cfg: dict, phase: str):
    if phase == "recycle-first":
        return ["tree_bv_recycle"]
    if phase == "main":
        return [entry["variant"]["name"] for entry in build_variants(cfg)
                if "main" in entry["groups"]]
    if phase == "complete":
        return None
    raise ValueError(f"Unknown Tree-BV suite phase: {phase}")


def plan_tree_suite(path: Path, phase="recycle-first", model_ids=None):
    models = load_tree_suite(path, model_ids)
    plans = {model["id"]: make_plan(
        model["config"], only_variants=tree_phase_variants(model["config"], phase)
    ) for model in models}
    return {
        "study": STUDY,
        "phase": phase,
        "model_count": len(plans),
        "models": plans,
        "model_variant_count": sum(plan["variant_count"] for plan in plans.values()),
        "evaluation_jobs": sum(plan["variant_count"] * len(plan["datasets"]) * len(plan["seeds"])
                               for plan in plans.values()),
        "expected_records": sum(plan["expected_records"] for plan in plans.values()),
        "expected_generations": sum(plan["expected_generations"] for plan in plans.values()),
        "decision_rule": "BRBV/DDTree paired throughput 95% bootstrap CI lower bound > 1 on both models",
        "note": "Counts are phase totals; completed records are reused on resume.",
    }


def run_tree_suite(path: Path, data_dir: Path, output: Path, device="cuda:0",
                   code_backend="docker", phase="recycle-first", model_ids=None):
    import torch
    from .audit import audit
    from .data import prepare
    from .preflight import check_model
    from .report import report
    from .runner import run
    from .scoring import score_run, validate_gold

    models = load_tree_suite(path, model_ids)
    plan = plan_tree_suite(path, phase, model_ids)
    if not torch.cuda.is_available() or torch.device(device).type != "cuda":
        raise RuntimeError("No CUDA GPU is available; suite planning works locally, formal runs require the GH200 environment")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / f"plan_{phase}.json", plan)
    cfg = models[0]["config"]
    prepare(cfg["datasets"], data_dir, cfg.get("evaluation"))
    audit(cfg, data_dir, [], output / "data_audit.json")
    validate_gold(data_dir, cfg["datasets"], output / "gold_audit.json", code_backend,
                  cfg["scoring"]["timeout_seconds"], cfg.get("evaluation"))
    summaries = []
    for model in models:
        cfg = model["config"]
        run_dir = output / model["id"]
        selected = tree_phase_variants(cfg, phase)
        print(f"Running {model['id']}: phase={phase}, variants={selected or 'all remaining'}", flush=True)
        check_model(cfg, run_dir / f"gpu_preflight_{phase}.json", device, code_backend, selected)
        gc.collect()
        torch.cuda.empty_cache()
        run(cfg, data_dir, run_dir, device, only_variants=selected)
        gc.collect()
        torch.cuda.empty_cache()
        settings = cfg["scoring"]
        score_run(run_dir, data_dir, code_backend, settings["workers"],
                  settings["timeout_seconds"], settings["lcb_timeout_seconds"])
        coverage = report(run_dir, run_dir / f"report_{phase}",
                          cfg.get("bootstrap_samples", 1000),
                          allow_partial=phase != "complete", plots=phase == "complete")
        rows = read_jsonl(run_dir / "results.jsonl")
        names = set(selected or [entry["variant"]["name"] for entry in build_variants(cfg)])
        phase_rows = [row for row in rows if row["variant"] in names]
        expected = plan["models"][model["id"]]["expected_records"]
        if len(phase_rows) != expected:
            raise RuntimeError("Tree-BV suite phase ended without all scheduled records")
        score_keys = {key(row) for row in read_jsonl(run_dir / "scores.jsonl")}
        if not {key(row) for row in phase_rows} <= score_keys:
            raise RuntimeError("Tree-BV suite phase ended without all scheduled scoring records")
        summaries.append({"model_id": model["id"], "phase_records": len(phase_rows),
                          "phase_generations": sum(row["turn_count"] for row in phase_rows),
                          "phase_complete": True, "full_experiment_complete": coverage["complete"]})
    result = {"study": STUDY, "phase": phase, "phase_complete": True, "models": summaries}
    if phase == "complete" and model_ids is None:
        from .tree_decision import write_decision
        result["decision"] = write_decision(output, [model["id"] for model in models],
                                             cfg.get("bootstrap_samples", 1000))
    write_json(output / f"phase_completed_{phase}.json", result)
    return result
