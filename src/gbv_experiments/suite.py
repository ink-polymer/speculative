"""Two-model GBV study with resumable scheduling phases and one frozen dataset."""
from __future__ import annotations

import gc
import json
from pathlib import Path
import re

from .common import read_jsonl, write_json
from .config import build_variants, load_config
from .data import evaluation_policy
from .runner import key, make_plan


def load_suite(path: Path, model_ids=None):
    spec = json.loads(path.read_text())
    if set(spec) != {"models"} or not isinstance(spec["models"], list) or not spec["models"]:
        raise ValueError("A suite requires a nonempty models list")
    models, seen, pairs = [], set(), set()
    for entry in spec["models"]:
        if set(entry) != {"id", "config"} or not re.fullmatch(r"[a-z0-9_]+", entry["id"]):
            raise ValueError("Invalid suite model entry")
        if entry["id"] in seen:
            raise ValueError("Duplicate suite model ID")
        seen.add(entry["id"])
        config_path = (path.parent / entry["config"]).resolve()
        cfg = load_config(config_path)
        pair = (cfg["model"]["target"], cfg["model"]["draft"])
        if pair in pairs:
            raise ValueError("Duplicate suite model pair")
        pairs.add(pair)
        gbv = next(e["variant"] for e in build_variants(cfg) if e["variant"]["name"] == "gbv")
        if (gbv["method"], gbv["paths"], gbv["length"], gbv["temperature"],
                gbv["draft_attention"], gbv["condition_features"]) != ("gbv", 3, 15, 1., "bidirectional", "target"):
            raise ValueError("This study requires unmodified three-path GBV at L=15, T=1")
        models.append({"id": entry["id"], "config_path": config_path, "config": cfg})
    reference = models[0]["config"]
    for model in models[1:]:
        cfg = model["config"]
        common = ("datasets", "seeds", "max_new_tokens", "scoring")
        if any(cfg.get(key) != reference.get(key) for key in common):
            raise ValueError("Suite models must use the same data, seeds, limits, and scoring")
        if evaluation_policy(cfg["datasets"], cfg.get("evaluation")) != evaluation_policy(reference["datasets"], reference.get("evaluation")):
            raise ValueError("Suite models must share the same sample selection")
    if model_ids is not None:
        if not model_ids or len(set(model_ids)) != len(model_ids) or set(model_ids) - seen:
            raise ValueError("Unknown or duplicate suite model selection")
        models = [m for m in models if m["id"] in model_ids]
    return models


def phase_variants(cfg, phase):
    if phase == "gbv-first":
        return ["gbv"]
    if phase == "main":
        return [e["variant"]["name"] for e in build_variants(cfg) if "main" in e["groups"]]
    if phase == "complete":
        return None
    raise ValueError(f"Unknown suite phase: {phase}")


def plan_suite(path: Path, phase="gbv-first", model_ids=None):
    models = load_suite(path, model_ids)
    plans = {m["id"]: make_plan(m["config"], only_variants=phase_variants(m["config"], phase)) for m in models}
    return {"phase": phase, "model_count": len(plans), "models": plans,
            "model_variant_count": sum(p["variant_count"] for p in plans.values()),
            "evaluation_jobs": sum(p["variant_count"] * len(p["datasets"]) * len(p["seeds"]) for p in plans.values()),
            "expected_records": sum(p["expected_records"] for p in plans.values()),
            "expected_generations": sum(p["expected_generations"] for p in plans.values()),
            "note": "Counts are phase totals; completed records are reused on resume. Model results remain separate."}


def run_suite(path: Path, data_dir: Path, output: Path, device="cuda:0", code_backend="docker",
              phase="gbv-first", model_ids=None):
    import torch
    from .audit import audit
    from .data import prepare
    from .preflight import check_model
    from .report import report
    from .runner import run
    from .scoring import score_run, validate_gold

    models = load_suite(path, model_ids)
    plan = plan_suite(path, phase, model_ids)
    if not torch.cuda.is_available() or torch.device(device).type != "cuda":
        raise RuntimeError("No CUDA GPU is available; suite planning works locally, formal runs require the H200 environment")
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
        selected = phase_variants(cfg, phase)
        print(f"Running {model['id']}: phase={phase}, variants={selected or 'all remaining'}", flush=True)
        check_model(cfg, run_dir / f"gpu_preflight_{phase}.json", device, code_backend, selected)
        gc.collect()
        torch.cuda.empty_cache()
        run(cfg, data_dir, run_dir, device, only_variants=selected)
        gc.collect()
        torch.cuda.empty_cache()
        settings = cfg["scoring"]
        score_run(run_dir, data_dir, code_backend, settings["workers"], settings["timeout_seconds"], settings["lcb_timeout_seconds"])
        coverage = report(run_dir, run_dir / f"report_{phase}", cfg.get("bootstrap_samples", 1000),
                          allow_partial=phase != "complete", plots=phase == "complete")
        rows = read_jsonl(run_dir / "results.jsonl")
        names = set(selected or [e["variant"]["name"] for e in build_variants(cfg)])
        phase_rows = [r for r in rows if r["variant"] in names]
        expected = plan["models"][model["id"]]["expected_records"]
        if len(phase_rows) != expected:
            raise RuntimeError("Suite phase ended without all scheduled records")
        score_keys = {key(r) for r in read_jsonl(run_dir / "scores.jsonl")}
        if not {key(r) for r in phase_rows} <= score_keys:
            raise RuntimeError("Suite phase ended without all scheduled scoring records")
        summaries.append({"model_id": model["id"], "phase_records": len(phase_rows),
                          "phase_generations": sum(r["turn_count"] for r in phase_rows),
                          "phase_complete": True, "full_experiment_complete": coverage["complete"]})
    result = {"phase": phase, "phase_complete": True, "models": summaries}
    write_json(output / f"phase_completed_{phase}.json", result)
    return result
