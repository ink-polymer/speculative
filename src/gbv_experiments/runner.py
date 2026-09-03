from __future__ import annotations

import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import traceback
from contextlib import contextmanager

from .common import canonical, digest, file_hash, prompt_seed, source_hashes, write_json
from .config import Variant, build_variants
from .data import DATASETS, evaluation_coverage, evaluation_policy, load_prepared


def key(record):
    return (record["variant"], record["dataset"], record["source_id"], record["seed"])


def resume_records(path: Path, run_id: str) -> dict:
    records = {}
    if not path.exists():
        return records
    # Only an unterminated final write is recoverable; other corruption is fatal.
    with path.open("rb+") as stream:
        while True:
            start = stream.tell()
            raw = stream.readline()
            if not raw:
                break
            if not raw.endswith(b"\n"):
                stream.truncate(start)
                break
            row = json.loads(raw)
            if row["run_id"] != run_id or key(row) in records:
                raise ValueError("Resume result run ID mismatch or duplicate key")
            records[key(row)] = row
    return records


def make_plan(cfg, groups=None):
    entries = build_variants(cfg, groups)
    policy = evaluation_policy(cfg["datasets"], cfg.get("evaluation"))
    counts = policy["counts"]
    turn_counts = {name: counts[name] * DATASETS[name].turns for name in cfg["datasets"]}
    return {"coverage": evaluation_coverage(policy), "evaluation": policy, "datasets": counts,
            "source_counts": {name: DATASETS[name].expected for name in cfg["datasets"]},
            "seeds": cfg["seeds"], "max_new_tokens": cfg["max_new_tokens"],
            "variants": entries, "variant_count": len(entries),
            "user_turns": turn_counts,
            "expected_records": sum(counts.values()) * len(cfg["seeds"]) * len(entries),
            "expected_generations": sum(turn_counts.values()) * len(cfg["seeds"]) * len(entries)}


def model_identity(cfg):
    from huggingface_hub import HfApi
    result = dict(cfg)
    for field in ("target", "draft"):
        path = Path(cfg[field]).expanduser()
        if path.is_dir():
            files = sorted(p for p in path.rglob("*") if p.is_file() and p.suffix in {".json", ".safetensors", ".bin", ".model"})
            result[field] = str(path.resolve())
            result[field + "_local_hashes"] = {str(p.relative_to(path)): file_hash(p) for p in files}
        else:
            revision = cfg.get(field + "_revision")
            if not revision or len(revision) != 40:
                result[field + "_revision"] = HfApi().model_info(cfg[field], revision=revision).sha
    return result


@contextmanager
def output_lock(directory):
    import fcntl
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".writer.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def run(cfg, data_dir: Path, output: Path, device: str, groups=None, smoke=False, profile=False):
    with output_lock(output):
        return _run(cfg, data_dir, output, device, groups, smoke, profile)


def _run(cfg, data_dir: Path, output: Path, device: str, groups=None, smoke=False, profile=False):
    import torch
    from .engine import load_models
    from .conversation import encode_messages, generate_conversation

    policy = evaluation_policy(cfg["datasets"], cfg.get("evaluation"))
    data_manifest, rows = load_prepared(data_dir, cfg["datasets"], policy)
    entries = build_variants(cfg, groups)
    if smoke:
        rows = [next(r for r in rows if r["dataset"] == name) for name in cfg["datasets"]]
        if "smoke" not in output.name.lower():
            raise ValueError("Smoke output directory must contain 'smoke'")
    if not torch.cuda.is_available() or torch.device(device).type != "cuda":
        raise RuntimeError("Formal inference needs CUDA; use the CPU tests for local verification")
    model = model_identity(cfg["model"])
    versions = {}
    for pkg in ("torch", "transformers", "datasets", "huggingface-hub", "numpy"):
        versions[pkg] = importlib.metadata.version(pkg)
    device_properties = torch.cuda.get_device_properties(torch.device(device))
    try:
        driver = subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True).strip().splitlines()[0]
    except (OSError, subprocess.CalledProcessError):
        driver = "unavailable"
    spec = {"schema": 3, "model": model, "variants": entries, "seeds": cfg["seeds"],
            "max_new_tokens": min(16, cfg["max_new_tokens"]) if smoke else cfg["max_new_tokens"],
            "coverage": "smoke" if smoke else evaluation_coverage(policy), "evaluation": policy, "profile": profile,
            "scoring": cfg.get("scoring", {}), "bootstrap_samples": cfg.get("bootstrap_samples", 1000),
            "data_manifest": data_manifest, "dataset_names": cfg["datasets"],
            "dataset_turn_counts": {name: DATASETS[name].turns for name in cfg["datasets"]},
            "prompt_ids": [[r["dataset"], r["source_id"], r["prompt_sha256"]] for r in rows],
            "source_hashes": source_hashes(), "versions": versions,
            "python": platform.python_version(), "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(torch.device(device)),
            "gpu_memory_bytes": device_properties.total_memory, "driver": driver,
            "torch_cpu_threads": torch.get_num_threads(),
            "expected_records": len(rows) * len(cfg["seeds"]) * len(entries),
            "expected_generations": sum(DATASETS[r["dataset"]].turns for r in rows) * len(cfg["seeds"]) * len(entries)}
    run_id = digest(spec)
    manifest = {"run_id": run_id, **spec}
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "run_manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text())["run_id"] != run_id:
            raise ValueError("Run configuration/data/code/environment changed; use a new output directory")
    else:
        if (output / "results.jsonl").exists():
            raise ValueError("Existing results have no run manifest")
        write_json(manifest_path, manifest)
    completed = resume_records(output / "results.jsonl", run_id)
    valid = {(e["variant"]["name"], r["dataset"], r["source_id"], seed)
             for e in entries for r in rows for seed in cfg["seeds"]}
    if set(completed) - valid:
        raise ValueError("Resume file contains unexpected experiments")
    if len(completed) == len(valid):
        print(f"Already complete: {len(completed)} question/conversation records")
        return
    engine, tokenizer = load_models(model, device)
    write_json(output / "model_parameters.json", {
        "target_parameters": sum(p.numel() for p in engine.target.parameters()),
        "draft_parameters": sum(p.numel() for p in engine.draft.parameters()),
        "trainable_parameters": sum(p.numel() for module in (engine.target, engine.draft) for p in module.parameters() if p.requires_grad),
        "draft_block_size": engine.draft.block_size,
        "target_feature_layers": engine.draft.target_layer_ids,
    })
    stop_ids = engine.target.generation_config.eos_token_id or tokenizer.eos_token_id
    stop_ids = [stop_ids] if isinstance(stop_ids, int) else list(stop_ids or [])

    warmup = encode_messages(tokenizer, [{"role": "user", "content": "Compute 1 + 1."}], model, device)
    for entry in entries:
        engine.generate(warmup, Variant(**entry["variant"]), cfg.get("warmup_tokens", 16),
                        stop_ids, seed=0, profile=profile)
    with (output / "results.jsonl").open("a", encoding="utf-8") as stream:
        for seed in cfg["seeds"]:
            for row in rows:
                per_prompt_seed = prompt_seed(seed, row["dataset"], row["source_id"])
                order = entries.copy()
                random.Random(per_prompt_seed).shuffle(order)
                for entry in order:
                    v = Variant(**entry["variant"])
                    ident = (v.name, row["dataset"], row["source_id"], seed)
                    if ident in completed:
                        continue
                    try:
                        result = generate_conversation(engine, tokenizer, row, v, spec["max_new_tokens"], stop_ids,
                                                       per_prompt_seed, model, profile)
                        record = {"run_id": run_id, "variant": v.name, "dataset": row["dataset"],
                                  "source_id": row["source_id"], "prompt_sha256": row["prompt_sha256"],
                                  "seed": seed, "sampling_seed": per_prompt_seed,
                                  **result}
                        stream.write(canonical(record) + "\n")
                        stream.flush()
                        os.fsync(stream.fileno())
                        completed[ident] = record
                        print(f"[{len(completed)}/{len(valid)}] {v.name} {row['dataset']}/{row['source_id']} seed={seed}", flush=True)
                    except Exception:
                        with (output / "errors.jsonl").open("a") as errors:
                            errors.write(canonical({"experiment": ident, "traceback": traceback.format_exc()}) + "\n")
                        raise
    write_json(output / "completed.json", {"run_id": run_id, "records": len(completed),
                                           "generations": sum(r["turn_count"] for r in completed.values())})
