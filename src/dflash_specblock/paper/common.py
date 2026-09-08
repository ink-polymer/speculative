from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
ORIGINAL_COMMIT = "9dd67698ad828b8c3fca8659e3a388f0b2dfbdf7"
VARIANTS = ("adaptive", "no_acceptance_calibration", "no_latency", "no_exploration", "frozen_after_warmup")
BASELINES = ("ar", "dflash", "ddtree", "fixed_30", "fixed_45", "fixed_80", "fixed_100", "fixed_128")
K = 15


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temp.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


@contextlib.contextmanager
def run_lock(directory: Path):
    import fcntl
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another process owns {directory}") from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def contract(directory: Path, metadata: dict) -> str:
    identity = digest(metadata)
    path = directory / "contract.json"
    if path.exists():
        if load_json(path) != {"identity": identity, "metadata": metadata}:
            raise ValueError(f"Resume contract changed: {path}; use a new run directory")
    else:
        atomic_json(path, {"identity": identity, "metadata": metadata})
    return identity


def verify_contract(directory: Path, metadata: dict) -> str:
    """Read-only lineage check; an absent upstream contract must never be created."""
    identity = digest(metadata)
    if load_json(directory / "contract.json") != {"identity": identity, "metadata": metadata}:
        raise ValueError(f"Upstream experiment contract mismatch: {directory}")
    return identity


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    ids = [row["id"] for row in rows]
    if not rows or len(ids) != len(set(ids)):
        raise ValueError(f"Empty dataset or duplicate IDs: {path}")
    return rows


def code_identity() -> str:
    paths = sorted((ROOT / "src/dflash_specblock").rglob("*.py"))
    return digest({str(p.relative_to(ROOT)): file_hash(p) for p in paths})


def load_config(path: Path) -> dict:
    cfg = load_json(path)
    expected = {"version", "method", "model_config", "temperature", "block_size",
                "baseline_budget", "budget_candidates", "initial_budget",
                "warmup_rounds_per_budget", "ewma_alpha", "exploration_interval",
                "max_new_tokens", "seeds", "variants", "eval_repeats",
                "warmup_runs", "bootstrap_samples"}
    if set(cfg) != expected or cfg["version"] != 3 or cfg["method"] != "latency_aware_ddtree":
        raise ValueError("Expected non-RL Adaptive DDTree v3; old RL configs are incompatible")
    if cfg["temperature"] != 0 or cfg["block_size"] != K or cfg["baseline_budget"] != 60:
        raise ValueError("Protocol requires T=0, K=15, fixed DDTree control B=60")
    if cfg["budget_candidates"] != [30, 45, 60, 80, 100, 128] or cfg["initial_budget"] != 60:
        raise ValueError("Use the original six budgets and initial budget 60")
    if cfg["warmup_rounds_per_budget"] != 1 or cfg["ewma_alpha"] != .2 or cfg["exploration_interval"] != 64:
        raise ValueError("Original controller parameters are fixed before evaluation")
    if (not cfg["seeds"] or len(set(cfg["seeds"])) != len(cfg["seeds"])
            or any(type(s) is not int or not 0 <= s < 2**32 for s in cfg["seeds"])):
        raise ValueError("seeds must be nonempty and unique")
    if (not cfg["variants"] or len(set(cfg["variants"])) != len(cfg["variants"])
            or "adaptive" not in cfg["variants"] or set(cfg["variants"]) - set(VARIANTS)):
        raise ValueError("Unknown/duplicate ablation or missing adaptive method")
    for key in ("max_new_tokens", "eval_repeats", "warmup_runs", "bootstrap_samples"):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    return cfg
