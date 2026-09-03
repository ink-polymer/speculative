from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
VARIANTS = (
    "full", "fixed_budget", "fixed_depth", "fixed_quotas", "fixed_width",
    "no_target", "no_history", "draft_only", "acceptance_reward",
    "local_ratio_reward", "no_pretrain", "no_online_rl",
)
BASELINES = ("ar", "dflash", "ddtree", "acceptance_budget_control", "static_layered")


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
    if cfg["temperature"] != 0:
        raise ValueError("This protocol proves and tests T=0 only")
    if cfg["baseline_budget"] != 60 or cfg["block_size"] != 15:
        raise ValueError("This version fixes the control to B=60 and K=15")
    if (not cfg["seeds"] or len(set(cfg["seeds"])) != len(cfg["seeds"])
            or any(type(s) is not int or not 0 <= s < 2**32 for s in cfg["seeds"])):
        raise ValueError("seeds must be nonempty and unique")
    if (not cfg["variants"] or len(set(cfg["variants"])) != len(cfg["variants"])
            or "full" not in cfg["variants"] or set(cfg["variants"]) - set(VARIANTS)):
        raise ValueError("Unknown/duplicate ablation or missing full method")
    for key in ("max_new_tokens", "train_epochs", "pretrain_epochs", "cf_actions", "cf_repeats",
                "eval_repeats", "warmup_runs", "bootstrap_samples"):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if cfg.get("gamma", 1.0) != 1.0:
        raise ValueError("Full-episode latency reward requires gamma=1")
    for key in ("learning_rate", "value_coef"):
        if not math.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f"Invalid {key}")
    if not 0 < cfg["validation_fraction"] < 1:
        raise ValueError("validation_fraction must be in (0,1)")
    budgets = cfg["oracle_budgets"]
    if (not budgets or 60 not in budgets or len(set(budgets)) != len(budgets)
            or any(type(b) is not int or not 1 <= b <= 400 for b in budgets)):
        raise ValueError("Counterfactual budgets must include 60 and be unique integers in [1,400]")
    return cfg
