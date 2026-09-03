from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class Variant:
    name: str
    method: str = "gbv"
    paths: int = 3
    length: int = 15  # Future tokens; clean anchor is not counted.
    temperature: float = 1.0
    draft_temperature: float | None = None
    share_prefixes: bool = True
    reuse_draft_cache: bool = True
    draft_attention: str = "bidirectional"
    condition_features: str = "target"
    probability_dtype: str = "float64"
    tree_budget: int = 60

    def validate(self) -> None:
        if self.method not in {"target", "token", "bv", "gbv", "ddtree"}:
            raise ValueError(f"Unknown method: {self.method}")
        if self.paths < 1 or self.length < 1 or self.temperature < 0:
            raise ValueError("paths/length must be positive and temperature nonnegative")
        if not math.isfinite(self.temperature) or any(not isinstance(v, int) or isinstance(v, bool) for v in (self.paths, self.length, self.tree_budget)):
            raise ValueError("Counts must be integers and temperature must be finite")
        if self.method in {"token", "bv"} and self.paths != 1:
            raise ValueError("Single-path token/BV baselines require paths=1")
        if self.draft_temperature is not None and (self.draft_temperature <= 0 or not math.isfinite(self.draft_temperature)):
            raise ValueError("Draft temperature must be positive")
        if self.draft_attention not in {"bidirectional", "causal"}:
            raise ValueError("Invalid draft_attention")
        if self.condition_features not in {"target", "zero"}:
            raise ValueError("Invalid condition_features")
        if self.probability_dtype not in {"float32", "float64"} or self.tree_budget < 1:
            raise ValueError("Invalid numerical precision/tree budget")

    def to_dict(self) -> dict:
        return asdict(self)


def load_config(path: Path) -> dict:
    cfg = json.loads(path.read_text())
    allowed = {"model", "datasets", "seeds", "max_new_tokens", "warmup_tokens", "main",
               "ablations", "bootstrap_samples", "scoring"}
    if set(cfg) - allowed:
        raise ValueError(f"Unknown configuration keys: {sorted(set(cfg)-allowed)}")
    if not cfg.get("datasets") or len(set(cfg["datasets"])) != len(cfg["datasets"]):
        raise ValueError("datasets must be a nonempty unique list")
    if not cfg.get("seeds") or len(set(cfg["seeds"])) != len(cfg["seeds"]):
        raise ValueError("seeds must be a nonempty unique list")
    if cfg.get("max_new_tokens", 0) < 1:
        raise ValueError("max_new_tokens must be positive")
    if cfg["model"].get("target_attention", "sdpa") not in {"sdpa", "eager"}:
        raise ValueError("Tree verification requires a backend supporting the explicit 4D branch mask")
    if cfg["model"].get("draft_attention", "sdpa") not in {"sdpa", "eager"}:
        raise ValueError("Attention ablations require sdpa or eager")
    if cfg.get("warmup_tokens", 16) < 2:
        raise ValueError("Warmup must execute at least one decode round")
    model_keys = {"target", "draft", "target_revision", "draft_revision", "dtype", "target_attention", "draft_attention", "enable_thinking", "allow_tf32"}
    if set(cfg["model"]) - model_keys:
        raise ValueError("Unknown model configuration keys")
    if cfg["model"].get("dtype", "bfloat16") not in {"bfloat16", "float16", "float32"}:
        raise ValueError("Unsupported model dtype")
    if any(not isinstance(seed, int) or isinstance(seed, bool) for seed in cfg["seeds"]):
        raise ValueError("Seeds must be integers")
    return cfg


def build_variants(cfg: dict, groups: list[str] | None = None) -> list[dict]:
    base = Variant(name="gbv", **cfg.get("main", {}))
    base.validate()
    variants: dict[str, dict] = {}

    def add(v: Variant, group: str):
        v.validate()
        key = tuple((k, val) for k, val in v.to_dict().items() if k != "name")
        for entry in variants.values():
            if entry["key"] == key:
                if group not in entry["groups"]:
                    entry["groups"].append(group)
                return
        if v.name in variants:
            raise ValueError(f"Duplicate variant name: {v.name}")
        variants[v.name] = {"variant": v, "groups": [group], "key": key}

    def target_at(temp: float, group: str):
        add(Variant(name=f"target_t{temp:g}", method="target", paths=1,
                    temperature=temp), group)

    target_at(base.temperature, "main")
    add(replace(base, name="dflash_token", method="token", paths=1), "main")
    add(replace(base, name="dflash_bv", method="bv", paths=1), "main")
    add(base, "main")
    add(replace(base, name="ddtree", method="ddtree", paths=1), "main")
    ablations = cfg.get("ablations", {})
    for k in ablations.get("paths", []):
        add(replace(base, name=f"gbv_k{k}", paths=k), "paths")
    for length in ablations.get("lengths", []):
        add(replace(base, name=f"gbv_l{length}", length=length), "lengths")
    for temp in ablations.get("temperatures", []):
        target_at(temp, "temperatures")
        add(replace(base, name=f"gbv_t{temp:g}", temperature=temp), "temperatures")
    switches = {
        "prefix_sharing": {"share_prefixes": False},
        "draft_cache": {"reuse_draft_cache": False},
        "bidirectional_attention": {"draft_attention": "causal"},
        "target_features": {"condition_features": "zero"},
        "probability_precision": {"probability_dtype": "float32"},
        "block_verification": {"method": "token", "paths": 1},
    }
    for name in ablations.get("switches", []):
        if name not in switches:
            raise ValueError(f"Unknown ablation: {name}")
        if name != "block_verification":
            add(base, name)
        # BV versus token verification is isolated at K=1.
        if name == "block_verification":
            add(replace(base, name="dflash_bv", method="bv", paths=1), name)
        add(replace(base, name=f"ablate_{name}", **switches[name]), name)
    known = {g for e in variants.values() for g in e["groups"]}
    if groups and set(groups) - known:
        raise ValueError(f"Unknown groups: {sorted(set(groups)-known)}")
    selected = [e for e in variants.values() if not groups or set(groups) & set(e["groups"])]
    # Always retain an AR baseline at every tested target temperature.
    temps = {e["variant"].temperature for e in selected}
    for temp in temps:
        for e in variants.values():
            if e["variant"].method == "target" and e["variant"].temperature == temp and e not in selected:
                selected.append(e)
    return [{"variant": e["variant"].to_dict(), "groups": e["groups"]} for e in selected]
