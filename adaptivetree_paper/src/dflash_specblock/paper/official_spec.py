"""Frozen DDTree T=0 protocol, including the authors' exact sampling limits."""
from __future__ import annotations

import importlib.util
import re
import sys
from functools import lru_cache
from types import SimpleNamespace

from .common import ROOT, VARIANTS, file_hash, load_json

COMMIT = "c96427a185677bf4133ed865dd1626a5041aef9b"
UPSTREAM = ROOT / "third_party/ddtree_pinned"
# Order and counts are copied from the pinned run_benchmark.sh.
LIMITS = {"gsm8k":128, "math500":128, "aime24":30, "aime25":30,
          "humaneval":164, "mbpp":128, "livecodebench":128, "swe-bench":128,
          "mt-bench":80, "alpaca":128}
SOURCES = {"gsm8k": ("openai/gsm8k", "main", "test"),
           "math500": ("HuggingFaceH4/MATH-500", None, "test"),
           "aime24": ("HuggingFaceH4/aime_2024", None, "train"),
           "aime25": ("MathArena/aime_2025", None, "train"),
           "humaneval": ("openai/openai_humaneval", None, "test"),
           "mbpp": ("google-research-datasets/mbpp", "sanitized", "test"),
           "livecodebench": ("livecodebench/code_generation_lite", None, "test"),
           "swe-bench": ("princeton-nlp/SWE-bench_Lite", None, "test"),
           "mt-bench": ("HuggingFaceH4/mt_bench_prompts", None, "train"),
           "alpaca": ("tatsu-lab/alpaca", None, "train")}
BUDGETS = [16,32,64,128,256,512,1024]
MODELS = [
    ("Qwen/Qwen3-4B", "z-lab/Qwen3-4B-DFlash-b16"),
    ("Qwen/Qwen3-8B", "z-lab/Qwen3-8B-DFlash-b16"),
    ("Qwen/Qwen3-Coder-30B-A3B-Instruct", "z-lab/Qwen3-Coder-30B-A3B-DFlash"),
]

# Retain the original 4B pair and the 8B pair used by the GBV paper suite.
# All methods within each pair load these same immutable revisions.
PINNED_MODEL_REVISIONS = {
    "Qwen/Qwen3-4B": "1cfa9a7208912126459214e8b04321603b3df60c",
    "z-lab/Qwen3-4B-DFlash-b16": "b74e3a329c4d963783143b1e970d95b002be72bd",
    "Qwen/Qwen3-8B": "b968826d9c46dd6066d109eabc6255188de91218",
    "z-lab/Qwen3-8B-DFlash-b16": "9b41424b7109f9c5413454f481b09a82b85333f4",
}


def load_config(path):
    config = load_json(path)
    if set(config) != {"version","protocol","official_commit","temperature","max_new_tokens",
                      "seed","datasets","sample_limits","tree_budgets","models","variants","adaptive"}:
        raise ValueError("Unexpected official protocol configuration fields")
    if (config.get("version") != 4 or config.get("protocol") != "ddtree_official_t0"
            or config["official_commit"] != COMMIT or config["temperature"] != 0
            or config["max_new_tokens"] != 2048 or config["seed"] != 0
            or config["datasets"] != list(LIMITS) or config["sample_limits"] != LIMITS
            or config["tree_budgets"] != BUDGETS
            or config["models"] != [list(pair) for pair in MODELS]
            or config["variants"] != list(VARIANTS)):
        raise ValueError("Expected pinned DDTree official T=0 matrix, not full-split/learned-policy config")
    a = config["adaptive"]
    if a != {"budget_candidates":[30,45,60,80,100,128], "initial_budget":60,
             "warmup_rounds_per_budget":1, "ewma_alpha":.2, "exploration_interval":64}:
        raise ValueError("The original Adaptive DDTree parameters must be preserved")
    return config


def verify_sources():
    manifest = load_json(UPSTREAM / "SOURCE_SHA256.json")
    if manifest["commit"] != COMMIT:
        raise ValueError("Wrong official source commit")
    for relative, sha in manifest["files"].items():
        if file_hash(UPSTREAM / relative) != sha:
            raise ValueError(f"Modified official source: {relative}")
    return manifest


def load_source(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def data_utils():
    verify_sources()
    # No import of the GPU model is required for dataset preparation.
    return load_source("_ddtree_pinned_data_utils", UPSTREAM / "model/utils.py")


@lru_cache(maxsize=1)
def upstream():
    verify_sources()
    # Upstream uses absolute local imports. Refuse collisions instead of silently
    # executing a different project's model/ddtree module.
    for name in ("model", "dflash", "ddtree", "distributed"):
        old = sys.modules.get(name)
        if old is not None and not str(getattr(old, "__file__", "")).startswith(str(UPSTREAM)):
            raise RuntimeError(f"Official module namespace collision: {name}; use a fresh process")
    sys.path.insert(0, str(UPSTREAM))
    import model, dflash, ddtree, distributed
    return SimpleNamespace(model=model, dflash=dflash, ddtree=ddtree, dist=distributed)
