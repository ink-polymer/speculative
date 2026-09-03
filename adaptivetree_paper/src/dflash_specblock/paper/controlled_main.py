from __future__ import annotations

import argparse
import importlib.metadata
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

from .common import BASELINES, ROOT, atomic_json, code_identity, contract, digest, load_config, load_json, read_rows, run_lock
from .data import SOURCES, check_manifest, expected_turns, is_evaluation_row, prepare, validate_evaluation


def environment(require_gpu=False):
    import torch
    versions = {name: importlib.metadata.version(name) for name in
                ("torch", "transformers", "datasets", "huggingface-hub", "accelerate", "numpy")}
    result = {"python": platform.python_version(), "platform": platform.platform(),
              "packages": versions, "cuda": torch.version.cuda,
              "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    if require_gpu:
        if sys.version_info[:2] not in {(3, 10), (3, 11)}:
            raise RuntimeError("Formal runtime requires Python 3.10/3.11")
        if versions["transformers"] != "4.57.1":
            raise RuntimeError("Formal runtime requires transformers==4.57.1")
        from packaging.version import Version
        if not Version("0.34") <= Version(versions["huggingface-hub"]) < Version("1"):
            raise RuntimeError("transformers 4.57.1 needs huggingface-hub>=0.34,<1")
        if not Version("2.20") <= Version(versions["datasets"]) < Version("4"):
            raise RuntimeError("Use the specified datasets>=2.20,<4 environment")
        if not result["gpu"]:
            raise RuntimeError("CUDA GPU is required; CPU tests are not formal model validation")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("A CUDA GPU with BF16 support is required")
        # Import, not just find_spec: detects broken optional dependency installations.
        import transformers
        from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, DynamicCache
        if transformers.__version__ != versions["transformers"]:
            raise RuntimeError("Imported Transformers code and installed package metadata disagree")
        result["transformers_import_ok"] = True
        properties = torch.cuda.get_device_properties(0)
        result["gpu_uuid"] = str(getattr(properties, "uuid", "unavailable"))
        result["gpu_memory_bytes"] = properties.total_memory
        result["compute_capability"] = list(torch.cuda.get_device_capability(0))
        try:
            driver = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                    capture_output=True, text=True, check=True, timeout=10)
        except (OSError, subprocess.SubprocessError) as exc:
            raise RuntimeError("Cannot record NVIDIA driver version for a formal run") from exc
        result["driver_versions"] = sorted(set(driver.stdout.strip().splitlines()))
    return result


def load_data(directory, smoke_count):
    manifest = check_manifest(directory)
    evaluation = []
    for name in SOURCES:
        rows = read_rows(directory / f"{name}.jsonl")
        if len(rows) != SOURCES[name][3] or any(not is_evaluation_row(row) for row in rows):
            raise ValueError(f"Not a full official test split: {name}")
        evaluation.extend(rows)
    validate_evaluation(evaluation)
    if smoke_count:
        evaluation = [r for name in SOURCES for r in [v for v in evaluation if v["dataset"] == name][:smoke_count]]
    return manifest, evaluation


def main(argv=None):
    parser = argparse.ArgumentParser(description="Paper T=0: original non-RL Adaptive DDTree, full data and paired ablations")
    parser.add_argument("stage", choices=["plan", "doctor", "prepare", "evaluate", "summarize", "all"])
    parser.add_argument("--config", type=Path, default=ROOT / "configs/paper_t0_controlled_legacy.json")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "datasets/adaptive_original_full")
    parser.add_argument("--run-dir", type=Path, default=ROOT / "outputs/adaptive_original_full")
    parser.add_argument("--smoke-count", type=int, default=0,
                        help="Explicitly non-publication run; take this many prompts per split")
    args = parser.parse_args(argv)
    cfg = load_config(args.config.resolve())
    if args.smoke_count < 0:
        parser.error("smoke-count must be nonnegative")
    if args.smoke_count:
        if "smoke" not in args.run_dir.name.lower():
            parser.error("Smoke results require a run directory whose name contains 'smoke'")
        cfg = {**cfg, "seeds": cfg["seeds"][:1],
               "max_new_tokens": min(32, cfg["max_new_tokens"]), "eval_repeats": 1, "bootstrap_samples": 100}
    if args.stage == "plan":
        import json
        print(json.dumps({"config": cfg, "test_counts": {k: v[3] for k, v in SOURCES.items()},
                          "turns_per_case": {name: expected_turns(name) for name in SOURCES},
                          "raw_train_count": 0,
                          "policy_checkpoints": 0, "method": "original non-RL latency-aware DDTree",
                          "test_generation_calls": (sum((min(v[3], args.smoke_count) if args.smoke_count else v[3])
                                                         * expected_turns(name) for name, v in SOURCES.items()) * len(cfg["seeds"])
                                                    * cfg["eval_repeats"] * (len(BASELINES) + len(cfg["variants"]))),
                          "full_data": not bool(args.smoke_count),
                          "launches_training": False}, ensure_ascii=False, indent=2))
        return
    if args.stage == "doctor":
        print(environment(require_gpu=True))
        return
    if args.stage in {"prepare", "all"}:
        prepare(args.data_dir)
        if args.stage == "prepare":
            print(check_manifest(args.data_dir))
            return
    manifest, test_rows = load_data(args.data_dir, args.smoke_count)
    model_config = load_json(ROOT / cfg["model_config"])
    protocol = {"version": 3, "training": False, "config": cfg, "model_config": model_config,
                "data_manifest_hash": digest(manifest), "code_hash": code_identity(),
                "smoke": bool(args.smoke_count), "smoke_count": args.smoke_count,
                "test_ids": digest([r["id"] for r in test_rows])}
    # Never regenerate metadata against modified code and silently reuse old runs.
    with run_lock(args.run_dir):
        contract(args.run_dir, protocol)
    if args.stage == "summarize":
        from .evaluation import summarize
        if not args.smoke_count and not (args.run_dir / "gpu_preflight.json").exists():
            raise RuntimeError("No successful GPU preflight: refusing a paper table")
        with run_lock(args.run_dir / "execution"):
            summarize(args.run_dir, test_rows, cfg, smoke=bool(args.smoke_count))
        return
    env = environment(require_gpu=True)
    with run_lock(args.run_dir):
        env_path = args.run_dir / "gpu_preflight.json"
        if env_path.exists() and load_json(env_path) != env:
            raise ValueError("Hardware/software changed during this experiment; start a separate run")
        atomic_json(env_path, env)
    # Shared per-device lock also prevents two different run directories from
    # contaminating each other's timings. This cannot stop unrelated GPU programs.
    gpu_lock = Path(tempfile.gettempdir()) / ("dflash-paper-gpu-" + digest(env["gpu_uuid"])[:20])
    with run_lock(args.run_dir / "execution"), run_lock(gpu_lock):
        execute_gpu(args, cfg, env, protocol, test_rows)


def execute_gpu(args, cfg, env, protocol, test_rows):
    from .runtime import PaperRuntime
    from .evaluation import evaluate, summarize
    import torch
    torch.set_num_threads(1)
    torch.manual_seed(cfg["seeds"][0])
    torch.cuda.manual_seed_all(cfg["seeds"][0])
    runtime = PaperRuntime(cfg)
    metadata = {**protocol, "environment": env}
    if args.stage in {"evaluate", "all"}:
        for seed in cfg["seeds"]:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            evaluate(runtime, test_rows, args.run_dir / "evaluation" / str(seed), metadata, seed)
    if args.stage == "all":
        summarize(args.run_dir, test_rows, cfg, smoke=bool(args.smoke_count))


if __name__ == "__main__":
    raise SystemExit("Historical controlled protocol only. Use python -m dflash_specblock.paper for official DDTree evaluation.")
