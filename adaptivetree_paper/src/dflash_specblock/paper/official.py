"""CLI for the exact official sampled T=0 protocol, extended with non-RL Adaptive."""
from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

from .common import (METHOD_SCHEMA_VERSION, PRIMARY_ADAPTIVE_METHOD, ROOT,
                     atomic_json, code_identity, contract, load_json, run_lock)
from .official_data import check_manifest, prepare
from .official_spec import LIMITS, MODELS, PINNED_MODEL_REVISIONS, load_config, verify_sources
from .controller import deprecated_experiment_flags
from .wandb_monitor import wandb_contract


def plan(config, model_indices, datasets, smoke_count, nproc,
         experimental_cost_attribution=False, experimental_extended_budgets=False):
    counts = {name:min(LIMITS[name], smoke_count) if smoke_count else LIMITS[name] for name in datasets}
    turns = sum(n * (2 if name=="mt-bench" else 1) for name,n in counts.items())
    # SDPA: baseline, DFlash, seven DDTree budgets, and the registered
    # AdaptiveTree method/ablation matrix.
    # FA2: baseline and DFlash only.
    legacy_cli_aliases = deprecated_experiment_flags(
        cost_attribution=experimental_cost_attribution,
        extended_budgets=experimental_extended_budgets,
    )
    result = {"protocol":config["protocol"], "temperature":0, "training":False,
            "data_sampling":"Dataset.shuffle(seed=0).select(range(limit)) only when full size > limit",
            "test_counts":counts, "cases":sum(counts.values()), "turns_per_method":turns,
            "models":[MODELS[i][0] for i in model_indices], "nproc_per_node":nproc,
            "model_pairs":[{"target":MODELS[i][0], "draft":MODELS[i][1],
                "target_revision":PINNED_MODEL_REVISIONS.get(MODELS[i][0], "locked during prepare"),
                "draft_revision":PINNED_MODEL_REVISIONS.get(MODELS[i][1], "locked during prepare")}
                for i in model_indices],
            "benchmark_process_groups":len(datasets)*len(model_indices)*2,
            "generation_calls":turns*len(model_indices)*(11+len(config["variants"])),
            "method_schema_version":METHOD_SCHEMA_VERSION,
            "primary_adaptive_method":PRIMARY_ADAPTIVE_METHOD,
            "full_split":False, "official_samples":not bool(smoke_count),
            "launches_models":False}
    if legacy_cli_aliases:
        result["deprecated_cli_aliases"] = list(legacy_cli_aliases)
    return result


def doctor(nproc):
    from .controlled_main import environment
    from .official_audit import gpu_identity
    import torch
    result = environment(require_gpu=True)
    import flash_attn
    for dependency in ("ninja", "loguru"):
        importlib.import_module(dependency)
    if torch.cuda.device_count() < nproc:
        raise RuntimeError(f"Official default requests {nproc} GPUs; only {torch.cuda.device_count()} visible. "
                           "An explicit --nproc-per-node override is recorded as a hardware deviation.")
    result["flash_attn"] = getattr(flash_attn, "__version__", "unknown")
    result["visible_gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    result["nproc_per_node"] = nproc
    result["benchmark_gpus"] = [gpu_identity(i, i) for i in range(nproc)]
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="Corrected non-RL AdaptiveTree under pinned DDTree official T=0 evaluation",
                                     allow_abbrev=False)
    parser.add_argument("stage", choices=["plan","doctor","prepare","evaluate","summarize","all","worker"])
    parser.add_argument("--config", type=Path, default=ROOT/"configs/paper_t0_full.json")
    parser.add_argument("--data-dir", type=Path, default=ROOT/"datasets/ddtree_official_t0")
    parser.add_argument("--run-dir", type=Path, default=ROOT/"outputs/adaptive_ddtree_official_t0")
    parser.add_argument("--nproc-per-node", type=int, default=int(os.environ.get("NPROC_PER_NODE","8")))
    parser.add_argument("--master-port", type=int, default=int(os.environ.get("MASTER_PORT","29600")))
    parser.add_argument("--model-index", type=int, choices=range(len(MODELS)))
    parser.add_argument("--dataset", choices=list(LIMITS))
    parser.add_argument("--smoke-count", type=int, default=0)
    parser.add_argument("--backend", choices=["sdpa","flash_attention_2"])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--identity")
    parser.add_argument("--greedy-audit-policy", choices=["strict", "record-bf16-mismatches"],
                        default="strict")
    parser.add_argument("--method-order-policy",
                        choices=["official-fixed", "balanced-rotation"],
                        default="official-fixed",
                        help="preserve upstream order or rotate every method through timed positions")
    parser.add_argument("--experimental-cost-attribution", action="store_true",
                        help="deprecated compatibility alias; adaptive_b128 is the formal primary")
    parser.add_argument("--experimental-extended-budgets", action="store_true",
                        help="deprecated compatibility alias; adaptive_b256 is the budget-extension ablation")
    parser.add_argument("--wandb-project",
                        help="stream non-secret progress metrics to this W&B project")
    parser.add_argument("--wandb-entity", help="optional W&B entity/team")
    parser.add_argument("--wandb-group", help="optional W&B run group")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.nproc_per_node < 1 or args.smoke_count < 0:
        parser.error("nproc must be positive and smoke-count nonnegative")
    if args.smoke_count and "smoke" not in args.run_dir.name.lower():
        parser.error("Smoke requires a separate run directory containing 'smoke'")
    legacy_cli_aliases = deprecated_experiment_flags(
        cost_attribution=args.experimental_cost_attribution,
        extended_budgets=args.experimental_extended_budgets,
    )
    wandb_settings = wandb_contract(args)
    if args.stage == "worker":
        if any(v is None for v in (args.model_index,args.dataset,args.backend,args.output,args.identity)):
            parser.error("Internal worker requires model, dataset, backend, output and identity")
        from .official_audit import validate_worker_contract
        validate_worker_contract(args, config)
        from .official_worker import worker
        worker(args, config)
        return
    models = [args.model_index] if args.model_index is not None else list(range(len(MODELS)))
    datasets = [args.dataset] if args.dataset else list(LIMITS)
    if args.stage == "plan":
        result = plan(config, models, datasets, args.smoke_count,
                      args.nproc_per_node, args.experimental_cost_attribution,
                      args.experimental_extended_budgets)
        if wandb_settings:
            result["wandb"] = wandb_settings
        print(json.dumps(result,
                         ensure_ascii=False, indent=2))
        return
    if args.stage == "doctor":
        print(json.dumps(doctor(args.nproc_per_node),ensure_ascii=False,indent=2))
        return
    if args.stage in {"prepare","all"}:
        prepare(args.data_dir)
        if args.stage == "prepare":
            print(json.dumps(check_manifest(args.data_dir),ensure_ascii=False,indent=2))
            return
    audit_policy = args.greedy_audit_policy
    manifest = check_manifest(args.data_dir)
    metadata = {"version":6, "config":config, "source_manifest":verify_sources(),
                "dataset_manifest":manifest, "code_identity":code_identity(),
                "nproc_per_node":args.nproc_per_node, "model_indices":models, "datasets":datasets,
                "smoke_count":args.smoke_count, "max_new_tokens":32 if args.smoke_count else 2048,
                "method_schema_version":METHOD_SCHEMA_VERSION,
                "primary_adaptive_method":PRIMARY_ADAPTIVE_METHOD,
                "greedy_audit_policy":audit_policy,
                "method_order_policy":args.method_order_policy}
    if legacy_cli_aliases:
        metadata["deprecated_cli_aliases"] = list(legacy_cli_aliases)
    if wandb_settings:
        metadata["wandb"] = wandb_settings
    with run_lock(args.run_dir):
        identity = contract(args.run_dir, metadata)
    from .official_reporting import load_completed, run_stem, summarize, validate_run_contract
    if args.stage == "summarize":
        with run_lock(args.run_dir/"execution"):
            summarize(args.run_dir,args.data_dir,config,identity,models,datasets,args.smoke_count)
        return
    env = doctor(args.nproc_per_node)
    with run_lock(args.run_dir/"execution"):
        path = args.run_dir/"environment.json"
        if path.exists() and load_json(path) != env:
            raise ValueError("Hardware/software changed; use a new run directory")
        atomic_json(path,env)
        for dataset in datasets:
            for model_index in models:
                for backend in ("sdpa","flash_attention_2"):
                    output = args.run_dir/(run_stem(dataset,model_index,backend)+".pt")
                    if output.with_suffix(".complete.json").exists():
                        run = load_completed(output,identity)
                        validate_run_contract(run, load_json(args.data_dir/"source_revisions.json"),
                                              args.nproc_per_node,args.smoke_count,env,audit_policy,
                                              (),
                                              metadata.get("wandb"),
                                              args.method_order_policy,
                                              metadata.get("deprecated_cli_aliases", ()))
                        continue
                    if output.exists():
                        raise FileExistsError("Incomplete artifact exists; use a new run directory, not silent overwrite")
                    worker_args = ["-m","dflash_specblock.paper","worker","--config",str(args.config.resolve()),
                        "--data-dir",str(args.data_dir.resolve()),"--run-dir",str(args.run_dir.resolve()),
                        "--model-index",str(model_index),"--dataset",dataset,"--backend",backend,
                        "--output",str(output.resolve()),"--identity",identity,
                        "--nproc-per-node",str(args.nproc_per_node),"--smoke-count",str(args.smoke_count),
                        "--greedy-audit-policy",audit_policy,
                        "--method-order-policy",args.method_order_policy]
                    if args.experimental_cost_attribution:
                        worker_args.append("--experimental-cost-attribution")
                    if args.experimental_extended_budgets:
                        worker_args.append("--experimental-extended-budgets")
                    if args.wandb_project:
                        worker_args += ["--wandb-project", args.wandb_project]
                    if args.wandb_entity:
                        worker_args += ["--wandb-entity", args.wandb_entity]
                    if args.wandb_group:
                        worker_args += ["--wandb-group", args.wandb_group]
                    if args.nproc_per_node == 1:
                        command = [sys.executable,*worker_args]
                    else:
                        command = [sys.executable,"-m","torch.distributed.run",
                            "--nproc_per_node",str(args.nproc_per_node),"--master_port",str(args.master_port),
                            *worker_args]
                    subprocess.run(command,check=True)
        if args.stage == "all":
            summarize(args.run_dir,args.data_dir,config,identity,models,datasets,args.smoke_count)


if __name__ == "__main__":
    main()
