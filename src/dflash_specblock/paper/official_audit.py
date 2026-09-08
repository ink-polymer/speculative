"""Read-only worker lineage and per-device benchmark identity checks."""
from __future__ import annotations

import os

from .common import code_identity, digest, load_json
from .official_data import check_manifest
from .official_spec import verify_sources


def validate_worker_contract(args, config):
    recorded = load_json(args.run_dir / "contract.json")
    metadata = recorded["metadata"]
    if recorded["identity"] != args.identity or digest(metadata) != args.identity:
        raise ValueError("Worker contract hash/identity mismatch")
    if (metadata["config"] != config or metadata["code_identity"] != code_identity()
            or metadata["source_manifest"] != verify_sources()
            or metadata["dataset_manifest"] != check_manifest(args.data_dir)
            or args.model_index not in metadata["model_indices"]
            or args.dataset not in metadata["datasets"]
            or args.nproc_per_node != metadata["nproc_per_node"]
            or args.smoke_count != metadata["smoke_count"]
            or metadata["max_new_tokens"] != (32 if args.smoke_count else 2048)
            or int(os.environ.get("WORLD_SIZE", "1")) != args.nproc_per_node):
        raise ValueError("Worker code/data/model scope differs from the parent contract")
    from .official_reporting import run_stem
    expected = args.run_dir / (run_stem(args.dataset, args.model_index, args.backend) + ".pt")
    if args.output.resolve() != expected.resolve():
        raise ValueError("Worker output path differs from its declared model/dataset/backend")


def gpu_identity(device, rank):
    import torch
    uuid = str(getattr(torch.cuda.get_device_properties(device), "uuid", "unknown"))
    if uuid.lower() in {"", "none", "unknown", "unavailable"}:
        raise RuntimeError("Cannot record GPU UUID; use a CUDA PyTorch build exposing device identity")
    return {"rank":rank, "gpu":torch.cuda.get_device_name(device), "uuid":uuid}


def validate_hardware(hardware, environment, nproc):
    expected = environment.get("benchmark_gpus", [])
    if (len(expected) != nproc or {h["rank"] for h in expected} != set(range(nproc))
            or any(h.get("uuid", "unknown").lower() in {"", "none", "unknown", "unavailable"}
                   for h in expected)):
        raise ValueError("Missing complete per-GPU UUID environment record")
    by_rank = {h["rank"]:h for h in expected}
    for actual in hardware:
        reference = by_rank.get(actual["rank"])
        if (reference is None or any(actual.get(k) != reference.get(k) for k in ("gpu", "uuid"))
                or actual.get("flash_attn") != environment.get("flash_attn")):
            raise ValueError("GPU identity or FlashAttention version changed during the benchmark")
