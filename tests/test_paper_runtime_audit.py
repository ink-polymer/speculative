from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

from dflash_specblock.paper import official_audit
from dflash_specblock.paper.common import atomic_json, contract, load_json
from dflash_specblock.paper.official import main
from dflash_specblock.paper.official_reporting import validate_run_contract
from test_paper_official_protocol import cfg, synthetic_environment, synthetic_run


def worker_fixture(tmp_path, monkeypatch):
    config = cfg()
    metadata = {"config":config, "code_identity":"current-test-code", "source_manifest":{"test":True},
        "dataset_manifest":{"test":True}, "model_indices":[1], "datasets":["gsm8k"],
        "nproc_per_node":1, "smoke_count":0, "max_new_tokens":2048}
    monkeypatch.setattr(official_audit,"code_identity",lambda:"current-test-code")
    monkeypatch.setattr(official_audit,"verify_sources",lambda:{"test":True})
    monkeypatch.setattr(official_audit,"check_manifest",lambda _: {"test":True})
    monkeypatch.setenv("WORLD_SIZE","1")
    identity = contract(tmp_path, metadata)
    args = SimpleNamespace(run_dir=tmp_path, data_dir=tmp_path/"data", identity=identity,
        model_index=1, dataset="gsm8k", backend="sdpa", nproc_per_node=1, smoke_count=0,
        output=tmp_path/"gsm8k__model1__temp0.0__sdpa.pt")
    return args, config


@pytest.mark.parametrize("change",["code","source","data","model","dataset","nproc","smoke","world","output","hash"])
def test_worker_rejects_stale_or_misrouted_parent_contract(tmp_path,monkeypatch,change):
    args, config = worker_fixture(tmp_path,monkeypatch)
    official_audit.validate_worker_contract(args,config)
    if change == "code": monkeypatch.setattr(official_audit,"code_identity",lambda:"modified-after-launch")
    elif change == "source": monkeypatch.setattr(official_audit,"verify_sources",lambda:{"modified":True})
    elif change == "data": monkeypatch.setattr(official_audit,"check_manifest",lambda _: {"modified":True})
    elif change == "model": args.model_index = 0
    elif change == "dataset": args.dataset = "math500"
    elif change == "nproc": args.nproc_per_node = 2
    elif change == "smoke": args.smoke_count = 2
    elif change == "world": monkeypatch.setenv("WORLD_SIZE","2")
    elif change == "output": args.output = tmp_path/"different-model.pt"
    elif change == "hash":
        recorded = load_json(tmp_path/"contract.json")
        recorded["metadata"]["code_identity"] = "changed-without-new-hash"
        atomic_json(tmp_path/"contract.json",recorded)
    with pytest.raises(ValueError):
        official_audit.validate_worker_contract(args,config)


def test_worker_cli_checks_before_entering_model_loader(tmp_path,monkeypatch):
    args, config = worker_fixture(tmp_path,monkeypatch)
    entered = []
    monkeypatch.setattr("dflash_specblock.paper.official_worker.worker",lambda a,c:entered.append(a.model_index))
    flags = ["worker","--model-index","1","--dataset","gsm8k","--backend","sdpa",
             "--run-dir",str(tmp_path),"--data-dir",str(args.data_dir),"--output",str(args.output),
             "--identity",args.identity,"--nproc-per-node","1"]
    main(flags)
    assert entered == [1]
    monkeypatch.setattr(official_audit,"code_identity",lambda:"changed")
    with pytest.raises(ValueError,match="parent contract"):
        main(flags)
    assert entered == [1]


@pytest.mark.parametrize("field,value",[("uuid","different-gpu"),("gpu","different-card"),
                                        ("flash_attn","different-library")])
def test_result_validator_rejects_different_gpu_or_library(field,value):
    run = synthetic_run()
    run["hardware"][0][field] = value
    with pytest.raises(ValueError,match="identity or FlashAttention"):
        validate_run_contract(run,{"test_only":True},1,0,synthetic_environment())


def test_second_gpu_is_checked_even_when_rank_zero_is_unchanged():
    environment = synthetic_environment()
    environment["benchmark_gpus"].append({"rank":1,"gpu":"synthetic","uuid":"gpu-1"})
    hardware = [*synthetic_run()["hardware"],
                {"rank":1,"gpu":"synthetic","uuid":"replacement","flash_attn":"test-only"}]
    with pytest.raises(ValueError,match="identity or FlashAttention"):
        official_audit.validate_hardware(hardware,environment,2)


def test_gpu_identity_requires_real_identifier(monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda,"get_device_properties",lambda _:SimpleNamespace())
    with pytest.raises(RuntimeError,match="GPU UUID"):
        official_audit.gpu_identity(0,0)
    monkeypatch.setattr(torch.cuda,"get_device_properties",lambda _:SimpleNamespace(uuid="GPU-test"))
    monkeypatch.setattr(torch.cuda,"get_device_name",lambda _:"Test device")
    assert official_audit.gpu_identity(0,0) == {"rank":0,"gpu":"Test device","uuid":"GPU-test"}


def test_legacy_environment_without_per_gpu_ids_is_rejected():
    environment = synthetic_environment()
    del environment["benchmark_gpus"]
    with pytest.raises(ValueError,match="per-GPU UUID"):
        validate_run_contract(synthetic_run(),{"test_only":True},1,0,environment)
