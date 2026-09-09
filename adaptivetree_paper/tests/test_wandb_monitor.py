from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest

from dflash_specblock.paper.wandb_monitor import (
    finish_wandb, initialize_wandb, log_response, wandb_contract)


class FakeRun:
    def __init__(self):
        self.summary = {}
        self.logs = []
        self.finished = False

    def log(self, values, step):
        self.logs.append((values, step))

    def finish(self):
        self.finished = True


def args(tmp_path, project="adaptive-test"):
    return SimpleNamespace(
        wandb_project=project, wandb_entity=None, wandb_group="group-1",
        run_dir=tmp_path, identity="a" * 64, dataset="gsm8k", smoke_count=2,
    )


def result(tpot, *, decisions=None):
    return SimpleNamespace(
        time_per_output_token=tpot,
        num_output_tokens=4,
        acceptance_lengths=[2, 3],
        stage_times={"draft":.004, "verify":.008},
        adaptive_decisions=decisions,
    )


def test_wandb_monitor_logs_per_response_and_aggregate_without_secret(
        tmp_path, monkeypatch):
    secret = "secret-must-not-enter-config"
    monkeypatch.setenv("WANDB_API_KEY", secret)
    fake_run = FakeRun()
    captured = {}

    def init(**kwargs):
        captured.update(kwargs)
        return fake_run

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(
        init=init, Settings=lambda **values:values))
    run = initialize_wandb(
        args(tmp_path), model_name="Qwen/Qwen3-4B", draft_name="draft",
        backend="sdpa", rank=0, world_size=1,
        diagnostic_variants=("extended",),
        controller_configs={"extended":{"budget_candidates":[128, 256]}},
    )
    assert run is fake_run
    assert secret not in json.dumps(captured)
    assert captured["project"] == "adaptive-test"
    assert captured["group"] == "group-1"
    assert captured["settings"] == {"x_disable_stats":True, "x_disable_meta":True}
    assert captured["save_code"] is False
    response = {
        "baseline":result(.020),
        "extended":result(.010, decisions=[
            {"decision":{"budget":128, "expected_draft_tokens":3.,
                         "predicted_round_ms":4., "predicted_tokens_per_ms":1.}},
            {"decision":{"budget":256, "expected_draft_tokens":4.,
                         "predicted_round_ms":5., "predicted_tokens_per_ms":1.}},
        ]),
    }
    log_response(run, response, step=0, index=7, turn=1)
    metrics, step = run.logs[0]
    assert step == 0
    assert metrics["methods/extended/speedup_vs_run_target"] == 2
    assert metrics["methods/extended/share_rounds_above_128"] == .5
    assert metrics["methods/extended/stage_ms_per_output_token/verify"] == 2
    finish_wandb(run, [response], list(response))
    assert run.finished and run.summary["status"] == "complete"
    assert run.summary["aggregate/extended/speedup_vs_run_target"] == 2


def test_wandb_monitor_requires_environment_secret_only_when_online(
        tmp_path, monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="WANDB_API_KEY"):
        initialize_wandb(
            args(tmp_path), model_name="model", draft_name="draft", backend="sdpa",
            rank=0, world_size=1, diagnostic_variants=(), controller_configs={})
    assert wandb_contract(args(tmp_path, project=None)) is None
