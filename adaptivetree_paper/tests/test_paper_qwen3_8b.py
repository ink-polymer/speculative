"""Qwen3-8B launch/routing and real config dimensions, without weight downloads."""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest
import torch

from dflash_specblock.paper.common import ROOT, load_json
from dflash_specblock.paper.official_data import validate_lock
from dflash_specblock.paper.official_spec import COMMIT, MODELS, PINNED_MODEL_REVISIONS, SOURCES, upstream
from dflash_specblock.paper import qwen3_8b


def test_qwen8b_default_only_plans_one_model(capsys, monkeypatch):
    monkeypatch.setenv("NPROC_PER_NODE", "8")
    qwen3_8b.main([])
    plan = json.loads(capsys.readouterr().out)
    assert plan["models"] == ["Qwen/Qwen3-8B"]
    pair = plan["model_pairs"][0]
    assert pair["draft"] == "z-lab/Qwen3-8B-DFlash-b16"
    assert pair["target_revision"] == PINNED_MODEL_REVISIONS[pair["target"]]
    assert pair["draft_revision"] == PINNED_MODEL_REVISIONS[pair["draft"]]
    assert plan["cases"] == 1072 and plan["turns_per_method"] == 1152
    assert plan["benchmark_process_groups"] == 20 and plan["generation_calls"] == 18432
    assert not plan["training"] and not plan["launches_models"]


def test_qwen8b_has_independent_output_and_accepts_explicit_smoke_directory(monkeypatch):
    calls = []
    monkeypatch.setattr(qwen3_8b, "official_main", lambda args: calls.append(args))
    qwen3_8b.main(["evaluate"])
    assert calls[-1] == ["evaluate", "--model-index", "1", "--run-dir",
                         str(ROOT/"outputs/adaptive_ddtree_official_t0_qwen3_8b")]
    qwen3_8b.main(["all", "--smoke-count", "2", "--run-dir", "outputs/8b_smoke"])
    assert calls[-1][-4:] == ["--smoke-count", "2", "--run-dir", "outputs/8b_smoke"]


@pytest.mark.parametrize("flags", [["--model-index", "0"], ["--model-index=0"], ["--model-index=1"]])
def test_qwen8b_does_not_silently_select_another_model(flags):
    with pytest.raises(ValueError, match="fixes --model-index=1"):
        qwen3_8b.main(["plan", *flags])


def test_abbreviated_model_override_is_rejected():
    with pytest.raises(SystemExit) as exc:
        qwen3_8b.main(["plan", "--model-i", "0"])
    assert exc.value.code == 2


def test_source_lock_rejects_a_different_8b_revision():
    lock = {"official_commit": COMMIT,
            "datasets": {s[0]: "a"*40 for s in SOURCES.values()},
            "models": {name: PINNED_MODEL_REVISIONS.get(name, "b"*40) for pair in MODELS for name in pair}}
    validate_lock(lock)
    lock["models"]["z-lab/Qwen3-8B-DFlash-b16"] = "c"*40
    with pytest.raises(ValueError, match="Pinned 4B/8B"):
        validate_lock(lock)


def test_qwen8b_actual_configs_construct_with_pinned_official_draft():
    # Meta tensors allocate no model weight storage; this is not a GPU forward test.
    from transformers import Qwen3Config, Qwen3ForCausalLM
    records = load_json(ROOT/"configs/paper_qwen3_8b_model_metadata.json")
    for record in records:
        assert record["revision"] == PINNED_MODEL_REVISIONS[record["repo"]]
    target_cfg, draft_cfg = [Qwen3Config(**r["config"]) for r in records]
    target_cfg._attn_implementation = draft_cfg._attn_implementation = "sdpa"
    with torch.device("meta"):
        target = Qwen3ForCausalLM(target_cfg)
        draft = upstream().model.DFlashDraftModel(draft_cfg)
    assert target.get_input_embeddings().weight.shape == (151936, 4096)
    assert target.lm_head.weight.shape == (151936, 4096)
    assert target_cfg.hidden_size == draft_cfg.hidden_size == 4096
    assert target_cfg.num_hidden_layers == draft_cfg.num_target_layers == 36
    assert len(draft.layers) == 5 and draft.block_size == 16
    assert draft.target_layer_ids == [1, 9, 17, 25, 33]
    assert draft.fc.in_features == 5*4096 and draft.fc.out_features == 4096
    assert all(p.is_meta for p in target.parameters()) and all(p.is_meta for p in draft.parameters())


def test_qwen8b_shell_entry_no_arguments_only_plans():
    result = subprocess.run(["bash", str(ROOT/"scripts/run_paper_t0_qwen3_8b.sh")],
        cwd=ROOT, env={**os.environ, "PAPER_PYTHON": sys.executable, "NPROC_PER_NODE": "1",
                       "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"},
        check=True, capture_output=True, text=True)
    plan = json.loads(result.stdout)
    assert plan["models"] == ["Qwen/Qwen3-8B"] and plan["nproc_per_node"] == 1
    assert not plan["launches_models"]
