from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
import torch

from gbv_experiments.common import ROOT, read_jsonl, write_json
from gbv_experiments.config import build_variants, load_config, select_variants
from gbv_experiments.suite import load_suite, phase_variants, plan_suite


SUITE = ROOT / "configs/gbv_paper_suite.json"


@pytest.mark.parametrize("phase,variants,jobs,records,generations", [
    ("gbv-first", 2, 42, 4716, 5196),
    ("main", 10, 210, 23580, 25980),
    ("complete", 20, 420, 47160, 51960),
])
def test_two_model_workload_and_focused_controls(phase, variants, jobs, records, generations):
    plan = plan_suite(SUITE, phase)
    assert (plan["model_count"], plan["model_variant_count"], plan["evaluation_jobs"],
            plan["expected_records"], plan["expected_generations"]) == (2, variants, jobs, records, generations)
    models = load_suite(SUITE)
    assert [len(build_variants(m["config"])) for m in models] == [13, 7]
    assert models[0]["config"]["evaluation"] == models[1]["config"]["evaluation"]
    for model in models:
        entries = build_variants(model["config"])
        by_name = {e["variant"]["name"]: e["variant"] for e in entries}
        assert all(v["draft_attention"] == "bidirectional" and v["condition_features"] == "target" for v in by_name.values())
        assert by_name["gbv"]["paths"] == 3
        assert by_name["ddtree"]["tree_budget"] == 3 * 15
        assert by_name["dflash_match"]["method"] == "dflash"
        assert "gbv_k1" not in by_name
        groups = {g for e in entries for g in e["groups"]}
        assert not groups & {"bidirectional_attention", "target_features", "draft_cache", "probability_precision"}
        if model["id"] == "qwen3_4b":
            assert by_name["single_token_rejection"]["method"] == "token"
            assert "main" not in next(e["groups"] for e in entries if e["variant"]["name"] == "single_token_rejection")
            assert "paths" in next(e["groups"] for e in entries if e["variant"]["name"] == "dflash_bv")
        if phase == "gbv-first":
            assert [e["variant"]["name"] for e in plan["models"][model["id"]]["variants"]] == ["gbv"]


def test_suite_rejects_mixed_data_and_duplicate_models(tmp_path):
    models = load_suite(SUITE)
    cfg4, cfg8 = [deepcopy(m["config"]) for m in models]
    write_json(tmp_path / "4.json", cfg4)
    cfg8["seeds"] = [17]
    write_json(tmp_path / "8.json", cfg8)
    path = tmp_path / "suite.json"
    write_json(path, {"models": [{"id": "a", "config": "4.json"}, {"id": "b", "config": "8.json"}]})
    with pytest.raises(ValueError, match="same data"):
        load_suite(path)
    write_json(path, {"models": [{"id": "a", "config": "4.json"}, {"id": "b", "config": "4.json"}]})
    with pytest.raises(ValueError, match="Duplicate suite model pair"):
        load_suite(path)
    for selected in ([], ["qwen8"], ["qwen3_8b", "qwen3_8b"]):
        with pytest.raises(ValueError, match="selection"):
            load_suite(SUITE, selected)
    assert list(plan_suite(SUITE, model_ids=["qwen3_8b"])["models"]) == ["qwen3_8b"]
    with pytest.raises(ValueError, match="phase"):
        phase_variants(cfg4, "invalid")
    with pytest.raises(ValueError):
        select_variants(build_variants(cfg4), ["missing"])


def test_qwen8b_pinned_configs_construct_with_vendored_draft():
    # Meta tensors verify full architecture dimensions without downloading weights.
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from gbv_experiments.engine import draft_model_class
    metadata = json.loads((ROOT / "experiments/gbv_paper/qwen3_8b_metadata.json").read_text())
    cfg = load_config(ROOT / "configs/gbv_paper_qwen3_8b.json")["model"]
    for item, kind in zip(metadata, ("target", "draft")):
        assert (item["repo"], item["revision"]) == (cfg[kind], cfg[kind + "_revision"])
    target_cfg, draft_cfg = [Qwen3Config(**item["config"]) for item in metadata]
    target_cfg._attn_implementation = draft_cfg._attn_implementation = "sdpa"
    with torch.device("meta"):
        target = Qwen3ForCausalLM(target_cfg)
        draft = draft_model_class()(draft_cfg)
    assert target.get_input_embeddings().weight.shape == (151936, 4096)
    assert draft.fc.in_features == 5 * 4096 and draft.fc.out_features == 4096
    assert draft.block_size == 16
    assert draft.target_layer_ids == [1, 9, 17, 25, 33]
    assert all(0 <= i < target_cfg.num_hidden_layers for i in draft.target_layer_ids)
    assert draft_cfg.num_target_layers == target_cfg.num_hidden_layers == 36
    assert target_cfg.hidden_size == draft_cfg.hidden_size == 4096
    assert target_cfg.vocab_size == draft_cfg.vocab_size == 151936


def test_gbv_first_resume_reuses_results_and_preserves_full_manifest(tmp_path, monkeypatch):
    import gbv_experiments.runner as runner
    import gbv_experiments.engine as engine_module
    import gbv_experiments.conversation as conversation

    cfg = load_config(ROOT / "configs/gbv_paper_ddtree_counts.json")
    cfg["datasets"], cfg["seeds"] = ["gsm8k"], [17]
    cfg.pop("evaluation")
    rows = [{"dataset": "gsm8k", "source_id": str(i), "prompt_sha256": "h"} for i in range(2)]
    monkeypatch.setattr(runner, "load_prepared", lambda *args: ({"fixture": True}, rows))
    monkeypatch.setattr(runner, "model_identity", lambda config: config)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: SimpleNamespace(total_memory=1))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "mock scheduler device")
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *args, **kwargs: "mock-driver")
    calls, loads = [], []
    target = torch.nn.Linear(1, 1)
    target.generation_config = SimpleNamespace(eos_token_id=0)
    draft = torch.nn.Linear(1, 1)
    draft.block_size, draft.target_layer_ids = 16, [1]
    engine = SimpleNamespace(target=target, draft=draft, generate=lambda *a, **k: None)
    tokenizer = SimpleNamespace(eos_token_id=0)
    def load(*args):
        loads.append(True)
        return engine, tokenizer
    def generate(engine, tokenizer, row, variant, *args):
        calls.append((variant.name, row["source_id"]))
        return {"turn_count": 1, "text": "fixed scheduler fixture"}
    monkeypatch.setattr(engine_module, "load_models", load)
    monkeypatch.setattr(conversation, "encode_messages", lambda *a: None)
    monkeypatch.setattr(conversation, "generate_conversation", generate)
    runner.run(cfg, tmp_path / "data", tmp_path, "cuda:0", only_variants=["gbv"])
    first_rows = read_jsonl(tmp_path / "results.jsonl")
    manifest = json.loads((tmp_path / "run_manifest.json").read_text())
    assert len(first_rows) == 2 and len(manifest["variants"]) == 13
    assert not (tmp_path / "completed.json").exists()
    stage = json.loads(next(tmp_path.glob("stage_completed_*.json")).read_text())
    assert stage["stage_complete"] and not stage["full_experiment_complete"]
    runner.run(cfg, tmp_path / "data", tmp_path, "cuda:0", only_variants=["gbv"])
    assert len(loads) == 1 and len(calls) == 2
    runner.run(cfg, tmp_path / "data", tmp_path, "cuda:0")
    complete_rows = read_jsonl(tmp_path / "results.jsonl")
    assert len(complete_rows) == len(calls) == 26 and len(loads) == 2
    assert [r for r in complete_rows if r["variant"] == "gbv"] == first_rows
    assert len({runner.key(r) for r in complete_rows}) == 26
    assert {r["run_id"] for r in complete_rows} == {manifest["run_id"]}
    assert json.loads((tmp_path / "completed.json").read_text())["records"] == 26
