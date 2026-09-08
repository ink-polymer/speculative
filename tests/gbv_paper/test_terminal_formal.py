from copy import deepcopy
from dataclasses import replace
import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from gbv_experiments.common import ROOT, canonical, digest, write_json
from gbv_experiments.terminal_protocol import (NAMES, check_group, compare, freeze_document,
                                              gpu_matches, load_study, method_names, paired_order,
                                              plan, primary_candidate, variants)


def test_formal_plan_reuses_data_and_includes_dflash_and_two_ablations():
    study = load_study(ROOT / "configs/terminal_mass_formal.json")
    value = plan(study)
    assert value["source_rows"] == 786
    assert value["turns_per_method_seed_repeat"] == 866
    assert value["expected_records"] == 42444
    assert value["expected_generations"] == 46764
    assert {v["method"] for v in value["variants"]} == {
        "target", "dflash", "ddtree", "ddtree_terminal_block",
        "ddtree_terminal_serial", "ddtree_terminal_dense"}
    assert not value["formal_complete"] and not value["gpu_executed"]


@pytest.mark.parametrize("name,family,expected", [
    ("NVIDIA H200", "H200", True), ("NVIDIA H200 NVL", "H200", True),
    ("NVIDIA GH200 120GB", "H200", False), ("NVIDIA GH200 120GB", "GH200", True),
    ("NVIDIA H100", "H200", False), ("NVIDIA H2000", "H200", False),
])
def test_gpu_identity_is_not_a_substring(name, family, expected):
    assert gpu_matches(name, family) is expected


def test_registration_refuses_changed_artifacts(tmp_path):
    path = tmp_path / "registration.json"
    freeze_document(path, {"seed": 17})
    freeze_document(path, {"seed": 17})
    with pytest.raises(ValueError, match="Frozen"):
        freeze_document(path, {"seed": 18})
    assert json.loads(path.read_text()) == {"seed": 17}


def test_paired_order_is_reproducible_and_rotates_without_losing_methods():
    order = paired_order(17, 0)
    assert set(order) == set(NAMES) and len(order) == len(NAMES)
    assert paired_order(17, 0) == order
    assert paired_order(17, 1) == order[1:] + order[:1]


def comparison_rows():
    rows = []
    for dataset in ("small", "large"):
        for source in range(3):
            for seed in (17, 29):
                for repeat in range(3):
                    for name, elapsed in (("ddtree", 20.), ("tm_full", 10.)):
                        rows.append({"variant": name, "dataset": dataset, "source_id": str(source),
                                     "seed": seed, "repeat": repeat, "decode_ms": elapsed,
                                     "decode_tokens": 10, "generated_tokens": 11, "e2e_ms": elapsed + 5})
    return rows


def test_bootstrap_uses_paired_prompt_clusters_not_repeated_timings():
    result = compare(comparison_rows(), "tm_full", "ddtree", ["small", "large"], samples=100)
    assert result["dataset_geomean_speedup"] == pytest.approx(2.)
    assert result["ci95"] == pytest.approx([2., 2.])
    assert result["matched_pairs"] == 36
    assert all(r["prompt_clusters"] == 3 for r in result["per_dataset"].values())
    e2e = compare(comparison_rows(), "tm_full", "ddtree", ["small", "large"], samples=100, metric="e2e")
    assert e2e["dataset_geomean_speedup"] == pytest.approx(25 / 15)


def test_geomean_endpoint_gives_each_dataset_equal_weight():
    rows = comparison_rows()
    for row in rows:
        if row["dataset"] == "large":
            row["decode_tokens"] *= 100
            row["decode_ms"] = 40. if row["variant"] == "tm_full" else 20.
    result = compare(rows, "tm_full", "ddtree", ["small", "large"], samples=30)
    assert result["dataset_geomean_speedup"] == pytest.approx(1.)
    assert result["ci95"] == pytest.approx([1., 1.])


@pytest.mark.parametrize("bad", ["missing", "duplicate", "nan", "negative", "missing_dataset"])
def test_statistics_fail_closed_on_invalid_evidence(bad):
    rows = comparison_rows()
    if bad == "missing":
        rows.pop()
    elif bad == "duplicate":
        rows.append(rows[0])
    elif bad == "nan":
        rows[0]["decode_ms"] = float("nan")
    elif bad == "negative":
        rows[0]["decode_ms"] = -1
    else:
        rows = [r for r in rows if r["dataset"] != "small"]
    with pytest.raises(ValueError):
        compare(rows, "tm_full", "ddtree", ["small", "large"], samples=20)


def test_group_rejects_partial_or_modified_records():
    rows = [{"variant": name, "run_id": "x", "dataset": "d", "source_id": "0", "seed": 17} for name in NAMES]
    group = {"run_id": "x", "repeat": 0, "group_key": ["d", "0", 17],
             "records": rows, "records_sha256": digest(rows)}
    assert check_group(group, "x", "d", "0", 17, 0) == rows
    bad = deepcopy(group)
    bad["records"].pop()
    with pytest.raises(ValueError, match="Incomplete"):
        check_group(bad, "x", "d", "0", 17, 0)
    bad = deepcopy(group)
    bad["records"][0]["extra"] = "modified"
    with pytest.raises(ValueError, match="content changed"):
        check_group(bad, "x", "d", "0", 17, 0)


def test_baseline_follows_vendored_official_traversal(monkeypatch):
    from gbv_experiments import sampling
    module = ast.parse((ROOT / "third_party/ddtree_official/ddtree.py").read_text())
    function = next(node for node in module.body if isinstance(node, ast.FunctionDef)
                    and node.name == "follow_verified_tree")
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "official_traversal", "exec"), namespace)
    parents, tokens = [-1, 0, 0, 1, 3], [0, 1, 2, 1]
    children = [{} for _ in parents]
    for node, parent in enumerate(parents[1:], 1):
        children[parent][tokens[node - 1]] = node
    for choices in ([0, 2, 0, 1, 2], [1, 0, 2, 1, 0], [2, 0, 1, 0, 2]):
        posterior = torch.tensor(choices)
        monkeypatch.setattr(sampling, "sample", lambda p, generator=None: posterior)
        nodes, out, bonus = sampling.tree_verify_ancestral_batched(parents, tokens,
                            torch.full((len(parents), 3), 1 / 3))
        expected_nodes, expected_bonus = namespace["follow_verified_tree"](children, posterior[None])
        assert [0] + nodes == expected_nodes and bonus == expected_bonus
        assert out == [tokens[node - 1] for node in expected_nodes[1:]]


def test_engine_diagnostic_observer_preserves_generation(tiny_engine):
    variant = replace(variants()[2], length=3, tree_budget=12)
    ids = torch.tensor([[1, 3, 5]])
    expected = tiny_engine.generate(ids, variant, 12, [], seed=17)
    states = []
    actual = tiny_engine.generate(ids, variant, 12, [], seed=17,
                                  tree_observer=lambda p, t, a: states.append((p, t, a.clone())))
    assert expected["generated_token_ids"] == actual["generated_token_ids"]
    assert len(states) == len(actual["rounds"])
    assert all(a.shape[0] == len(p) == len(t) + 1 for p, t, a in states)


@pytest.mark.parametrize("study_file", ["terminal_mass_formal.json", "protected_tree_formal.json", "atom_tree_formal.json",
                                      "diffusion_tree_t03.json", "diffusion_tree_t06.json", "diffusion_tree_t10.json",
                                      "diffusion_scaffold_t03.json", "diffusion_scaffold_t06.json", "diffusion_scaffold_t10.json"])
def test_interrupted_timing_group_is_replayed_whole_on_resume(tmp_path, monkeypatch, study_file):
    from gbv_experiments import terminal_formal as formal
    import gbv_experiments.conversation as conversation
    study = load_study(ROOT / "configs" / study_file)
    study["config"]["datasets"] = ["gsm8k"]
    study["config"]["evaluation"] = {"protocol": "full"}
    study["spec"]["seeds"] = [17]
    study["spec"]["max_new_tokens"] = 3
    data = [{"dataset": "gsm8k", "source_id": "0", "prompt": "x", "prompt_sha256": digest("x")}]
    reg = {"registration_id": "r", "data_manifest": {"coverage": "full_evaluation_split"},
           "prompt_ids": [["gsm8k", "0", digest("x")]]}
    monkeypatch.setattr(formal, "registered", lambda *args: (reg, data))
    monkeypatch.setattr(formal, "gate", lambda *args: {"passed": True, "runtime": {}})
    monkeypatch.setattr(formal, "runtime", lambda *args: {})
    monkeypatch.setattr(formal, "telemetry", lambda *args: {})
    monkeypatch.setattr(formal, "allocation_gate", lambda *args: None)
    monkeypatch.setattr(formal, "model_gate", lambda *args: None)
    monkeypatch.setattr(formal, "stop_token_ids", lambda *args: [])
    monkeypatch.setattr(formal, "load_frozen_engine", lambda *args: (
        SimpleNamespace(generate=lambda *args, **kw: None), None))
    monkeypatch.setattr(conversation, "encode_messages", lambda *args: None)
    write_json(tmp_path / "runtime.json", {})
    calls = []
    interrupt = [True]

    def generate(engine, tokenizer, row, variant, *args, **kwargs):
        assert kwargs["profile"] is False
        calls.append(variant.name)
        if interrupt[0] and len(calls) == 3:
            raise RuntimeError("interrupted")
        return {"generated_tokens": 3, "generated_token_ids": [1, 2, 3],
                "decode_tokens": 2, "prefill_ms": 1., "decode_ms": 2., "e2e_ms": 3.,
                "rounds": [], "turn_count": 1, "finish_reason": "length", "text": "x"}

    monkeypatch.setattr(conversation, "generate_conversation", generate)
    with pytest.raises(RuntimeError, match="interrupted"):
        formal.run_repeat(study, tmp_path, tmp_path, "cpu", 0)
    assert not list((tmp_path / "repeat_0/groups").glob("*.json"))
    assert list((tmp_path / "repeat_0/attempts").glob("*.json"))
    interrupt[0] = False
    calls.clear()
    result = formal.run_repeat(study, tmp_path, tmp_path, "cpu", 0)
    assert len(calls) == len(variants(study)) == result["records"]
    calls.clear()
    assert formal.run_repeat(study, tmp_path, tmp_path, "cpu", 0) == result
    assert calls == []


def test_formal_report_cannot_skip_missing_gates(tmp_path, monkeypatch):
    from gbv_experiments import terminal_formal as formal
    study = load_study(ROOT / "configs/terminal_mass_formal.json")
    monkeypatch.setattr(formal, "registered", lambda *args: ({"registration_id": "r"}, []))
    with pytest.raises(FileNotFoundError):
        formal.final_report(study, tmp_path, tmp_path)
    assert not (tmp_path / "formal_report.json").exists()


@pytest.mark.parametrize("tm_ms,achieved", [(10., True), (25., False)])
@pytest.mark.parametrize("study_file", ["terminal_mass_formal.json", "protected_tree_formal.json", "atom_tree_formal.json",
                                      "diffusion_tree_t03.json", "diffusion_tree_t06.json", "diffusion_tree_t10.json",
                                      "diffusion_scaffold_t03.json", "diffusion_scaffold_t06.json", "diffusion_scaffold_t10.json"])
@pytest.mark.parametrize("dflash_ms", [20., 5.])
def test_complete_pipeline_decision_and_tamper_detection(tmp_path, monkeypatch, tm_ms, achieved, study_file, dflash_ms):
    """Exercise exports and the final report using explicitly synthetic timings."""
    from gbv_experiments import terminal_formal as formal
    import gbv_experiments.conversation as conversation
    from gbv_experiments.common import file_hash
    study = load_study(ROOT / "configs" / study_file)
    study["config"]["datasets"] = ["gsm8k", "math500"]
    study["config"]["evaluation"] = {"protocol": "full"}
    study["config"]["seeds"] = study["spec"]["seeds"] = [17]
    study["spec"]["max_new_tokens"] = 3
    study["spec"]["bootstrap_samples"] = 20
    data = [{"dataset": dataset, "source_id": str(i), "prompt": "x", "prompt_sha256": digest("x")}
            for dataset in study["config"]["datasets"] for i in range(2)]
    reg = {"registration_id": "synthetic", "data_manifest": {"coverage": "full_evaluation_split"},
           "prompt_ids": [[r["dataset"], r["source_id"], r["prompt_sha256"]] for r in data]}
    monkeypatch.setattr(formal, "registered", lambda *args: (reg, data))
    monkeypatch.setattr(formal, "runtime", lambda *args: {})
    monkeypatch.setattr(formal, "telemetry", lambda *args: {})
    monkeypatch.setattr(formal, "allocation_gate", lambda *args: None)
    monkeypatch.setattr(formal, "model_gate", lambda *args: None)
    monkeypatch.setattr(formal, "stop_token_ids", lambda *args: [])
    monkeypatch.setattr(formal, "load_frozen_engine", lambda *args: (
        SimpleNamespace(generate=lambda *args, **kw: None), None))
    monkeypatch.setattr(conversation, "encode_messages", lambda *args: None)
    for name, value in (("runtime", {}), ("registration", reg), ("capture_index", [])):
        write_json(tmp_path / f"{name}.json", value)
    for name in ("unit_gate", "data_gate", "diagnostics_gate", "scoring_gate"):
        write_json(tmp_path / f"{name}.json", {"registration_id": "synthetic", "passed": True, "runtime": {}})

    def generate(engine, tokenizer, row, variant, *args, **kwargs):
        elapsed = tm_ms if variant.name == primary_candidate(study) else 20.
        if variant.name == "dflash_match":
            elapsed = dflash_ms
        return {"generated_tokens": 3, "generated_token_ids": [1, 2, 3],
                "decode_tokens": 2, "prefill_ms": 5., "decode_ms": elapsed, "e2e_ms": elapsed + 5.,
                "rounds": [{"accepted_draft_tokens": 1, "committed_tokens": 2, "tree_nodes": 2}],
                "target_forward_calls": 2, "draft_forward_calls": 1,
                "peak_allocated_bytes": 1024, "turn_count": 1, "finish_reason": "length", "text": "x"}

    monkeypatch.setattr(conversation, "generate_conversation", generate)
    for repeat in range(3):
        formal.run_repeat(study, tmp_path, tmp_path, "cpu", repeat)
    rows = formal.read_jsonl(tmp_path / "repeat_0/results.jsonl")
    scores = [{key: row[key] for key in ("run_id", "variant", "dataset", "source_id", "seed")}
              | {"prediction_sha256": digest(row["text"]), "passed": True} for row in rows]
    (tmp_path / "repeat_0/scores.jsonl").write_text("".join(canonical(row) + "\n" for row in scores))
    result = formal.final_report(study, tmp_path, tmp_path)
    both_baselines = study_file == "atom_tree_formal.json" or study_file.startswith("diffusion_")
    expected_achieved = achieved and (not both_baselines or dflash_ms > tm_ms)
    assert result["formal_complete"] and result["target_achieved"] is expected_achieved
    if both_baselines:
        assert result["baseline_speed_gates"] == {"ddtree": achieved, "dflash_match": dflash_ms > tm_ms}
    if study_file.startswith("diffusion_tree_"):
        assert result["temperature"] == study["spec"]["temperature"]
        assert result["novelty_established"] is False and result["research_goal_achieved"] is False
        assert result["universal_acceptance_dominance"] == "disproved_by_counterexample"
        assert result["conditional_acceptance_advantage"] == "proved_under_unverified_distribution_conditions"
        assert result["theory_comparison_document"] == "docs/DIFFUSION_TREE_THEORY_AUDIT.md"
    if study_file.startswith("diffusion_scaffold_"):
        assert result["universal_acceptance_dominance"] == "disproved_vs_ddtree_by_scaffold_counterexample"
        assert result["dflash_acceptance_dominance"] == "proved_at_fixed_context_under_exact_oracle"
        assert result["novelty_established"] is False and result["research_goal_achieved"] is False
        assert result["theory_document"] == "docs/DIFFUSION_SCAFFOLD_BV.md"
    assert result["primary"]["ci95"] == pytest.approx([20 / tm_ms, 20 / tm_ms])
    assert len(result["comparisons"]) == 2 * (len(variants(study)) - 1)
    assert result["records"] == 12 * len(variants(study))
    index = formal.read(tmp_path / "artifact_index.json")
    assert index["sha256"]["formal_report.json"] == file_hash(tmp_path / "formal_report.json")
    assert (tmp_path / "comparisons.csv").is_file()
    assert result["decision"] in (tmp_path / "report.md").read_text()
    path = tmp_path / "repeat_2/results.jsonl"
    path.write_text(path.read_text().replace('"decode_ms":20.0', '"decode_ms":1.0', 1))
    with pytest.raises(ValueError, match="changed formal results"):
        formal.final_report(study, tmp_path, tmp_path)


def test_protected_plan_includes_controls_and_rotates_all_eleven_methods():
    study = load_study(ROOT / "configs/protected_tree_formal.json")
    value = plan(study)
    assert value["expected_records"] == 786 * 11 * 3 * 3
    assert value["expected_generations"] == 866 * 11 * 3 * 3
    assert primary_candidate(study) == "protected_full"
    assert {"dflash_match", "ddtree", "rm_full", "rm_early_root", "rm_shared_ddtree"} <= set(method_names(study))
    assert set(paired_order(17, 0, study)) == set(method_names(study))
    assert len(method_names(study)) == 11


def test_protected_capture_retains_actual_q_and_replays_all_controls(tiny_engine):
    from gbv_experiments.terminal_formal import capture_states, replay_functions
    study = load_study(ROOT / "configs/protected_tree_formal.json")
    for entry in study["config"]["explicit_variants"]:
        entry["variant"]["length"] = 3
        entry["variant"]["tree_budget"] = 12
    states = capture_states(study, tiny_engine, torch.tensor([[1, 3, 5]]), 17, limit=1, tokens=8)
    assert len(states) == 2
    shared = states[1]
    assert shared["kind"] == "shared_suffix" and shared["q"].shape == (3, 17)
    assert shared["paths"].shape == (3, 3)
    assert shared["all_p"].shape == (10, 17)
    methods = replay_functions(shared)
    assert set(methods) == {"protected_full", "rm_shared_ddtree", "rm_full", "rm_serial_ref", "rm_token", "rm_early_root"}
    for name, function in methods.items():
        a = function(torch.Generator().manual_seed(5))
        b = function(torch.Generator().manual_seed(5))
        assert a == b
        if name != "rm_shared_ddtree":
            branch, suffix, bonus = a
            assert -1 <= branch < 3 and 0 <= suffix <= 2 and 0 <= bonus < 17


def test_atom_formal_plan_and_joint_proposal_replay(tiny_engine):
    from gbv_experiments.terminal_formal import capture_states, replay_functions
    from gbv_experiments import atom_tree_bv
    study = load_study(ROOT / "configs/atom_tree_formal.json")
    value = plan(study)
    assert value["expected_records"] == 70740
    assert value["expected_generations"] == 77940
    assert primary_candidate(study) == "atom_full"
    assert study["config"]["model"]["target"] == "Qwen/Qwen3-8B"
    assert {"atom_full", "atom_fixed", "atom_aligned", "atom_no_pool", "atom_ancestral",
            "protected_control", "gbv", "target_t1", "ddtree", "dflash_match"} == set(method_names(study))
    for entry in study["config"]["explicit_variants"]:
        entry["variant"]["length"] = 3
        entry["variant"]["tree_budget"] = 12
    states = capture_states(study, tiny_engine, torch.tensor([[1, 3, 5]]), 17, limit=1, tokens=8)
    assert [s["kind"] for s in states] == ["probability_tree", "latent_atoms"]
    state = states[1]
    proposal = atom_tree_bv.AtomProposal(**{key: state["atom_" + key]
                                          for key in ("roots", "tokens", "slots", "source", "draws")})
    proposal.validate(17)
    assert proposal.source.shape == (2, 25)
    assert proposal.paths().shape == (3, 3)
    functions = replay_functions(state)
    assert set(functions) == {"atom_full", "atom_no_pool", "atom_ancestral"}
    for name, function in functions.items():
        assert function(torch.Generator().manual_seed(5)) == function(torch.Generator().manual_seed(5))
