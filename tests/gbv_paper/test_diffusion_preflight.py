"""Fail-closed positive-temperature diagnostics for candidates AND baselines."""
import json

import pytest
import torch

from gbv_experiments import diffusion_preflight as preflight
from gbv_experiments.common import ROOT, write_json
from gbv_experiments.config import DIFFUSION_SCAFFOLD_METHODS, Variant
from gbv_experiments.terminal_protocol import load_study


def variant(method, temperature=.6):
    return Variant(name=method, method=method, paths=1 if method in {"target", "dflash", "ddtree", *DIFFUSION_SCAFFOLD_METHODS} else 3,
                   length=3, tree_budget=9, temperature=temperature, draft_temperature=temperature)


@pytest.mark.parametrize("temperature", [.3, .6, 1.])
@pytest.mark.parametrize("method", ["target", "dflash", "ddtree", "gbv", "atom_tree_bv"])
def test_real_baselines_receive_positive_temperature_checks(tiny_engine, method, temperature):
    v = variant(method, temperature)
    result = preflight.probe(tiny_engine, torch.tensor([[1, 3, 5]]), v, tokens=16, tv_limit=1e-6)
    assert result["passed"] and result["temperature"] == temperature
    assert result["finite_probabilities_checked"] and result["forward_call_counts_passed"]
    assert len(result["ar_checks"]) == (3 if method == "target" else 1)
    assert result["captured_rounds"] == (0 if method == "target" else 3)
    assert result["max_node_total_variation"] < 1e-6


@pytest.mark.parametrize("seed", [101, 104])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("method", ["diffusion_tree_bv", "dflash", "ddtree"])
def test_invalid_unselected_leaf_cannot_pass(tiny_engine, monkeypatch, seed, bad, method):
    original = tiny_engine.target_forward

    def corrupt(*args, **kwargs):
        output = original(*args, **kwargs)
        if kwargs.get("mask") is not None:
            output.logits[0, -1, :] = bad
        return output

    monkeypatch.setattr(tiny_engine, "target_forward", corrupt)
    with pytest.raises(FloatingPointError, match="invalid or nonfinite"):
        preflight.probe(tiny_engine, torch.tensor([[1, 3, 5]]), variant(method, 1.),
                        tokens=16, seed=seed, tv_limit=1e-6)


@pytest.mark.parametrize("method", ["target", "dflash", "ddtree", "gbv", "atom_tree_bv"])
def test_finite_but_wrong_baseline_conditionals_fail(tiny_engine, monkeypatch, method):
    generate, forward = tiny_engine.generate, tiny_engine.target_forward
    active = False

    def wrapped_generate(*args, **kwargs):
        nonlocal active
        active = True
        try:
            return generate(*args, **kwargs)
        finally:
            active = False

    def corrupt(*args, **kwargs):
        output = forward(*args, **kwargs)
        # Only the measured run is corrupted, never the independent reference.
        # Speculative methods keep a valid prefill and corrupt their tree rows.
        if active and (method == "target" or kwargs.get("mask") is not None):
            output.logits[..., 0] += 10
        return output

    monkeypatch.setattr(tiny_engine, "generate", wrapped_generate)
    monkeypatch.setattr(tiny_engine, "target_forward", corrupt)
    result = preflight.probe(tiny_engine, torch.tensor([[1, 3, 5]]), variant(method), tokens=16)
    assert not result["passed"] and result["max_node_total_variation"] > .1
    assert result["same_seed_repeatable"]  # Repeatability alone must not pass.


def test_zero_probability_tokens_are_not_rejected():
    p = preflight._checked_probabilities(torch.tensor([[0., -float("inf")]]), .3, "valid zero")
    assert torch.equal(p, torch.tensor([[1., 0.]], dtype=torch.float64))
    assert preflight._node_tv(p, p) == 0


@pytest.mark.parametrize("temperature", [1e-320, 1., 1e308])
def test_extreme_finite_temperature_is_stable_and_shared(temperature):
    from gbv_experiments import diffusion_tree_bv as diffusion
    logits = torch.tensor([[1e308, -1e308]], dtype=torch.float64)
    expected = (torch.tensor([[1., -1.]], dtype=torch.float64).softmax(-1)
                if temperature == 1e308 else torch.tensor([[1., 0.]], dtype=torch.float64))
    p = preflight._checked_probabilities(logits, temperature, "extreme but valid")
    torch.testing.assert_close(p, expected, rtol=1e-14, atol=0)
    law = diffusion.DiffusionBlockLaw.from_logits(logits, torch.tensor([0, 1]), temperature, 1)
    torch.testing.assert_close(law.weights, expected, rtol=1e-14, atol=0)
    assert law.retained_block_mass() == 1


@pytest.mark.parametrize("temperature", [float("nan"), float("inf"), -float("inf"), -1.])
def test_invalid_temperature_does_not_silently_make_a_distribution(temperature):
    from gbv_experiments.sampling import probabilities
    with pytest.raises(ValueError, match="finite positive temperature"):
        probabilities(torch.tensor([[1., 0.]]), temperature)


@pytest.mark.parametrize("temperature", [1e-320, 1., 1e308])
def test_large_common_logit_offset_preserves_retained_support_mass(temperature):
    from gbv_experiments.diffusion_tree_bv import DiffusionBlockLaw
    logits = torch.full((2, 2), 1e308, dtype=torch.float64)
    law = DiffusionBlockLaw.from_logits(logits, torch.tensor([0, 1, 1]), temperature, 1, support_size=1)
    torch.testing.assert_close(law.retained_mass, torch.full((2,), .5, dtype=torch.float64), rtol=1e-14, atol=0)
    assert float(law.retained_block_mass()) == pytest.approx(.25)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.])
def test_nonfinite_or_negative_tv_inputs_fail_before_max(bad):
    actual = torch.tensor([[.5, .5]], dtype=torch.float64)
    reference = torch.tensor([[.5, bad]], dtype=torch.float64)
    with pytest.raises(FloatingPointError, match="invalid or nonfinite"):
        preflight._node_tv(actual, reference)


def mock_checkpoint_environment(monkeypatch, engine=None):
    monkeypatch.setattr("gbv_experiments.preflight.check_environment", lambda *a: {})
    monkeypatch.setattr(preflight, "numerical_probe", lambda device: {"passed": True})
    monkeypatch.setattr("gbv_experiments.engine.load_models", lambda *a: (engine, None))
    monkeypatch.setattr("gbv_experiments.conversation.encode_messages", lambda *a: torch.tensor([[1, 3, 5]]))


@pytest.mark.parametrize("tag,temperature", [("t03", .3), ("t06", .6), ("t10", 1.)])
@pytest.mark.parametrize("family,count", [("diffusion_tree", 11), ("diffusion_scaffold", 10)])
def test_checkpoint_gate_exercises_every_registered_method(tiny_engine, tmp_path, monkeypatch, tag, temperature, family, count):
    cfg = load_study(ROOT / f"configs/{family}_{tag}.json")["config"]
    for entry in cfg["explicit_variants"]:
        entry["variant"]["length"] = 3
        entry["variant"]["tree_budget"] = 9
    mock_checkpoint_environment(monkeypatch, tiny_engine)
    output = tmp_path / "checks.json"
    result = preflight.check_diffusion_model(cfg, output, "cpu")
    expected = {entry["variant"]["name"] for entry in cfg["explicit_variants"]}
    assert result["passed"] and set(result["required_variants"]) == expected
    assert len(expected) == count and len(result["checks"]) == 2 * (count + 1)
    prompts = {check["prompt_sha256"] for check in result["checks"]}
    assert len(prompts) == 2
    for prompt in prompts:
        checks = [check for check in result["checks"] if check["prompt_sha256"] == prompt]
        assert {check["variant"]["name"] for check in checks} == expected
        assert all(check["temperature"] == temperature and check["passed"] for check in checks)
        uncached = [check for check in checks if not check["variant"]["reuse_draft_cache"]]
        assert len(uncached) == 1 and uncached[0]["variant"]["name"] == "diffusion_full"
        for check in checks:
            if check["variant"]["method"] in DIFFUSION_SCAFFOLD_METHODS:
                assert check["scaffold_rounds_checked"] >= check["captured_rounds"] > 0
                assert all(row["scaffold_contract"]["labelled_paths_checked"] for row in check["round_checks"])
    assert json.loads(output.read_text()) == result


@pytest.mark.parametrize("failing", ["target_sample", "dflash_match", "ddtree", "atom_control", "gbv"])
@pytest.mark.parametrize("failure_kind", ["mismatch", "nonfinite"])
def test_failed_baseline_replaces_stale_success_evidence(tmp_path, monkeypatch, failing, failure_kind):
    cfg = load_study(ROOT / "configs/diffusion_tree_t03.json")["config"]
    mock_checkpoint_environment(monkeypatch)
    seen = []

    def check(engine, ids, v):
        seen.append(v.name)
        if v.name == failing:
            if failure_kind == "nonfinite":
                raise FloatingPointError("invalid or nonfinite target logits")
            return {"passed": False, "max_node_total_variation": .5}
        return {"passed": True}

    monkeypatch.setattr(preflight, "probe", check)
    output = tmp_path / "checks.json"
    write_json(output, {"passed": True, "stale": True})
    with pytest.raises(RuntimeError, match="checkpoint check failed"):
        preflight.check_diffusion_model(cfg, output, "cpu")
    evidence = json.loads(output.read_text())
    assert not evidence["passed"] and "stale" not in evidence
    assert evidence["checks"][-1]["variant"]["name"] == failing
    assert not evidence["checks"][-1]["passed"] and seen[-1] == failing
    if failure_kind == "nonfinite":
        assert evidence["checks"][-1]["error"]["type"] == "FloatingPointError"
    else:
        assert evidence["checks"][-1]["max_node_total_variation"] == .5


@pytest.mark.parametrize("temperature", [0., .6])
def test_checkpoint_gate_rejects_unmatched_or_zero_baseline_temperature(tmp_path, temperature):
    cfg = load_study(ROOT / "configs/diffusion_tree_t03.json")["config"]
    cfg["explicit_variants"][1]["variant"]["temperature"] = temperature
    with pytest.raises(RuntimeError, match="checkpoint check failed"):
        preflight.check_diffusion_model(cfg, tmp_path / "checks.json", "cpu")
    assert json.loads((tmp_path / "checks.json").read_text())["passed"] is False


def test_ar_observer_does_not_change_samples(tiny_engine):
    ids = torch.tensor([[1, 3, 5]])
    for method in ("target", "diffusion_tree_bv"):
        v = variant(method)
        expected = tiny_engine.generate(ids, v, 16, [], seed=101)
        seen = []
        actual = tiny_engine.generate(ids, v, 16, [], seed=101,
                                      ar_observer=lambda offset, logits: seen.append((offset, logits.clone())))
        assert actual["generated_token_ids"] == expected["generated_token_ids"]
        assert [offset for offset, logits in seen] == (list(range(16)) if method == "target" else [0])


def test_formal_diagnostics_marks_exception_as_failed(tmp_path, monkeypatch):
    from gbv_experiments import terminal_formal as formal
    study = load_study(ROOT / "configs/diffusion_tree_t03.json")
    monkeypatch.setattr(formal, "registered", lambda *a: ({"registration_id": "test"}, []))
    monkeypatch.setattr(formal, "gate", lambda *a: {})
    monkeypatch.setattr(formal, "runtime", lambda *a: {})
    monkeypatch.setattr(formal, "allocation_gate", lambda *a: None)

    def fail(*args):
        raise RuntimeError("nonfinite target")

    monkeypatch.setattr(preflight, "check_diffusion_model", fail)
    write_json(tmp_path / "diagnostics_gate.json", {"registration_id": "test", "passed": True})
    with pytest.raises(RuntimeError, match="nonfinite target"):
        formal.diagnostics(study, tmp_path, tmp_path, "cpu")
    assert json.loads((tmp_path / "diagnostics_gate.json").read_text())["passed"] is False


@pytest.mark.parametrize("mutation", ["missing_greedy", "no_recycling", "missing_witness", "late_no_recycling"])
def test_scaffold_proof_premises_fail_closed(tiny_engine, monkeypatch, mutation):
    from gbv_experiments import diffusion_tree_bv as diffusion
    from gbv_experiments.tree import sampled_tree

    if mutation == "missing_greedy":
        monkeypatch.setattr(diffusion, "scaffold_tree", lambda proposal, *a, **kw: sampled_tree(proposal.paths()))
        error, message = ValueError, "greedy path is missing"
    elif mutation == "missing_witness":
        original = tiny_engine.generate

        def generate(*a, **kw):
            kw.pop("scaffold_observer", None)
            return original(*a, **kw)

        monkeypatch.setattr(tiny_engine, "generate", generate)
        error, message = RuntimeError, "missing scaffold witnesses"
    else:
        # The nearly uniform tiny random model often has no in-tree residual
        # at all, so disabling continuation there is observationally inert.
        # Keep its real cache/forward path but use the explicit p=.99/q=.51
        # counterexample to exercise a genuine correction-recycling event.
        forward = tiny_engine.target_forward

        def draft_logits(hidden):
            logits = hidden.new_full((*hidden.shape[:-1], 17), -float("inf"))
            logits[..., :2] = hidden.new_tensor([.51, .49]).log() * .6
            return logits

        def target_logits(*a, **kw):
            output = forward(*a, **kw)
            output.logits[..., :2] = output.logits.new_tensor([.99, .01]).log() * .6
            return output

        monkeypatch.setattr(tiny_engine.target.get_output_embeddings(), "forward", draft_logits)
        monkeypatch.setattr(tiny_engine, "target_forward", target_logits)
        assert preflight.probe(tiny_engine, torch.tensor([[1, 3, 5]]), variant("diffusion_scaffold_bv"),
                               tokens=48, seed=42, tv_limit=1e-6)["passed"]
        original = diffusion.verify_scaffold_logits
        calls = 0

        def stop_early(*a, **kw):
            nonlocal calls
            calls += 1
            if mutation != "late_no_recycling" or calls > 3:
                kw["recycle"] = False
            return original(*a, **kw)

        monkeypatch.setattr(diffusion, "verify_scaffold_logits", stop_early)
        error, message = ValueError, "before maximal exit"
    with pytest.raises(error, match=message):
        preflight.probe(tiny_engine, torch.tensor([[1, 3, 5]]), variant("diffusion_scaffold_bv"),
                        tokens=48, seed=42, tv_limit=1e-6)


def test_scaffold_observer_is_noninterfering_and_checks_pre_cap_selection(tiny_engine):
    v, ids = variant("diffusion_scaffold_bv"), torch.tensor([[1, 3, 5]])
    seen = []

    def observe(tree, proposal, logits, nodes, selected, bonus):
        seen.append(preflight.scaffold_contract(tree, proposal, v, logits.argmax(-1).tolist(),
                                               nodes, selected, bonus, logits.shape[-1]))

    for cap in (2, 24):
        expected = tiny_engine.generate(ids, v, cap, [], seed=42)
        seen.clear()
        actual = tiny_engine.generate(ids, v, cap, [], seed=42, scaffold_observer=observe)
        assert actual["generated_token_ids"] == expected["generated_token_ids"]
        assert actual["rounds"] == expected["rounds"]
        assert len(seen) == len(actual["rounds"]) > 0
        assert all(check["maximal_exit_checked"] for check in seen)
