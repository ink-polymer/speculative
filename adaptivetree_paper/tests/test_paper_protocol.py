from __future__ import annotations

import copy
import itertools
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from dflash_specblock.ddtree_builder import DDTreeBuilder, LatencyAwareDDTreeBuilder
from dflash_specblock.paper.common import BASELINES, ROOT, VARIANTS, atomic_json, contract, load_config, load_json
from dflash_specblock.paper.controller import (B128_ABLATION_VARIANT,
    B256_ABLATION_VARIANT,
    CONTEXTUAL_V8_VARIANT, EXTENDED_BUDGETS, FixedBudgetBuilder,
    PaperAdaptiveBuilder, make_builder, make_paper_builder)
from dflash_specblock.paper.adaptive_official import (
    _compact_dynamic_cache_with_tensor)
from dflash_specblock.paper.data import make_row
from dflash_specblock.paper.evaluation import evaluate, paired_bootstrap, summarize, validate_states
from dflash_specblock.paper.runtime import PaperRuntime, commit
from dflash_specblock.paper.triton_cache import BatchedTritonCacheCompactor

torch.set_num_threads(1)


def cfg():
    return load_config(ROOT / "configs/paper_t0_controlled_legacy.json")


def paths(tree):
    result = []
    for n in tree.nodes:
        result.append((() if n.parent < 0 else result[n.parent]) + (n.token_id,))
    return result


def test_adaptive_cache_compaction_reuses_supplied_index_tensor():
    class Cache:
        def __init__(self):
            base = torch.arange(8, dtype=torch.float32).view(1, 1, 8, 1)
            self.key_cache = [base.clone()]
            self.value_cache = [(base + 100).clone()]

        def crop(self, length):
            self.key_cache = [tensor[..., :length, :]
                              for tensor in self.key_cache]
            self.value_cache = [tensor[..., :length, :]
                                for tensor in self.value_cache]

    seen_indices = []

    def compact(cache_tensor, past_length, keep_indices):
        seen_indices.append(keep_indices)
        selected = cache_tensor[..., past_length:, :].index_select(
            -2, keep_indices)
        cache_tensor[..., past_length:past_length + keep_indices.numel(), :].copy_(
            selected)

    cache = Cache()
    indices = torch.tensor([0, 2, 4], dtype=torch.long)
    _compact_dynamic_cache_with_tensor(
        SimpleNamespace(_compact_appended_window=compact), cache, 2, indices)

    assert all(item is indices for item in seen_indices)
    assert cache.key_cache[0].flatten().tolist() == [0, 1, 2, 4, 6]
    assert cache.value_cache[0].flatten().tolist() == [100, 101, 102, 104, 106]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_batched_triton_cache_compaction_is_bit_exact():
    compactor = BatchedTritonCacheCompactor(
        6, torch.device("cuda"), maximum_copy_elements=6 * 2 * 4 * 4,
        element_size=2)
    if not compactor.enabled:
        pytest.skip("Triton is unavailable")
    caches = [
        (torch.arange(2 * 17 * 4, device="cuda", dtype=torch.int32)
         .add(1000 * index).to(torch.bfloat16).view(2, 17, 4).contiguous())
        for index in range(6)
    ]
    originals = [cache.clone() for cache in caches]
    accepted = torch.tensor([0, 2, 8, 13], device="cuda", dtype=torch.long)

    assert compactor.compact(caches, 3, accepted)
    torch.cuda.synchronize()

    for actual, original in zip(caches, originals):
        expected = original[:, 3:, :].index_select(1, accepted)
        assert torch.equal(actual[:, 3:7, :], expected)
    assert compactor.batched_calls == 1


def test_full_method_matches_original_150_rounds_and_resume():
    config = cfg()
    original = LatencyAwareDDTreeBuilder(15, 128, (30, 45, 60, 80, 100, 128), 60)
    current = PaperAdaptiveBuilder(config)
    rng = torch.Generator().manual_seed(3)
    for i in range(150):
        logits = torch.randn(15, 160, generator=rng) * 3
        a, b = original.build_from_logits(logits), current.build_from_logits(logits)
        assert paths(a) == paths(b)
        assert original.last_decision == current.last_decision
        obs = dict(tree_nodes=len(a), draft_ms=.2 + i % 7, verify_ms=.1 + len(a) / 20,
                   accepted_draft_tokens=i % 16)
        original.observe(**obs)
        current.observe(**obs)
        assert original._fixed_ms == current._fixed_ms
        assert original._verify_ms == current._verify_ms
        assert original._acceptance_scale == current._acceptance_scale
        if i in (2, 63, 127):
            restored = PaperAdaptiveBuilder(config)
            restored.load_state_dict(json.loads(json.dumps(current.state_dict())))
            assert restored.state_dict() == current.state_dict()
            current = restored


def test_original_warmup_scoring_and_ewma_equations():
    builder = PaperAdaptiveBuilder(cfg())
    scores = np.linspace(-1, -8, 128)
    seen = []
    for _ in range(6):
        b = builder._select_node_count(scores)
        seen.append(b)
        builder.observe(tree_nodes=b, draft_ms=2, verify_ms=b/10, accepted_draft_tokens=3)
    assert seen == [60, 45, 80, 30, 100, 128]
    mass = np.cumsum(np.exp(scores))
    best = max(seen, key=lambda b: ((1 + min(15, builder._acceptance_scale * mass[b-1]))/(2+b/10), -b))
    assert builder._select_node_count(scores) == best
    old_scale = builder._acceptance_scale
    builder.observe(tree_nodes=best, draft_ms=7, verify_ms=9, accepted_draft_tokens=2)
    assert builder._fixed_ms == pytest.approx(.8*2 + .2*7)
    assert builder._verify_ms[best] == pytest.approx(.8*(best/10) + .2*9)
    assert builder._acceptance_scale == pytest.approx(.8*old_scale + .2*min(2, 2/mass[best-1]))
    builder._decision_count = 64
    expected = min(seen, key=lambda b: (builder._observations[b], b))
    assert builder._select_node_count(scores) == expected


def test_budget_aware_stage_attribution_is_isolated_and_exact():
    legacy = PaperAdaptiveBuilder(cfg(), "no_exploration")
    corrected = PaperAdaptiveBuilder(
        cfg(), "no_exploration", timing_partition="budget_aware"
    )
    scores = np.linspace(-1, -8, 128)
    assert legacy._select_node_count(scores) == corrected._select_node_count(scores) == 60
    stages = dict(tree_nodes=60, draft_ms=2, tree_build_ms=3,
                  tree_compile_ms=5, target_verify_ms=7, commit_ms=11,
                  accepted_draft_tokens=4)
    legacy.observe(tree_nodes=60, draft_ms=5, verify_ms=23,
                   accepted_draft_tokens=4)
    corrected.observe_stages(**stages)
    assert legacy._fixed_ms == 5
    assert legacy._verify_ms[60] == 23
    assert corrected._fixed_ms == 2
    assert corrected._verify_ms[60] == 26
    assert legacy._fixed_ms + legacy._verify_ms[60] == 28
    assert corrected._fixed_ms + corrected._verify_ms[60] == 28
    assert legacy.identity != corrected.identity
    assert "raw_stage_ms" not in legacy.trace[-1]
    assert corrected.trace[-1]["raw_stage_ms"]["tree_build"] == 3
    with pytest.raises(ValueError, match="identity/schema mismatch"):
        corrected.load_state_dict(legacy.state_dict())
    restored = PaperAdaptiveBuilder(
        cfg(), "no_exploration", timing_partition="budget_aware"
    )
    restored.load_state_dict(corrected.state_dict())
    assert restored.state_dict() == corrected.state_dict()
    factory = make_paper_builder(cfg(), B128_ABLATION_VARIANT)
    assert factory.variant == "guarded_raw_prefix"
    assert factory.reserve_greedy_chain is False
    assert factory.timing_partition == "budget_aware"


def test_budget_extension_ablation_builds_all_256_nodes():
    builder = make_paper_builder(cfg(), B256_ABLATION_VARIANT)
    assert builder.budget_candidates == EXTENDED_BUDGETS
    assert builder.tree_budget == 256
    logits = torch.randn(15, 320, generator=torch.Generator().manual_seed(91))
    selected = []
    for _ in EXTENDED_BUDGETS:
        tree = builder.build_from_logits(logits)
        selected.append(len(tree.nodes))
        builder.observe(tree_nodes=len(tree.nodes), draft_ms=1, verify_ms=1,
                        accepted_draft_tokens=1)
    assert set(selected) == set(EXTENDED_BUDGETS)
    assert selected[-1] == 256
    restored = make_paper_builder(cfg(), B256_ABLATION_VARIANT)
    restored.load_state_dict(builder.state_dict())
    assert restored.state_dict() == builder.state_dict()
    with pytest.raises(ValueError, match="identity/schema mismatch"):
        make_paper_builder(cfg(), B128_ABLATION_VARIANT).load_state_dict(
            builder.state_dict())


def test_guarded_raw_prefix_calibrates_robustly_and_resumes():
    builder = make_paper_builder(cfg(), B128_ABLATION_VARIANT)
    scores = np.concatenate((np.full(80, -2.), np.full(48, -20.)))
    latency = {30:4., 45:4.5, 60:5., 80:5.5, 100:9., 128:10.}
    budget = builder._select_node_count(scores)
    assert budget == 128
    builder.observe(tree_nodes=budget, draft_ms=2., verify_ms=latency[budget],
                    accepted_draft_tokens=5,
                    accepted_node_indices=[0, 1, 2, 3, 4, 5])
    assert all(builder._acceptance_observations[value] == 1
               for value in builder.budget_candidates)
    assert [builder._select_node_count(scores) for _ in range(3)] == [128] * 3
    builder._decision_count = builder.reevaluation_interval
    pilot = builder._select_node_count(scores)
    assert pilot < 128
    assert builder._guard_diagnostics["reason"] == "counterfactual_pilot"
    # With materially lower end-to-end latency and essentially the same
    # proposal mass, B80 is an admissible challenger.
    for budget in builder.budget_candidates:
        builder._observations[budget] = builder.minimum_latency_samples
        builder._latency_samples[budget] = [latency[budget]] * 3
    builder._decision_count = builder.reevaluation_interval
    assert builder._select_node_count(scores) == 80
    assert builder._guard_diagnostics["reason"] == "qualified_challenger"

    # A B128 observation reveals the realized target path for every nested
    # prefix budget, so all smaller acceptance calibrators can update without
    # extra target-model calls.
    builder._last_selected_budget = 128
    builder._last_expected_draft_tokens = builder._last_mass_by_budget[128]
    builder._evaluated_last_decision = True
    before = dict(builder._acceptance_observations)
    builder.observe(tree_nodes=128, draft_ms=2., verify_ms=10.,
                    accepted_draft_tokens=2,
                    accepted_node_indices=[0, 1, 85])
    assert all(builder._acceptance_observations[budget] == before[budget] + 1
               for budget in builder.budget_candidates)

    restored = make_paper_builder(cfg(), B128_ABLATION_VARIANT)
    restored.load_state_dict(json.loads(json.dumps(builder.state_dict())))
    assert restored.state_dict() == builder.state_dict()


def test_contextual_v8_uses_free_safe_labels_refreshes_and_falls_back():
    builder = make_paper_builder(cfg(), CONTEXTUAL_V8_VARIANT)
    scores = np.concatenate((np.full(80, -2.), np.full(48, -20.)))
    safe_path = [0, 1, 2, 3, 4, 5, 6]

    # Three B128 rounds warm the safe arm and label every nested prefix without
    # spending an additional target verification.
    for _ in range(builder.contextual_minimum_history):
        budget = builder._select_node_count(scores)
        assert budget == 128
        builder.observe(tree_nodes=budget, draft_ms=2., verify_ms=10.,
                        accepted_draft_tokens=6,
                        accepted_node_indices=safe_path)
    assert builder._safe_required_budgets == [6] * builder.contextual_minimum_history
    assert all(builder._acceptance_observations[value]
               == builder.contextual_minimum_history
               for value in builder.budget_candidates)

    # The tail carries negligible mass and every safe label fits, so B100 is
    # admitted.  Low realized acceptance immediately forces a B128 fallback.
    assert builder._select_node_count(scores) == 100
    assert builder._guard_diagnostics["reason"] == "contextual_high_retention_prefix"
    assert .999 <= builder._guard_diagnostics["mass_retention_ratio"] <= 1.
    builder.observe(tree_nodes=100, draft_ms=2., verify_ms=7.,
                    accepted_draft_tokens=1,
                    accepted_node_indices=[0, 1])
    assert builder._select_node_count(scores) == 128
    assert builder._guard_diagnostics["reason"] == "contextual_acceptance_fallback"

    restored = make_paper_builder(cfg(), CONTEXTUAL_V8_VARIANT)
    restored.load_state_dict(json.loads(json.dumps(builder.state_dict())))
    assert restored.state_dict() == builder.state_dict()
    assert restored.identity != make_paper_builder(
        cfg(), B128_ABLATION_VARIANT).identity


def test_guarded_raw_tree_matches_fixed_ddtree_at_b128():
    from dflash_specblock.paper.adaptive_official import build_with_controller
    builder = make_paper_builder(cfg(), B128_ABLATION_VARIANT)
    generator = torch.Generator().manual_seed(812)
    for _ in range(3):
        logits = torch.randn(15, 256, generator=generator)
        expected = build_with_controller(logits, DDTreeBuilder(15, 128))
        actual = build_with_controller(logits, builder)
        for index in (0, 1, 4):
            assert torch.equal(expected[index], actual[index])
        assert expected[2:4] == actual[2:4]


@pytest.mark.parametrize("seed", range(6))
def test_prefix_closure_mass_identity_and_greedy_tree_semantics(seed):
    # Exhaust every full string and every budget for a finite factorized proposal.
    logits = torch.randn(3, 3, generator=torch.Generator().manual_seed(seed))
    q = logits.double().softmax(-1).numpy()
    full = DDTreeBuilder(3, 39).build_from_logits(logits)
    all_paths = paths(full)
    for budget in range(1, 40):
        kept = all_paths[:budget]
        assert all(not p[:-1] or p[:-1] in kept for p in kept)
        mass = sum(math.prod(q[d,t] for d,t in enumerate(p)) for p in kept)
        expected = 0.
        for seq in itertools.product(range(3), repeat=3):
            length = sum(seq[:d] in kept for d in (1,2,3))
            expected += math.prod(q[d,t] for d,t in enumerate(seq)) * length
            # Greedy verification accepts this exact path until the first absent child.
            cursor, walked = (), []
            for token in seq:
                if cursor + (token,) not in kept:
                    break
                walked.append(token)
                cursor = tuple(walked)
            assert walked == list(seq[:length])
        assert mass == pytest.approx(expected, abs=1e-12)
    # Global top-B scores, with distinct random scores; don't assert torch.topk tie order.
    brute = sorted((math.prod(q[d,t] for d,t in enumerate(p)), p)
                   for depth in (1,2,3) for p in itertools.product(range(3), repeat=depth))
    for b in (1,3,8,20):
        selected_mass = sum(math.prod(q[d,t] for d,t in enumerate(p)) for p in all_paths[:b])
        assert selected_mass == pytest.approx(sum(x[0] for x in brute[-b:]), rel=1e-6)


@pytest.mark.parametrize("variant", VARIANTS)
def test_ablations_are_nonlearned_and_state_isolated(variant):
    b = PaperAdaptiveBuilder(cfg(), variant)
    scores = np.linspace(-1, -9, 128)
    for _ in range(6):
        n = b._select_node_count(scores)
        b.observe(tree_nodes=n, draft_ms=1, verify_ms=n, accepted_draft_tokens=2)
    prior = b.state_dict()
    n = b._select_node_count(scores)
    b.observe(tree_nodes=n, draft_ms=19, verify_ms=33, accepted_draft_tokens=15)
    if variant == "no_acceptance_calibration":
        assert b._acceptance_scale == 1
    if variant == "no_exploration":
        assert b.exploration_interval == 0
    if variant == "frozen_after_warmup":
        assert b._verify_ms == {int(k):v for k,v in prior["verify_ms"].items()}
        assert b._fixed_ms == prior["fixed_ms"]
        assert b._acceptance_scale == prior["acceptance_scale"]
        assert sum(b._observations.values()) == 7
    if variant == "no_latency":
        assert n == 128
    assert not hasattr(b, "parameters") and not hasattr(b, "optimizer")
    restored = PaperAdaptiveBuilder(cfg(), variant)
    restored.load_state_dict(b.state_dict())
    assert restored.state_dict() == b.state_dict()


def test_state_corruption_and_wrong_variant_rejected():
    state = PaperAdaptiveBuilder(cfg()).state_dict()
    with pytest.raises(ValueError):
        PaperAdaptiveBuilder(cfg(), "no_latency").load_state_dict(state)
    state["acceptance_scale"] = float("nan")
    with pytest.raises(ValueError):
        PaperAdaptiveBuilder(cfg()).load_state_dict(state)
    with pytest.raises(ValueError):
        validate_states({}, cfg())


@pytest.mark.parametrize("method", [m for m in BASELINES if m not in ("ar", "dflash")])
def test_fixed_control_never_interpolates_budget(method):
    b = make_builder(cfg(), method)
    assert isinstance(b, FixedBudgetBuilder) and b.manages_budget
    assert b.tree_budget == (60 if method == "ddtree" else int(method.split("_")[1]))


@pytest.mark.parametrize("limit,stops", [(1,set()), (17,set()), (80,set()), (80,{9}), (80,{6})])
def test_original_adaptive_real_engine_mock_greedy_eos_and_cap(limit, stops):
    from test_ddtree_integration import _create_ddtree_engine
    from test_integration_speedup import _baseline_greedy
    engine, target = _create_ddtree_engine(torch.device("cpu"))
    engine.tree_builder = PaperAdaptiveBuilder(cfg())
    ids = torch.tensor([[5]])
    expected, _ = _baseline_greedy(target, ids, limit, torch.device("cpu"))
    expected = commit(expected, limit, stops)
    result = engine.generate(ids, limit, stops)
    assert result.generated_ids[0].tolist() == expected
    assert all(i.tree_nodes in cfg()["budget_candidates"] for i in result.iterations)


class FakeRuntime:
    def __init__(self, config, fail=False):
        self.cfg, self.fail = config, fail
        self.device = torch.device("cpu")
        self.tokenizer = SimpleNamespace(decode=lambda tokens, **_: "answer")
        self.states = {}
        self.calls = 0
    def warmup(self): self.states = {}
    def controller_states(self): return copy.deepcopy(self.states)
    def restore_controllers(self, states): self.states = copy.deepcopy(states)
    def encode(self, prompt): return torch.tensor([[len(prompt)]])
    def encode_messages(self, messages): return torch.tensor([[len(messages)]])
    def generate(self, ids, method, replica=0):
        self.calls += 1
        if method in VARIANTS:
            key = f"{method}:{replica}"
            b = PaperAdaptiveBuilder(self.cfg, method)
            b.load_state_dict(self.states[key])
            n = b._select_node_count(np.linspace(-1, -8, 128))
            b.observe(tree_nodes=n, draft_ms=1, verify_ms=n/10, accepted_draft_tokens=2)
            self.states[key] = b.state_dict()
        return {"tokens": [7 if self.fail and method == "adaptive" else 6],
                "wall_ms": 10 if method == "ar" else 2, "rounds": []}


def test_evaluation_resume_state_chain_full_methods_and_smoke_table(tmp_path):
    config = {**cfg(), "seeds": [17], "eval_repeats": 1, "bootstrap_samples": 10}
    rows = [make_row("gsm8k", {"question": str(i)}, i, "test", "openai/gsm8k", "main") for i in range(3)]
    runtime = FakeRuntime(config)
    directory = tmp_path / "evaluation/17"
    evaluate(runtime, rows, directory, {"smoke": True}, 17)
    calls = runtime.calls
    end = runtime.controller_states()
    evaluate(runtime, rows, directory, {"smoke": True}, 17)
    assert runtime.calls == calls
    assert runtime.controller_states() == end
    summarize(tmp_path, rows, config, smoke=True)
    report = load_json(tmp_path / "tables.json")
    assert not report["publication_eligible"] and report["training"] is False
    assert {r["method"] for r in report["rows"]} == set(BASELINES + tuple(VARIANTS))
    path = directory / "prompts/000000.json"
    bad = load_json(path)
    bad["state_before_sha256"] = "tampered"
    atomic_json(path, bad)
    with pytest.raises(ValueError, match="state chain"):
        evaluate(runtime, rows, directory, {"smoke": True}, 17)


def test_mismatch_saved_but_no_completion_or_official_table(tmp_path):
    config = {**cfg(), "seeds": [17], "eval_repeats": 1}
    rows = [make_row("gsm8k", {"question": "q"}, 0, "test", "openai/gsm8k", "main")]
    with pytest.raises(RuntimeError, match="Greedy mismatch"):
        evaluate(FakeRuntime(config, True), rows, tmp_path, {}, 17)
    assert (tmp_path / "prompts/000000.json").exists()
    assert not (tmp_path / "complete.json").exists()
    with pytest.raises(ValueError, match="Formal matrix"):
        summarize(tmp_path, rows, config)


def test_config_rejects_rl_and_contract_change(tmp_path):
    config = cfg()
    config["train_epochs"] = 2
    atomic_json(tmp_path / "bad.json", config)
    with pytest.raises(ValueError, match="non-RL"):
        load_config(tmp_path / "bad.json")
    contract(tmp_path, {"method":"original"})
    with pytest.raises(ValueError, match="Resume contract"):
        contract(tmp_path, {"method":"rl"})


def test_paired_bootstrap_and_lengths():
    assert paired_bootstrap([2,4], [1,2], 10) == [2,2]
    assert commit([1,2,3], 2, {2}) == [1,2]


def test_all_stage_dispatch_needs_no_training_inputs(tmp_path, monkeypatch):
    from dflash_specblock.paper import controlled_main as entry, runtime, evaluation
    events = []
    config = cfg()
    monkeypatch.setattr(runtime, "PaperRuntime", lambda config: FakeRuntime(config))
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda seed: None)
    monkeypatch.setattr(evaluation, "evaluate", lambda rt, rows, directory, metadata, seed:
                        events.append(("evaluate", seed, rows)))
    monkeypatch.setattr(evaluation, "summarize", lambda *args, **kwargs: events.append(("summarize",)))
    entry.execute_gpu(SimpleNamespace(stage="all", run_dir=tmp_path, smoke_count=0),
                      config, {}, {}, ["test-only-row"])
    assert events == [("evaluate", seed, ["test-only-row"]) for seed in config["seeds"]] + [("summarize",)]


def test_every_method_runs_through_actual_cpu_runtime():
    from test_ddtree_integration import _create_ddtree_engine
    from dflash_specblock.vanilla_engine import VanillaDFlashEngine
    engine, target = _create_ddtree_engine(torch.device("cpu"))
    runtime = object.__new__(PaperRuntime)
    runtime.cfg = {**cfg(), "max_new_tokens": 38}
    runtime.target, runtime.adapter, runtime.verifier = target, engine.adapter, engine.verifier
    runtime.device, runtime.engines, runtime.stops = torch.device("cpu"), {}, set()
    runtime.vanilla = VanillaDFlashEngine(target, engine.adapter, runtime.device, torch.float32)
    ids = torch.tensor([[5]])
    reference = runtime.generate(ids, "ar")["tokens"]
    for method in (*BASELINES, *VARIANTS):
        result = runtime.generate(ids, method)
        assert result["tokens"] == reference
        assert result["wall_ms"] > 0
        if method in VARIANTS:
            assert result["decisions"]
    states = runtime.controller_states()
    runtime.reset_controllers()
    runtime.restore_controllers(states)
    assert runtime.controller_states() == states


def test_legacy_cli_config_supports_original_adaptive():
    from dflash_specblock.config import ExperimentConfig
    config = ExperimentConfig.from_json(ROOT / "configs/qwen3_4b_cuda_ddtree_adaptive.json")
    assert config.tree_mode == "ddtree_adaptive" and config.tree_budget == 128
    assert config.rank_checkpoint is None
    config.ddtree_reserve_greedy_chain = True
    with pytest.raises(ValueError):
        config.validate()
