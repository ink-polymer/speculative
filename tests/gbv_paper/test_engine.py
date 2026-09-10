from dataclasses import replace
from itertools import product
from types import SimpleNamespace

import pytest
import torch

from gbv_experiments import engine as engine_module
from gbv_experiments.common import ROOT, digest, file_hash
from gbv_experiments.config import SHARED_SUFFIX_METHODS as ROOT_MARGINAL_METHODS, Variant
from gbv_experiments.engine import Engine
from gbv_experiments.sampling import (
    tree_verify_ancestral_batched,
    tree_verify_internal_ancestral_batched,
)
from gbv_experiments.preflight import (
    _same_tree_runtime_witness,
    classify_greedy_mismatch,
)
from gbv_experiments.tree import (adaptive_path_proposal, adaptive_prefix_proposal,
                                  block_aligned_spine_tree,
                                  budgeted_prefix_proposal, compact_cache,
                                  probability_tree, sampled_tree)


def test_tree_merge_keeps_candidate_multiplicity():
    paths = torch.tensor([[1, 2, 3], [1, 2, 3], [1, 4, 5]])
    tree = sampled_tree(paths)
    assert len(tree.tokens) == 5
    assert tree.path_nodes[0] == tree.path_nodes[1]
    visible = tree.visibility()
    assert not visible[tree.path_nodes[2][-1], tree.path_nodes[0][-1]]
    assert len(sampled_tree(paths, False).tokens) == 9


@pytest.mark.parametrize(
    "method",
    ["ddtree_fused", "ddtree_fused_parallel", "ddtree_fused_scan",
     "ddtree_sparse_exit_fused_scan",
     "ddtree_same_draw_fused",
     "ddtree_direct_logits_fused_scan",
     "ddtree_lazy_target", "ddtree_lazy_target_deferred_leaf",
     "ddtree_lazy_target_prefetch1",
     "ddtree_lazy_target_prefetch2",
     "ddtree_lazy_target_aligned32",
     "ddtree_lazy_target_aligned40",
     "ddtree_lazy_projection", "ddtree_lazy_softmax_fused_scan",
     "ddtree_lazy_projection_fused_scan"],
)
def test_fused_tree_methods_are_valid_probability_tree_variants(method):
    variant = Variant(
        name=method, method=method, paths=1, length=15,
        temperature=1.0, draft_temperature=1.0, tree_budget=45,
        probability_dtype="float64",
    )
    variant.validate()


@pytest.mark.parametrize("method,attribute", [
    ("ddtree_fused", "tree_verify_ancestral_fused"),
    ("ddtree_fused_parallel", "tree_verify_ancestral_fused_parallel"),
    ("ddtree_fused_scan", "tree_verify_ancestral_fused_scan"),
    ("ddtree_sparse_exit_fused_scan",
     "tree_verify_ancestral_sparse_exit_fused_scan"),
    ("ddtree_same_draw_fused", "tree_verify_ancestral_same_draw_fused"),
])
def test_fused_tree_methods_dispatch_in_generation(
        tiny_engine, monkeypatch, method, attribute):
    monkeypatch.setattr(engine_module, attribute, tree_verify_ancestral_batched)
    observed = []
    ids = torch.tensor([[1, 4, 2, 6]])
    reference = tiny_engine.generate(
        ids, Variant(name="target", method="target", paths=1, length=3,
                     temperature=0),
        12, [], seed=19,
    )
    result = tiny_engine.generate(
        ids, Variant(name=method, method=method, paths=1, length=3,
                     temperature=0, tree_budget=12),
        12, [], seed=19, verifier_observer=observed.append,
    )
    assert result["generated_token_ids"] == reference["generated_token_ids"]
    assert len(result["rounds"]) > 1
    assert len(observed) == 1
    event = observed[0]
    assert {key:event[key] for key in (
        "module", "qualname", "callable", "source_file", "source_sha256",
    )} == {
        "module":"gbv_experiments.sampling",
        "qualname":"tree_verify_ancestral_batched",
        "callable":"gbv_experiments.sampling.tree_verify_ancestral_batched",
        "source_file":"src/gbv_experiments/sampling.py",
        "source_sha256":file_hash(ROOT / "src/gbv_experiments/sampling.py"),
    }
    assert event["observed_call_index"] == 1
    assert event["input_sha256"] == digest(event["input"])
    assert event["output_sha256"] == digest(event["output"])
    assert event["input"]["probability_shape"] == [13, 17]
    assert event["input"]["probability_dtype"] == "torch.float64"
    assert event["input"]["generator_before"]["initial_seed"] == 19
    assert event["input"]["generator_before"]["device"] == "cpu"
    assert len(event["input"]["generator_before"]["state_sha256"]) == 64
    assert event["input"]["validate"] is False


@pytest.mark.parametrize("method,attribute", [
    ("ddtree_lazy_softmax_fused_scan",
     "tree_verify_ancestral_lazy_softmax_fused_scan"),
    ("ddtree_lazy_projection_fused_scan",
     "tree_verify_ancestral_lazy_projection_fused_scan"),
])
def test_lazy_fused_tree_methods_dispatch_and_record_saved_rows(
        tiny_engine, monkeypatch, method, attribute):
    calls = []

    def fake_verifier(parents, tokens, *args, **kwargs):
        calls.append((list(parents), list(tokens), kwargs))
        internal = len(set(parents[1:]))
        return [], [], 0, {
            "internal_projected_rows": internal,
            "leaf_projected_rows": 0,
            "projected_rows": internal,
            "total_tree_rows": len(parents),
            "lm_head_rows": internal,
            "probability_rows": internal,
        }

    monkeypatch.setattr(engine_module, attribute, fake_verifier)
    result = tiny_engine.generate(
        torch.tensor([[1, 4, 2, 6]]),
        Variant(
            name=method, method=method, paths=1, length=3,
            temperature=1.0, draft_temperature=1.0,
            probability_dtype="float64", tree_budget=12,
        ),
        6, [], seed=19,
    )
    assert calls
    assert all(call[2]["validate"] is False for call in calls)
    assert all(
        row["posterior_probability_rows"]
        <= row["full_vocabulary_projection_rows"]
        for row in result["rounds"]
    )
    assert any(
        row["posterior_probability_rows"]
        < row["full_vocabulary_projection_rows"]
        for row in result["rounds"]
    )


def test_internal_only_tree_walk_defers_exactly_the_reached_leaf():
    parents = [-1, 0, 0, 1, 1]
    tokens = [1, 2, 3, 4]
    # Internal rows are original nodes 0 and 1.  Root chooses node 1 and
    # node 1 chooses leaf node 4; no other leaf posterior is needed.
    internal_p = torch.zeros((2, 7), dtype=torch.float64)
    internal_p[0, 1] = 1
    internal_p[1, 4] = 1
    nodes, output_tokens, bonus, leaf, stats = (
        tree_verify_internal_ancestral_batched(
            parents, tokens, internal_p,
            torch.Generator().manual_seed(1),
        )
    )
    assert (nodes, output_tokens, bonus, leaf) == ([1, 4], [1, 4], -1, 4)
    assert stats == {
        "internal_probability_rows":2,
        "leaf_probability_rows":1,
        "total_tree_rows":5,
    }

    # An internal-node exit has its complete bonus already and needs no
    # deferred Target call.
    internal_p[1].zero_()
    internal_p[1, 6] = 1
    nodes, output_tokens, bonus, leaf, stats = (
        tree_verify_internal_ancestral_batched(
            parents, tokens, internal_p,
            torch.Generator().manual_seed(1),
        )
    )
    assert (nodes, output_tokens, bonus, leaf) == ([1], [1], 6, -1)
    assert stats["leaf_probability_rows"] == 0


def test_lazy_target_verifies_only_internal_rows_and_matches_full_rows(
        tiny_engine):
    full_captures = []
    lazy_captures = []

    def capture_full(parents, tokens, p):
        full_captures.append((list(parents), list(tokens), p.detach().clone()))

    def capture_lazy(parents, tokens, internal_nodes, p, leaf, leaf_p):
        lazy_captures.append((
            list(parents), list(tokens), list(internal_nodes),
            p.detach().clone(), leaf,
            None if leaf_p is None else leaf_p.detach().clone(),
        ))

    ids = torch.tensor([[1, 4, 2, 6]])
    common = dict(
        paths=1, length=3, temperature=1.0, draft_temperature=1.0,
        probability_dtype="float64", tree_budget=12,
    )
    tiny_engine.generate(
        ids, Variant(name="ddtree", method="ddtree", **common),
        6, [], seed=19, tree_observer=capture_full,
    )
    result = tiny_engine.generate(
        ids, Variant(
            name="ddtree_lazy_target", method="ddtree_lazy_target", **common,
        ),
        6, [], seed=19, lazy_target_observer=capture_lazy,
    )
    full_parents, full_tokens, full_p = full_captures[0]
    (lazy_parents, lazy_tokens, internal_nodes, internal_p,
     leaf, leaf_p) = lazy_captures[0]
    assert (lazy_parents, lazy_tokens) == (full_parents, full_tokens)
    torch.testing.assert_close(
        internal_p,
        full_p.index_select(0, torch.tensor(internal_nodes)),
        rtol=0,
        atol=1e-7,
    )
    if leaf >= 0:
        torch.testing.assert_close(
            leaf_p, full_p[leaf], rtol=0, atol=1e-7,
        )
    assert all(
        row["target_verified_rows"] <= row["full_target_tree_rows"]
        for row in result["rounds"]
    )
    assert any(
        row["target_verified_rows"] < row["full_target_tree_rows"]
        for row in result["rounds"]
    )


@pytest.mark.parametrize("method", [
    "ddtree_lazy_target_deferred_leaf",
    "ddtree_lazy_target_aligned32",
    "ddtree_lazy_target_aligned40",
])
def test_deferred_leaf_uses_exactly_one_target_tree_forward_per_round(
        tiny_engine, method):
    result = tiny_engine.generate(
        torch.tensor([[1, 4, 2, 6]]),
        Variant(
            name="deferred", method=method,
            paths=1, length=3, temperature=1.0, draft_temperature=1.0,
            probability_dtype="float64", tree_budget=12,
        ),
        24, [], seed=19,
    )
    assert result["generated_tokens"] == 24
    assert result["target_forward_calls"] == 1 + len(result["rounds"])
    assert any(
        row["deferred_leaf_stop"] or row["prefetched_leaf_hit"]
        for row in result["rounds"]
    )
    assert all(
        not row["lazy_leaf_target_forward"] for row in result["rounds"]
    )


@pytest.mark.parametrize(
    "method,attribute",
    [
        ("ddtree", "tree_verify_ancestral_batched"),
        ("ddtree_fused_scan", "tree_verify_ancestral_fused_scan"),
    ],
)
def test_verifier_observer_runs_only_after_success(
        tiny_engine, monkeypatch, method, attribute):
    observed = []

    def failing_verifier(*args, **kwargs):
        raise RuntimeError("verifier sentinel")

    monkeypatch.setattr(
        engine_module, attribute, failing_verifier,
    )
    ids = torch.tensor([[1, 4, 2, 6]])
    variant = Variant(
        name=method, method=method, paths=1, length=3,
        temperature=0, tree_budget=12,
    )
    with pytest.raises(RuntimeError, match="verifier sentinel"):
        tiny_engine.generate(
            ids, variant, 12, [], seed=19,
            verifier_observer=observed.append,
        )
    assert observed == []


def test_no_verifier_observer_does_not_inspect_callable(tiny_engine, monkeypatch):
    def forbidden_identity(*args, **kwargs):
        raise AssertionError("normal timing path inspected verifier identity")

    monkeypatch.setattr(
        Engine, "_runtime_verifier_identity", staticmethod(forbidden_identity),
    )
    result = tiny_engine.generate(
        torch.tensor([[1, 4, 2, 6]]),
        Variant(
            name="ddtree", method="ddtree", paths=1, length=3,
            temperature=0, tree_budget=12,
        ),
        12, [], seed=19,
    )
    assert result["generated_tokens"] == 12


@pytest.mark.parametrize(
    "candidate_callable,repetitions,passed",
    [
        (engine_module.tree_verify_ancestral_fused_scan, 1, True),
        (tree_verify_ancestral_batched, 1, False),
        (engine_module.tree_verify_ancestral_fused_scan, 2, False),
    ],
)
def test_same_tree_preflight_binds_exactly_one_actual_verifier(
        candidate_callable, repetitions, passed):
    class Tokenizer:
        @staticmethod
        def apply_chat_template(*args, **kwargs):
            return torch.tensor([[1, 2]])

    class DiagnosticEngine:
        target = SimpleNamespace(config=SimpleNamespace(vocab_size=17))

        @staticmethod
        def generate(ids, variant, maximum, stops, **kwargs):
            del ids, maximum, stops
            parents = [-1, 0, 0, 0]
            tree_tokens = [1, 2, 3]
            probability_rows = torch.full(
                (4, 17), 1 / 17, dtype=torch.float64,
            )
            kwargs["tree_observer"](parents, tree_tokens, probability_rows)
            verifier = (
                tree_verify_ancestral_batched
                if variant.method == "ddtree" else candidate_callable
            )
            event = Engine._runtime_verifier_identity(
                verifier, parents, tree_tokens, probability_rows,
                [1], [tree_tokens[0]], 4,
                generator_before=Engine._runtime_generator_identity(
                    torch.Generator(device="cpu").manual_seed(
                        kwargs["seed"]
                    )
                ),
                validate=False,
            )
            for _ in range(repetitions):
                kwargs["verifier_observer"](event)

    variants = [
        Variant(
            name="ddtree_t1p0", method="ddtree", paths=1, length=2,
            temperature=1.0, draft_temperature=1.0,
            probability_dtype="float64", tree_budget=3,
        ),
        Variant(
            name="tree_block_verification_t1p0", method="ddtree_fused_scan",
            paths=1, length=2, temperature=1.0, draft_temperature=1.0,
            probability_dtype="float64", tree_budget=3,
        ),
    ]
    witness = _same_tree_runtime_witness(
        DiagnosticEngine(), Tokenizer(), {"enable_thinking":False},
        variants, [], "cpu",
    )
    assert witness["passed"] is passed
    if repetitions == 1:
        assert witness["verifier_routes_correct"] is passed
        assert witness["verifier_generator_inputs_equal"]
    else:
        assert witness["verifier_observer_calls"] == {
            "ddtree_t1p0":2, "tree_block_verification_t1p0":2,
        }


def test_same_tree_preflight_rejects_actual_candidate_misroute(
        tiny_engine, monkeypatch):
    class Tokenizer:
        @staticmethod
        def apply_chat_template(*args, **kwargs):
            return torch.tensor([[1, 2]])

    monkeypatch.setattr(
        engine_module, "tree_verify_ancestral_fused_scan",
        tree_verify_ancestral_batched,
    )
    variants = [
        Variant(
            name="ddtree_t1p0", method="ddtree", paths=1, length=2,
            temperature=1.0, draft_temperature=1.0,
            probability_dtype="float64", tree_budget=3,
        ),
        Variant(
            name="tree_block_verification_t1p0", method="ddtree_fused_scan",
            paths=1, length=2, temperature=1.0, draft_temperature=1.0,
            probability_dtype="float64", tree_budget=3,
        ),
    ]
    witness = _same_tree_runtime_witness(
        tiny_engine, Tokenizer(), {"enable_thinking":False},
        variants, [], "cpu",
    )
    assert not witness["passed"]
    assert not witness["verifier_routes_correct"]


def test_budgeted_prefix_proposal_caps_final_verification_tree():
    q = torch.tensor([
        [0.45, 0.30, 0.15, 0.10],
        [0.40, 0.35, 0.15, 0.10],
        [0.38, 0.32, 0.20, 0.10],
    ], dtype=torch.float64)
    proposal = budgeted_prefix_proposal(q, budget=7)
    tree = sampled_tree(proposal.paths)
    assert len(tree.tokens) <= 7
    assert proposal.paths.shape[1] == 3
    assert proposal.leaf_probabilities.sum() == pytest.approx(1.0)
    for path in proposal.paths:
        rows = proposal.conditional_rows(path)
        torch.testing.assert_close(rows.sum(-1), torch.ones(3, dtype=torch.float64))
        assert (rows.gather(1, path[:, None]) > 0).all()


@torch.inference_mode()
@pytest.mark.parametrize("share", [True, False])
@pytest.mark.parametrize("shared_suffix", [False, True])
def test_tree_logits_and_compacted_cache_equal_sequential(tiny_engine, share, shared_suffix):
    engine = tiny_engine
    prefix = torch.tensor([[1, 2, 3]])
    paths = torch.tensor([[4, 5, 6], [7, 5, 6], [8, 5, 6]] if shared_suffix
                         else [[4, 5, 6], [4, 7, 8], [4, 5, 6]])
    tree = sampled_tree(paths, share)
    cache = engine.cache_factory()
    engine.target_forward(prefix, cache, hidden=False)
    ids = torch.tensor([[9] + tree.tokens])
    output = engine.target_forward(ids, cache, hidden=True,
              positions=(torch.tensor(tree.depths) + 3)[None], mask=tree.mask(3, torch.float32, "cpu"))
    for path, nodes in zip(paths, tree.path_nodes):
        sequence = torch.cat((prefix, torch.tensor([[9]]), path[None]), dim=1)
        reference = engine.target(sequence, output_hidden_states=True)
        torch.testing.assert_close(output.logits[:, [0] + nodes], reference.logits[:, 3:], atol=2e-6, rtol=1e-5)
        torch.testing.assert_close(engine.features(output.hidden_states, torch.tensor([0] + nodes)),
                                   engine.features(reference.hidden_states)[:, 3:], atol=2e-6, rtol=1e-5)
    keep = [0] + tree.path_nodes[1][:2]
    compact_cache(cache, 3, keep, "cpu")
    actual = engine.target_forward(torch.tensor([[11]]), cache, hidden=False)
    reference = engine.target(torch.tensor([[1, 2, 3, 9] + paths[1, :2].tolist() + [11]]))
    torch.testing.assert_close(actual.logits[0, -1], reference.logits[0, -1], atol=2e-6, rtol=1e-5)


@pytest.mark.parametrize("method,paths", [("dflash", 1), ("token", 1), ("bv", 1),
                                           ("gbv", 3), ("tree_gbv", 3),
                                           ("tree_gbv_full", 3),
                                           ("tree_gbv_full_dense_ref", 3),
                                           ("tree_gbv_full_sparse_ref", 3),
                                           ("tree_gbv_full_sparse_lazy", 3),
                                           ("tree_gbv_recycle", 3),
                                           ("tree_gbv_prefix_recycle", 3),
                                           ("tree_gbv_budgeted_prefix_recycle", 3),
                                           ("tree_gbv_budgeted_prefix_recycle_sparse_lazy", 3),
                                           ("tree_gbv_budgeted_prefix_recycle_sparse_lazy_prefetch", 3),
                                           ("tree_gbv_budgeted_prefix_recycle_host", 3),
                                           ("tree_gbv_prefix_recycle_packed", 3),
                                           ("tree_gbv_packed", 3),
                                           ("tree_gbv_prefix", 3),
                                           ("ddtree_terminal_block", 1),
                                           ("ddtree_terminal_serial", 1),
                                           ("ddtree_terminal_dense", 1),
                                           ("ddtree_lazy_projection", 1),
                                           ("ddtree", 1)] + [(name, 3) for name in sorted(ROOT_MARGINAL_METHODS)])
@pytest.mark.parametrize("options", [{}, {"reuse_draft_cache": False}, {"share_prefixes": False}, {"draft_attention": "causal"}, {"condition_features": "zero"}])
def test_greedy_matches_target_across_cache_attention_and_verifiers(tiny_engine, method, paths, options):
    ids = torch.tensor([[1, 4, 2, 6]])
    base = Variant(name="ar", method="target", paths=1, length=3, temperature=0)
    expected = tiny_engine.generate(ids, base, 19, [], seed=19)
    v = replace(base, name=method, method=method, paths=paths, tree_budget=12, **options)
    result = tiny_engine.generate(ids, v, 19, [], seed=19)
    assert result["generated_token_ids"] == expected["generated_token_ids"]
    assert result["generated_tokens"] == 19
    assert result["e2e_ms"] == pytest.approx(result["prefill_ms"] + result["decode_ms"])
    assert result["draft_forward_calls"] == len(result["rounds"])


def test_ratio_transport_requires_loaded_architecture(tiny_engine):
    ids = torch.tensor([[1, 4, 2, 6]])
    variant = Variant(
        name="ratio_transport", method="tree_gbv_ratio_transport_recycle",
        paths=3, length=3, temperature=0, tree_budget=12,
    )
    with pytest.raises(RuntimeError, match="requires a loaded ratio head"):
        tiny_engine.generate(ids, variant, 8, [], seed=19)


def test_greedy_audit_compares_tree_rows_with_sequential_forward(tiny_engine):
    ids = torch.tensor([[1, 4, 2, 6]])
    variant = Variant(name="gbv", method="gbv", paths=3, length=3, temperature=0)
    result = tiny_engine.generate(ids, variant, 12, [], seed=19, audit_greedy=True)
    assert result["greedy_audit"]
    assert all(all(round_["argmax_equal"]) for round_ in result["greedy_audit"])
    assert all(len(round_["tree_token_ids"]) == len(round_["sequential_token_ids"])
               for round_ in result["greedy_audit"])


def test_target_greedy_audit_records_decision_margins(tiny_engine):
    ids = torch.tensor([[1, 4, 2, 6]])
    variant = Variant(name="ar", method="target", paths=1, temperature=0)
    result = tiny_engine.generate(ids, variant, 12, [], seed=19, audit_greedy=True)
    assert len(result["target_greedy_trace"]) == len(result["generated_token_ids"])
    assert [point["generated_index"] for point in result["target_greedy_trace"]] == list(range(12))
    assert [point["token_id"] for point in result["target_greedy_trace"]] == result["generated_token_ids"]
    assert all(point["top1_margin"] >= 0 for point in result["target_greedy_trace"])


def test_greedy_trace_uses_argmax_tie_break_instead_of_topk_order():
    point = Engine.greedy_trace_point(torch.tensor([1.0, 2.0, 2.0]), generated_index=4)
    assert point == {"generated_index": 4, "token_id": 1,
                     "runner_up_token_id": 2, "top1_margin": 0.0}


def numerical_mismatch_fixture():
    return {
        "first_index": 7,
        "reference_token_id": 11,
        "result_token_id": 12,
        "reference_repeat_equal": True,
        "result_audit_repeat_equal": True,
        "greedy_logit_audit": [{
            "generated_start_index": 6,
            "tree_token_ids": [20, 12],
            "sequential_token_ids": [20, 11],
            "max_absolute_logit_error": [0.1, 0.25],
            "tree_top1_margin": [0.1, 0.125],
            "sequential_top1_margin": [0.1, 0.125],
        }],
    }, [{"generated_index": 7, "token_id": 11, "runner_up_token_id": 12,
         "top1_margin": 0.125}]


def test_bounded_bf16_near_tie_is_reported_without_hiding_exact_mismatch():
    mismatch, trace = numerical_mismatch_fixture()
    diagnosis = classify_greedy_mismatch(mismatch, trace, dtype="bfloat16")
    assert diagnosis["gate_passed"]
    assert diagnosis["classification"] == "bounded_bf16_near_tie"


@pytest.mark.parametrize("mutation,failed_condition", [
    ({"reference_repeat_equal": False}, "reference_repeat_stable"),
    ({"result_audit_repeat_equal": False}, "candidate_repeat_stable"),
    ({"dtype": "float32"}, "bf16_checkpoint"),
    ({"max_error": 0.75}, "bounded_logit_error"),
    ({"reference_margin": 0.75}, "reference_is_near_tie"),
    ({"tree_margin": 0.75}, "tree_is_near_tie"),
])
def test_unexplained_greedy_mismatch_fails_closed(mutation, failed_condition):
    mismatch, trace = numerical_mismatch_fixture()
    dtype = mutation.get("dtype", "bfloat16")
    mismatch.update({key: value for key, value in mutation.items()
                     if key in ("reference_repeat_equal", "result_audit_repeat_equal")})
    if "max_error" in mutation:
        mismatch["greedy_logit_audit"][0]["max_absolute_logit_error"][1] = mutation["max_error"]
    if "tree_margin" in mutation:
        mismatch["greedy_logit_audit"][0]["tree_top1_margin"][1] = mutation["tree_margin"]
    if "reference_margin" in mutation:
        trace[0]["top1_margin"] = mutation["reference_margin"]
    diagnosis = classify_greedy_mismatch(mismatch, trace, dtype=dtype)
    assert not diagnosis["gate_passed"]
    assert failed_condition in diagnosis["failed_conditions"]


@pytest.mark.parametrize("method", ["gbv"] + sorted(ROOT_MARGINAL_METHODS))
def test_stochastic_repeat_cache_and_prefix_equivalence(tiny_engine, method):
    ids = torch.tensor([[1, 3, 5]])
    v = Variant(name=method, method=method, length=3, paths=3)
    expected = tiny_engine.generate(ids, v, 23, [], seed=4)["generated_token_ids"]
    for modified in (v, replace(v, reuse_draft_cache=False), replace(v, share_prefixes=False)):
        assert tiny_engine.generate(ids, modified, 23, [], seed=4)["generated_token_ids"] == expected


def test_dflash_uses_greedy_draft_at_nonzero_target_temperature(tiny_engine, monkeypatch):
    import gbv_experiments.engine as module
    from gbv_experiments.sampling import matching_verify
    observed = []
    handle = tiny_engine.draft.register_forward_hook(
        lambda model, args, output: observed.append(tiny_engine.target.get_output_embeddings()(output[:, 1:4]).argmax(-1)[0]))
    calls = []
    def verify(path, p, generator):
        assert torch.equal(path, observed[-1])
        calls.append(path.tolist())
        return matching_verify(path, p, generator)
    monkeypatch.setattr(module, "matching_verify", verify)
    try:
        v = Variant(name="dflash", method="dflash", paths=1, length=3, temperature=1.)
        tiny_engine.generate(torch.tensor([[1, 3, 5]]), v, 17, [], seed=4)
    finally:
        handle.remove()
    assert calls and len(calls) == len(observed)


@pytest.mark.parametrize("max_tokens", [1, 2, 3, 7])
@pytest.mark.parametrize("method", ["gbv"] + sorted(ROOT_MARGINAL_METHODS))
def test_length_and_eos(tiny_engine, max_tokens, method):
    ids = torch.tensor([[1, 3, 5]])
    v = Variant(name=method, method=method, length=3, paths=3)
    result = tiny_engine.generate(ids, v, max_tokens, [], seed=4)
    assert result["generated_tokens"] == max_tokens
    eos = result["generated_token_ids"][0]
    stopped = tiny_engine.generate(ids, v, max_tokens, [eos], seed=4)
    assert stopped["generated_token_ids"] == [eos]
    assert stopped["draft_forward_calls"] == 0


@pytest.mark.parametrize("method", sorted(ROOT_MARGINAL_METHODS))
@pytest.mark.parametrize("length", [1, 3])
def test_root_marginal_full_root_coverage_multiround_eos_and_budget(tiny_engine, method, length):
    # Include every vocabulary root so the inside path and its cache handling
    # are exercised even with an untrained, almost-uniform tiny checkpoint.
    vocab = tiny_engine.target.config.vocab_size
    v = Variant(name=method, method=method, paths=vocab, length=length,
                tree_budget=vocab * length)
    ids = torch.tensor([[1, 3, 5]])
    result = tiny_engine.generate(ids, v, 23, [], seed=9)
    tokens = result["generated_token_ids"]
    assert len(tokens) == 23 and len(result["rounds"]) > 1
    assert result["target_forward_calls"] == len(result["rounds"]) + 1
    assert result["draft_forward_calls"] == len(result["rounds"])
    for round_ in result["rounds"]:
        assert round_["tree_nodes"] == vocab * length
        assert round_["verify_tokens"] == vocab * length + 1
        assert round_["proposed_tokens"] == vocab * length
    eos = next(token for token in tokens if token != tokens[0])
    stopped = tiny_engine.generate(ids, v, 23, [eos], seed=9)
    assert stopped["generated_token_ids"] == tokens[:tokens.index(eos) + 1]
    assert stopped["draft_forward_calls"] > 0


def test_probability_tree_has_top_prefix_masses():
    q = torch.tensor([[.63, .37], [.71, .29], [.57, .43]])
    tree = probability_tree(q, 8)
    masses = [1.]
    for node in range(1, len(tree.parents)):
        masses.append(masses[tree.parents[node]] * float(q[tree.depths[node]-1, tree.tokens[node-1]]))
    expected = []
    for length in (1, 2, 3):
        for path in product(range(2), repeat=length):
            expected.append(float(torch.prod(torch.stack([q[i, x] for i, x in enumerate(path)]))))
    assert sorted(masses[1:], reverse=True) == pytest.approx(sorted(expected, reverse=True)[:8])


def test_probability_tree_depth_reward_allocates_block_utility_to_depth():
    q = torch.tensor([[.6, .4], [.6, .4], [.6, .4]])
    baseline = probability_tree(q, 3)
    rewarded = probability_tree(q, 3, depth_reward=1.0)
    assert baseline.depths == [0, 1, 1, 2]
    assert rewarded.depths == [0, 1, 2, 3]


def test_probability_tree_adaptive_budget_prunes_confident_blocks():
    q = torch.tensor([[0.9, 0.1], [0.8, 0.2], [0.9, 0.1]])
    full = probability_tree(q, 5)
    pruned = probability_tree(
        q, 5, adaptive_min_budget=2, confidence_threshold=0.7,
    )
    assert len(full.tokens) == 5
    assert len(pruned.tokens) == 2


def test_block_aligned_spine_tree_keeps_full_paths_within_budget():
    q = torch.tensor([
        [.6, .4], [.7, .3], [.55, .45], [.8, .2],
    ], dtype=torch.float64)
    tree = block_aligned_spine_tree(q, budget=10, max_spines=4)
    assert len(tree.tokens) <= 10
    assert max(tree.depths) == 4
    assert all(len(nodes) == 4 for nodes in tree.path_nodes)


def test_adaptive_path_proposal_has_top_full_paths_and_consistent_conditionals():
    q = torch.tensor([[.63, .37], [.71, .29], [.57, .43]], dtype=torch.float64)
    proposal = adaptive_path_proposal(q, 5)
    all_paths = list(product(range(2), repeat=3))
    ranked = sorted(all_paths, key=lambda path: (
        float(torch.prod(torch.stack([q[i, token] for i, token in enumerate(path)]))), path
    ), reverse=True)[:5]
    assert {tuple(path) for path in proposal.paths.tolist()} == set(ranked)
    torch.testing.assert_close(proposal.leaf_probabilities.sum(), torch.tensor(1., dtype=q.dtype))
    for path, leaf_probability, token_probabilities in zip(
            proposal.paths, proposal.leaf_probabilities, proposal.token_probabilities):
        rows = proposal.conditional_rows(path)
        sparse_tokens, sparse_probabilities = proposal.conditional_sparse(path)
        sparse_rows = torch.zeros_like(rows).scatter_add_(
            1, sparse_tokens, sparse_probabilities
        )
        torch.testing.assert_close(sparse_rows, rows)
        torch.testing.assert_close(rows.sum(-1), torch.ones(3, dtype=q.dtype))
        gathered = rows.gather(1, path[:, None])[:, 0]
        torch.testing.assert_close(gathered, token_probabilities)
        torch.testing.assert_close(gathered.prod(), leaf_probability)


def test_adaptive_path_proposal_sampling_preserves_duplicate_candidates():
    q = torch.tensor([[.8, .2], [.6, .4]], dtype=torch.float64)
    proposal = adaptive_path_proposal(q, 3)
    paths, token_probabilities = proposal.sample(
        7, torch.Generator().manual_seed(3)
    )
    assert paths.shape == token_probabilities.shape == (7, 2)
    assert len(sampled_tree(paths).path_nodes) == 7
    for path, probabilities in zip(paths, token_probabilities):
        torch.testing.assert_close(
            probabilities,
            proposal.conditional_rows(path).gather(1, path[:, None])[:, 0],
        )


def test_adaptive_prefix_proposal_is_normalized_and_root_diverse():
    q = torch.tensor([[.63, .37], [.71, .29], [.57, .43]], dtype=torch.float64)
    proposal = adaptive_prefix_proposal(q, 8)
    assert proposal.paths.shape[1] == 3
    assert set(proposal.paths[:, 0].tolist()) == {0, 1}
    torch.testing.assert_close(proposal.leaf_probabilities.sum(), torch.tensor(1., dtype=q.dtype))
    for path, probability in zip(proposal.paths, proposal.leaf_probabilities):
        rows = proposal.conditional_rows(path)
        torch.testing.assert_close(rows.sum(-1), torch.ones(3, dtype=q.dtype))
        torch.testing.assert_close(rows.gather(1, path[:, None])[:, 0].prod(), probability)


def test_selected_leaf_distribution_sparse_conditionals_match_dense():
    q = torch.tensor([[.63, .37], [.71, .29], [.57, .43]], dtype=torch.float64)
    proposal = adaptive_path_proposal(q, 5)
    selected = torch.tensor([.04, .13, .21, .27, .35], dtype=q.dtype)
    for path in proposal.paths:
        dense = proposal.conditional_rows(path, selected)
        tokens, probabilities = proposal.conditional_sparse(path, selected)
        reconstructed = torch.zeros_like(dense).scatter_add_(1, tokens, probabilities)
        torch.testing.assert_close(reconstructed, dense, rtol=1e-12, atol=1e-13)
