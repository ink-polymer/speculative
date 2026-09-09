from dataclasses import replace
from itertools import product

import pytest
import torch

from gbv_experiments import engine as engine_module
from gbv_experiments.config import SHARED_SUFFIX_METHODS as ROOT_MARGINAL_METHODS, Variant
from gbv_experiments.engine import Engine
from gbv_experiments.sampling import tree_verify_ancestral_batched
from gbv_experiments.preflight import classify_greedy_mismatch
from gbv_experiments.tree import (adaptive_path_proposal, adaptive_prefix_proposal,
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
    ["ddtree_fused", "ddtree_fused_parallel", "ddtree_lazy_projection"],
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
])
def test_fused_tree_methods_dispatch_in_generation(
        tiny_engine, monkeypatch, method, attribute):
    monkeypatch.setattr(engine_module, attribute, tree_verify_ancestral_batched)
    ids = torch.tensor([[1, 4, 2, 6]])
    reference = tiny_engine.generate(
        ids, Variant(name="target", method="target", paths=1, length=3,
                     temperature=0),
        12, [], seed=19,
    )
    result = tiny_engine.generate(
        ids, Variant(name=method, method=method, paths=1, length=3,
                     temperature=0, tree_budget=12),
        12, [], seed=19,
    )
    assert result["generated_token_ids"] == reference["generated_token_ids"]


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
