from dataclasses import replace
from itertools import product

import pytest
import torch

from gbv_experiments.config import Variant
from gbv_experiments.tree import sampled_tree, compact_cache, probability_tree


def test_tree_merge_keeps_candidate_multiplicity():
    paths = torch.tensor([[1, 2, 3], [1, 2, 3], [1, 4, 5]])
    tree = sampled_tree(paths)
    assert len(tree.tokens) == 5
    assert tree.path_nodes[0] == tree.path_nodes[1]
    visible = tree.visibility()
    assert not visible[tree.path_nodes[2][-1], tree.path_nodes[0][-1]]
    assert len(sampled_tree(paths, False).tokens) == 9


@torch.inference_mode()
@pytest.mark.parametrize("share", [True, False])
def test_tree_logits_and_compacted_cache_equal_sequential(tiny_engine, share):
    engine = tiny_engine
    prefix = torch.tensor([[1, 2, 3]])
    paths = torch.tensor([[4, 5, 6], [4, 7, 8], [4, 5, 6]])
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
    reference = engine.target(torch.tensor([[1, 2, 3, 9, 4, 7, 11]]))
    torch.testing.assert_close(actual.logits[0, -1], reference.logits[0, -1], atol=2e-6, rtol=1e-5)


@pytest.mark.parametrize("method,paths", [("token", 1), ("bv", 1), ("gbv", 3), ("ddtree", 1)])
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


def test_stochastic_repeat_cache_and_prefix_equivalence(tiny_engine):
    ids = torch.tensor([[1, 3, 5]])
    v = Variant(name="gbv", length=3, paths=3)
    expected = tiny_engine.generate(ids, v, 23, [], seed=4)["generated_token_ids"]
    for modified in (v, replace(v, reuse_draft_cache=False), replace(v, share_prefixes=False)):
        assert tiny_engine.generate(ids, modified, 23, [], seed=4)["generated_token_ids"] == expected


@pytest.mark.parametrize("max_tokens", [1, 2, 3, 7])
def test_length_and_eos(tiny_engine, max_tokens):
    ids = torch.tensor([[1, 3, 5]])
    v = Variant(name="gbv", length=3, paths=3)
    result = tiny_engine.generate(ids, v, max_tokens, [], seed=4)
    assert result["generated_tokens"] == max_tokens
    eos = result["generated_token_ids"][0]
    stopped = tiny_engine.generate(ids, v, max_tokens, [eos], seed=4)
    assert stopped["generated_token_ids"] == [eos]
    assert stopped["draft_forward_calls"] == 0


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
