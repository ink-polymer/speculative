"""Research-only oracle checks; synthetic examples are NOT speed benchmarks."""
from collections import defaultdict
from fractions import Fraction as F
from itertools import product

import pytest

from gbv_experiments.tree_coupling_oracle import (
    completed_law, demo, example_rows, example_trees, fixed_tree_committed,
    prefix_tree, probability, solve,
)


def test_rational_full_block_certificate():
    # P(00)=P(11)=9/20; P(01)=P(10)=1/20. Shared-second trees:
    # after choosing s uniformly, choose first=s with probability 9/10.
    rows = {(): (F(1, 2), F(1, 2))}
    for a in range(2):
        rows[(a,)] = tuple(F(9, 10) if b == a else F(1, 10) for b in range(2))
        for b in range(2):
            rows[(a, b)] = (F(1, 3), F(2, 3))
    events = {}
    trees = example_trees("shared_second")
    for s, a, bonus in product(range(2), repeat=3):
        events[(s, (a, s, bonus))] = F(1, 2) * rows[(s,)][a] * rows[(a, s)][bonus]
    for s in range(2):
        assert sum(m for (i, _), m in events.items() if i == s) == F(1, 2)
    assert all(emitted[:-1] in trees[i] for i, emitted in events)
    assert completed_law(rows, 3, events) == {
        seq: probability(rows, seq) for seq in product(range(2), repeat=3)}
    assert sum(len(emitted) * m for (_, emitted), m in events.items()) == 3


def test_rational_rejection_and_correction_certificate():
    rows = {(): (F(1, 2), F(1, 2))}
    for a in range(2):
        rows[(a,)] = tuple(F(9, 10) if b == a else F(1, 10) for b in range(2))
        for b in range(2):
            rows[(a, b)] = (F(1, 3), F(2, 3))
    events = {}
    trees = example_trees("xor_second")
    for a, bonus in product(range(2), repeat=2):
        events[(0, (a, a, bonus))] = F(1, 4) * rows[(a, a)][bonus]
        events[(1, (a, 1 - a, bonus))] = F(1, 20) * rows[(a, 1 - a)][bonus]
    for a in range(2):
        events[(1, (a, a))] = F(1, 5)  # correct at the second token, then return
    assert all(emitted[:-1] in trees[i] for i, emitted in events)
    for i in range(2):
        assert sum(m for (j, _), m in events.items() if j == i) == F(1, 2)
    assert completed_law(rows, 3, events) == {
        seq: probability(rows, seq) for seq in product(range(2), repeat=3)}
    assert sum(len(emitted) * m for (_, emitted), m in events.items()) == F(13, 5)


@pytest.mark.parametrize("family", ["shared_second", "xor_second"])
def test_same_budget_and_indexed_draft_marginals(family):
    trees = example_trees(family)
    marginals = [defaultdict(F), defaultdict(F)]
    for tree in trees:
        assert len(tree) - 1 == 4
        leaves = [node for node in tree if len(node) == 2]
        assert len(leaves) == 2
        for swap in range(2):
            for index in range(2):
                marginals[index][leaves[index ^ swap]] += F(1, 4)
    expected = {seq: F(1, 4) for seq in product(range(2), repeat=2)}
    assert marginals == [expected, expected]


@pytest.mark.parametrize("regime", ["copy", "zero"])
def test_correlation_can_help_or_hurt(regime):
    pytest.importorskip("scipy")
    rows = example_rows(regime)
    good = "shared_second" if regime == "copy" else "xor_second"
    for family in ("shared_second", "xor_second"):
        trees = example_trees(family)
        result = solve(rows, trees, [0.5, 0.5], 3)
        assert result.committed_tokens == pytest.approx(3 if family == good else 2.6)
        assert result.target_law_error < 1e-12
        assert result.equality_error < 1e-12
        stronger = solve(rows, trees, [0.5, 0.5], 3, conditional_on_tree=True)
        assert stronger.committed_tokens == pytest.approx(2.5)
        assert stronger.committed_tokens == pytest.approx(sum(
            fixed_tree_committed(rows, t) for t in trees) / 2)


@pytest.mark.parametrize("strength", [0, 0.1, 0.5, 0.9, 1])
def test_fixed_tree_bound_including_zero_support(strength):
    pytest.importorskip("scipy")
    rows = example_rows(strength=strength)
    for tree in [prefix_tree([()]), prefix_tree([(0, 0)]),
                 prefix_tree([(0, 0), (1, 1)]), prefix_tree(list(product(range(2), repeat=2)))]:
        result = solve(rows, [tree], [1], 3)
        assert result.committed_tokens == pytest.approx(fixed_tree_committed(rows, tree))


def test_demo_is_explicitly_not_a_gpu_or_novelty_result():
    pytest.importorskip("scipy")
    result = demo()
    assert result["new_algorithm_proved"] is False
    assert result["gpu_benchmarked"] is False
    for stats in result["regimes"].values():
        assert stats["best_fixed_tree_ddtree_committed"] == pytest.approx(2.9)


@pytest.mark.parametrize("trees,weights", [
    ([prefix_tree([(0,)])], [0.5]),
    ([prefix_tree([(0,)])], [-1]),
    ([prefix_tree([(0,)])], [float("nan")]),
    ([((), (0, 0))], [1]),
    ([((), (), (0,))], [1]),
    ([prefix_tree([(0, 0, 0)])], [1]),
])
def test_invalid_mixture_rejected(trees, weights):
    pytest.importorskip("scipy")
    with pytest.raises(ValueError):
        solve(example_rows(), trees, weights, 3)


def test_wrong_target_rows_rejected():
    pytest.importorskip("scipy")
    rows = example_rows()
    rows[(0,)] = (0.7, 0.7)
    with pytest.raises(ValueError, match="normalized"):
        solve(rows, example_trees("shared_second"), [0.5, 0.5], 3)
