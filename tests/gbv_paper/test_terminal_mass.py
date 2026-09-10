"""Exhaustive block-event laws, including rare tails and irregular trees."""
from collections import defaultdict
from fractions import Fraction
from itertools import product

import pytest
import torch

from gbv_experiments import sampling
from gbv_experiments.fused_tree_sampling import (tree_verify_ancestral_fused,
                                                 tree_verify_ancestral_fused_parallel,
                                                 tree_verify_ancestral_fused_scan,
                                                 tree_verify_ancestral_sparse_exit_fused_scan,
                                                 tree_verify_ancestral_logits_fused_scan,
                                                 tree_verify_ancestral_lazy_projection_fused_scan,
                                                 tree_verify_ancestral_lazy_softmax_fused_scan)


class ZeroMass(Exception):
    pass


def _assert_fused_tree_sampler_matches_inverse_cdf_paths(function):
    parents = [-1, 0, 0, 1]
    tokens = [0, 1, 2]
    p = torch.tensor(
        [[0.2, 0.3, 0.5], [0.4, 0.1, 0.5],
         [0.3, 0.6, 0.1], [0.7, 0.2, 0.1]],
        dtype=torch.float64, device="cuda",
    )
    for seed in range(32):
        first = torch.Generator(device="cuda").manual_seed(seed)
        second = torch.Generator(device="cuda").manual_seed(seed)
        uniforms = torch.rand(4, dtype=torch.float64, device="cuda", generator=first)
        expected_nodes, node = [], 0
        children = {(parent, tokens[index - 1]): index
                    for index, parent in enumerate(parents[1:], 1)}
        expected_bonus = None
        for uniform in uniforms.cpu().tolist():
            row = p[node].cpu()
            expected_bonus = int(torch.searchsorted(row.cumsum(0), uniform))
            child = children.get((node, expected_bonus))
            if child is None:
                break
            expected_nodes.append(child)
            node = child
        result = function(parents, tokens, p, second, validate=True)
        assert result == (
            expected_nodes,
            [tokens[index - 1] for index in expected_nodes],
            expected_bonus,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA extension test")
@pytest.mark.parametrize("function", [tree_verify_ancestral_fused,
                                      tree_verify_ancestral_fused_parallel])
def test_fused_tree_sampler_matches_inverse_cdf_paths(function):
    _assert_fused_tree_sampler_matches_inverse_cdf_paths(function)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA extension test")
def test_fused_scan_sampler_matches_inverse_cdf_paths():
    """Dedicated formal-doctor node for the registered fast verifier."""
    _assert_fused_tree_sampler_matches_inverse_cdf_paths(
        tree_verify_ancestral_fused_scan
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA extension test")
def test_sparse_exit_fused_scan_matches_exact_conditional_construction():
    parents = [-1, 0, 0, 1]
    tokens = [0, 1, 2]
    p = torch.tensor(
        [[0.2, 0.3, 0.5], [0.4, 0.1, 0.5],
         [0.3, 0.6, 0.1], [0.7, 0.2, 0.1]],
        dtype=torch.float64, device="cuda",
    )
    children = {
        0: [(0, 1), (1, 2)],
        1: [(2, 3)],
    }
    for seed in range(64):
        expected_generator = torch.Generator(device="cuda").manual_seed(seed)
        actual_generator = torch.Generator(device="cuda").manual_seed(seed)
        uniforms = torch.rand(
            8, dtype=torch.float64, device="cuda",
            generator=expected_generator,
        ).cpu().tolist()
        expected_nodes, node, bonus = [], 0, None
        for depth in range(4):
            outgoing = children.get(node, [])
            cumulative = 0.0
            child = None
            for token, candidate in outgoing:
                cumulative += float(p[node, token])
                if child is None and uniforms[2 * depth] < cumulative:
                    child = candidate
            if child is not None:
                expected_nodes.append(child)
                node = child
                continue
            weights = p[node].cpu().clone()
            for token, _ in outgoing:
                weights[token] = 0
            threshold = uniforms[2 * depth + 1] * float(weights.sum())
            bonus = int(torch.searchsorted(weights.cumsum(0), threshold))
            break
        actual = tree_verify_ancestral_sparse_exit_fused_scan(
            parents, tokens, p, actual_generator, validate=True,
        )
        assert actual == (
            expected_nodes,
            [tokens[index - 1] for index in expected_nodes],
            bonus,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA extension test")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32, torch.float64])
def test_direct_logits_fused_scan_matches_fp64_inverse_cdf(dtype):
    parents = [-1, 0, 0, 1]
    tokens = [0, 1, 2]
    raw_logits = torch.tensor(
        [[2., 3., 5.], [4., 1., 5.], [3., 6., 1.], [7., 2., 1.]],
        dtype=dtype, device="cuda",
    ).log()
    reference = torch.softmax(raw_logits.to(torch.float64), dim=-1)
    children = {(parent, tokens[index - 1]): index
                for index, parent in enumerate(parents[1:], 1)}
    for seed in range(64):
        expected_generator = torch.Generator(device="cuda").manual_seed(seed)
        actual_generator = torch.Generator(device="cuda").manual_seed(seed)
        uniforms = torch.rand(
            4, dtype=torch.float64, device="cuda",
            generator=expected_generator,
        ).cpu().tolist()
        expected_nodes, node, bonus = [], 0, None
        for uniform in uniforms:
            bonus = int(torch.searchsorted(
                reference[node].cpu().cumsum(0),
                torch.tensor(uniform, dtype=torch.float64),
            ))
            child = children.get((node, bonus))
            if child is None:
                break
            expected_nodes.append(child)
            node = child
        actual = tree_verify_ancestral_logits_fused_scan(
            parents, tokens, raw_logits, 1.0, torch.float64,
            actual_generator, validate=True,
        )
        assert actual[:3] == (
            expected_nodes,
            [tokens[index - 1] for index in expected_nodes],
            bonus,
        )
        assert actual[3]["visited_probability_rows"] == len(expected_nodes) + 1
        assert actual[3]["total_tree_rows"] == len(parents)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA extension test")
def test_lazy_softmax_fused_scan_matches_inverse_cdf_and_skips_leaf_rows():
    parents = [-1, 0, 0, 1]
    tokens = [0, 1, 2]
    logits = torch.tensor(
        [[2., 3., 5.], [4., 1., 5.], [3., 6., 1.], [7., 2., 1.]],
        dtype=torch.float64, device="cuda",
    ).log()
    p = torch.softmax(logits, dim=-1, dtype=torch.float64)
    children = {(parent, tokens[index - 1]): index
                for index, parent in enumerate(parents[1:], 1)}
    saw_leaf = False
    for seed in range(64):
        expected_generator = torch.Generator(device="cuda").manual_seed(seed)
        actual_generator = torch.Generator(device="cuda").manual_seed(seed)
        uniforms = torch.rand(
            4, dtype=torch.float64, device="cuda",
            generator=expected_generator,
        ).cpu().tolist()
        expected_nodes, node, bonus = [], 0, None
        for uniform in uniforms:
            bonus = int(torch.searchsorted(
                p[node].cpu().cumsum(0),
                torch.tensor(uniform, dtype=torch.float64),
            ))
            child = children.get((node, bonus))
            if child is None:
                break
            expected_nodes.append(child)
            node = child
        actual_nodes, actual_tokens, actual_bonus, stats = (
            tree_verify_ancestral_lazy_softmax_fused_scan(
                parents, tokens, logits, 1.0, torch.float64,
                actual_generator, validate=True,
            )
        )
        assert (actual_nodes, actual_tokens, actual_bonus) == (
            expected_nodes,
            [tokens[index - 1] for index in expected_nodes],
            bonus,
        )
        assert stats["internal_projected_rows"] == 2
        assert stats["projected_rows"] < stats["total_tree_rows"]
        saw_leaf |= bool(stats["leaf_projected_rows"])

        projection_result = tree_verify_ancestral_lazy_projection_fused_scan(
            parents, tokens, logits, torch.nn.Identity(), 1.0,
            torch.float64,
            torch.Generator(device="cuda").manual_seed(seed),
            validate=True,
        )
        assert projection_result[:3] == (
            expected_nodes,
            [tokens[index - 1] for index in expected_nodes],
            bonus,
        )
        assert projection_result[3]["lm_head_rows"] < len(parents)
    assert saw_leaf


def test_official_ddtree_batched_posterior_has_exact_ancestral_law(monkeypatch):
    parents = [-1, 0, 0, 1]
    tokens = [0, 1, 2]
    rows = [[2, 3, 5], [4, 1, 5], [3, 6, 1], [7, 2, 1]]
    rational = [[Fraction(value, sum(row)) for value in row] for row in rows]
    p = torch.tensor([[float(value) for value in row] for row in rational],
                     dtype=torch.float64)
    observed = defaultdict(Fraction)

    for choices in product(range(p.shape[-1]), repeat=len(parents)):
        probability = Fraction(1)
        for node, token in enumerate(choices):
            probability *= rational[node][token]

        calls = 0

        def forced(weights, generator=None):
            nonlocal calls
            calls += 1
            assert weights.shape == p.shape
            return torch.tensor(choices, device=weights.device)

        monkeypatch.setattr(sampling, "sample", forced)
        nodes, output_tokens, bonus = sampling.tree_verify_ancestral_batched(
            parents, tokens, p
        )
        assert calls == 1
        observed[(tuple(nodes), tuple(output_tokens), bonus)] += probability

    expected = {}
    paths = [()]
    prefix_mass = [Fraction(1)]
    children = [set() for _ in parents]
    for node, parent in enumerate(parents[1:], 1):
        edge_token = tokens[node - 1]
        paths.append(paths[parent] + (node,))
        prefix_mass.append(prefix_mass[parent] * rational[parent][edge_token])
        children[parent].add(edge_token)
    for node in range(len(parents)):
        output_tokens = tuple(tokens[index - 1] for index in paths[node])
        for token in range(p.shape[-1]):
            if token not in children[node]:
                expected[(paths[node], output_tokens, token)] = (
                    prefix_mass[node] * rational[node][token]
                )

    assert observed == expected
    assert sum(observed.values()) == 1


def test_lazy_projection_has_exact_ancestral_law_and_skips_leaf_heads(monkeypatch):
    parents = [-1, 0, 0, 1]
    tokens = [0, 1, 2]
    rows = [[2, 3, 5], [4, 1, 5], [3, 6, 1], [7, 2, 1]]
    rational = [[Fraction(value, sum(row)) for value in row] for row in rows]
    p = torch.tensor(
        [[float(value) for value in row] for row in rational],
        dtype=torch.float64,
    )
    hidden = p.log()
    observed = defaultdict(float)

    def execute(internal_choices, probability, leaf_choice=None):
        calls = 0

        def forced(weights, generator=None):
            nonlocal calls
            calls += 1
            if calls == 1:
                assert weights.shape == (2, 3)
                return torch.tensor(internal_choices)
            assert leaf_choice is not None and weights.shape == (3,)
            return torch.tensor(leaf_choice)

        monkeypatch.setattr(sampling, "sample", forced)
        nodes, output_tokens, bonus, stats = (
            sampling.tree_verify_ancestral_lazy_projection(
                parents, tokens, hidden, torch.nn.Identity(), 1.0,
                torch.float64, validate=True,
            )
        )
        observed[(tuple(nodes), tuple(output_tokens), bonus)] += float(probability)
        assert stats["internal_projected_rows"] == 2
        assert stats["projected_rows"] < stats["total_tree_rows"]
        assert stats["leaf_projected_rows"] == (leaf_choice is not None)

    children = {(0, 0): 1, (0, 1): 2, (1, 2): 3}
    for choices in product(range(3), repeat=2):
        probability = rational[0][choices[0]] * rational[1][choices[1]]
        node = children.get((0, choices[0]))
        if node == 1:
            node = children.get((1, choices[1]))
        if node is None:
            execute(choices, probability)
            continue
        for leaf_choice in range(3):
            execute(
                choices,
                probability * rational[node][leaf_choice],
                leaf_choice,
            )

    expected = {}
    paths = [(), (1,), (2,), (1, 3)]
    prefix_mass = [Fraction(1), rational[0][0], rational[0][1],
                   rational[0][0] * rational[1][2]]
    child_tokens = [{0, 1}, {2}, set(), set()]
    for node, path in enumerate(paths):
        output_tokens = tuple(tokens[index - 1] for index in path)
        for token in range(3):
            if token not in child_tokens[node]:
                expected[(path, output_tokens, token)] = float(
                    prefix_mass[node] * rational[node][token]
                )

    assert set(observed) == set(expected)
    for event, probability in expected.items():
        assert observed[event] == pytest.approx(probability, abs=1e-13)
    assert sum(observed.values()) == pytest.approx(1, abs=1e-13)


@pytest.mark.parametrize("validate", [False, True])
@pytest.mark.parametrize("mode", [{}, {"prefix_mode": "serial"}, {"exit_mode": "dense"},
                                  {"exit_mode": "complement"}])
@pytest.mark.parametrize("parents,tokens,rows", [
    ([-1], [], [[2, 3, 5]]),
    ([-1, 0, 1, 2], [0, 1, 0], [[2, 3, 5], [7, 2, 1], [3, 4, 3], [1, 8, 1]]),
    # Full root support, repeated labels under different parents, uneven depth.
    ([-1, 0, 0, 0, 2, 4, 1], [0, 1, 2, 0, 2, 0],
     [[2, 3, 5], [7, 2, 1], [3, 4, 3], [1, 8, 1], [4, 1, 5], [1, 1, 8], [3, 3, 4]]),
    # An unreachable subtree and an internal node with zero exit probability.
    ([-1, 0, 0, 1, 2], [0, 2, 1, 0],
     [[0, 7, 3], [2, 3, 5], [10, 0, 0], [1, 1, 8], [3, 3, 4]]),
])
def test_exact_terminal_events_and_accepted_length(monkeypatch, validate, mode, parents, tokens, rows):
    rational = [[Fraction(value, sum(row)) for value in row] for row in rows]
    p = torch.tensor([[float(value) for value in row] for row in rational],
                     dtype=torch.float64)
    original = p.clone()
    prefix_mass, paths = [Fraction(1)], [[]]
    children = [set() for _ in parents]
    for node, parent in enumerate(parents[1:], 1):
        prefix_mass.append(prefix_mass[parent] * rational[parent][tokens[node - 1]])
        paths.append(paths[parent] + [node])
        children[parent].add(tokens[node - 1])

    total = accepted = 0.0
    for node in range(len(parents)):
        for token in range(p.shape[-1]):
            choices = iter([node, token] if len(parents) > 1 else [token])
            event_mass = 1.0

            def forced(weights, generator=None):
                nonlocal event_mass
                choice = next(choices)
                weights = weights / weights.sum()
                if weights[choice] == 0:
                    raise ZeroMass
                event_mass *= float(weights[choice])
                return torch.tensor(choice, device=weights.device)

            monkeypatch.setattr(sampling, "sample", forced)
            expected = (prefix_mass[node] * rational[node][token]
                        if token not in children[node] else Fraction(0))
            if expected == 0:
                with pytest.raises(ZeroMass):
                    sampling.tree_block_verify_terminal_mass(parents, tokens, p, validate=validate, **mode)
                continue
            nodes, output_tokens, bonus = sampling.tree_block_verify_terminal_mass(
                parents, tokens, p, validate=validate, **mode
            )
            assert nodes == paths[node]
            assert output_tokens == [tokens[index - 1] for index in paths[node]]
            assert bonus == token
            assert event_mass == pytest.approx(float(expected), rel=1e-13, abs=0)
            with pytest.raises(StopIteration):
                next(choices)
            total += event_mass
            accepted += len(nodes) * event_mass
    assert total == pytest.approx(1, abs=1e-13)
    assert accepted == pytest.approx(float(sum(prefix_mass[1:])), abs=1e-13)
    torch.testing.assert_close(p, original, rtol=0, atol=0)


@pytest.mark.parametrize("validate", [False, True])
@pytest.mark.parametrize("mode", [{}, {"prefix_mode": "serial"}, {"exit_mode": "dense"}])
@pytest.mark.parametrize("tail", [1e-10, 1e-16, 1e-20, 1e-200])
def test_terminal_mass_preserves_positive_exit_tail(monkeypatch, validate, mode, tail):
    p = torch.tensor([[1 - tail, tail], [0.25, 0.75]], dtype=torch.float64)
    calls = []

    def force_rare_exit(weights, generator=None):
        calls.append(weights)
        return torch.tensor(0 if len(calls) == 1 else 1)

    monkeypatch.setattr(sampling, "sample", force_rare_exit)
    assert sampling.tree_block_verify_terminal_mass(
        [-1, 0], [0], p, validate=validate, **mode
    ) == ([], [], 1)
    assert float(calls[0][0]) == pytest.approx(tail, rel=1e-13, abs=0)
    torch.testing.assert_close(calls[1], torch.tensor([0., 1.], dtype=p.dtype))


@pytest.mark.parametrize("validate", [False, True])
@pytest.mark.parametrize("tail", [1e-10, 1e-16, 1e-20, 1e-200])
def test_joint_terminal_draw_preserves_positive_exit_tail(monkeypatch, validate, tail):
    p = torch.tensor([[1 - tail, tail], [0.25, 0.75]], dtype=torch.float64)
    calls = []

    def force_rare_exit(weights, generator=None):
        calls.append(weights / weights.sum())
        return torch.tensor(1)

    monkeypatch.setattr(sampling, "sample", force_rare_exit)
    assert sampling.tree_block_verify_terminal_mass(
        [-1, 0], [0], p, validate=validate, exit_mode="joint"
    ) == ([], [], 1)
    assert len(calls) == 1
    assert float(calls[0][1]) == pytest.approx(tail, rel=1e-13, abs=0)


@pytest.mark.parametrize("validate", [False, True])
def test_joint_terminal_draw_has_exact_terminal_event_law(monkeypatch, validate):
    parents, tokens = [-1, 0, 0, 1], [0, 1, 2]
    rows = [[2, 3, 5], [4, 1, 5], [3, 6, 1], [7, 2, 1]]
    rational = [[Fraction(value, sum(row)) for value in row] for row in rows]
    p = torch.tensor([[float(value) for value in row] for row in rational],
                     dtype=torch.float64)
    observed = defaultdict(float)
    for node in range(len(parents)):
        for token in range(p.shape[-1]):
            choice = node * p.shape[-1] + token

            def forced(weights, generator=None):
                normalized = weights / weights.sum()
                if normalized[choice] == 0:
                    raise ZeroMass
                observed[(node, token)] += float(normalized[choice])
                return torch.tensor(choice)

            monkeypatch.setattr(sampling, "sample", forced)
            try:
                result = sampling.tree_block_verify_terminal_mass(
                    parents, tokens, p, validate=validate, exit_mode="joint"
                )
            except ZeroMass:
                continue
            assert result[2] == token

    prefix_mass = [Fraction(1)]
    children = [set() for _ in parents]
    for node, parent in enumerate(parents[1:], 1):
        prefix_mass.append(
            prefix_mass[parent] * rational[parent][tokens[node - 1]]
        )
        children[parent].add(tokens[node - 1])
    expected = {
        (node, token): float(prefix_mass[node] * rational[node][token])
        for node in range(len(parents))
        for token in range(p.shape[-1])
        if token not in children[node]
    }
    assert observed == pytest.approx(expected, rel=1e-13, abs=0)


@pytest.mark.parametrize("parents,tokens,p,exception", [
    ([], [], torch.empty(0, 2), ValueError),
    ([-1], [], torch.empty(1, 0), ValueError),
    ([0], [], torch.ones(1, 1), ValueError),
    ([-1, 1], [0], torch.ones(2, 1), ValueError),
    ([-1, 0, 0], [0, 0], torch.ones(3, 1), ValueError),
    ([-1, 0], [1], torch.ones(2, 1), ValueError),
    ([-1], [], torch.ones(1, 1, dtype=torch.long), TypeError),
    ([-1], [], torch.tensor([[float("nan"), 1.]]), FloatingPointError),
    ([-1], [], torch.tensor([[-1., 2.]]), FloatingPointError),
    ([-1], [], torch.tensor([[0., 0.]]), FloatingPointError),
    ([-1], [], torch.tensor([[0.2, 0.2]]), FloatingPointError),
])
def test_terminal_mass_rejects_invalid_inputs(parents, tokens, p, exception):
    with pytest.raises(exception):
        sampling.tree_block_verify_terminal_mass(parents, tokens, p)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_terminal_mass_reproducible_and_noncontiguous(dtype):
    p = torch.tensor([[0.2, 0.3, 0.5], [0.1, 0.4, 0.5], [0.7, 0.2, 0.1]],
                     dtype=dtype).T.contiguous().T
    assert not p.is_contiguous()
    first = torch.Generator().manual_seed(912)
    second = torch.Generator().manual_seed(912)
    assert [sampling.tree_block_verify_terminal_mass([-1, 0, 1], [0, 2], p, first)
            for _ in range(16)] == [
                sampling.tree_block_verify_terminal_mass([-1, 0, 1], [0, 2], p, second)
                for _ in range(16)]
