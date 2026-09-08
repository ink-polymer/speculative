"""Exhaustive block-event laws, including rare tails and irregular trees."""
from collections import defaultdict
from fractions import Fraction
from itertools import product

import pytest
import torch

from gbv_experiments import sampling


class ZeroMass(Exception):
    pass


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


@pytest.mark.parametrize("validate", [False, True])
@pytest.mark.parametrize("mode", [{}, {"prefix_mode": "serial"}, {"exit_mode": "dense"}])
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
