from collections import defaultdict
from fractions import Fraction as F
from itertools import product

import pytest
import torch

from gbv_experiments import sampling


def target(prefix):
    # Autoregressive dependence varies with the entire prefix, including position.
    n = (sum((i + 1) * x for i, x in enumerate(prefix)) + len(prefix)) % 5 + 1
    return [F(n, 7), 1 - F(n, 7)]


Q = [[F(1, 3), F(2, 3)], [F(3, 5), F(2, 5)]]
PATHS = list(product(range(2), repeat=2))


def mass(path):
    return Q[0][path[0]] * Q[1][path[1]]


def order(path):
    return tuple((target(path[:i])[x] / Q[i][x], x) for i, x in enumerate(path))


def tensor(value):
    return torch.tensor([[float(x) for x in row] for row in value], dtype=torch.float64)


def selected_distribution(k):
    distribution = defaultdict(F)
    for candidates in product(PATHS, repeat=k):
        chosen = max(candidates, key=order)
        probability = F(1)
        for path in candidates:
            probability *= mass(path)
        distribution[chosen] += probability
    assert sum(distribution.values()) == 1
    return distribution


@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_gbv_all_candidates_against_fraction_enumeration(k):
    distribution = selected_distribution(k)
    q = tensor(Q)
    for candidates in product(PATHS, repeat=k):
        paths = torch.tensor(candidates)
        p = torch.stack([tensor([target(path[:i]) for i in range(3)]) for path in candidates])
        chosen, r = sampling.select_and_reweight(paths, p, q)
        path = candidates[chosen]
        assert path == max(candidates, key=order)
        for i in range(2):
            prefix_mass = sum(w for y, w in distribution.items() if y[:i] == path[:i])
            expected = [sum(w for y, w in distribution.items() if y[:i] == path[:i] and y[i] == x) / prefix_mass for x in range(2)]
            torch.testing.assert_close(r[i], torch.tensor([float(x) for x in expected], dtype=torch.float64), rtol=1e-12, atol=1e-13)


class Impossible(Exception):
    pass


@pytest.mark.parametrize("k", [1, 3, 4])
def test_actual_block_verifier_full_output_law(monkeypatch, k):
    # Enumerate every actual multinomial outcome, then complete from Target.
    law = defaultdict(float)
    for path, path_probability in selected_distribution(k).items():
        paths = torch.tensor([path] * k)
        p = tensor([target(path[:i]) for i in range(3)])
        _, r = sampling.select_and_reweight(paths, p[None].expand(k, -1, -1), tensor(Q))
        for choices in product(range(3), repeat=3):
            probability, cursor = 1.0, 0

            def forced(weights, generator=None):
                nonlocal probability, cursor
                index = choices[cursor]
                cursor += 1
                if index >= weights.numel() or weights[index] == 0:
                    raise Impossible
                probability *= float(weights[index])
                return torch.tensor(index)

            monkeypatch.setattr(sampling, "sample", forced)
            try:
                accepted, bonus = sampling.block_verify(torch.tensor(path), p, r)
            except Impossible:
                continue
            emitted = path[:accepted] + (bonus,)
            for tail in product(range(2), repeat=3-len(emitted)):
                sequence, weight = emitted, float(path_probability) * probability
                for token in tail:
                    weight *= float(target(sequence)[token])
                    sequence += (token,)
                law[sequence] += weight
    assert sum(law.values()) == pytest.approx(1, abs=1e-12)
    for sequence in product(range(2), repeat=3):
        expected = F(1)
        for i, token in enumerate(sequence):
            expected *= target(sequence[:i])[token]
        assert law[sequence] == pytest.approx(float(expected), abs=1e-12)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_deep_prefix_no_zero_distribution(dtype):
    q = torch.full((15, 128), 1 / 128, dtype=dtype)
    paths = torch.zeros((3, 15), dtype=torch.long)
    paths[:, 0] = torch.tensor([96, 80, 0])
    p = torch.full((3, 16, 128), 1 / 128, dtype=dtype)
    chosen, r = sampling.select_and_reweight(paths, p, q)
    assert chosen == 0
    assert bool((r > 0).all())
    torch.testing.assert_close(r.sum(-1), torch.ones(15, dtype=dtype))


def test_ties_follow_token_order_and_k1_identity():
    q = torch.full((2, 3), 1/3, dtype=torch.float64)
    p = torch.full((2, 3, 3), 1/3, dtype=torch.float64)
    chosen, _ = sampling.select_and_reweight(torch.tensor([[1, 2], [2, 0]]), p, q)
    assert chosen == 1
    _, r = sampling.select_and_reweight(torch.tensor([[2, 0]]), p[:1], q)
    torch.testing.assert_close(q, r)


def test_zero_draft_mass_fails_instead_of_silent_fallback():
    with pytest.raises(FloatingPointError):
        sampling.select_and_reweight(torch.tensor([[0]]), torch.ones(1, 2, 2) / 2, torch.tensor([[1., 0.]]))


def test_matching_verifier_preserves_target_output_law(monkeypatch):
    draft = (0, 1)
    p = [target(draft[:i]) for i in range(3)]
    law = defaultdict(float)
    for posterior in product(range(2), repeat=3):
        probability = float(p[0][posterior[0]] * p[1][posterior[1]] * p[2][posterior[2]])
        monkeypatch.setattr(sampling, "sample", lambda weights, generator=None: torch.tensor(posterior))
        accepted, bonus = sampling.matching_verify(torch.tensor(draft), tensor(p))
        emitted = draft[:accepted] + (bonus,)
        for tail in product(range(2), repeat=3-len(emitted)):
            sequence, weight = emitted, probability
            for token in tail:
                weight *= float(target(sequence)[token])
                sequence += (token,)
            law[sequence] += weight
    assert sum(law.values()) == pytest.approx(1)
    for sequence in product(range(2), repeat=3):
        expected = F(1)
        for i, token in enumerate(sequence):
            expected *= target(sequence[:i])[token]
        assert law[sequence] == pytest.approx(float(expected), abs=1e-12)
