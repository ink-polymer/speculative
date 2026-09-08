"""Rational forward-flow oracle vs all actual backward sampling branches."""
from collections import defaultdict
from fractions import Fraction as F
from itertools import product
from random import Random

import pytest
import torch

from gbv_experiments import protected_tree_bv as protected
from gbv_experiments import root_marginalized_bv as rm
from test_root_marginal_adversarial import (expected_length, negative_example, normalized,
                                          probability, round_law, target_complete)
from test_root_marginalized_bv import enumerate_method, tensor


def rational_law(roots, rows, q):
    """Integrate the future analytically via forward residual conservation.

    No terminal-row sampler, floating point, or production helpers are used.
    The returned scores certify the branchwise early-BV lower bound at every
    possible shared proposal prefix, not only the final expected length.
    """
    vocab, m = len(rows[()]), len(q)
    coverage = sum(rows[()][a] for a in roots)
    law = defaultdict(F, {(x,): p for x, p in enumerate(rows[()]) if x not in roots and p})
    if not coverage:
        return dict(law)
    alpha = [rows[()][a] / coverage for a in roots]
    stack = [((), F(1), alpha, alpha)]
    while stack:
        prefix, prefix_mass, floor, score = stack.pop()
        j = len(prefix)
        p = [rows[(a,) + prefix] for a in roots]
        proposal = q[j] if j < m else [F(0)] * vocab
        flow = [[F(0)] * vocab for _ in roots]
        for x in range(vocab):
            base = [min(floor[k] * p[k][x], alpha[k] * proposal[x]) for k in range(len(roots))]
            extra = [score[k] * p[k][x] - base[k] for k in range(len(roots))]
            assert all(d >= 0 for d in extra)
            left = proposal[x] - sum(base)
            factor = min(F(1), left / sum(extra)) if sum(extra) else F(0)
            for k, a in enumerate(roots):
                flow[k][x] = base[k] + factor * extra[k]
                residual = score[k] * p[k][x] - flow[k][x]
                if residual:
                    law[(a,) + prefix + (x,)] += coverage * prefix_mass * residual
            assert sum(row[x] for row in flow) <= proposal[x]
        if j < m:
            for x in range(vocab):
                if not proposal[x]:
                    continue
                next_floor = [min(alpha[k], floor[k] * p[k][x] / proposal[x]) for k in range(len(roots))]
                next_score = [flow[k][x] / proposal[x] for k in range(len(roots))]
                assert all(s >= e for s, e in zip(next_score, next_floor))
                assert sum(next_score) <= 1
                stack.append((prefix + (x,), prefix_mass * proposal[x], next_floor, next_score))
    assert sum(law.values()) == 1
    return dict(law)


def check(monkeypatch, roots, rows, q):
    oracle = rational_law(roots, rows, q)
    early = round_law(roots, rows, q, early=True)
    for length in range(1, len(q) + 3):
        assert sum(p for s, p in oracle.items() if len(s) >= length) >= sum(
            p for s, p in early.items() if len(s) >= length)
    completed = target_complete(oracle, rows, len(q) + 2)
    for sequence in product(range(len(rows[()])), repeat=len(q) + 2):
        assert completed.get(sequence, F(0)) == probability(rows, sequence)
    actual = defaultdict(float)
    float_rows = {s: list(map(float, row)) for s, row in rows.items()}
    float_q = [list(map(float, row)) for row in q]
    for path in product(range(len(rows[()])), repeat=len(q)):
        mass = F(1)
        for j, x in enumerate(path):
            mass *= q[j][x]
        if not mass:
            continue
        for emitted, conditional in enumerate_method(monkeypatch, roots, path, float_rows,
                                                     float_q, protected.verify).items():
            actual[emitted] += float(mass) * conditional
    for emitted in actual.keys() | oracle.keys():
        assert actual.get(emitted, 0) == pytest.approx(float(oracle.get(emitted, 0)), abs=1e-12)
    return oracle


@pytest.mark.parametrize("case", range(48))
def test_joint_law_and_every_length_tail_dominate_early_bv(monkeypatch, case):
    rng = Random(2026090800 + case)
    vocab, m = 2 + case % 2, case % 4
    rows = {s: normalized([rng.randrange(5) for _ in range(vocab)])
            for d in range(m + 2) for s in product(range(vocab), repeat=d)}
    q = [normalized([rng.randrange(5) for _ in range(vocab)]) for _ in range(m)]
    roots = tuple((case + j) % vocab for j in range(1 + (case // 2) % vocab))
    check(monkeypatch, roots, rows, q)


@pytest.mark.parametrize("positive", [False, True])
def test_fixes_both_recorded_counterexamples(monkeypatch, positive):
    rows, q = negative_example(positive)
    fixed = check(monkeypatch, (0, 1), rows, q)
    early = round_law((0, 1), rows, q, early=True)
    old = round_law((0, 1), rows, q)
    assert expected_length(fixed) >= expected_length(early) > expected_length(old)


@pytest.mark.parametrize("m", [1, 2, 3, 4])
def test_retains_strict_gain_on_noisy_parity_family(monkeypatch, m):
    rows = {}
    for d in range(m + 2):
        for s in product(range(2), repeat=d):
            preferred = sum(s) % 2
            rows[s] = [F(9, 10) if x == preferred else F(1, 10) for x in range(2)] if d == m else [F(1, 2)] * 2
    q = [[F(1, 2)] * 2] * m
    fixed = check(monkeypatch, (0, 1), rows, q)
    assert expected_length(fixed) == m + 2
    assert expected_length(fixed) > expected_length(round_law((0, 1), rows, q, early=True))


def test_single_root_equals_existing_bv_block_law(monkeypatch):
    rows, q = negative_example(True)
    floats = {s: list(map(float, row)) for s, row in rows.items()}
    q = [list(map(float, row)) for row in q]
    for path in product(range(2), repeat=2):
        expected = enumerate_method(monkeypatch, (0,), path, floats, q, rm.verify)
        actual = enumerate_method(monkeypatch, (0,), path, floats, q, protected.verify)
        assert dict(actual) == pytest.approx(dict(expected), abs=1e-12)


def test_long_prefix_scores_protect_each_branch_and_total_capacity():
    g = torch.Generator().manual_seed(92)
    for m in (0, 14, 128):
        p = torch.randn(3, m + 1, 17, generator=g, dtype=torch.float64).softmax(-1)
        q = torch.randn(m, 17, generator=g, dtype=torch.float64).softmax(-1)
        path = torch.zeros(m, dtype=torch.long)
        alpha = tensor([0.2, 0.3, 0.5])
        for zero in (False, True):
            if zero and m:
                p[0, m // 2, 0] = 0
                p = p / p.sum(-1, keepdim=True)
            floors, scores = protected._scores(alpha, path, p, q)
            assert bool((scores >= floors - 1e-12).all())
            assert bool((scores.sum(0) <= 1 + 1e-12).all())
            residual, capacity = protected._residuals(alpha, floors, scores, p, q)
            assert bool(torch.isfinite(residual).all())
            assert bool((residual >= 0).all() & (residual <= capacity + 1e-12).all())
