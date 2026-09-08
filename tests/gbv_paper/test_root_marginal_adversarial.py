"""Independent rational-flow oracle and explicit negative RM-BV examples.

Audit only: production inference is not modified. Fractions check finite laws
exactly; comparisons to actual PyTorch categorical probabilities use tolerance.
"""
from collections import defaultdict
from fractions import Fraction as F
from functools import lru_cache
from itertools import product
from random import Random

import pytest

from gbv_experiments import root_marginalized_bv as method
from test_root_marginalized_bv import enumerate_method


def normalized(raw):
    total = sum(raw)
    return [F(x, total) for x in raw] if total else [F(1, len(raw))] * len(raw)


def probability(rows, sequence, start=0):
    result = F(1)
    for j in range(start, len(sequence)):
        result *= rows[sequence[:j]][sequence[j]]
    return result


def conditional_flow_law(roots, path, rows, q):
    """Construct joint endpoint/root/correction flows, without mixture/BV code.

    Each shared symbol has one source capacity q[x], allocated proportionally
    to root-specific target capacities. Backward failure products give the
    endpoint law. This is an independent expression of the audited LV reduction,
    not a claim that the original LV forward algorithm implements this flow.
    """
    vocab, m = len(rows[()]), len(path)
    law = defaultdict(F)
    coverage = sum(rows[()][a] for a in roots)
    for x in range(vocab):
        if x not in roots and rows[()][x]:
            law[(x,)] = rows[()][x]
    if not coverage:
        return dict(law)
    scores = [rows[()][a] / coverage for a in roots]
    stops, failures = [], []
    for j in range(m + 1):
        cap = [[scores[k] * rows[(a,) + path[:j]][x]
                for x in range(vocab)] for k, a in enumerate(roots)]
        column = [sum(row[x] for row in cap) for x in range(vocab)]
        source = q[j] if j < m else [F(0)] * vocab
        flow = [[cap[k][x] * min(F(1), source[x] / column[x])
                 if column[x] else F(0) for x in range(vocab)]
                for k in range(len(roots))]
        total_flow = sum(map(sum, flow))
        denominator = 1 - total_flow
        if denominator:
            stop = [[(cap[k][x] - flow[k][x]) / denominator
                     for x in range(vocab)] for k in range(len(roots))]
            failure = (1 - sum(scores)) / denominator
        else:
            # This row is unreachable after the deeper backward decisions.
            stop, failure = cap, F(0)
        assert sum(map(sum, stop)) + failure == 1
        stops.append(stop)
        failures.append(failure)
        if j < m:
            scores = [flow[k][path[j]] / source[path[j]] for k in range(len(roots))]
    tail = F(1)
    for j in reversed(range(m + 1)):
        for k, a in enumerate(roots):
            for x, mass in enumerate(stops[j][k]):
                if mass * tail:
                    law[(a,) + path[:j] + (x,)] += coverage * tail * mass
        tail *= failures[j]
    assert sum(law.values()) == 1
    return dict(law)


def round_law(roots, rows, q, *, early=False):
    law = defaultdict(F)
    for path in product(range(len(rows[()])), repeat=len(q)):
        proposal_mass = F(1)
        for j, x in enumerate(path):
            proposal_mass *= q[j][x]
        if not proposal_mass:
            continue
        if early:
            # Sum single-root BV laws, removing their mutually duplicated
            # outside-root fallback; this leaves each inside-root weighted law.
            conditional = defaultdict(F)
            for a in roots:
                for emitted, mass in conditional_flow_law((a,), path, rows, q).items():
                    if len(emitted) > 1:
                        conditional[emitted] += mass
            for x, mass in enumerate(rows[()]):
                if x not in roots and mass:
                    conditional[(x,)] = mass
        else:
            conditional = conditional_flow_law(roots, path, rows, q)
        for emitted, mass in conditional.items():
            law[emitted] += proposal_mass * mass
    assert sum(law.values()) == 1
    return dict(law)


def target_complete(law, rows, horizon):
    completed = defaultdict(F)
    for emitted, mass in law.items():
        if len(emitted) >= horizon:
            completed[emitted[:horizon]] += mass
            continue
        for suffix in product(range(len(rows[()])), repeat=horizon - len(emitted)):
            sequence = emitted + suffix
            completed[sequence] += mass * probability(rows, sequence, len(emitted))
    return dict(completed)


def negative_example(positive=False):
    rows = {s: [F(1, 2)] * 2 for d in range(4) for s in product(range(2), repeat=d)}
    rows[()] = [F(3, 5), F(2, 5)]
    rows[(0,)] = [F(3, 10), F(7, 10)]
    rows[(1,)] = [F(99, 100), F(1, 100)] if positive else [F(1), F(0)]
    for x in range(2):
        rows[(0, x)] = [F(1, 100), F(99, 100)] if positive else [F(0), F(1)]
        rows[(1, x)] = [F(99, 100), F(1, 100)] if positive else [F(1), F(0)]
    q = [[F(1, 4), F(3, 4)], rows[(0, 0)]]
    return rows, q


def expected_length(law):
    return sum(len(s) * mass for s, mass in law.items())


@pytest.mark.parametrize("case", range(24))
@pytest.mark.parametrize("verifier", [method.verify, method.verify_tensorized])
def test_independent_rational_flow_and_completed_joint_law(monkeypatch, case, verifier):
    rng = Random(2026090700 + case)
    vocab, m = 2 + case % 2, case % 3
    rows = {s: normalized([rng.randrange(5) for _ in range(vocab)])
            for d in range(m + 2) for s in product(range(vocab), repeat=d)}
    q = [normalized([rng.randrange(5) for _ in range(vocab)]) for _ in range(m)]
    roots = tuple((case + j) % vocab for j in range(1 + (case // 2) % vocab))
    float_rows = {s: list(map(float, row)) for s, row in rows.items()}
    float_q = [list(map(float, row)) for row in q]
    law = round_law(roots, rows, q)
    completed = target_complete(law, rows, m + 2)
    for seq in product(range(vocab), repeat=m + 2):
        assert completed.get(seq, F(0)) == probability(rows, seq)
    for path in product(range(vocab), repeat=m):
        if any(not q[j][x] for j, x in enumerate(path)):
            continue
        expected = conditional_flow_law(roots, path, rows, q)
        actual = enumerate_method(monkeypatch, roots, path, float_rows, float_q, verifier)
        for emitted in expected.keys() | actual.keys():
            assert actual.get(emitted, 0) == pytest.approx(float(expected.get(emitted, 0)), abs=1e-12)


@pytest.mark.parametrize("positive", [False, True])
def test_multistep_delayed_root_can_lose_to_early_root(monkeypatch, positive):
    rows, q = negative_example(positive)
    late = round_law((0, 1), rows, q)
    early = round_law((0, 1), rows, q, early=True)
    assert expected_length(early) > expected_length(late)
    if not positive:
        assert expected_length(late) == F(4593, 1450)
        assert expected_length(early) == F(81, 25)
        assert expected_length(early) - expected_length(late) == F(21, 290)
    float_rows = {s: list(map(float, row)) for s, row in rows.items()}
    float_q = [list(map(float, row)) for row in q]
    for verifier, expected in [(method.verify_tensorized, late), (method.verify_early_root, early)]:
        actual = defaultdict(float)
        for path in product(range(2), repeat=2):
            proposal_mass = q[0][path[0]] * q[1][path[1]]
            if not proposal_mass:
                continue
            for emitted, mass in enumerate_method(monkeypatch, (0, 1), path, float_rows,
                                                  float_q, verifier).items():
                actual[emitted] += float(proposal_mass) * mass
        for emitted in actual.keys() | expected.keys():
            assert actual.get(emitted, 0) == pytest.approx(float(expected.get(emitted, 0)), abs=1e-12)


@pytest.mark.parametrize("zeros", [False, True])
def test_adaptive_multiround_law_and_eos_truncation(zeros):
    """Context-dependent root sets/proposals, variable blocks, and EOS stopping."""
    horizon = 5
    rng = Random(59)
    rows = {s: normalized([rng.randrange(0 if zeros else 1, 5) for _ in range(2)])
            for d in range(horizon + 3) for s in product(range(2), repeat=d)}

    @lru_cache(None)
    def decode(context):
        if len(context) >= horizon:
            return {context[:horizon]: F(1)}
        m = 1 + len(context) % 2
        suffix_rows = {s: rows[context + s] for d in range(m + 2)
                       for s in product(range(2), repeat=d)}
        roots = (0, 1) if (sum(context) + len(context)) % 2 else ((len(context) // 2) % 2,)
        q = [normalized([1 + (sum(context) + j) % 3, 1 + (len(context) + j) % 4])
             for j in range(m)]
        result = defaultdict(F)
        for emitted, mass in round_law(roots, suffix_rows, q).items():
            for sequence, continuation_mass in decode(context + emitted).items():
                result[sequence] += mass * continuation_mass
        return dict(result)

    actual = decode(())
    for seq in product(range(2), repeat=horizon):
        assert actual.get(seq, F(0)) == probability(rows, seq)
    # EOS is a deterministic map of the complete sequence, not a fresh sample.
    def stopped(law):
        result = defaultdict(F)
        for seq, mass in law.items():
            if not mass:
                continue
            terminal = seq[:seq.index(1) + 1] if 1 in seq else seq
            result[terminal] += mass
        return dict(result)
    target = {s: probability(rows, s) for s in product(range(2), repeat=horizon)}
    assert stopped(actual) == stopped(target)


def test_postselecting_roots_from_suffix_is_biased():
    """A forbidden tempting adaptation: S={z} after observing suffix z."""
    rows = {s: [F(1, 2)] * 2 for d in range(3) for s in product(range(2), repeat=d)}
    q = [[F(1, 2)] * 2]
    bad = defaultdict(F)
    for z in range(2):
        for emitted, mass in conditional_flow_law((z,), (z,), rows, q).items():
            bad[emitted] += F(1, 2) * mass
    complete = target_complete(bad, rows, 2)
    assert complete == {(0, 0): F(3, 8), (0, 1): F(1, 8),
                        (1, 0): F(1, 8), (1, 1): F(3, 8)}
    assert sum(abs(mass - F(1, 4)) for mass in complete.values()) / 2 == F(1, 4)
