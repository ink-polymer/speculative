"""Exhaust every actual categorical draw, then target-complete each output."""
from collections import defaultdict
from functools import partial
from itertools import product

import pytest
import torch

from gbv_experiments import root_marginalized_bv as method
from gbv_experiments import sampling
from gbv_experiments.tree_coupling_oracle import example_rows, probability


class BranchNeeded(Exception):
    def __init__(self, probabilities):
        self.probabilities = probabilities


def tensor(values):
    return torch.tensor(values, dtype=torch.float64)


def enumerate_method(monkeypatch, roots, path, rows, q, verifier=method.verify):
    """No sampled frequency estimates: enumerate all categorical decisions."""
    branch_rows = [[rows[(root,) + path[:j]] for j in range(len(path) + 1)] for root in roots]
    stack = [((), 1.0)]
    emitted_law = defaultdict(float)
    while stack:
        choices, mass = stack.pop()
        cursor = 0

        def forced(weights, generator=None):
            nonlocal cursor
            assert weights.ndim == 1
            assert float(weights.sum()) == pytest.approx(1, abs=1e-12)
            if cursor == len(choices):
                raise BranchNeeded(weights.tolist())
            index = choices[cursor]
            cursor += 1
            assert float(weights[index]) > 0
            return torch.tensor(index)

        monkeypatch.setattr(sampling, "sample", forced)
        try:
            branch, accepted, bonus = verifier(
                torch.tensor(roots, dtype=torch.long), tensor(rows[()]),
                torch.tensor(path, dtype=torch.long), tensor(branch_rows),
                tensor(q).reshape(len(path), len(rows[()])))
        except BranchNeeded as request:
            stack.extend((choices + (i,), mass * p) for i, p in enumerate(request.probabilities) if p > 0)
            continue
        assert cursor == len(choices)
        if branch == -1:
            assert bonus not in roots
            output = (bonus,)
        else:
            assert 0 <= accepted <= len(path)
            output = (roots[branch],) + path[:accepted] + (bonus,)
        emitted_law[output] += mass
    assert sum(emitted_law.values()) == pytest.approx(1, abs=1e-12)
    return emitted_law


def aggregate(monkeypatch, roots, rows, q, verifier=method.verify):
    m, vocab = len(q), len(rows[()])
    output_law = defaultdict(float)
    for path in product(range(vocab), repeat=m):
        path_mass = 1
        for j, token in enumerate(path):
            path_mass *= q[j][token]
        if path_mass == 0:
            continue
        for emitted, mass in enumerate_method(monkeypatch, roots, path, rows, q, verifier).items():
            output_law[emitted] += path_mass * mass
    complete = defaultdict(float)
    for emitted, mass in output_law.items():
        for suffix in product(range(vocab), repeat=m + 2 - len(emitted)):
            sequence = emitted + suffix
            complete[sequence] += mass * probability(rows, sequence, len(emitted))
    for sequence in product(range(vocab), repeat=m + 2):
        assert complete[sequence] == pytest.approx(probability(rows, sequence), abs=1e-12)
    committed = sum(len(s) * p for s, p in output_law.items())
    # The covered-root gate is sampled before BV; outside S only one token is
    # returned. Even perfect suffix acceptance cannot exceed this bound.
    coverage = sum(rows[()][root] for root in roots)
    assert committed <= 1 + (m + 1) * coverage + 1e-12
    return committed


@pytest.mark.parametrize("roots", [(0,), (0, 1), (1, 2), (0, 1, 2)])
@pytest.mark.parametrize("length", [0, 1, 2])
@pytest.mark.parametrize("verifier", [method.verify, method.verify_tensorized,
                                    method.verify_early_root,
                                    partial(method.verify_tensorized, rule="token")])
def test_full_autoregressive_law_including_partial_root_support(monkeypatch, roots, length, verifier):
    vocab = 3
    rows = {}
    for depth in range(length + 2):
        for prefix in product(range(vocab), repeat=depth):
            raw = [(1 + (t + sum((i + 1) * x for i, x in enumerate(prefix)) + depth) % 5)
                   for t in range(vocab)]
            rows[prefix] = [x / sum(raw) for x in raw]
    q = [[0.2, 0.3, 0.5], [0.4, 0.5, 0.1]][:length]
    aggregate(monkeypatch, roots, rows, q, verifier)


@pytest.mark.parametrize("strength", [0, 0.1, 0.5, 0.9, 1])
def test_copy_family_reaches_the_full_block_bound(monkeypatch, strength):
    rows = example_rows("copy", strength)
    committed = aggregate(monkeypatch, (0, 1), rows, [[0.5, 0.5]])
    assert committed == pytest.approx(3)


def test_same_mechanism_also_has_a_failure_regime(monkeypatch):
    # Delaying the root choice cannot help when that root reveals no information
    # about the suffix. This must remain in reports, not just the positive case.
    committed = aggregate(monkeypatch, (0, 1), example_rows("zero"), [[0.5, 0.5]])
    assert committed == pytest.approx(2.6)


@pytest.mark.parametrize("m,fixed_best", [(1, 2.9), (2, 3.0), (3, 3.25), (4, 3.5)])
@pytest.mark.parametrize("verifier", [method.verify, method.verify_tensorized])
def test_noisy_parity_family_multistep_separation(monkeypatch, m, fixed_best, verifier):
    # A is 90% the parity of m IID fair suffix bits. The suffix marginal is
    # exactly IID fair, although fixing A induces a late conditional mismatch.
    rows = {}
    for depth in range(m + 2):
        for prefix in product(range(2), repeat=depth):
            if depth == m:
                preferred = (prefix[0] + sum(prefix[1:])) % 2
                rows[prefix] = [0.9 if x == preferred else 0.1 for x in range(2)]
            else:
                rows[prefix] = [0.5, 0.5]
    committed = aggregate(monkeypatch, (0, 1), rows, [[0.5, 0.5]] * m, verifier)
    assert committed == pytest.approx(m + 2)
    # For any prefix tree, ancestor probabilities are >= descendant ones.
    # The highest-mass B nodes (ancestors first on ties) form a valid optimal
    # fixed tree. This strong baseline can inspect the full synthetic target.
    candidates = [s for d in range(1, m + 2) for s in product(range(2), repeat=d)]
    ordered = sorted(candidates, key=lambda s: (-probability(rows, s), len(s), s))
    selected = set(ordered[:2 * (m + 1)])
    assert all(len(s) == 1 or s[:-1] in selected for s in selected)
    assert 1 + sum(probability(rows, s) for s in selected) == pytest.approx(fixed_best)
    assert committed > fixed_best


@pytest.mark.parametrize("root_p", [[0.0, 1.0], [1.0, 0.0]])
@pytest.mark.parametrize("verifier", [method.verify, method.verify_tensorized,
                                    method.verify_early_root,
                                    partial(method.verify_tensorized, rule="token")])
def test_zero_covered_or_zero_outside_mass(monkeypatch, root_p, verifier):
    rows = example_rows()
    rows[()] = root_p
    aggregate(monkeypatch, (0,), rows, [[0.0, 1.0]], verifier)


@pytest.mark.parametrize("verifier", [method.verify, method.verify_tensorized,
                                    partial(method.verify_tensorized, rule="token")])
def test_selected_branch_conditions_on_correction_too(monkeypatch, verifier):
    rows = example_rows("copy")
    # In the p=q marginalized case, the sole proposed second token is accepted,
    # and root selection must still use the extra correction token's likelihood.
    rows[(0, 0)] = [0.2, 0.8]
    rows[(1, 0)] = [0.9, 0.1]
    rows[(0, 1)] = [0.8, 0.2]
    rows[(1, 1)] = [0.1, 0.9]
    aggregate(monkeypatch, (0, 1), rows, [[0.5, 0.5]], verifier)


@pytest.mark.parametrize("problem", ["duplicate_roots", "bad_rows", "impossible_proposal", "fp32"])
@pytest.mark.parametrize("verifier", [method.verify, method.verify_tensorized, method.verify_early_root])
def test_invalid_inputs_fail_closed(problem, verifier):
    roots = torch.tensor([0, 1])
    root_p = tensor([0.5, 0.5])
    path = torch.tensor([0])
    branch_p = tensor([[[0.5, 0.5], [0.5, 0.5]]] * 2)
    q = tensor([[0.5, 0.5]])
    if problem == "duplicate_roots":
        roots[1] = 0
    elif problem == "bad_rows":
        branch_p[1, 0, 0] = float("nan")
    elif problem == "impossible_proposal":
        q = tensor([[0.0, 1.0]])
    else:
        root_p = root_p.float()
    with pytest.raises(ValueError):
        verifier(roots, root_p, path, branch_p, q)


@pytest.mark.parametrize("strength", [0, 0.1, 0.5, 0.9, 1])
@pytest.mark.parametrize("family", ["copy", "zero"])
def test_tensorized_matches_full_conditional_emitted_block_law(monkeypatch, strength, family):
    rows = example_rows(family, strength)
    for roots in [(0,), (0, 1)]:
        for path in [(0,), (1,)]:
            expected = enumerate_method(monkeypatch, roots, path, rows, [[0.5, 0.5]])
            actual = enumerate_method(monkeypatch, roots, path, rows, [[0.5, 0.5]], method.verify_tensorized)
            assert dict(actual) == pytest.approx(dict(expected), abs=1e-12)


@pytest.mark.parametrize("length", [1, 3, 15])
def test_tensorized_uses_three_categorical_calls_independent_of_depth(monkeypatch, length):
    original = sampling.sample
    shapes = []

    def traced(weights, generator=None):
        shapes.append(tuple(weights.shape))
        return original(weights, generator)

    monkeypatch.setattr(sampling, "sample", traced)
    method.verify_tensorized(torch.tensor([0, 1]), tensor([0.4, 0.4, 0.2]),
                             torch.zeros(length, dtype=torch.long),
                             torch.full((2, length + 1, 3), 1 / 3, dtype=torch.float64),
                             torch.full((length, 3), 1 / 3, dtype=torch.float64))
    assert shapes == [(length + 2,), (3,), (2,)]


def test_proposal_has_distinct_preselected_roots_and_one_shared_suffix(monkeypatch):
    q = tensor([[0.6, 0.3, 0.1], [0.2, 0.1, 0.7], [0.8, 0.1, 0.1]])
    calls = []

    def forced(rows, generator=None):
        calls.append(rows)
        return torch.tensor([2, 0])

    monkeypatch.setattr(sampling, "sample", forced)
    paths = method.propose(q, 2)
    assert paths.tolist() == [[0, 2, 0], [1, 2, 0]]
    assert len(calls) == 1
    torch.testing.assert_close(calls[0], q[1:])
    assert method.propose(q[:1], 2).tolist() == [[0], [1]]
    with pytest.raises(ValueError, match="branch count"):
        method.propose(q, 4)


def test_parallel_prefix_weights_match_serial_with_zeros_and_long_prefixes():
    generator = torch.Generator().manual_seed(59)
    p = torch.randn(129, 3, dtype=torch.float64, generator=generator).softmax(-1)
    q = torch.randn(128, 3, dtype=torch.float64, generator=generator).softmax(-1)
    path = torch.zeros(128, dtype=torch.long)
    for zero_depth in [None, 0, 65, 127]:
        rows = p.clone()
        if zero_depth is not None:
            rows[zero_depth] = tensor([0, 0.5, 0.5])
        endpoint, _ = method._terminal_rows(path, rows, q, "bv")
        w = 1.0
        success, failure = [], []
        for i in range(129):
            residual = (w * rows[i] - (q[i] if i < 128 else 0)).clamp_min(0).sum().item()
            total = residual + 1 - w
            success.append(residual / total if total else 1.0)
            failure.append((1 - w) / total if total else 0.0)
            if i < 128:
                w = min(1.0, w * rows[i, 0].item() / q[i, 0].item())
        tail = 1.0
        expected = [0.0] * 129
        for i in reversed(range(129)):
            expected[i] = success[i] * tail
            tail *= failure[i]
        torch.testing.assert_close(endpoint, tensor(expected), atol=1e-12, rtol=1e-10)


def test_explicit_outside_tail_is_not_lost(monkeypatch):
    gate = []

    def forced(weights, generator=None):
        if not gate:
            gate.append(weights.clone())
            return torch.tensor(0)
        return weights.argmax()

    monkeypatch.setattr(sampling, "sample", forced)
    result = method.verify_tensorized(torch.tensor([0]), tensor([1.0, 1e-20]),
                                     torch.tensor([0]), tensor([[[0.5, 0.5], [0.5, 0.5]]]),
                                     tensor([[0.5, 0.5]]))
    assert gate[0][0].item() == pytest.approx(1e-20, rel=1e-12, abs=0)
    assert result == (-1, 0, 1)


@pytest.mark.parametrize("location", ["root", "branch", "proposal"])
def test_hot_path_does_not_replace_nonfinite_probabilities_with_uniform(location):
    root = tensor([0.5, 0.5])
    branches = torch.full((2, 2, 2), 0.5, dtype=torch.float64)
    q = tensor([[0.5, 0.5]])
    {"root": root, "branch": branches, "proposal": q}[location].reshape(-1)[0] = float("nan")
    with pytest.raises(RuntimeError):
        method.verify_tensorized(torch.tensor([0, 1]), root, torch.tensor([0]),
                                 branches, q, validate=False)
