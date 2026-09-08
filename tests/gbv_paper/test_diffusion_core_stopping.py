"""Exact stopped-kernel laws and falsifiable conditional theory, without a GPU."""
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction as F
from itertools import product
from random import Random

import pytest
import torch

from gbv_experiments import diffusion_core_stopping as stopping
from gbv_experiments.tree import probability_tree
from test_diffusion_scaffold import enumerate_scaffold
from test_diffusion_theory_comparison import (actual_proposal, geometric_tail, prefixes,
                                             product_probability, rational_atoms, rational_kernel)
from test_diffusion_tree_bv import fraction_rows, probability, tensor


def exact_stopped_law(rows, q, fixed, budget, *, stop=None):
    """Independent Fraction stopping + existing independent Fraction BV oracle.

    Integrate the FULL product law, not the conditional law given the horizon.
    ``stop`` is only an adversarial oracle hook, absent from the implementation.
    """
    atoms = [rational_atoms(row, [F(0)]) for row in q]
    law, horizons = defaultdict(F), defaultdict(F)
    for draws in product(*(range(len(row)) for row in atoms)):
        path = tuple(atoms[j][draws[j]][1][0] for j in range(len(q)))
        mass = product_probability(q, path)
        present, horizon = set(fixed), 0
        if stop is None:
            while horizon < len(q) and len(present) < budget:
                horizon += 1
                present.add(path[:horizon])
        else:
            horizon = stop(path)
            present.update(path[:j] for j in range(1, horizon + 1))
        horizons[horizon] += mass
        conditional = (rational_kernel(rows, atoms[:horizon], draws[:horizon], pool=True)
                       if horizon else {(x,): p for x, p in enumerate(rows[()]) if p})
        queue = [(output, mass * value) for output, value in conditional.items()]
        while queue:
            output, value = queue.pop()
            if output in present:
                queue.extend((output + (x,), value * p) for x, p in enumerate(rows[output]) if p)
            else:
                law[output] += value
    assert sum(law.values()) == sum(horizons.values()) == 1
    return dict(law), dict(horizons)


def completed(rows, law, length):
    result = defaultdict(F)
    for output, mass in law.items():
        for suffix in product(range(len(rows[()])), repeat=length - len(output)):
            seq = output + suffix
            result[seq] += mass * probability(rows, seq, start=len(output))
    return dict(result)


@pytest.mark.parametrize("budget,core_budget", [(3, 0), (4, 0), (5, 2), (6, 0), (9, 6)])
@pytest.mark.parametrize("case", range(8))
def test_exact_stopped_joint_law_and_actual_kernel(monkeypatch, case, budget, core_budget):
    length, vocab = 3, 2 + (case == 7)
    rows = fraction_rows(Random(55417 + case), vocab, length)
    q = [[F(2, 3), F(1, 3)], [F(3, 5), F(2, 5)], [F(4, 7), F(3, 7)]]
    if case == 0:
        rows = {prefix: [F(1), F(0)] for prefix in rows}
    elif case == 1:
        q = [[F(1), F(0)]] * length
    elif case == 2:
        rows = {prefix: q[min(len(prefix), length - 1)] for prefix in rows}
    elif case == 7:  # Strictly positive target mass outside the proposal support.
        q = [row + [F(0)] for row in q]
    atoms = [rational_atoms(row, [F(0)]) for row in q]
    temperature = [.3, .6, 1.][case % 3]
    proposal = actual_proposal(q, atoms, temperature)
    if case == 3:  # Core uses original retained denoising mass, BV uses normalized Q.
        proposal.law.retained_mass = tensor([F(1, 4), F(3, 4), F(2, 3)])
    greedy = tensor(q).argmax(-1)
    fixed = stopping.core_scaffold(proposal.law, greedy, budget, core_budget)
    expected, horizons = exact_stopped_law(rows, q, fixed, budget)
    target = {seq: probability(rows, seq) for seq in product(range(vocab), repeat=length + 1)}
    exact_complete = completed(rows, expected, length + 1)
    assert all(exact_complete.get(seq, 0) == mass for seq, mass in target.items())
    actual = enumerate_scaffold(
        monkeypatch, proposal, rows, greedy, budget, temperature=temperature,
        build=lambda p: stopping.stopped_tree(p, greedy, budget, core_budget),
        verify=lambda logits, tree, p: stopping.verify_logits(
            logits, tree, p, temperature, greedy=greedy, budget=budget, core_budget=core_budget))
    assert sum(actual.values()) == pytest.approx(1., abs=1e-11)
    for seq in set(actual) | set(expected):
        assert actual.get(seq, 0.) == pytest.approx(float(expected.get(seq, 0)), abs=1e-11, rel=0)
    for d in range(1, length + 1):
        tail = sum(mass for seq, mass in expected.items() if len(seq) > d)
        coverage = sum(probability(rows, prefix) for prefix in fixed if len(prefix) == d)
        assert tail >= coverage >= probability(rows, tuple(greedy[:d].tolist()))
        if d <= min(horizons):
            cut_bound = sum(min(product_probability(q, seq[:k])
                                * probability(rows, seq, start=k) for k in range(d + 1))
                            for seq in product(range(vocab), repeat=d))
            assert tail >= cut_bound
    if core_budget == 0 and budget == length:
        assert horizons == {0: F(1)}
    if core_budget == 0 and budget == length + 1 and case == 4:
        assert len(horizons) > 1  # Actually exercised data-dependent stopping.


def test_peeking_at_future_before_stopping_is_detectably_biased():
    q = [[F(1, 2), F(1, 2)]] * 2
    rows = {prefix: q[0] for d in range(3) for prefix in product(range(2), repeat=d)}
    safe, _ = exact_stopped_law(rows, q, set(), 2, stop=lambda path: 1 if path[0] == 0 else 2)
    unsafe, _ = exact_stopped_law(rows, q, set(), 2, stop=lambda path: 0 if path[0] == 0 else 2)
    assert set(completed(rows, safe, 3).values()) == {F(1, 8)}
    unsafe_complete = completed(rows, unsafe, 3)
    assert sum(mass for seq, mass in unsafe_complete.items() if seq[0] == 0) == F(1, 4)
    assert sum(mass for seq, mass in unsafe_complete.items() if seq[0] == 1) == F(3, 4)


@pytest.mark.parametrize("temperature", [.3, .6, 1.])
def test_main_budget_all_diffusion_draws_preserve_core_and_stop_predictably(temperature):
    length, budget, core_budget = 15, 45, 30
    q = [[F(51, 100), F(49, 100)]] * length
    atoms = [rational_atoms(row, [F(0)]) for row in q]
    proposal = actual_proposal(q, atoms, temperature)
    greedy = torch.zeros(length).long()
    fixed = stopping.core_scaffold(proposal.law, greedy, budget, core_budget)
    assert len(fixed) == 41
    assert all(prefix in fixed for d in range(1, 5) for prefix in product(range(2), repeat=d))
    # Every full latent path, not a Monte Carlo estimate of minimum horizon.
    stopped_prefixes, minimum = {}, length
    for path in product(range(2), repeat=length):
        current = replace(proposal, draws=torch.tensor(path))
        tree = stopping.stopped_tree(current, greedy, budget, core_budget)
        horizon = len(tree.path_nodes[0])
        minimum = min(minimum, horizon)
        assert fixed <= set(prefixes(tree)) and len(tree.tokens) <= budget
        assert tree.path_nodes == [list(range(1, horizon + 1))]
        assert prefixes(tree)[horizon] == path[:horizon]
        if horizon < length:
            assert len(tree.tokens) == budget
        # Once stopped, all full witnesses with the same past have the same tree.
        previous = stopped_prefixes.setdefault(path[:horizon], tree)
        assert tree == previous
        difficult = (0, 0, 0, 0, 1) + (0,) * 10
        present = set(prefixes(tree))
        for d in range(5, 9):
            assert (difficult[:d] in present) == (path[:d] == difficult[:d])
        assert difficult[:9] not in present
    assert minimum == 8


def test_strict_advantage_on_a_full_support_neighborhood():
    lo, hi, length = F(49, 100), F(51, 100), 15
    assert hi ** 5 < lo ** 4 and hi ** 6 < lo ** 5
    # Allows arbitrary AR prefix dependence of P within these per-token limits.
    new_lower = 4 + sum((lo / hi) ** d for d in range(5, 9))
    dd_upper = 4 + 15 * hi ** 5
    df_upper = geometric_tail(hi, length)
    assert new_lower > dd_upper > df_upper
    assert float(new_lower) == pytest.approx(7.087185165863393)
    assert float(dd_upper) == pytest.approx(4.5175378765)
    # Cost ratios are sufficient conditional thresholds, NOT measured speedups.
    assert float((1 + new_lower) / (1 + dd_upper)) == pytest.approx(1.4657235431600564)


@pytest.mark.parametrize("p0", [F(99, 100), F(1, 1000)])
def test_both_previous_opposite_alignment_counterexamples_are_repaired(p0):
    q0, length = F(51, 100), 15
    dd = probability_tree(tensor([[q0, 1 - q0]] * length), 45)
    assert Counter(dd.depths[1:]) == {1: 2, 2: 4, 3: 8, 4: 16, 5: 15}
    dd_yield = sum(p0 ** p.count(0) * (1 - p0) ** p.count(1) for p in prefixes(dd)[1:])
    if p0 > q0:
        lower = 4 + sum(p0 ** d for d in range(5, length + 1))
    else:
        # Full four levels plus the cut bound's all-ones contribution at 5..8.
        lower = 4 + sum((1 - q0) ** d for d in range(5, 9))
    assert lower > dd_yield and lower > geometric_tail(p0, length)


def test_not_a_universal_dominance_claim_new_construction_has_counterexample():
    # L3/B6/M3 is the scaled-down policy. Some DD prefixes are present only
    # when sampled by Q; a concentrated positive P can expose this weakness.
    q = [[F(51, 100), F(49, 100)]] * 3
    proposal = actual_proposal(q, [rational_atoms(row, [F(0)]) for row in q], 1.)
    fixed = stopping.core_scaffold(proposal.law, torch.zeros(3).long(), 6, 3)
    dd = set(prefixes(probability_tree(tensor(q), 6))) - {()}
    # Find a strict counterexample by exact exhaustive target policies, not by
    # rounding a selected empirical throughput comparison.
    found = False
    for preferred in product(range(2), repeat=3):
        rows = {p: [F(999, 1000) if preferred[min(d, 2)] == x else F(1, 1000)
                    for x in range(2)] for d in range(4) for p in product(range(2), repeat=d)}
        law, _ = exact_stopped_law(rows, q, fixed, 6)
        actual = sum((len(seq) - 1) * mass for seq, mass in law.items())
        baseline = sum(probability(rows, prefix) for prefix in dd)
        if actual < baseline:
            found = True
            break
    assert found


def test_main_budget_still_has_an_explicit_positive_target_counterexample():
    q0, p_correct = F(51, 100), F(999, 1000)
    # Preferred sequence is 000010...; DD contains its first five prefixes.
    # New tree covers depths 5..8 only if X matches; depths >=9 are impossible
    # because four non-core, non-greedy nodes have exhausted its spare budget.
    new_upper = (4 + sum(q0 ** (d - 1) * (1 - q0) for d in range(5, 9))
                 + sum(1 - p_correct ** d for d in range(5, 16)))
    dd_lower = sum(p_correct ** d for d in range(1, 6))
    assert new_upper < dd_lower
    assert float(new_upper) < 4.18 and float(dd_lower) > 4.98


@pytest.mark.parametrize("corruption", ["path", "leaf_nan", "greedy", "temperature", "branches"])
def test_invalid_inputs_are_rejected(corruption):
    q = [[F(2, 3), F(1, 3)]] * 3
    proposal = actual_proposal(q, [rational_atoms(row, [F(0)]) for row in q], 1.)
    greedy, temperature = torch.zeros(3).long(), 1.
    tree = stopping.stopped_tree(proposal, greedy, 5, 2)
    logits = torch.zeros((len(tree.parents), 2)).double()
    if corruption == "path":
        tree.path_nodes[0] = []
    elif corruption == "leaf_nan":
        logits[-1, 0] = float("nan")
    elif corruption == "greedy":
        greedy[-1] = 2
    elif corruption == "temperature":
        temperature = 0.
    else:
        proposal = replace(proposal, slots=proposal.slots.expand(2, -1, -1))
    with pytest.raises(ValueError):
        stopping.verify_logits(logits, tree, proposal, temperature,
                               greedy=greedy, budget=5, core_budget=2)


def test_full_witness_retained_and_no_future_peek_at_exhausted_budget():
    q = [[F(2, 3), F(1, 3)]] * 3
    proposal = actual_proposal(q, [rational_atoms(row, [F(0)]) for row in q], 1.)
    proposal.draws[:] = 0
    before = deepcopy(proposal)
    # G already consumes all three nodes: do not peek at X1=0 and accept free G.
    tree = stopping.stopped_tree(proposal, torch.zeros(3).long(), 3, 0)
    assert tree.path_nodes == [[]]
    assert torch.equal(proposal.draws, before.draws) and torch.equal(proposal.source, before.source)
    assert proposal.law.tokens.shape[0] == 3
