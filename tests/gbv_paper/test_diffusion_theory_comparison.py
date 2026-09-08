"""Exact rational theory audit; toy acceptance bounds are NOT GPU timings.

The independent Fraction oracle enumerates the backward endpoint kernel and
target-completed joint sequence law. A separate comparison enumerates draws
from the actual production verifier. No empirical frequency is used as proof.
"""
from collections import Counter, defaultdict
from fractions import Fraction as F
from itertools import accumulate, product
from random import Random

import pytest
import torch

from gbv_experiments import diffusion_tree_bv as diffusion
from gbv_experiments.tree import probability_tree, sampled_tree
from test_diffusion_tree_bv import (enumerate_sampler, fraction_rows, probability,
                                    tensor)


def geometric_tail(ratio, length=15):
    return sum((ratio ** d for d in range(1, length + 1)), F(0))


def prefixes(tree):
    result = [()]
    for node in range(1, len(tree.parents)):
        result.append(result[tree.parents[node]] + (tree.tokens[node - 1],))
    return result


def rational_atoms(q, shifts):
    """Exact shifted inverse CDF, independent of the torch partition builder."""
    cdf = list(accumulate(q))
    boundaries = sorted({F(0), F(1)} | {(edge - shift) % 1
                                      for edge in cdf for shift in shifts})
    return [(right - left, tuple(next(x for x, edge in enumerate(cdf)
                                     if ((left + right) / 2 + shift) % 1 < edge)
                                for shift in shifts))
            for left, right in zip(boundaries, boundaries[1:])]


def rational_kernel(rows, atoms, draws, *, pool):
    """Conditional law given every latent draw, including future-dependent exits."""
    branches, length, vocab = len(atoms[0][0][1]), len(atoms), len(rows[()])
    alpha = F(1, branches)
    floors = scores = [alpha] * branches
    paths = [tuple(atoms[j][draws[j]][1][a] for j in range(length))
             for a in range(branches)]
    residuals, failures = [], []
    for j in range(length + 1):
        assert all(w >= e >= 0 for w, e in zip(scores, floors))
        assert sum(scores) <= 1
        failures.append(1 - sum(scores))
        residual = [[scores[a] * p for p in rows[paths[a][:j]]]
                    for a in range(branches)]
        if j < length:
            marginal = [[sum(s for s, mapped in atoms[j] if mapped[a] == x)
                         for x in range(vocab)] for a in range(branches)]
            next_floors, next_scores = None, None
            for h, (s, mapped) in enumerate(atoms[j]):
                lift = [rows[paths[a][:j]][mapped[a]] * s / marginal[a][mapped[a]]
                        for a in range(branches)]
                base = [min(floors[a] * lift[a], alpha * s) for a in range(branches)]
                extra = [scores[a] * lift[a] - base[a] for a in range(branches)]
                factor = min(F(1), (s - sum(base)) / sum(extra)) if sum(extra) else F(0)
                flow = [base[a] + (extra[a] * factor if pool else 0)
                        for a in range(branches)]
                assert all(0 <= flow[a] <= scores[a] * lift[a] for a in range(branches))
                assert sum(flow) <= s
                for a in range(branches):
                    residual[a][mapped[a]] -= flow[a]
                if h == draws[j]:
                    next_floors = [value / s for value in base]
                    next_scores = [value / s for value in flow]
            floors, scores = next_floors, next_scores
        assert all(value >= 0 for row in residual for value in row)
        residuals.append(residual)
    law, later, endpoint_total = defaultdict(F), F(1), F(0)
    for j in reversed(range(length + 1)):
        mass = sum(map(sum, residuals[j]))
        total = mass + failures[j]
        # Same harmless convention as production for an unreachable 0/0 row.
        endpoint = later * (mass / total if total else 1)
        endpoint_total += endpoint
        if mass:
            for a, row in enumerate(residuals[j]):
                for token, value in enumerate(row):
                    if value:
                        law[paths[a][:j] + (token,)] += endpoint * value / mass
        else:
            assert endpoint == 0
        later *= failures[j] / total if total else 0
    assert endpoint_total == sum(law.values()) == 1
    return {seq: mass for seq, mass in law.items() if mass}


def exact_law(rows, atoms, *, pool):
    law = defaultdict(F)
    for draws in product(*(range(len(row)) for row in atoms)):
        mass = F(1)
        for j, h in enumerate(draws):
            mass *= atoms[j][h][0]
        for seq, conditional in rational_kernel(rows, atoms, draws, pool=pool).items():
            law[seq] += mass * conditional
    return dict(law)


def actual_proposal(q, atoms, temperature):
    """Feed the identical rational partition to the real verifier, rounded to FP64."""
    length, vocab = len(q), len(q[0])
    branches = len(atoms[0][0][1])
    columns = max(map(len, atoms))
    source = torch.zeros((length, columns), dtype=torch.float64)
    slots = torch.zeros((branches, length, columns), dtype=torch.long)
    for j, row in enumerate(atoms):
        for h, (s, mapped) in enumerate(row):
            source[j, h] = float(s)
            slots[:, j, h] = torch.tensor(mapped)
    law = diffusion.DiffusionBlockLaw(
        torch.arange(vocab).expand(length, -1), tensor(q), torch.ones(length).double(),
        torch.tensor([0] + [vocab - 1] * length), temperature, vocab - 1)
    proposal = diffusion.DiffusionProposal(law, slots, source, torch.zeros(length).long())
    proposal.validate(vocab)
    return proposal


@pytest.mark.parametrize("case", range(12))
@pytest.mark.parametrize("pool", [False, True])
def test_exact_rational_joint_law_and_production_kernel(monkeypatch, case, pool):
    length, branches = 2, 3
    rows = fraction_rows(Random(92300 + case), 2, length)
    q = [[F(1, 2), F(1, 2)], [F(2, 3), F(1, 3)]]
    if case == 0:  # Deterministic target: endpoint 0/0 and zero residuals.
        rows = {prefix: [F(1), F(0)] for prefix in rows}
    elif case == 1:  # Exact denoiser, complete terminal acceptance.
        rows = {prefix: q[len(prefix)] if len(prefix) < length else q[0] for prefix in rows}
    elif case == 2:  # Target mass outside the proposal support.
        q = [[F(1), F(0)]] * length
    shifts = [[F(0) if case % 2 else F((a * (j + 1)) % branches, branches)
               for a in range(branches)] for j in range(length)]
    atoms = [rational_atoms(q[j], shifts[j]) for j in range(length)]
    expected = exact_law(rows, atoms, pool=pool)
    coverage = defaultdict(F)
    for draws in product(*(range(len(row)) for row in atoms)):
        latent_mass = F(1)
        for j, h in enumerate(draws):
            latent_mass *= atoms[j][h][0]
        present = {tuple(atoms[j][draws[j]][1][a] for j in range(d))
                   for a in range(branches) for d in range(1, length + 1)}
        for prefix in present:
            coverage[prefix] += latent_mass
    completed = defaultdict(F)
    for output, mass in expected.items():
        for suffix in product(range(2), repeat=length + 1 - len(output)):
            sequence = output + suffix
            completed[sequence] += mass * probability(rows, sequence, start=len(output))
    assert sum(expected.values()) == 1
    assert all(completed[seq] == probability(rows, seq)
               for seq in product(range(2), repeat=length + 1))
    # Exact cut-measure bound, checked independently of the rational flow code.
    for d in range(1, length + 1):
        bound = sum(min(product_probability(q, seq[:k]) * probability(rows, seq, start=k)
                        for k in range(d + 1)) for seq in product(range(2), repeat=d))
        tail = sum(mass for seq, mass in expected.items() if len(seq) > d)
        assert tail >= bound if pool else tail == bound
        for seq in product(range(2), repeat=d):
            union_bound = branches * product_probability(q, seq)
            assert coverage[seq] <= union_bound
            assert tail <= min(F(1), coverage[seq] + 1 - probability(rows, seq))
    temperature = [0.3, 0.6, 1.0][case % 3]
    actual = enumerate_sampler(monkeypatch, actual_proposal(q, atoms, temperature), rows,
                               pool=pool, temperature=temperature)
    for seq in set(actual) | set(expected):
        assert actual.get(seq, 0.) == pytest.approx(float(expected.get(seq, 0)), abs=1e-11, rel=0)


def product_probability(q, sequence):
    mass = F(1)
    for j, token in enumerate(sequence):
        mass *= q[j][token]
    return mass


@pytest.mark.parametrize("temperature", [0.3, 0.6, 1.0])
def test_strict_same_budget_counterexample(temperature):
    p0, q0, length, branches, budget = F(99, 100), F(51, 100), 15, 3, 45
    q = [[q0, 1 - q0]] * length
    law = diffusion.DiffusionBlockLaw.from_logits(
        tensor(q).log() * temperature, torch.tensor([0] + [1] * length), temperature, 1)
    dense = torch.zeros((length, 2), dtype=torch.float64).scatter(1, law.tokens, law.weights)
    tree = probability_tree(dense, budget)
    assert Counter(tree.depths[1:]) == {1: 2, 2: 4, 3: 8, 4: 16, 5: 15}
    # All four shallow levels precede ANY depth-five node, independent of ties.
    assert q0 ** 5 < (1 - q0) ** 4
    upper = sum(min(F(1), branches * q0 ** d + 1 - p0 ** d)
                for d in range(1, length + 1))
    ddtree = sum(product_probability([[p0, 1 - p0]] * length, prefix)
                 for prefix in prefixes(tree)[1:])
    dflash = geometric_tail(p0, length)
    assert upper == F(1864048920924898330259604926727, 500000000000000000000000000000)
    assert ddtree == F(9999786239, 2000000000)
    assert upper < 4 <= ddtree < dflash
    assert float(upper) == pytest.approx(3.7280978418497965)
    assert float(dflash) == pytest.approx(13.854222890512435)


def test_conditional_strict_advantage_exact_bounds():
    a, b = F(49, 50), F(51, 100)
    floor = geometric_tail(a)
    dd_upper, df_upper = 4 + 15 * b ** 5, geometric_tail(b)
    assert floor > dd_upper > df_upper
    assert float(floor) == pytest.approx(12.810113970375209)
    assert dd_upper == F(9035075753, 2000000000)
    assert float(df_upper) == pytest.approx(1.0407735774540776)
    # Expected-yield ratios are cost thresholds, NOT measured throughput ratios.
    assert (1 + floor) / (1 + dd_upper) > F(5, 2)
    assert (1 + floor) / (1 + df_upper) > F(67, 10)
    tree = probability_tree(tensor([[F(1, 2), F(1, 2)]] * 15), 45)
    assert sum(F(1, 2) ** d for d in tree.depths[1:]) == F(143, 32)
    assert geometric_tail(F(1, 2)) == 1 - F(1, 2) ** 15


@pytest.mark.parametrize("case", range(9))
@pytest.mark.parametrize("pool", [False, True])
def test_registered_length_pointwise_floor_and_exact_drafter(case, pool):
    temperature = [0.3, 0.6, 1.0][case % 3]
    law = diffusion.DiffusionBlockLaw.from_logits(
        torch.zeros((15, 2), dtype=torch.float64), torch.tensor([0] + [1] * 15), temperature, 1)
    proposal = diffusion.propose(law, 3, torch.Generator().manual_seed(case),
                                 coupling="aligned" if case % 2 else "depth_permuted")
    tree = sampled_tree(proposal.paths())
    rng = Random(9971 + case)
    p0 = [F(1, 2) if case < 3 else F(rng.randrange(4900, 5101), 10000)
          for _ in tree.parents]
    p = tensor([[p, 1 - p] for p in p0])
    # Same trie node always has the same conditional, including shared prefixes.
    row_nodes = torch.tensor([[0] + path[:-1] for path in tree.path_nodes])
    support = p[row_nodes].gather(-1, proposal.tokens)
    state = diffusion.transport.plan(tensor([F(1, 3)] * 3), support, proposal, pool=pool)
    lower = tensor([F(1, 3) * F(49, 50) ** d for d in range(16)])
    assert bool((state.scores >= state.floors - 1e-12).all())
    assert bool((state.floors >= lower[None] - 1e-12).all())
    if case < 3:
        assert torch.allclose(state.scores, tensor([[F(1, 3)] * 16] * 3), atol=1e-12, rtol=0)
        assert float(state.endpoint[-1]) == pytest.approx(1., abs=1e-12)
        _, accepted, _ = diffusion.verify_logits(
            p.log() * temperature, tree, proposal, temperature,
            torch.Generator().manual_seed(case), pool=pool)
        assert len(accepted) == 15


def test_theory_evidence_is_in_formal_gate_and_source_identity():
    from gbv_experiments.terminal_formal import UNIT_FILES
    from gbv_experiments.terminal_protocol import study_sources

    assert "tests/gbv_paper/test_diffusion_theory_comparison.py" in UNIT_FILES
    assert "docs/DIFFUSION_TREE_THEORY_AUDIT.md" in study_sources()
