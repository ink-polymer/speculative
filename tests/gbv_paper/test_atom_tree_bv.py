"""Independent rational conservation oracle and exhaustive actual sampler paths."""
from collections import defaultdict
from dataclasses import replace
from fractions import Fraction as F
from itertools import product
from random import Random

import pytest
import torch

from gbv_experiments import atom_tree_bv as atom
from gbv_experiments.config import ATOM_TREE_METHODS, Variant


def tensor(x):
    return torch.tensor(x, dtype=torch.float64)


def normalize(values):
    if not sum(values):
        values = [1] * len(values)
    return [F(v, sum(values)) for v in values]


def target_probability(rows, sequence):
    mass = F(1)
    for j, x in enumerate(sequence):
        mass *= rows[sequence[:j]][x]
    return mass


def complete(law, rows, horizon):
    result = defaultdict(F)
    stack = list(law.items())
    while stack:
        sequence, mass = stack.pop()
        if len(sequence) == horizon:
            result[sequence] += mass
        else:
            for x, p in enumerate(rows[sequence]):
                if p:
                    stack.append((sequence + (x,), mass * p))
    return dict(result)


def oracle(roots, tokens, slots, source, rows, *, pool=True):
    """Enumerate latent prefixes and CONSERVE forward target flow (not LV draws).

    No production plan, floating arithmetic or backward endpoint formula is used.
    Sources can have zeros, and several atoms may map to one support token.
    """
    k, m, vocab = len(roots), len(source), len(rows[()])
    coverage = sum(rows[()][a] for a in roots)
    law = defaultdict(F, {(x,): p for x, p in enumerate(rows[()]) if x not in roots and p})
    if not coverage:
        return dict(law)
    alpha = [rows[()][a] / coverage for a in roots]
    stack = [((), F(1), alpha, alpha)]
    while stack:
        history, history_mass, floor, score = stack.pop()
        j = len(history)
        prefixes = [(a,) + tuple(tokens[b][d][slots[b][d][h]] for d, h in enumerate(history))
                    for b, a in enumerate(roots)]
        p = [rows[s] for s in prefixes]
        q = [[F(0)] * vocab for _ in roots]
        s = source[j] if j < m else []
        mapped = [[tokens[b][j][slots[b][j][h]] for h in range(len(s))] for b in range(k)]
        for b in range(k):
            for h, mass in enumerate(s):
                q[b][mapped[b][h]] += mass
        flows, bases = [[F(0)] * len(s) for _ in roots], [[F(0)] * len(s) for _ in roots]
        for h, mass in enumerate(s):
            lifted = [p[b][mapped[b][h]] * mass / q[b][mapped[b][h]]
                      if q[b][mapped[b][h]] else F(0) for b in range(k)]
            base = [min(floor[b] * lifted[b], alpha[b] * mass) for b in range(k)]
            extra = [score[b] * lifted[b] - base[b] for b in range(k)]
            assert all(x >= 0 for x in extra) and sum(base) <= mass
            factor = min(F(1), (mass - sum(base)) / sum(extra)) if pool and sum(extra) else F(0)
            for b in range(k):
                bases[b][h] = base[b]
                flows[b][h] = base[b] + factor * extra[b]
            assert sum(f[h] for f in flows) <= mass
        for b in range(k):
            for x in range(vocab):
                residual = score[b] * p[b][x] - sum(flows[b][h] for h in range(len(s)) if mapped[b][h] == x)
                assert residual >= 0
                if residual:
                    law[prefixes[b] + (x,)] += coverage * history_mass * residual
        for h, mass in enumerate(s):
            if mass:
                next_floor = [b[h] / mass for b in bases]
                next_score = [f[h] / mass for f in flows]
                assert sum(next_score) <= 1
                assert all(w >= e for w, e in zip(next_score, next_floor))
                stack.append((history + (h,), history_mass * mass, next_floor, next_score))
    assert sum(law.values()) == 1
    return dict(law)


def actual_law(monkeypatch, roots, tokens, slots, source, rows, *, pool=True, logits=False):
    k, m, vocab = len(roots), len(source), len(rows[()])
    r = len(tokens[0][0]) if m else 1
    c = len(source[0]) if m else 1
    base = atom.AtomProposal(torch.tensor(roots), torch.tensor(tokens).long().reshape(k, m, r),
                             torch.tensor(slots).long().reshape(k, m, c), tensor(source).reshape(m, c),
                             torch.zeros(m, dtype=torch.long))
    result = defaultdict(float)

    class NeedDraw(Exception):
        def __init__(self, probabilities):
            self.probabilities = probabilities

    for draws in product(range(c), repeat=m):
        mass = F(1)
        for j, h in enumerate(draws):
            mass *= source[j][h]
        if not mass:
            continue
        proposal = replace(base, draws=torch.tensor(draws, dtype=torch.long))
        paths = proposal.paths().tolist()
        p = tensor([[list(map(float, rows[tuple(path[:j + 1])])) for j in range(m + 1)] for path in paths])
        root = tensor(list(map(float, rows[()])))
        queue = [((), 1.)]
        while queue:
            choices, verifier_mass = queue.pop()
            used = []

            def draw(probabilities, generator=None):
                if len(used) == len(choices):
                    raise NeedDraw(probabilities.tolist())
                choice = choices[len(used)]
                used.append(choice)
                return torch.tensor(choice, dtype=torch.long)

            with monkeypatch.context() as patch:
                patch.setattr(atom.sampling, "sample", draw)
                try:
                    if logits:
                        branch, depth, token = atom.verify_logits(root.log() * 1.3, p.log() * 1.3,
                                                                  proposal, 1.3, pool=pool)
                    else:
                        branch, depth, token = atom.verify(root, p, proposal, pool=pool)
                except NeedDraw as request:
                    assert sum(request.probabilities) == pytest.approx(1.)
                    for choice, probability in enumerate(request.probabilities):
                        if probability > 1e-16:
                            queue.append((choices + (choice,), verifier_mass * probability))
                    continue
            emitted = (token,) if branch == -1 else tuple(paths[branch][:depth + 1]) + (token,)
            result[emitted] += float(mass) * verifier_mass
    return dict(result)


@pytest.mark.parametrize("case", range(24))
def test_complete_joint_law_zeros_duplicate_atoms_and_branch_specific_proposals(monkeypatch, case):
    rng = Random(89120 + case)
    vocab, m = 2 + case % 2, case % 3
    k, r, c = 1 + case % 2, 1 + (case // 2) % 2, 2
    roots = tuple(range(k))
    rows = {s: normalize([rng.randrange(5) for _ in range(vocab)])
            for d in range(m + 2) for s in product(range(vocab), repeat=d)}
    tokens = [[rng.sample(range(vocab), r) for _ in range(m)] for _ in roots]
    slots = [[[rng.randrange(r) for _ in range(c)] for _ in range(m)] for _ in roots]
    source = [normalize([rng.randrange(4) for _ in range(c)]) for _ in range(m)]
    for pool in (False, True):
        expected = oracle(roots, tokens, slots, source, rows, pool=pool)
        completed = complete(expected, rows, m + 2)
        for sequence in product(range(vocab), repeat=m + 2):
            assert completed.get(sequence, F(0)) == target_probability(rows, sequence)
        actual = actual_law(monkeypatch, roots, tokens, slots, source, rows, pool=pool, logits=bool(case % 2))
        # FP64 exp/log may create ~1e-17 exits that are zero in the exact oracle.
        # Compare the UNION of outcomes, not just the nonzero dictionary keys.
        for s in actual.keys() | expected.keys():
            assert actual.get(s, 0.) == pytest.approx(float(expected.get(s, 0)), abs=1e-11, rel=0)
    protected = oracle(roots, tokens, slots, source, rows)
    floor = oracle(roots, tokens, slots, source, rows, pool=False)
    for length in range(1, m + 3):
        assert sum(p for s, p in protected.items() if len(s) >= length) >= sum(
            p for s, p in floor.items() if len(s) >= length)


def test_coupling_architecture_has_gain_and_counterexample_not_universal_dominance(monkeypatch):
    roots, tokens, source = (0, 1), [[[0, 1]], [[0, 1]]], [[F(1, 2), F(1, 2)]]
    aligned, stratified = [[[0, 1]], [[0, 1]]], [[[0, 1]], [[1, 0]]]
    for copy_root, expected in ((False, (F(13, 5), F(3))), (True, (F(3), F(13, 5)))):
        rows = {s: [F(1, 2)] * 2 for d in range(3) for s in product(range(2), repeat=d)}
        for a in roots:
            preferred = a if copy_root else 0
            rows[(a,)] = [F(9, 10) if x == preferred else F(1, 10) for x in roots]
        lengths = []
        for slots in (aligned, stratified):
            law = oracle(roots, tokens, slots, source, rows)
            lengths.append(sum(len(s) * p for s, p in law.items()))
            assert actual_law(monkeypatch, roots, tokens, slots, source, rows) == pytest.approx(
                {s: float(p) for s, p in law.items()}, abs=1e-12, rel=0)
        assert tuple(lengths) == expected


@pytest.mark.parametrize("aligned", [False, True])
def test_cdf_partition_preserves_each_branch_actual_marginal(aligned):
    g = torch.Generator().manual_seed(41)
    weights = torch.randn(3, 4, 5, generator=g, dtype=torch.float64).softmax(-1)
    weights[0, 0, 1] = 0
    weights /= weights.sum(-1, keepdim=True)
    tokens = torch.arange(5).expand(3, 4, 5)
    shifts = tensor([0., 0., 0.]) if aligned else None
    proposal = atom.couple(torch.arange(3), tokens, weights, g, shifts=shifts)
    proposal.validate(5)
    torch.testing.assert_close(proposal.marginals(), weights, rtol=0, atol=1e-14)
    assert proposal.source.shape == (4, 16)
    assert proposal.paths().shape == (3, 5)


def test_aligned_and_stratified_draft_proposals_keep_same_marginals_not_same_tree():
    q = tensor([[.5, .5], [.5, .5], [.5, .5]])
    aligned = atom.propose(q, 2, torch.Generator().manual_seed(3), coupling="aligned")
    different = atom.propose(q, 2, torch.Generator().manual_seed(3))
    torch.testing.assert_close(aligned.marginals(), different.marginals())
    assert torch.equal(aligned.paths()[0, 1:], aligned.paths()[1, 1:])
    assert not torch.equal(different.paths()[0, 1:], different.paths()[1, 1:])


def test_depth_permuted_coupling_reassigns_strata_across_depths_and_keeps_marginals():
    q = tensor([[.6, .3, .1]] * 4)
    fixed = atom.propose(q, 3, torch.Generator().manual_seed(11), coupling="fixed")
    permuted = atom.propose(q, 3, torch.Generator().manual_seed(11))
    torch.testing.assert_close(fixed.marginals(), permuted.marginals(), rtol=0, atol=1e-14)
    assert torch.equal(fixed.slots[:, 0], fixed.slots[:, 1])
    assert not torch.equal(permuted.slots[:, 0], permuted.slots[:, 1])
    assert set(atom.depth_permuted_shifts(3, 4)[:, 0].tolist()) == {0., 1 / 3, 2 / 3}
    assert set(atom.depth_permuted_shifts(3, 4)[:, 1].tolist()) == {0., 1 / 3, 2 / 3}


def test_logits_path_materializes_only_root_and_one_correction_distribution(monkeypatch):
    g = torch.Generator().manual_seed(7)
    q = torch.randn(5, 17, generator=g, dtype=torch.float64).softmax(-1)
    proposal = atom.propose(q, 3, g)
    root = torch.randn(17, generator=g, dtype=torch.float64)
    branch = torch.randn(3, 5, 17, generator=g, dtype=torch.float64)
    original, shapes = atom.sampling.probabilities, []

    def record(logits, *args):
        shapes.append(tuple(logits.shape))
        return original(logits, *args)

    monkeypatch.setattr(atom.sampling, "probabilities", record)
    atom.verify_logits(root, branch, proposal, 1., g)
    assert shapes == [(17,), (17,)]


@pytest.mark.parametrize("method", sorted(ATOM_TREE_METHODS))
def test_real_tiny_qwen_generation_greedy_cache_eos_and_budget(tiny_engine, method):
    ids = torch.tensor([[1, 4, 2, 6]])
    v = Variant(name=method, method=method, paths=3, length=3, tree_budget=9, temperature=0)
    reference = tiny_engine.generate(ids, replace(v, method="target", paths=1), 19, [], seed=19)
    result = tiny_engine.generate(ids, v, 19, [], seed=19)
    assert result["generated_token_ids"] == reference["generated_token_ids"]
    assert result["target_forward_calls"] == len(result["rounds"]) + 1
    assert result["draft_forward_calls"] == len(result["rounds"])
    assert all(r["tree_nodes"] <= 9 and r["verify_tokens"] <= 10 for r in result["rounds"])
    token = reference["generated_token_ids"][4]
    stopped = tiny_engine.generate(ids, v, 19, [token], seed=19)
    assert stopped["generated_token_ids"] == reference["generated_token_ids"][:reference["generated_token_ids"].index(token) + 1]
    for setting in (replace(v, reuse_draft_cache=False), replace(v, share_prefixes=False)):
        assert tiny_engine.generate(ids, setting, 19, [], seed=19)["generated_token_ids"] == reference["generated_token_ids"]


@pytest.mark.parametrize("method", sorted(ATOM_TREE_METHODS))
def test_stochastic_repeatability_and_cache_compaction(tiny_engine, method):
    ids = torch.tensor([[1, 3, 5]])
    v = Variant(name=method, method=method, length=3, paths=3, tree_budget=9)
    expected = tiny_engine.generate(ids, v, 23, [], seed=4)["generated_token_ids"]
    for setting in (v, replace(v, reuse_draft_cache=False), replace(v, share_prefixes=False)):
        assert tiny_engine.generate(ids, setting, 23, [], seed=4)["generated_token_ids"] == expected


def test_invalid_inputs_fail_before_sampling():
    q = tensor([[.5, .5], [1., 0.]])
    proposal = atom.propose(q, 2)
    root, p = tensor([.5, .5]), tensor([[[.5, .5], [.5, .5]]] * 2)
    with pytest.raises(ValueError, match="distinct"):
        atom.verify(root, p, replace(proposal, roots=torch.tensor([0, 0])))
    with pytest.raises(ValueError, match="dtype"):
        atom.verify(root.float(), p, proposal)
    with pytest.raises(ValueError, match="normalized"):
        atom.verify(root, p * 2, proposal)
    with pytest.raises(ValueError, match="budget"):
        Variant(name="bad", method="atom_tree_bv", paths=3, length=3, tree_budget=8).validate()
    with pytest.raises(ValueError, match="support size"):
        atom.propose(q, 2, coupling="unknown")


def test_empty_suffix_and_zero_root_coverage(monkeypatch):
    rows = {(): [F(0), F(1)], (0,): [F(1), F(0)], (1,): [F(0), F(1)]}
    expected = oracle((0,), [[]], [[]], [], rows)
    assert expected == {(1,): F(1)}
    assert actual_law(monkeypatch, (0,), [[]], [[]], [], rows) == {(1,): 1.}
    proposal = atom.propose(tensor([[.5, .5]]), 2)
    proposal.validate(2)
    assert proposal.paths().shape == (2, 1)
