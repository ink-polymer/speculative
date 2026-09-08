"""Scaffold coverage, complete output laws, and cached-engine integration."""
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction as F
from itertools import product
from random import Random

import pytest
import torch

from gbv_experiments import diffusion_tree_bv as diffusion
from gbv_experiments.config import DIFFUSION_SCAFFOLD_METHODS, Variant
from gbv_experiments.tree import probability_tree
from test_diffusion_tree_bv import complete, fraction_rows, probability, tensor
from test_diffusion_theory_comparison import (actual_proposal, geometric_tail, prefixes,
                                             product_probability, rational_atoms, rational_kernel)


def enumerate_scaffold(monkeypatch, proposal, rows, greedy, budget, *, fill=True, recycle=True, continuation="terminal", temperature=.6, build=None, verify=None):
    """Enumerate actual endpoint, correction, and batched continuation draws."""
    result = defaultdict(float)

    class DrawNeeded(Exception):
        def __init__(self, probs):
            self.probs = probs

    for indices in product(*[torch.nonzero(row > 0).flatten().tolist() for row in proposal.source]):
        current = replace(proposal, draws=torch.tensor(indices))
        latent_mass = float(current.latent_log_probability().exp())
        tree = (build(current) if build else
                diffusion.scaffold_tree(current, greedy, budget, fill=fill))
        logits = tensor([rows[prefix] for prefix in prefixes(tree)]).log() * temperature
        queue = [((), 1.)]
        while queue:
            choices, mass = queue.pop()
            used = []

            def draw(probs, generator=None):
                if probs.ndim == 2:
                    return torch.stack([draw(row) for row in probs])
                assert probs.ndim == 1
                assert float(probs.sum()) == pytest.approx(1., abs=1e-12)
                assert bool((probs >= 0).all())
                if len(used) == len(choices):
                    raise DrawNeeded(probs.tolist())
                choice = choices[len(used)]
                used.append(choice)
                return torch.tensor(choice)

            with monkeypatch.context() as patch:
                patch.setattr(diffusion.sampling, "sample", draw)
                try:
                    nodes, tokens, bonus = (verify(logits, tree, current) if verify else
                        diffusion.verify_scaffold_logits(logits, tree, current, temperature,
                                                          recycle=recycle, continuation=continuation))
                except DrawNeeded as needed:
                    for index, p in enumerate(needed.probs):
                        if p > 1e-15:
                            queue.append((choices + (index,), mass * p))
                    continue
            assert [tree.tokens[n - 1] for n in nodes] == tokens
            assert all(tree.parents[n] == parent for n, parent in zip(nodes, [0] + nodes[:-1]))
            if recycle:
                assert (nodes[-1] if nodes else 0, bonus) not in set(zip(tree.parents[1:], tree.tokens))
            result[tuple(tokens) + (bonus,)] += latent_mass * mass
    return dict(result)


@pytest.mark.parametrize("case", range(8))
def test_complete_joint_law_and_simultaneous_acceptance_floors(monkeypatch, case):
    vocab, length, branches = 2 + case % 2, 2, 1 + case % 2
    rows = fraction_rows(Random(8713 + case), vocab, length)
    q = tensor([[2, 1] if vocab == 2 else [3, 2, 1]] * length)
    q /= q.sum(-1, keepdim=True)
    if case == 0:
        rows = {prefix: [F(1), F(0)] for prefix in rows}
    law = diffusion.DiffusionBlockLaw.from_logits(
        q.log() * .6, torch.tensor([0] + [vocab - 1] * length), .6, vocab - 1,
        support_size=1 if case == 1 else vocab)
    proposal = diffusion.propose(law, branches, torch.Generator().manual_seed(case))
    greedy, budget = q.argmax(-1), (branches + 1) * length + 1
    laws = [enumerate_scaffold(monkeypatch, proposal, rows, greedy, budget, **kw)
            for kw in ({}, {"fill": False}, {"recycle": False})]
    ancestral = enumerate_scaffold(monkeypatch, proposal, rows, greedy, budget, continuation="ancestral")
    for seq in set(ancestral) | set(laws[0]):
        assert ancestral.get(seq, 0.) == pytest.approx(laws[0].get(seq, 0.), abs=1e-10, rel=0)
    expected = {seq: float(probability(rows, seq))
                for seq in product(range(vocab), repeat=length + 1)}
    for actual in laws:
        assert sum(actual.values()) == pytest.approx(1., abs=1e-10)
        completed = complete(actual, rows, length + 1)
        assert {seq: completed.get(seq, 0.) for seq in expected} == pytest.approx(expected, abs=1e-10, rel=0)
    for d in range(1, length + 1):
        full, no_fill, base = [sum(p for seq, p in law.items() if len(seq) > d) for law in laws]
        assert full >= no_fill - 1e-10 >= base - 2e-10
        assert no_fill >= float(probability(rows, tuple(greedy[:d].tolist()))) - 1e-10


@pytest.mark.parametrize("seed", range(12))
def test_same_budget_counterexample_is_repaired_and_fixed_filler_is_retained(seed):
    q0, p0, length, budget = F(51, 100), F(99, 100), 15, 45
    q = tensor([[q0, 1 - q0]] * length)
    law = diffusion.DiffusionBlockLaw.from_logits(q.log(), torch.tensor([0] + [1] * length), 1., 1)
    proposal = diffusion.propose(law, 1, torch.Generator().manual_seed(seed))
    tree = diffusion.scaffold_tree(proposal, q.argmax(-1), budget)
    guaranteed = set(prefixes(probability_tree(q, budget - 2 * length)))
    guaranteed.update((0,) * d for d in range(1, length + 1))
    assert guaranteed <= set(prefixes(tree))
    assert len(tree.tokens) == budget and max(tree.depths) == length
    # Exact fixed-scaffold lower bound, not a sample-average estimate.
    lower = sum(p0 ** prefix.count(0) * (1 - p0) ** prefix.count(1)
                for prefix in guaranteed if prefix)
    assert lower == 3 + sum(p0 ** d for d in range(4, length + 1))
    assert lower > geometric_tail(p0) > F(9999786239, 2000000000)
    assert float(lower) == pytest.approx(13.913823890512435)


@pytest.mark.parametrize("temperature", [.3, .6, 1.])
@pytest.mark.parametrize("branches", [1, 2])
def test_exact_diffusion_keeps_full_block_acceptance(temperature, branches):
    law = diffusion.DiffusionBlockLaw.from_logits(
        torch.zeros((15, 2)).double(), torch.tensor([0] + [1] * 15), temperature, 1)
    proposal = diffusion.propose(law, branches, torch.Generator().manual_seed(37))
    tree = diffusion.scaffold_tree(proposal, torch.zeros(15).long(), 45)
    _, accepted, _ = diffusion.verify_scaffold_logits(
        torch.zeros((len(tree.parents), 2)).double(), tree, proposal, temperature)
    assert len(accepted) == 15


@pytest.mark.parametrize("temperature", [.3, .6, 1.])
@pytest.mark.parametrize("method", sorted(DIFFUSION_SCAFFOLD_METHODS))
@pytest.mark.parametrize("branches", [1, 2])
def test_scaffold_tiny_qwen_caches_eos_caps_and_checkpoint(tiny_engine, temperature, method, branches):
    from gbv_experiments.diffusion_preflight import probe

    v = Variant(name=method, method=method, paths=branches, length=3, tree_budget=9,
                temperature=temperature, draft_temperature=temperature)
    ids = torch.tensor([[1, 3, 5]])
    result = tiny_engine.generate(ids, v, 24, [], seed=42)
    assert result["target_forward_calls"] == 1 + result["draft_forward_calls"]
    assert result["draft_forward_calls"] == len(result["rounds"])
    assert all(r["tree_nodes"] <= 9 and r["fixed_greedy_scaffold"] for r in result["rounds"])
    assert tiny_engine.generate(ids, replace(v, reuse_draft_cache=False), 24, [], seed=42)["generated_token_ids"] == result["generated_token_ids"]
    for cap in (1, 2, 7):
        assert tiny_engine.generate(ids, v, cap, [], seed=42)["generated_token_ids"] == result["generated_token_ids"][:cap]
    eos = result["generated_token_ids"][5]
    end = result["generated_token_ids"].index(eos) + 1
    assert tiny_engine.generate(ids, v, 24, [eos], seed=42)["generated_token_ids"] == result["generated_token_ids"][:end]
    assert probe(tiny_engine, ids, v, tokens=18, seed=42, tv_limit=1e-6)["passed"]


@pytest.mark.parametrize("corruption", ["nan_leaf", "duplicate", "bad_parent", "bad_depth"])
def test_scaffold_rejects_invalid_unused_nodes(corruption):
    law = diffusion.DiffusionBlockLaw.from_logits(tensor([[.7, .3]]).log(), torch.tensor([0, 1]), 1., 1)
    proposal = diffusion.propose(law, 1, torch.Generator().manual_seed(1))
    tree = diffusion.scaffold_tree(proposal, torch.tensor([0]), 3)
    logits = torch.zeros((len(tree.parents), 2)).double()
    if corruption == "nan_leaf":
        logits[-1, 0] = float("nan")
    elif corruption == "duplicate":
        tree.tokens[-1] = tree.tokens[0]
    elif corruption == "bad_parent":
        tree.parents[-1] = len(tree.parents)
    else:
        tree.depths[-1] += 1
    with pytest.raises(ValueError):
        diffusion.verify_scaffold_logits(logits, tree, proposal, 1.)


@pytest.mark.parametrize("tag", ["t03", "t06", "t10"])
def test_new_study_identity_and_capture_replay(tiny_engine, tag):
    from gbv_experiments.common import ROOT
    from gbv_experiments.terminal_formal import UNIT_FILES, capture_states, replay_functions
    from gbv_experiments.terminal_protocol import load_study, variants

    study = load_study(ROOT / f"configs/diffusion_scaffold_{tag}.json")
    declared = variants(study)
    assert len(declared) == 10
    assert next(v for v in declared if v.name == "diffusion_full").method == "diffusion_scaffold_bv"
    assert next(v for v in declared if v.name == "diffusion_old").paths == 3
    # Small real-model replay while retaining the full registered method set.
    for entry in study["config"]["explicit_variants"]:
        entry["variant"].update(length=3, tree_budget=9)
    states = capture_states(study, tiny_engine, torch.tensor([[1, 3, 5]]), seed=12, limit=1, tokens=12)
    state = next(s for s in states if s["kind"] == "scaffold_diffusion_trie")
    assert state["scaffold_witness"]["checks"]["maximal_exit_checked"]
    funcs = replay_functions(state)
    for func in funcs.values():
        assert func(torch.Generator().manual_seed(8)) == func(torch.Generator().manual_seed(8))
    missing = deepcopy(state)
    del missing["scaffold_witness"]
    with pytest.raises(ValueError, match="recapture"):
        replay_functions(missing)
    invalid = deepcopy(state)
    invalid["scaffold_witness"]["greedy_tokens"] = [-1] * 3
    with pytest.raises(ValueError, match="greedy path is missing"):
        replay_functions(invalid)
    assert "tests/gbv_paper/test_diffusion_scaffold.py" in UNIT_FILES


@pytest.mark.parametrize("tag,temperature", [("t03", .3), ("t06", .6), ("t10", 1.)])
def test_qwen3_4b_registration_is_separate_and_matched(tag, temperature):
    from gbv_experiments.common import ROOT
    from gbv_experiments.terminal_protocol import load_study, variants

    study = load_study(ROOT / f"configs/diffusion_scaffold_qwen3_4b_{tag}.json")
    assert study["model_id"] == "qwen3_4b"
    assert study["config"]["model"]["target"] == "Qwen/Qwen3-4B"
    assert {v.temperature for v in variants(study)} == {temperature}
    assert len(variants(study)) == 10


def test_scaffold_budget_and_sharing_contract():
    with pytest.raises(ValueError):
        Variant(name="bad", method="diffusion_scaffold_bv", paths=3, length=15, tree_budget=45).validate()
    with pytest.raises(ValueError):
        Variant(name="bad", method="diffusion_scaffold_bv", paths=1, share_prefixes=False).validate()


def test_filler_tie_order_is_independent_of_random_node_numbers():
    law = diffusion.DiffusionBlockLaw.from_logits(torch.zeros((3, 3)).double(), torch.tensor([0, 2, 2, 2]), 1., 2)
    ordered = sorted((prefix for d in range(1, 4) for prefix in product(range(3), repeat=d)),
                     key=lambda prefix: (-F(1, 3) ** len(prefix), prefix))
    fixed = set(ordered[:9])  # B15 - two mandatory length-three paths.
    for seed in range(24):
        proposal = diffusion.propose(law, 1, torch.Generator().manual_seed(seed))
        tree = diffusion.scaffold_tree(proposal, torch.zeros(3).long(), 15)
        assert fixed <= set(prefixes(tree))


def test_greedy_path_can_be_outside_truncated_support_on_ties():
    law = diffusion.DiffusionBlockLaw.from_logits(torch.zeros((2, 17)).double(), torch.tensor([0, 16, 16]), 1., 16,
                                                 support_size=1)
    greedy_token = next(x for x in range(17) if x != int(law.tokens[0, 0]))
    # Every logit ties; the passed baseline's tie rule, not top-k ordering, wins.
    proposal = diffusion.propose(law, 1)
    tree = diffusion.scaffold_tree(proposal, torch.tensor([greedy_token] * 2), 6)
    assert (greedy_token, greedy_token) in prefixes(tree)


def rational_scaffold_law(rows, q, atoms, greedy, budget, *, fill):
    """Independent exact-rational tree/set construction and target continuation.

    No production scaffold builder, float probabilities, terminal-mass kernel,
    or Monte Carlo frequencies are used by this oracle.
    """
    length, vocab, branches = len(q), len(q[0]), len(atoms[0][0][1])
    ordered = sorted((seq for d in range(1, length + 1) for seq in product(range(vocab), repeat=d)
                      if product_probability(q, seq)),
                     key=lambda seq: (-product_probability(q, seq), seq))
    result = defaultdict(F)
    for draws in product(*(range(len(row)) for row in atoms)):
        mass = F(1)
        for j, draw in enumerate(draws):
            mass *= atoms[j][draw][0]
        present = {tuple(greedy[:d]) for d in range(1, length + 1)}
        present.update(tuple(atoms[j][draws[j]][1][a] for j in range(d))
                       for a in range(branches) for d in range(1, length + 1))
        assert len(present) <= budget
        if fill:
            for seq in ordered:
                if len(present) == budget:
                    break
                present.add(seq)
        # Keep the dependence between the proposal witness, initial BV output,
        # and random scaffold. Only appended target draws are fresh.
        queue = list(rational_kernel(rows, atoms, draws, pool=True).items())
        while queue:
            seq, conditional = queue.pop()
            if seq in present:
                queue.extend((seq + (token,), conditional * p)
                             for token, p in enumerate(rows[seq]) if p)
            else:
                result[seq] += mass * conditional
    return dict(result)


@pytest.mark.parametrize("case", range(6))
@pytest.mark.parametrize("fill", [False, True])
def test_independent_rational_scaffold_law_variable_positions(monkeypatch, case, fill):
    length, branches, temperature = 3, 1 + case % 2, [.3, .6, 1.][case % 3]
    q = [[F(2, 3), F(1, 3)], [F(1, 4), F(3, 4)], [F(3, 5), F(2, 5)]]
    rows = fraction_rows(Random(202609070 + case), 2, length)
    if case == 0:
        rows = {prefix: [F(0), F(1)] for prefix in rows}
    elif case == 1:
        q = [[F(1), F(0)], [F(0), F(1)], [F(1, 2), F(1, 2)]]
    elif case == 2:
        rows = {prefix: q[len(prefix) % length] for prefix in rows}
    shifts = [[F((a * (j + 1)) % branches, branches) for a in range(branches)]
              for j in range(length)]
    atoms = [rational_atoms(q[j], shifts[j]) for j in range(length)]
    greedy = [max(range(2), key=q[j].__getitem__) for j in range(length)]
    budget = (branches + 1) * length
    expected = rational_scaffold_law(rows, q, atoms, greedy, budget, fill=fill)
    assert sum(expected.values()) == 1
    completed = defaultdict(F)
    for output, mass in expected.items():
        for suffix in product(range(2), repeat=length + 1 - len(output)):
            seq = output + suffix
            completed[seq] += mass * probability(rows, seq, start=len(output))
    assert all(completed[seq] == probability(rows, seq)
               for seq in product(range(2), repeat=length + 1))
    for depth in range(1, length + 1):
        assert sum(p for seq, p in expected.items() if len(seq) > depth) >= probability(rows, tuple(greedy[:depth]))
    actual = enumerate_scaffold(monkeypatch, actual_proposal(q, atoms, temperature), rows,
                                torch.tensor(greedy), budget, fill=fill, temperature=temperature)
    for seq in set(expected) | set(actual):
        assert actual.get(seq, 0.) == pytest.approx(float(expected.get(seq, 0)), abs=1e-11, rel=0)


@pytest.mark.parametrize("temperature", [.3, .6, 1.])
def test_positive_scaffold_counterexample_exact_output_law(monkeypatch, temperature):
    """The NEW scaffold can lose to DDTree while staying exact and beating DFlash."""
    q = [[F(51, 100), F(49, 100)]] * 3
    rows = {prefix: ([F(99, 100), F(1, 100)] if len(prefix) % 2 == 0
                     else [F(1, 100), F(99, 100)])
            for depth in range(4) for prefix in product(range(2), repeat=depth)}
    atoms = [rational_atoms(row, [F(0)]) for row in q]
    expected = rational_scaffold_law(rows, q, atoms, [0, 0, 0], 6, fill=True)
    accepted = sum((len(seq) - 1) * mass for seq, mass in expected.items())
    dflash = sum(probability(rows, (0,) * depth) for depth in range(1, 4))
    ddtree = sum(probability(rows, prefix) for prefix in prefixes(probability_tree(tensor(q), 6))[1:])
    assert dflash < accepted == F(8373, 5000) < ddtree == 2
    completed = defaultdict(F)
    for output, mass in expected.items():
        for suffix in product(range(2), repeat=4 - len(output)):
            seq = output + suffix
            completed[seq] += mass * probability(rows, seq, start=len(output))
    assert all(completed[seq] == probability(rows, seq) for seq in product(range(2), repeat=4))
    actual = enumerate_scaffold(monkeypatch, actual_proposal(q, atoms, temperature), rows,
                                torch.zeros(3, dtype=torch.long), 6, temperature=temperature)
    for seq in set(expected) | set(actual):
        assert actual.get(seq, 0.) == pytest.approx(float(expected.get(seq, 0)), abs=1e-11, rel=0)


@pytest.mark.parametrize("temperature", [.3, .6, 1.])
def test_registered_scaffold_coverage_counterexample_all_32768_paths(temperature):
    """Check the actual L15/K1/B45 builder on EVERY possible binary draft.

    Fractions give a target/proposal coupling-independent upper bound, not a
    Monte Carlo average or an exponential enumeration of full target outputs.
    """
    length, budget, q0, q1, p1 = 15, 45, F(51, 100), F(49, 100), F(999, 1000)
    assert q0 ** 5 < q1 ** 4 and q0 ** 6 < q1 ** 5
    law = diffusion.DiffusionBlockLaw.from_logits(
        tensor([[q0, q1]] * length).log() * temperature,
        torch.tensor([0] + [1] * length), temperature, 1)
    proposal = diffusion.propose(law, 1, torch.Generator().manual_seed(17))
    # Resolve atom labels instead of assuming the padding columns' positions.
    atom_ids = [next(h for h, s in enumerate(proposal.source[0])
                     if s > 0 and int(proposal.tokens[0, 0, proposal.slots[0, 0, h]]) == token)
                for token in range(2)]
    greedy = torch.zeros(length, dtype=torch.long)
    targets = [(1,) * depth for depth in range(1, length + 1)]
    counts = [Counter() for _ in targets]
    for path in product(range(2), repeat=length):
        current = replace(proposal, draws=torch.tensor([atom_ids[token] for token in path]))
        tree = diffusion.scaffold_tree(current, greedy, budget)
        present = set(prefixes(tree))
        assert len(tree.tokens) == budget and current.paths().tolist() == [list(path)]
        for depth, prefix in enumerate(targets, 1):
            covered = prefix in present
            if depth <= 3:
                expected = True
            elif depth == 4:
                expected = path[:4] == (1,) * 4 or path[:11] == (0,) * 11
            else:
                expected = path[:depth] == prefix
            assert covered == expected
            if covered:
                counts[depth - 1][path.count(0)] += 1
    coverage = [sum(count * q0 ** zeros * q1 ** (length - zeros)
                    for zeros, count in counter.items()) for counter in counts]
    assert coverage[:3] == [1, 1, 1]
    assert coverage[3] == q1 ** 4 + q0 ** 11
    assert coverage[4:] == [q1 ** depth for depth in range(5, length + 1)]
    # No false independence assumption: Pr(D>=d) <= Pr(prefix in random tree)
    # + Pr(target prefix differs), for ANY coupling with the exact target law.
    upper = sum(min(F(1), cov + 1 - p1 ** depth)
                for depth, cov in enumerate(coverage, 1))
    assert upper == F(3227066589282414577190211036982819440119984001, 10 ** 45)
    assert float(upper) == pytest.approx(3.2270665892824146)
    dd = probability_tree(tensor([[q0, q1]] * length), budget)
    assert Counter(dd.depths[1:]) == {1: 2, 2: 4, 3: 8, 4: 16, 5: 15}
    assert upper < 4 <= sum(p1 ** prefix.count(1) * (1 - p1) ** prefix.count(0)
                            for prefix in prefixes(dd)[1:])
