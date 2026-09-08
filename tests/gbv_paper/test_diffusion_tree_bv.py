"""Full joint-law enumeration, diffusion identities, and real-model trie checks."""
from collections import defaultdict
from dataclasses import replace
from fractions import Fraction as F
from itertools import product
import math
from random import Random

import pytest
import torch

from gbv_experiments import diffusion_tree_bv as diffusion
from gbv_experiments.config import DIFFUSION_TREE_METHODS, Variant
from gbv_experiments.tree import sampled_tree


def tensor(x):
    return torch.tensor(x, dtype=torch.float64)


def fraction_rows(rng, vocab, length):
    rows = {}
    for d in range(length + 1):
        for prefix in product(range(vocab), repeat=d):
            counts = [rng.randrange(5) for _ in range(vocab)]
            counts = counts if sum(counts) else [1] * vocab
            rows[prefix] = [F(v, sum(counts)) for v in counts]
    return rows


def probability(rows, sequence, start=0):
    result = F(1)
    for j in range(start, len(sequence)):
        result *= rows[sequence[:j]][sequence[j]]
    return result


def complete(law, rows, horizon):
    result = defaultdict(float)
    stack = list(law.items())
    while stack:
        prefix, mass = stack.pop()
        if len(prefix) == horizon:
            result[prefix] += mass
        else:
            for token, p in enumerate(rows[prefix]):
                if p:
                    stack.append((prefix + (token,), mass * float(p)))
    return result


def enumerate_sampler(monkeypatch, proposal, rows, *, pool, temperature):
    """Enumerate the ACTUAL endpoint and correction draws, not a copied recurrence."""
    result = defaultdict(float)

    class DrawNeeded(Exception):
        def __init__(self, probs):
            self.probs = probs

    length, columns = proposal.source.shape
    supports = [torch.nonzero(row > 0).flatten().tolist() for row in proposal.source]
    for indices in product(*supports):
        current = replace(proposal, draws=torch.tensor(indices))
        latent_mass = float(current.latent_log_probability().exp())
        tree = sampled_tree(current.paths())
        prefixes = [()]
        for n in range(1, len(tree.parents)):
            prefixes.append(prefixes[tree.parents[n]] + (tree.tokens[n - 1],))
        logits = tensor([[float(p) for p in rows[prefix]] for prefix in prefixes]).log() * temperature
        queue = [((), 1.)]
        while queue:
            choices, mass = queue.pop()
            used = []

            def draw(probs, generator=None):
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
                    _, tokens, bonus = diffusion.verify_logits(logits, tree, current, temperature, pool=pool)
                except DrawNeeded as draw_needed:
                    for index, p in enumerate(draw_needed.probs):
                        if p > 1e-15:
                            queue.append((choices + (index,), mass * p))
                    continue
            assert len(used) == 2
            result[tuple(tokens) + (bonus,)] += latent_mass * mass
    return dict(result)


@pytest.mark.parametrize("case", range(12))
def test_full_joint_law_and_diffusion_bv_bound(monkeypatch, case):
    rng = Random(16001 + case)
    vocab, length, branches = 2 + case % 2, 1 + case % 3, 1 + case % 3
    # Includes target-zero tokens and proposal truncation, with full residuals.
    rows = fraction_rows(rng, vocab, length + 1)
    q = tensor([[1 + rng.randrange(4) for _ in range(vocab)] for _ in range(length)])
    q /= q.sum(-1, keepdim=True)
    temperature = [0.3, 0.6, 1.0][case % 3]
    law = diffusion.DiffusionBlockLaw.from_logits(
        q.log() * temperature, torch.tensor([0] + [vocab - 1] * length), temperature,
        vocab - 1, support_size=1 + case % vocab)
    proposal = diffusion.propose(law, branches, torch.Generator().manual_seed(case),
                                 coupling="aligned" if case % 4 == 0 else "depth_permuted")
    proposal.validate(vocab)
    expected = {seq: float(probability(rows, seq))
                for seq in product(range(vocab), repeat=length + 1)}
    laws = []
    for pool in (False, True):
        actual = enumerate_sampler(monkeypatch, proposal, rows, pool=pool, temperature=temperature)
        assert sum(actual.values()) == pytest.approx(1., abs=1e-11)
        completed = complete(actual, rows, length + 1)
        assert {seq: completed.get(seq, 0.) for seq in expected} == pytest.approx(expected, abs=1e-10, rel=0)
        laws.append(actual)
    # Independent cut-measure formula (13), no calls to transport.plan.
    dense_q = torch.zeros((length, vocab), dtype=torch.float64).scatter(1, law.tokens, law.weights)
    for d in range(1, length + 1):
        bound = 0.
        for seq in product(range(vocab), repeat=d):
            cuts = []
            q_prefix = 1.
            for k in range(d + 1):
                cuts.append(q_prefix * float(probability(rows, seq, start=k)))
                if k < d:
                    q_prefix *= float(dense_q[k, seq[k]])
            bound += min(cuts)
        no_pool, full = [sum(p for seq, p in item.items() if len(seq) >= d + 1) for item in laws]
        assert no_pool == pytest.approx(bound, abs=1e-10)
        assert full >= no_pool - 1e-10


def test_one_step_diffusion_joint_law_and_exact_truncation_tv():
    q = tensor([[.5, .3, .2], [.2, .2, .6]])
    law = diffusion.DiffusionBlockLaw.from_logits(q.log(), torch.tensor([0, 2, 2]), 1., 2, support_size=2)
    proposal = diffusion.propose(law, 3, torch.Generator().manual_seed(7))
    proposal.validate(3)
    expected_q = torch.zeros_like(q).scatter(1, law.tokens, law.weights)
    marginals = [defaultdict(float) for _ in range(3)]
    for indices in product(*[torch.nonzero(row > 0).flatten().tolist() for row in proposal.source]):
        current = replace(proposal, draws=torch.tensor(indices))
        mass = float(current.latent_log_probability().exp())
        for a, path in enumerate(current.paths().tolist()):
            marginals[a][tuple(path)] += mass
    tv = 0.
    for seq in product(range(3), repeat=2):
        full = float(q[0, seq[0]] * q[1, seq[1]])
        truncated = float(expected_q[0, seq[0]] * expected_q[1, seq[1]])
        tv += abs(full - truncated) / 2
        for marginal in marginals:
            assert marginal.get(seq, 0.) == pytest.approx(truncated, abs=1e-12)
    assert tv == pytest.approx(1 - float(law.retained_block_mass()))


@pytest.mark.parametrize("case", ["exact", "close", "truncated", "zero_q"])
def test_denoiser_cross_entropy_acceptance_bound(case):
    """Independent finite sums for (15)-(16), including a nontrivial >0 bound."""
    q = tensor([[.8, .2], [.4, .6]])
    if case == "zero_q":
        q[0] = tensor([1., 0.])
    rows = {}
    for d in range(2):
        for prefix in product(range(2), repeat=d):
            p = [float(value) for value in q[d]]
            if case != "exact":
                change = -.02 if p[0] > .5 else .01
                p = [p[0] + change, p[1] - change]
            rows[prefix] = p
    law = diffusion.DiffusionBlockLaw.from_logits(q.log(), torch.tensor([0, 1, 1]),
                                                 1., 1, support_size=1 if case == "truncated" else 2)
    truncated = torch.zeros_like(q).scatter(1, law.tokens, law.weights)
    tv_sum, penalty_sum = 0., 0.
    for d in (1, 2):
        tv, kl, entropy, denoising_cross_entropy, acceptance = 0., 0., 0., 0., 0.
        for seq in product(range(2), repeat=d):
            p = float(probability(rows, seq))
            full = math.prod(float(q[j, seq[j]]) for j in range(d))
            sparse = math.prod(float(truncated[j, seq[j]]) for j in range(d))
            tv += abs(p - sparse) / 2
            if p:
                entropy -= p * math.log(p)
                kl += p * math.log(p / full) if full else math.inf
                denoising_cross_entropy += (p * sum(-math.log(float(q[j, seq[j]])) for j in range(d))
                                            if full else math.inf)
            acceptance += min(math.prod(float(truncated[j, seq[j]]) for j in range(k))
                              * float(probability(rows, seq, start=k)) for k in range(d + 1))
        if math.isfinite(kl):
            assert kl == pytest.approx(denoising_cross_entropy - entropy, abs=1e-12)
        else:
            assert math.isinf(denoising_cross_entropy)
        penalty = math.sqrt(max(0., kl) / 2) + 1 - float(law.retained_mass[:d].prod())
        assert tv <= penalty + 1e-12
        tv_sum += tv
        penalty_sum += penalty
        assert acceptance >= max(0., 1 - tv_sum) - 1e-12
        assert acceptance >= max(0., 1 - penalty_sum) - 1e-12
        if case in {"exact", "close"}:
            assert 1 - penalty_sum > .8


def test_random_first_token_and_rebranching_trie_are_not_distinct_root_chains():
    q = tensor([[.9, .1], [.5, .5], [.5, .5]])
    law = diffusion.DiffusionBlockLaw.from_logits(q.log(), torch.tensor([0, 1, 1, 1]), 1., 1)
    proposal = diffusion.propose(law, 2)
    supported = [torch.nonzero(row > 0).flatten().tolist() for row in proposal.source]
    found = False
    for indices in product(*supported):
        current = replace(proposal, draws=torch.tensor(indices))
        paths = current.paths()
        if paths[0, 0] == paths[1, 0] and paths[0, 1] != paths[1, 1]:
            tree = sampled_tree(paths)
            assert tree.path_nodes[0][0] == tree.path_nodes[1][0]
            assert len(tree.tokens) < paths.numel()
            assert tree.path_nodes[0][1] != tree.path_nodes[1][1]
            found = True
            break
    assert found


@pytest.mark.parametrize("temperature", [0.3, 0.6, 1.0])
@pytest.mark.parametrize("method", sorted(DIFFUSION_TREE_METHODS))
def test_real_qwen_positive_sampling_cache_eos_caps(tiny_engine, temperature, method):
    ids = torch.tensor([[1, 3, 5]])
    variant = Variant(name=method, method=method, paths=3, length=3, tree_budget=9,
                      temperature=temperature, draft_temperature=temperature)
    expected = tiny_engine.generate(ids, variant, 23, [], seed=4)
    assert expected["draft_forward_calls"] == len(expected["rounds"])
    assert expected["target_forward_calls"] == 1 + len(expected["rounds"])
    assert all(r["tree_nodes"] <= 9 and r["denoising_steps"] == 1 for r in expected["rounds"])
    for alternative in (variant, replace(variant, reuse_draft_cache=False), replace(variant, share_prefixes=False)):
        assert tiny_engine.generate(ids, alternative, 23, [], seed=4)["generated_token_ids"] == expected["generated_token_ids"]
    first = expected["generated_token_ids"][0]
    stopped = tiny_engine.generate(ids, variant, 23, [first], seed=4)
    assert stopped["generated_token_ids"] == [first] and stopped["draft_forward_calls"] == 0
    for cap in (1, 2, 7):
        capped = tiny_engine.generate(ids, variant, cap, [], seed=4)
        assert capped["generated_token_ids"] == expected["generated_token_ids"][:cap]
    eos = expected["generated_token_ids"][5]
    stopped = tiny_engine.generate(ids, variant, 23, [eos], seed=4)
    end = expected["generated_token_ids"].index(eos) + 1
    assert stopped["generated_token_ids"] == expected["generated_token_ids"][:end]


def test_positive_checkpoint_probe_on_real_qwen(tiny_engine):
    from gbv_experiments.diffusion_preflight import probe
    v = Variant(name="diffusion_full", method="diffusion_tree_bv", length=3, paths=3, tree_budget=9)
    result = probe(tiny_engine, torch.tensor([[1, 3, 5]]), v, tokens=16, tv_limit=1e-6)
    assert result["passed"] and result["max_node_total_variation"] < 1e-6
    assert result["captured_rounds"] == 3
    assert result["round_checks"][1]["generated_prefix_tokens"] > 1


def test_positive_probe_detects_later_round_target_corruption(tiny_engine, monkeypatch):
    from gbv_experiments.diffusion_preflight import probe
    original = tiny_engine.target_forward
    calls = []

    def corrupted(*args, **kwargs):
        output = original(*args, **kwargs)
        if kwargs.get("mask") is not None:
            calls.append(True)
            if len(calls) == 2:
                output.logits[..., 0] += 10
        return output

    monkeypatch.setattr(tiny_engine, "target_forward", corrupted)
    v = Variant(name="diffusion_full", method="diffusion_tree_bv", length=3, paths=3, tree_budget=9)
    result = probe(tiny_engine, torch.tensor([[1, 3, 5]]), v, tokens=16, tv_limit=1e-6)
    assert not result["passed"]
    assert result["round_checks"][0]["max_node_total_variation"] < 1e-6
    assert result["round_checks"][1]["max_node_total_variation"] > 1e-3


def test_full_depth_numerical_probe_cpu():
    from gbv_experiments.diffusion_preflight import numerical_probe
    result = numerical_probe("cpu")
    assert result["passed"] and result["cases"] == 16
    assert result["max_absolute_error"] == 0 and result["stochastic_first_token"]


def test_tiny_positive_temperature_does_not_overflow():
    law = diffusion.DiffusionBlockLaw.from_logits(tensor([[1e308, -1e308]]),
                                                 torch.tensor([0, 1]), 1e-320, 1)
    proposal = diffusion.propose(law, 3, torch.Generator().manual_seed(1))
    proposal.validate(2)
    assert torch.equal(law.weights, tensor([[1., 0.]]))
    tree = sampled_tree(proposal.paths())
    logits = tensor([[-1e308, 1e308]] * len(tree.parents))
    _, tokens, bonus = diffusion.verify_logits(logits, tree, proposal, 1e-320)
    assert tokens == [] and bonus == 1


def test_cli_temperature_specific_default_output(monkeypatch, capsys):
    from gbv_experiments import terminal_formal as formal
    from gbv_experiments.common import ROOT
    for tag in ("t03", "t06", "t10"):
        outputs = []
        monkeypatch.setattr(formal, "status", lambda study, output: outputs.append(output) or {})
        monkeypatch.setattr("sys.argv", ["terminal_formal", "status", "--study",
                                        str(ROOT / f"configs/diffusion_tree_{tag}.json")])
        formal.main()
        assert outputs == [ROOT / f"outputs/diffusion_tree_{tag}"]
    capsys.readouterr()


def test_snapshot_replay_and_reject_inconsistent_diffusion_law():
    law = diffusion.DiffusionBlockLaw.from_logits(tensor([[0., 1.], [1., 0.]]),
                                                 torch.tensor([0, 1, 1]), .6, 1)
    proposal = diffusion.propose(law, 3, torch.Generator().manual_seed(17))
    restored = diffusion.restore(diffusion.snapshot(proposal))
    assert torch.equal(restored.paths(), proposal.paths())
    bad_law = replace(law, weights=law.weights.flip(-1))
    with pytest.raises(ValueError, match="marginals"):
        replace(proposal, law=bad_law).validate(2)
    with pytest.raises(ValueError, match="masked"):
        diffusion.DiffusionBlockLaw.from_logits(tensor([[1., 2.]]), torch.tensor([0, 0]), 1., 1)
    with pytest.raises(ValueError, match="positive"):
        diffusion.verify_logits(torch.zeros(1, 2), sampled_tree(proposal.paths()), proposal, 0.)
    for override in ({"temperature": 0}, {"condition_features": "zero"}, {"draft_attention": "causal"}):
        with pytest.raises(ValueError):
            Variant(name="bad", method="diffusion_tree_bv", **override).validate()


def test_diffusion_formal_temperatures_and_replay(tiny_engine):
    from gbv_experiments.common import ROOT
    from gbv_experiments.terminal_protocol import load_study, plan, primary_candidate
    from gbv_experiments.terminal_formal import capture_states, replay_functions
    for tag, temperature in (("t03", .3), ("t06", .6), ("t10", 1.)):
        study = load_study(ROOT / f"configs/diffusion_tree_{tag}.json")
        assert primary_candidate(study) == "diffusion_full"
        assert plan(study)["expected_records"] == 786 * 11 * 9
        assert plan(study)["temperature"] == [temperature]
        for entry in study["config"]["explicit_variants"]:
            assert entry["variant"]["temperature"] == temperature
            entry["variant"]["length"] = 3
            entry["variant"]["tree_budget"] = 9
        states = capture_states(study, tiny_engine, torch.tensor([[1, 3, 5]]), 17, limit=1, tokens=8)
        state = states[1]
        assert state["kind"] == "one_step_diffusion_trie"
        assert diffusion.restore(state).law.draft_temperature == temperature
        functions = replay_functions(state)
        assert set(functions) == {"diffusion_full", "diffusion_no_pool", "diffusion_ancestral"}
        for fn in functions.values():
            assert fn(torch.Generator().manual_seed(7)) == fn(torch.Generator().manual_seed(7))
