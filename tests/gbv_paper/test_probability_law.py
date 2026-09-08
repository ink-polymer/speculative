from collections import defaultdict
from fractions import Fraction as F
from itertools import product

import pytest
import torch

from gbv_experiments import sampling
from gbv_experiments.tree import AdaptivePathProposal, sampled_tree


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


AR_Q = {
    (): [F(2, 5), F(3, 5)],
    (0,): [F(4, 5), F(1, 5)],
    (1,): [F(1, 4), F(3, 4)],
}


def ar_mass(path):
    result, prefix = F(1), ()
    for token in path:
        result *= AR_Q[prefix][token]
        prefix += (token,)
    return result


def ar_order(path):
    prefix = ()
    result = []
    for token in path:
        result.append((target(prefix)[token] / AR_Q[prefix][token], token))
        prefix += (token,)
    return tuple(result)


def ar_rows(path):
    prefix, rows = (), []
    for token in path:
        rows.append(AR_Q[prefix])
        prefix += (token,)
    return tensor(rows)


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
def test_branch_conditional_gbv_against_fraction_enumeration(k):
    selected_distribution = defaultdict(F)
    for candidates in product(PATHS, repeat=k):
        chosen = max(candidates, key=ar_order)
        probability = F(1)
        for candidate in candidates:
            probability *= ar_mass(candidate)
        selected_distribution[chosen] += probability
    assert sum(selected_distribution.values()) == 1

    for candidates in product(PATHS, repeat=k):
        paths = torch.tensor(candidates)
        target_by_path = torch.stack([
            tensor([target(path[:i]) for i in range(3)]) for path in candidates
        ])
        q_by_path = torch.stack([ar_rows(path) for path in candidates])
        chosen, correction = sampling.select_and_reweight_autoregressive(
            paths, target_by_path, q_by_path
        )
        path = candidates[chosen]
        assert path == max(candidates, key=ar_order)
        for depth in range(2):
            prefix_mass = sum(
                weight for other, weight in selected_distribution.items()
                if other[:depth] == path[:depth]
            )
            expected = [
                sum(
                    weight for other, weight in selected_distribution.items()
                    if other[:depth] == path[:depth] and other[depth] == token
                ) / prefix_mass
                for token in range(2)
            ]
            torch.testing.assert_close(
                correction[depth],
                torch.tensor([float(x) for x in expected], dtype=torch.float64),
                rtol=1e-12,
                atol=1e-13,
            )


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


class NeedSampleBranch(Exception):
    def __init__(self, options):
        self.options = options


def sparse_block_verify_from_dense(path, p, r, generator=None):
    width = max(int(row.count_nonzero()) for row in r)
    tokens = torch.zeros((r.shape[0], width), dtype=torch.long, device=r.device)
    probabilities = torch.zeros_like(tokens, dtype=r.dtype)
    for depth, row in enumerate(r):
        indices = row.nonzero()[:, 0]
        tokens[depth, :indices.numel()] = indices
        probabilities[depth, :indices.numel()] = row[indices]
    return sampling.block_verify_sparse(path, p, tokens, probabilities, generator)


def lazy_sparse_block_verify_from_dense(path, p, r, generator=None):
    width = max(int(row.count_nonzero()) for row in r)
    tokens = torch.zeros((r.shape[0], width), dtype=torch.long, device=r.device)
    probabilities = torch.zeros_like(tokens, dtype=r.dtype)
    for depth, row in enumerate(r):
        indices = row.nonzero()[:, 0]
        tokens[depth, :indices.numel()] = indices
        probabilities[depth, :indices.numel()] = row[indices]
    return sampling.block_verify_sparse_lazy(
        path, p, tokens, probabilities, generator
    )


def enumerate_lazy_verifier_law(monkeypatch, selected_paths, build_rows):
    """Enumerate every Bernoulli row and final correction-token outcome."""
    law = defaultdict(float)
    for path, path_probability in selected_paths.items():
        p, r = build_rows(path)
        row_count, vocab = p.shape
        for binary_choices in product(range(2), repeat=row_count):
            for correction_token in range(vocab):
                probability, call = 1.0, 0

                def forced(weights, generator=None):
                    nonlocal probability, call
                    if call == 0:
                        assert weights.shape == (row_count, 2)
                        indices = torch.tensor(binary_choices, device=weights.device)
                        for row, index in zip(weights, binary_choices):
                            if row[index] == 0:
                                raise Impossible
                            probability *= float(row[index])
                        call += 1
                        return indices
                    assert call == 1 and weights.shape == (vocab,)
                    if weights[correction_token] == 0:
                        raise Impossible
                    probability *= float(weights[correction_token])
                    call += 1
                    return torch.tensor(correction_token, device=weights.device)

                monkeypatch.setattr(sampling, "sample", forced)
                try:
                    accepted, bonus = lazy_sparse_block_verify_from_dense(
                        torch.tensor(path), p, r
                    )
                except Impossible:
                    continue
                assert call == 2
                emitted = path[:accepted] + (bonus,)
                for tail in product(range(vocab), repeat=row_count-len(emitted)):
                    sequence, weight = emitted, float(path_probability) * probability
                    for token in tail:
                        weight *= float(target(sequence)[token])
                        sequence += (token,)
                    law[sequence] += weight
    return law


@pytest.mark.parametrize(
    "verifier",
    [sampling.block_verify, sampling.block_verify_batched, sparse_block_verify_from_dense],
    ids=["reference", "batched", "sparse"],
)
@pytest.mark.parametrize("k", [1, 3, 4])
def test_actual_block_verifier_full_output_law(monkeypatch, k, verifier):
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
                if weights.ndim == 1:
                    index = choices[cursor]
                    cursor += 1
                    if index >= weights.numel() or weights[index] == 0:
                        raise Impossible
                    probability *= float(weights[index])
                    return torch.tensor(index)
                indices = torch.tensor(choices)
                for row, index in zip(weights, choices):
                    if index >= row.numel() or row[index] == 0:
                        raise Impossible
                    probability *= float(row[index])
                return indices

            monkeypatch.setattr(sampling, "sample", forced)
            try:
                accepted, bonus = verifier(torch.tensor(path), p, r)
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


@pytest.mark.parametrize("k", [1, 3, 4])
def test_lazy_sparse_block_verifier_full_output_law(monkeypatch, k):
    def rows(path):
        paths = torch.tensor([path] * k)
        p = tensor([target(path[:i]) for i in range(3)])
        _, r = sampling.select_and_reweight(
            paths, p[None].expand(k, -1, -1), tensor(Q)
        )
        return p, r

    law = enumerate_lazy_verifier_law(monkeypatch, selected_distribution(k), rows)
    assert sum(law.values()) == pytest.approx(1, abs=1e-12)
    for sequence in product(range(2), repeat=3):
        expected = F(1)
        for i, token in enumerate(sequence):
            expected *= target(sequence[:i])[token]
        assert law[sequence] == pytest.approx(float(expected), abs=1e-12)


def test_sparse_residual_total_identity_with_duplicate_tokens():
    p = torch.tensor([
        [.05, .15, .30, .10, .40],
        [.20, .10, .25, .35, .10],
        [.40, .05, .15, .10, .30],
        [.10, .20, .30, .25, .15],
    ], dtype=torch.float64)
    prefix_weights = torch.tensor([1., .8, .35, .2], dtype=torch.float64)
    proposal_tokens = torch.tensor([
        [1, 1, 3, 0],
        [2, 4, 2, 4],
        [0, 3, 3, 0],
    ])
    proposal_probabilities = torch.tensor([
        [.20, .10, .70, .00],
        [.25, .15, .35, .25],
        [.10, .20, .30, .40],
    ], dtype=torch.float64)
    dense_proposal = torch.zeros((4, 5), dtype=torch.float64)
    dense_proposal[:-1].scatter_add_(
        1, proposal_tokens, proposal_probabilities
    )
    expected = (prefix_weights[:, None] * p - dense_proposal).clamp_min(0).sum(-1)
    actual = sampling._sparse_residual_totals(
        p, prefix_weights, proposal_tokens, proposal_probabilities
    )
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-13)


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


TREE_LEAF_MASS = {
    (0, 0): F(1, 10),
    (0, 1): F(2, 10),
    (1, 0): F(3, 10),
    (1, 1): F(4, 10),
}


def tree_proposal(prefix):
    if not prefix:
        return [F(3, 10), F(7, 10)]
    if prefix == (0,):
        return [F(1, 3), F(2, 3)]
    if prefix == (1,):
        return [F(3, 7), F(4, 7)]
    raise AssertionError("Only length-two proposal prefixes are valid")


def tree_order(path):
    return tuple((target(path[:i])[token] / tree_proposal(path[:i])[token], token)
                 for i, token in enumerate(path))


def tree_selected_distribution(k):
    distribution = defaultdict(F)
    for candidates in product(PATHS, repeat=k):
        chosen = max(candidates, key=tree_order)
        probability = F(1)
        for path in candidates:
            probability *= TREE_LEAF_MASS[path]
        distribution[chosen] += probability
    assert sum(distribution.values()) == 1
    return distribution


@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_autoregressive_tree_gbv_against_fraction_enumeration(k):
    distribution = tree_selected_distribution(k)
    for candidates in product(PATHS, repeat=k):
        paths = torch.tensor(candidates)
        p = torch.stack([tensor([target(path[:i]) for i in range(3)])
                         for path in candidates])
        chosen_target = p[:, :2].gather(2, paths[:, :, None])[:, :, 0]
        chosen_proposal = tensor([
            [tree_proposal(path[:i])[token] for i, token in enumerate(path)]
            for path in candidates
        ])
        chosen = sampling.select_greedy_path(paths, chosen_target, chosen_proposal)
        path = candidates[chosen]
        proposal_rows = tensor([tree_proposal(path[:i]) for i in range(2)])
        r = sampling.reweight_selected_path(paths[chosen], p[chosen], proposal_rows, k)
        assert path == max(candidates, key=tree_order)
        for i in range(2):
            prefix_mass = sum(w for y, w in distribution.items() if y[:i] == path[:i])
            expected = [
                sum(w for y, w in distribution.items()
                    if y[:i] == path[:i] and y[i] == token) / prefix_mass
                for token in range(2)
            ]
            torch.testing.assert_close(
                r[i], torch.tensor([float(x) for x in expected], dtype=torch.float64),
                rtol=1e-12, atol=1e-13,
            )


def test_autoregressive_tree_reweight_allows_sparse_support():
    path = torch.tensor([0, 1])
    proposal = torch.tensor([[1., 0., 0.], [.25, .75, 0.]], dtype=torch.float64)
    target_rows = torch.tensor([
        [.2, .3, .5], [.4, .4, .2], [.1, .2, .7]
    ], dtype=torch.float64)
    r = sampling.reweight_selected_path(path, target_rows, proposal, 3)
    assert bool(torch.isfinite(r).all())
    torch.testing.assert_close(r.sum(-1), torch.ones(2, dtype=torch.float64))
    assert r[0, 0] == 1
    assert r[0, 1] == r[0, 2] == 0


@pytest.mark.parametrize("k", [1, 2, 3, 4, 8])
def test_full_tree_gbv_analytically_marginalizes_candidate_draws(k):
    paths = torch.tensor(PATHS)
    p = torch.stack([tensor([target(path[:i]) for i in range(3)]) for path in PATHS])
    chosen_target = p[:, :2].gather(2, paths[:, :, None])[:, :, 0]
    chosen_proposal = tensor([
        [tree_proposal(path[:i])[token] for i, token in enumerate(path)]
        for path in PATHS
    ])
    leaf_probabilities = torch.tensor(
        [float(TREE_LEAF_MASS[path]) for path in PATHS], dtype=torch.float64
    )
    actual = sampling.greedy_max_distribution(
        paths, chosen_target, chosen_proposal, leaf_probabilities, k
    )
    expected_law = tree_selected_distribution(k)
    expected = torch.tensor([float(expected_law[path]) for path in PATHS], dtype=torch.float64)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-13)

    proposal = AdaptivePathProposal(
        paths=paths,
        leaf_probabilities=leaf_probabilities,
        token_probabilities=chosen_proposal,
        children={},
        vocab_size=2,
    )
    for index, path in enumerate(PATHS):
        if actual[index] == 0:
            continue
        rows = proposal.conditional_rows(paths[index], actual)
        for depth in range(2):
            prefix_mass = sum(actual[j] for j, other in enumerate(PATHS)
                              if other[:depth] == path[:depth])
            expected_row = torch.stack([
                sum(actual[j] for j, other in enumerate(PATHS)
                    if other[:depth] == path[:depth] and other[depth] == token) / prefix_mass
                for token in range(2)
            ])
            torch.testing.assert_close(rows[depth], expected_row, rtol=1e-12, atol=1e-13)


@pytest.mark.parametrize(
    "verifier",
    [sampling.block_verify, sampling.block_verify_batched, sparse_block_verify_from_dense],
    ids=["reference", "batched", "sparse"],
)
@pytest.mark.parametrize("k", [1, 3])
def test_tree_gbv_and_block_verifier_full_output_law(monkeypatch, k, verifier):
    law = defaultdict(float)
    for path, path_probability in tree_selected_distribution(k).items():
        p = tensor([target(path[:i]) for i in range(3)])
        proposal_rows = tensor([tree_proposal(path[:i]) for i in range(2)])
        r = sampling.reweight_selected_path(torch.tensor(path), p, proposal_rows, k)
        for choices in product(range(3), repeat=3):
            probability, cursor = 1.0, 0

            def forced(weights, generator=None):
                nonlocal probability, cursor
                if weights.ndim == 1:
                    index = choices[cursor]
                    cursor += 1
                    if index >= weights.numel() or weights[index] == 0:
                        raise Impossible
                    probability *= float(weights[index])
                    return torch.tensor(index)
                indices = torch.tensor(choices)
                for row, index in zip(weights, choices):
                    if index >= row.numel() or row[index] == 0:
                        raise Impossible
                    probability *= float(row[index])
                return indices

            monkeypatch.setattr(sampling, "sample", forced)
            try:
                accepted, bonus = verifier(torch.tensor(path), p, r)
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


@pytest.mark.parametrize("k", [1, 3])
def test_tree_gbv_and_lazy_verifier_full_output_law(monkeypatch, k):
    def rows(path):
        p = tensor([target(path[:i]) for i in range(3)])
        proposal_rows = tensor([tree_proposal(path[:i]) for i in range(2)])
        r = sampling.reweight_selected_path(
            torch.tensor(path), p, proposal_rows, k
        )
        return p, r

    law = enumerate_lazy_verifier_law(monkeypatch, tree_selected_distribution(k), rows)
    assert sum(law.values()) == pytest.approx(1, abs=1e-12)
    for sequence in product(range(2), repeat=3):
        expected = F(1)
        for i, token in enumerate(sequence):
            expected *= target(sequence[:i])[token]
        assert law[sequence] == pytest.approx(float(expected), abs=1e-12)


def test_terminal_mass_tree_block_full_output_law(monkeypatch):
    """The two-draw tree block has the exact Target AR continuation law."""
    paths = torch.tensor([[0, 0]])
    tree = sampled_tree(paths)
    all_p = torch.stack((
        tensor([target(())])[0],
        tensor([target((0,))])[0],
        tensor([target((0, 0))])[0],
    ))

    completed = []
    for terminal_node in range(len(tree.parents)):
        for bonus in range(2):
            choices = iter((terminal_node, bonus))
            probability = 1.0

            def forced(weights, generator=None):
                nonlocal probability
                choice = next(choices)
                if weights[choice] <= 0:
                    raise Impossible
                probability *= float(weights[choice])
                return torch.tensor(choice, device=weights.device)

            monkeypatch.setattr(sampling, "sample", forced)
            try:
                result = sampling.tree_block_verify_terminal_mass(
                    tree.parents, tree.tokens, all_p
                )
            except Impossible:
                continue
            with pytest.raises(StopIteration):
                next(choices)
            completed.append((result, probability))

    law = defaultdict(float)
    for (_, emitted_tokens, bonus), probability in completed:
        emitted = tuple(emitted_tokens) + (bonus,)
        for tail in product(range(2), repeat=3-len(emitted)):
            sequence, weight = emitted, probability
            for token in tail:
                weight *= float(target(sequence)[token])
                sequence += (token,)
            law[sequence] += weight

    assert sum(law.values()) == pytest.approx(1, abs=1e-12)
    for sequence in product(range(2), repeat=3):
        expected = F(1)
        for depth, token in enumerate(sequence):
            expected *= target(sequence[:depth])[token]
        assert law[sequence] == pytest.approx(float(expected), abs=1e-12)


@pytest.mark.parametrize("k", [1, 3])
@pytest.mark.parametrize("precompute_subtrees", [False, True])
def test_tree_block_recycling_full_output_law(monkeypatch, k, precompute_subtrees):
    paths = torch.tensor(PATHS)
    proposal_token_probabilities = tensor([
        [tree_proposal(path[:i])[token] for i, token in enumerate(path)]
        for path in PATHS
    ])
    leaf_probabilities = torch.tensor(
        [float(TREE_LEAF_MASS[path]) for path in PATHS], dtype=torch.float64
    )
    tree = sampled_tree(paths)
    path_nodes = torch.tensor(tree.path_nodes)
    children = {
        (tree.parents[i], tree.tokens[i - 1]): i
        for i in range(1, len(tree.parents))
    }
    all_p = torch.empty((len(tree.parents), 2), dtype=torch.float64)
    all_p[0] = torch.tensor(
        [float(x) for x in target(())], dtype=torch.float64
    )
    assigned = {0}
    for path, nodes in zip(PATHS, tree.path_nodes):
        for depth, node in enumerate(nodes, 1):
            row = torch.tensor(
                [float(x) for x in target(path[:depth])], dtype=torch.float64
            )
            if node in assigned:
                torch.testing.assert_close(all_p[node], row)
            else:
                all_p[node] = row
                assigned.add(node)

    def run():
        return sampling.tree_block_verify_recycle(
            paths, path_nodes, children, all_p, leaf_probabilities,
            proposal_token_probabilities, k, precompute_subtrees=precompute_subtrees,
        )

    completed = []
    frontier = [((), 1.0)]
    while frontier:
        plan, branch_probability = frontier.pop()
        cursor = 0

        def forced(weights, generator=None):
            nonlocal cursor
            if cursor < len(plan):
                value = plan[cursor]
                cursor += 1
                return torch.tensor(value, device=weights.device)
            if weights.ndim == 1:
                options = [
                    (index, float(weight))
                    for index, weight in enumerate(weights)
                    if weight > 0
                ]
            else:
                row_options = [
                    [index for index, weight in enumerate(row) if weight > 0]
                    for row in weights
                ]
                options = []
                for indices in product(*row_options):
                    probability = 1.0
                    for row, index in zip(weights, indices):
                        probability *= float(row[index])
                    options.append((indices, probability))
            raise NeedSampleBranch(options)

        monkeypatch.setattr(sampling, "sample", forced)
        try:
            result = run()
        except NeedSampleBranch as branch:
            for value, probability in branch.options:
                frontier.append((plan + (value,), branch_probability * probability))
            continue
        assert cursor == len(plan)
        completed.append((result, branch_probability))

    law = defaultdict(float)
    assert any(result[3]["recycled_corrections"] > 0 for result, _ in completed)
    for (nodes, tokens, bonus, recycle_stats), probability in completed:
        parent = 0
        for node, token in zip(nodes, tokens):
            assert children[(parent, token)] == node
            parent = node
        emitted = tuple(tokens) + (bonus,)
        assert 1 <= len(emitted) <= 3
        assert recycle_stats["segments"] >= 1
        assert recycle_stats["recycled_corrections"] >= 0
        for tail in product(range(2), repeat=3-len(emitted)):
            sequence, weight = emitted, probability
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


@pytest.mark.parametrize("segment_verifier", ["sparse", "sparse_lazy"])
@pytest.mark.parametrize("control_device", ["device", "cpu"])
@pytest.mark.parametrize("precompute_subtrees", [False, True])
@pytest.mark.parametrize("k", [1, 3])
def test_tree_block_recycling_segment_verifier(
    monkeypatch, k, control_device, precompute_subtrees, segment_verifier
):
    paths = torch.tensor(PATHS)
    proposal_token_probabilities = tensor([
        [tree_proposal(path[:i])[token] for i, token in enumerate(path)]
        for path in PATHS
    ])
    leaf_probabilities = torch.tensor(
        [float(TREE_LEAF_MASS[path]) for path in PATHS], dtype=torch.float64
    )
    tree = sampled_tree(paths)
    path_nodes = torch.tensor(tree.path_nodes)
    children = {
        (tree.parents[i], tree.tokens[i - 1]): i
        for i in range(1, len(tree.parents))
    }
    all_p = torch.empty((len(tree.parents), 2), dtype=torch.float64)
    all_p[0] = torch.tensor(
        [float(x) for x in target(())], dtype=torch.float64
    )
    assigned = {0}
    for path, nodes in zip(PATHS, tree.path_nodes):
        for depth, node in enumerate(nodes, 1):
            row = torch.tensor(
                [float(x) for x in target(path[:depth])], dtype=torch.float64
            )
            if node in assigned:
                torch.testing.assert_close(all_p[node], row)
            else:
                all_p[node] = row
                assigned.add(node)

    def run():
        return sampling.tree_block_verify_recycle(
            paths, path_nodes, children, all_p, leaf_probabilities,
            proposal_token_probabilities, k,
            segment_verifier=segment_verifier,
            control_device=control_device,
            host_generator=(
                torch.Generator(device="cpu").manual_seed(1)
                if control_device == "cpu" else None
            ),
            precompute_subtrees=precompute_subtrees,
        )

    completed = []
    frontier = [((), 1.0)]
    while frontier:
        plan, branch_probability = frontier.pop()
        cursor = 0

        def forced(weights, generator=None):
            nonlocal cursor
            if cursor < len(plan):
                value = plan[cursor]
                cursor += 1
                return torch.tensor(value, device=weights.device)
            if weights.ndim == 1:
                options = [
                    (index, float(weight))
                    for index, weight in enumerate(weights)
                    if weight > 0
                ]
            else:
                row_options = [
                    [index for index, weight in enumerate(row) if weight > 0]
                    for row in weights
                ]
                options = []
                for indices in product(*row_options):
                    probability = 1.0
                    for row, index in zip(weights, indices):
                        probability *= float(row[index])
                    options.append((indices, probability))
            raise NeedSampleBranch(options)

        monkeypatch.setattr(sampling, "sample", forced)
        try:
            result = run()
        except NeedSampleBranch as branch:
            for value, probability in branch.options:
                frontier.append((plan + (value,), branch_probability * probability))
            continue
        assert cursor == len(plan)
        completed.append((result, branch_probability))

    law = defaultdict(float)
    assert any(result[3]["recycled_corrections"] > 0 for result, _ in completed)
    for (nodes, tokens, bonus, recycle_stats), probability in completed:
        parent = 0
        for node, token in zip(nodes, tokens):
            assert children[(parent, token)] == node
            parent = node
        emitted = tuple(tokens) + (bonus,)
        assert 1 <= len(emitted) <= 3
        assert recycle_stats["segments"] >= 1
        assert recycle_stats["recycled_corrections"] >= 0
        for tail in product(range(2), repeat=3-len(emitted)):
            sequence, weight = emitted, probability
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
