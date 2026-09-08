"""Tiny, exhaustive research oracle; NOT a production verification algorithm.

Enumerates all block-return events and all target-completed strings. The LP
uses floating-point SciPy/HiGHS; independent rational witnesses can be checked
with ``completed_law`` without invoking the solver. Off-tree target rows are
available to this oracle, so its optimum is not an implementability claim.

The generic coupling/LP formulation is background, not a novelty claim; see
docs/TREE_COUPLING_RESEARCH.md for prior work and the restricted experiment.
"""
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations, product
import json
import math


def prefix_tree(paths):
    return tuple(sorted({path[:i] for path in paths for i in range(len(path) + 1)},
                        key=lambda p: (len(p), p)))


def probability(rows, sequence, start=0):
    """Conditional suffix probability, with no division by zero-mass prefixes."""
    value = 1
    for i in range(start, len(sequence)):
        value *= rows[sequence[:i]][sequence[i]]
    return value


def completed_law(rows, horizon, events):
    """Events map (tree_index, emitted_prefix) to JOINT mass including P(tree).

    Extend every emitted prefix with ordinary target sampling to ``horizon``.
    Fraction-valued inputs give a genuinely exact rational check. This helper
    does not itself validate the tree support or its marginal.
    """
    vocab = len(rows[()])
    result = defaultdict(lambda: 0)
    for (_, emitted), mass in events.items():
        if not 1 <= len(emitted) <= horizon:
            raise ValueError("emitted prefix outside the declared horizon")
        for suffix in product(range(vocab), repeat=horizon - len(emitted)):
            sequence = emitted + suffix
            result[sequence] += mass * probability(rows, sequence, len(emitted))
    return dict(result)


def fixed_tree_committed(rows, tree):
    """DDTree expected committed tokens, including one final correction token."""
    return 1 + sum(probability(rows, node) for node in tree if node)


@dataclass
class OracleResult:
    committed_tokens: float
    accepted_tokens: float
    equality_error: float
    target_law_error: float
    events: dict


def solve(rows, trees, tree_weights, horizon, *, conditional_on_tree=False):
    """Maximize accepted-prefix length plus one under target-completion equality.

    conditional_on_tree=True imposes the stronger DDTree-style condition that
    each realized tree separately returns a target-correct continuation.
    Otherwise only the mixture over random trees must be target-correct.
    Neither condition permits emitting more than a tree prefix plus one token.
    """
    import numpy as np
    from scipy.optimize import linprog

    if horizon < 1 or not trees or len(trees) != len(tree_weights):
        raise ValueError("invalid horizon or tree mixture")
    vocab = len(rows.get((), ()))
    if vocab < 1:
        raise ValueError("missing root probabilities")
    for depth in range(horizon):
        for node in product(range(vocab), repeat=depth):
            row = rows.get(node, ())
            if (len(row) != vocab or any(not math.isfinite(float(x)) or x < 0 for x in row)
                    or not math.isclose(float(sum(row)), 1, abs_tol=1e-12, rel_tol=0)):
                raise ValueError("target must specify normalized rows at every prefix")
    weights = [float(w) for w in tree_weights]
    if (any(not math.isfinite(w) or w <= 0 for w in weights)
            or not math.isclose(sum(weights), 1, abs_tol=1e-12, rel_tol=0)):
        raise ValueError("tree weights must be positive and sum to one")
    for tree in trees:
        nodes = set(tree)
        if len(nodes) != len(tree) or () not in nodes:
            raise ValueError("tree must contain root and unique nodes")
        for node in nodes:
            if (len(node) >= horizon or any(t not in range(vocab) for t in node)
                    or (node and node[:-1] not in nodes)):
                raise ValueError("tree must be prefix-closed within the horizon")

    events = [(i, node + (token,)) for i, tree in enumerate(trees)
              for node in tree for token in range(vocab)]
    count_sequences = vocab ** horizon
    nrows = len(trees) + count_sequences * (len(trees) if conditional_on_tree else 1)
    if nrows * len(events) > 2_000_000:
        raise ValueError("exhaustive oracle limited to tiny problems")
    sequences = list(product(range(vocab), repeat=horizon))
    equations, rhs = [], []
    for i, weight in enumerate(weights):
        equations.append([float(j == i) for j, _ in events])
        rhs.append(weight)
    for tree_index in range(len(trees)) if conditional_on_tree else [None]:
        for sequence in sequences:
            coefficients = []
            for i, emitted in events:
                matches = (tree_index is None or i == tree_index)
                matches &= sequence[:len(emitted)] == emitted
                coefficients.append(float(probability(rows, sequence, len(emitted)))
                                    if matches else 0.0)
            equations.append(coefficients)
            rhs.append(float(probability(rows, sequence)) *
                       (1 if tree_index is None else weights[tree_index]))
    matrix, rhs = np.asarray(equations), np.asarray(rhs)
    reward = np.array([len(emitted) for _, emitted in events], dtype=float)
    answer = linprog(-reward, A_eq=matrix, b_eq=rhs, bounds=(0, None), method="highs",
                     options={"dual_feasibility_tolerance": 1e-9,
                              "primal_feasibility_tolerance": 1e-9})
    if not answer.success:
        raise RuntimeError(f"oracle solver failed: {answer.message}")
    if float(answer.x.min()) < -1e-10:
        raise ArithmeticError("negative mass in numerical LP result")
    witness = {event: float(value) for event, value in zip(events, answer.x)}
    law = completed_law(rows, horizon, witness)
    error = max(abs(float(law.get(s, 0)) - float(probability(rows, s))) for s in sequences)
    equality_error = float(np.max(np.abs(matrix @ answer.x - rhs)))
    if max(error, equality_error) > 1e-8:
        raise ArithmeticError("numerical LP result fails full distribution check")
    committed = float(reward @ answer.x)
    return OracleResult(committed, committed - 1, equality_error, error, witness)


def example_rows(regime="copy", strength=0.9):
    """Binary length-two example, followed by an ordinary target bonus token."""
    if regime not in {"copy", "zero"} or not 0 <= strength <= 1:
        raise ValueError("unknown regime or invalid strength")
    rows = {(): (0.5, 0.5)}
    for first in range(2):
        preferred = first if regime == "copy" else 0
        rows[(first,)] = tuple(strength if t == preferred else 1 - strength for t in range(2))
        for second in range(2):
            rows[(first, second)] = (0.5, 0.5)
    return rows


def example_trees(kind):
    """Each realization has two leaves, both root children, four non-root nodes.

    Randomly swapping path indices makes each indexed path uniform on 00,01,10,11
    under either family. This is a correlation-only change in the path proposal.
    """
    if kind == "shared_second":
        return [prefix_tree([(0, s), (1, s)]) for s in range(2)]
    if kind == "xor_second":
        return [prefix_tree([(0, s), (1, 1 - s)]) for s in range(2)]
    raise ValueError("unknown tree family")


def demo():
    result = {"status": "finite_toy_oracle_not_an_online_algorithm",
              "new_algorithm_proved": False, "gpu_benchmarked": False,
              "budget_nonroot_nodes": 4, "draft_depth": 2,
              "each_indexed_path_law": "uniform after a fair random path-index swap",
              "regimes": {}}
    possible_nodes = [s for d in (1, 2) for s in product(range(2), repeat=d)]
    fixed_trees = [tuple([()] + list(nodes)) for nodes in combinations(possible_nodes, 4)
                   if all(len(s) == 1 or s[:-1] in nodes for s in nodes)]
    for regime in ("copy", "zero"):
        rows = example_rows(regime)
        stats = {"best_fixed_tree_ddtree_committed": max(
            fixed_tree_committed(rows, t) for t in fixed_trees)}
        for family in ("shared_second", "xor_second"):
            trees = example_trees(family)
            oracle = solve(rows, trees, [0.5, 0.5], 3)
            conditional = solve(rows, trees, [0.5, 0.5], 3, conditional_on_tree=True)
            stats[family] = {"joint_block_oracle_committed": oracle.committed_tokens,
                             "conditional_on_tree_oracle_committed": conditional.committed_tokens,
                             "ddtree_same_random_tree_committed": sum(
                                 fixed_tree_committed(rows, t) for t in trees) / 2,
                             "maximum_target_law_error": oracle.target_law_error,
                             "maximum_equality_error": oracle.equality_error}
        result["regimes"][regime] = stats
    return result


if __name__ == "__main__":
    print(json.dumps(demo(), indent=2, sort_keys=True))
