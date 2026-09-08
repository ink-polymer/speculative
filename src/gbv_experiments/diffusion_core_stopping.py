"""Research reference: fixed coverage plus budget-stopped diffusion BV (K=1).

Not registered as a GPU experiment or a novel verification kernel. The full
diffusion witness is retained; the stopping rule never conditions/reweights Q
on the observed horizon. See docs/DIFFUSION_CORE_STOPPING.md for the proof,
conditional improvements, counterexample, and substantial prior-art overlap.
"""
from dataclasses import replace
import math

import torch

from . import diffusion_tree_bv as diffusion, sampling
from .tree import Tree, probability_tree, sampled_tree


def core_scaffold(law, greedy, budget, core_budget):
    """Fixed DDTree-style core union the full original-logit greedy chain."""
    length = law.tokens.shape[0]
    if (greedy.shape != (length,) or greedy.dtype != torch.long
            or bool((greedy < 0).any()) or type(budget) is not int
            or type(core_budget) is not int or core_budget < 0
            or budget < core_budget + length):
        raise ValueError("Core scaffold requires B >= M + L and a valid greedy chain")
    fixed = {tuple(greedy[:d].tolist()) for d in range(1, length + 1)}
    if core_budget:
        core = probability_tree(law.weights * law.retained_mass[:, None], core_budget)
        support, paths = law.tokens.tolist(), [()]
        for node, slot in enumerate(core.tokens, 1):
            prefix = paths[core.parents[node]] + (support[core.depths[node] - 1][slot],)
            paths.append(prefix)
            fixed.add(prefix)
    return fixed


def stopped_tree(proposal, greedy, budget, core_budget):
    """Reveal one sample prefix until its new-node capacity is exhausted.

    A continue/stop decision is made BEFORE inspecting the next token. Existing
    core nodes are free only while capacity is positive. In particular, once
    capacity reaches zero we do not peek ahead for another free overlap.
    """
    if proposal.slots.shape[0] != 1:
        raise ValueError("Budget-stopped reference supports exactly one diffusion path")
    fixed = core_scaffold(proposal.law, greedy, budget, core_budget)
    present, prefix, horizon = set(fixed), (), 0
    for token in proposal.paths()[0].tolist():
        if len(present) == budget:
            break
        prefix += (token,)
        present.add(prefix)
        horizon += 1
    # Put the original sampled prefix first, preserving the existing BV layout.
    tree = sampled_tree(proposal.paths()[:, :horizon])
    nodes, prefix = {(): 0}, ()
    for node, token in enumerate(tree.tokens, 1):
        prefix += (token,)
        nodes[prefix] = node
    for prefix in sorted(fixed, key=lambda p: (len(p), p)):
        if prefix not in nodes:
            nodes[prefix] = len(tree.parents)
            tree.tokens.append(prefix[-1])
            tree.parents.append(nodes[prefix[:-1]])
            tree.depths.append(len(prefix))
    return tree


def verify_logits(node_logits, tree, proposal, temperature, generator=None, *,
                  greedy, budget, core_budget, validate=True):
    """Stopped ordinary BV then fresh target continuation on the full scaffold.

    This is NOT arbitrary post-hoc pruning. With validation enabled, reconstruct
    the exact predictable tree from the retained full diffusion witness.
    """
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Budget-stopped diffusion BV requires positive target temperature")
    if (proposal.slots.shape[0] != 1 or node_logits.ndim != 2
            or not node_logits.is_floating_point() or node_logits.shape[-1] < 1
            or node_logits.shape[0] != len(tree.parents)
            or node_logits.device != proposal.source.device
            or len(tree.path_nodes) != 1):
        raise ValueError("Stopped diffusion target/tree shape mismatch")
    if validate:
        proposal.validate(node_logits.shape[-1])
        if (bool((greedy >= node_logits.shape[-1]).any())
                or tree != stopped_tree(proposal, greedy, budget, core_budget)
                or bool(torch.isnan(node_logits).any() | torch.isposinf(node_logits).any()
                        | ~torch.isfinite(node_logits).any(-1).all())):
            raise ValueError("Invalid stopped tree, greedy path, or target rows")
    horizon = len(tree.path_nodes[0])
    if not horizon:
        p = sampling.probabilities(node_logits.double(), temperature)
        return sampling.tree_block_verify_terminal_mass(
            tree.parents, tree.tokens, p, generator, validate=False)
    # Only a view used by the established kernel; never discard the full witness
    # or pretend that Q(X[:tau] | tau) is the original product distribution.
    law = replace(proposal.law, tokens=proposal.law.tokens[:horizon],
                  weights=proposal.law.weights[:horizon],
                  retained_mass=proposal.law.retained_mass[:horizon])
    prefix_proposal = replace(proposal, law=law, slots=proposal.slots[:, :horizon],
                              source=proposal.source[:horizon], draws=proposal.draws[:horizon])
    count = horizon + 1
    base = Tree(tree.tokens[:horizon], tree.parents[:count], tree.depths[:count], tree.path_nodes)
    nodes, tokens, bonus = diffusion.verify_logits(
        node_logits[:count], base, prefix_proposal, temperature, generator, validate=validate)
    return diffusion.continue_scaffold_logits(
        node_logits, tree, nodes, tokens, bonus, temperature, generator)
