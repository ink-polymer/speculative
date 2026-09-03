"""The exact layerwise construction in equations (3)-(7), not a DDTree prefix cut."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F

from ..tree import DraftTree

K = 15
TARGET_FEATURES = 16
HISTORY_FEATURES = 6
FEATURE_DIM = K * 4 + TARGET_FEATURES + HISTORY_FEATURES + 2


@dataclass(frozen=True)
class Action:
    budget: int
    depth: int
    quotas: tuple[int, ...]
    widths: tuple[int, ...]

    def validate(self, slots: int, vocab: int) -> None:
        values = (self.budget, self.depth, *self.quotas, *self.widths)
        if any(type(value) is not int for value in values):
            raise ValueError("Structure parameters must be integers")
        if not 0 <= self.budget <= 400 or not 0 <= self.depth <= slots:
            raise ValueError("Invalid budget/depth")
        if len(self.quotas) != self.depth or len(self.widths) != self.depth:
            raise ValueError("One quota/width is required per active layer")
        if any(q < 0 or q > 400 for q in self.quotas):
            raise ValueError("Invalid layer quota")
        if any(w < 1 or w > vocab for w in self.widths):
            raise ValueError("Invalid layer width")

    def json(self):
        return asdict(self)


def build_layered(logits: torch.Tensor, action: Action) -> DraftTree:
    if logits.ndim != 2 or not torch.isfinite(logits).all():
        raise ValueError("Expected finite [K,V] logits")
    action.validate(*logits.shape)
    tree = DraftTree()
    if not action.budget or not action.depth:
        return tree
    # Top-k in the common case; repair ties at the cutoff to enforce token-ID order.
    # Do not sort the entire 151K-token vocabulary just to keep <=8 children.
    scores = logits[:action.depth].float()
    width = max(action.widths)
    top_values, ids = torch.topk(scores, width, -1)
    cutoff = top_values[:, -1:]
    ties_at_cutoff = (scores >= cutoff).sum(-1).cpu().tolist()
    normalizers = torch.logsumexp(scores, -1).cpu().tolist()
    raw_values = top_values.cpu().tolist()
    tokens = ids.cpu().tolist()
    values = []
    for d in range(action.depth):
        if ties_at_cutoff[d] > width:
            strict = [(v, token) for v, token in zip(raw_values[d], tokens[d]) if v > raw_values[d][-1]]
            tied_ids = (scores[d] == cutoff[d]).nonzero().flatten()[:width - len(strict)].cpu().tolist()
            pairs = strict + [(raw_values[d][-1], token) for token in tied_ids]
        else:
            pairs = list(zip(raw_values[d], tokens[d]))
        pairs.sort(key=lambda pair: (-pair[0], pair[1]))
        tokens[d] = [token for _, token in pairs]
        values.append([v for v, _ in pairs])
    previous = [(-1, (), 0.0)]
    expanded = 0
    for d in range(action.depth):
        remaining = action.budget - len(tree)
        if not remaining or not previous:
            break
        candidates = [
            (score + values[d][rank], path + (tokens[d][rank],), parent, rank)
            for parent, path, score in previous for rank in range(action.widths[d])
        ]
        expanded += len(candidates)
        candidates.sort(key=lambda item: (-item[0], item[1]))
        previous = []
        for score, path, parent, rank in candidates[:min(action.quotas[d], remaining)]:
            # All candidates at this depth share sum(log Z_j). Rank by raw
            # prefix logits so subtracting normalizers cannot break exact ties.
            log_probability = score - math.fsum(normalizers[:d + 1])
            index = tree.add_node(path[-1], parent, log_probability, 0, d, rank)
            previous.append((index, path, score))
    tree.expanded_candidates = expanded
    assert expanded <= width * (action.budget + 1)
    tree.validate(budget=action.budget)
    return tree


def features(logits, target_context, history, prefix_length, remaining):
    if logits.ndim != 2 or logits.shape[0] != K or logits.shape[1] < 2:
        raise ValueError("The policy contract requires [15,V] logits with V>=2")
    logp = torch.log_softmax(logits.float(), -1)
    p = logp.exp()
    top = torch.topk(p, min(8, p.shape[-1]), -1).values
    entropy = -(p * logp).sum(-1) / math.log(logits.shape[-1])
    slot = torch.stack((top[:, 0], top[:, 0] - top[:, 1], entropy,
                        top.sum(-1)), -1)
    # Already-computed, last cached target position; never an extra target forward.
    hidden = target_context[0, -1].float()
    hidden = hidden / hidden.square().mean().sqrt().clamp_min(1e-6)
    pooled = F.adaptive_avg_pool1d(hidden[None, None], TARGET_FEATURES).flatten()
    recent = history[-8:]
    if recent:
        h = [recent[-1]["accepted"] / K,
             sum(v["accepted"] for v in recent) / (K * len(recent)),
             sum(v["greedy_disagreement"] for v in recent) / len(recent),
             recent[-1]["nodes"] / 120,
             min(recent[-1]["latency_ms"] / 1000, 10),
             min(len(history) / 128, 1)]
    else:
        h = [0.] * HISTORY_FEATURES
    result = torch.cat((slot.flatten().cpu(), pooled.cpu(), torch.tensor(h),
                       torch.tensor([min(prefix_length / 8192, 4), min(remaining / 2048, 4)]))).float()
    if not torch.isfinite(result).all():
        raise FloatingPointError("Nonfinite policy features")
    return result


class StructurePolicy(nn.Module):
    """One MLP forward, factorized categorical heads; inactive heads have zero loss."""
    def __init__(self, variant="full", hidden=64):
        super().__init__()
        from .common import VARIANTS
        if variant not in VARIANTS:
            raise ValueError(f"Unknown policy variant: {variant}")
        self.variant = variant
        self.choices = [
            (60,) if variant == "fixed_budget" else (30, 45, 60, 90, 120),
            (15,) if variant == "fixed_depth" else (4, 8, 12, 15),
        ]
        self.choices += [(4,) if variant == "fixed_quotas" else (1, 2, 4, 8, 12, 16, 24, 32)] * K
        self.choices += [(4,) if variant == "fixed_width" else (1, 2, 4, 8)] * K
        self.body = nn.Sequential(nn.Linear(FEATURE_DIM, hidden), nn.Tanh(),
                                  nn.Linear(hidden, hidden), nn.Tanh())
        self.actor = nn.Linear(hidden, sum(map(len, self.choices)))
        self.value = nn.Linear(hidden, 1)
        nn.init.zeros_(self.actor.bias)
        nn.init.normal_(self.actor.weight, std=0.01)

    def forward(self, x):
        x = x.clone()
        if self.variant in {"no_target", "draft_only"}:
            x[..., K * 4:K * 4 + TARGET_FEATURES] = 0
        if self.variant in {"no_history", "draft_only"}:
            x[..., K * 4 + TARGET_FEATURES:-2] = 0
        h = self.body(x)
        return self.actor(h).split([len(v) for v in self.choices], -1), self.value(h).squeeze(-1)

    def decode(self, indices):
        values = [choices[int(i)] for choices, i in zip(self.choices, indices)]
        depth = values[1]
        return Action(values[0], depth, tuple(values[2:2 + depth]),
                      tuple(values[2 + K:2 + K + depth]))

    @torch.no_grad()
    def choose(self, x, *, sample=False, generator=None):
        heads, _ = self(x)
        indices = [int(torch.multinomial(h.softmax(-1), 1, generator=generator))
                   if sample else int(h.argmax(-1)) for h in heads]
        return self.decode(indices), indices

    def random_action(self, rng):
        indices = [rng.randrange(len(c)) for c in self.choices]
        return self.decode(indices), indices

    def log_prob_value(self, x, indices):
        heads, value = self(x)
        if x.ndim == 1:
            heads = [h.unsqueeze(0) for h in heads]
            indices = indices.unsqueeze(0)
        depths = torch.tensor(self.choices[1])[indices[:, 1]]
        terms = []
        for j, head in enumerate(heads):
            active = torch.ones_like(depths, dtype=torch.bool) if j < 2 else ((j - 2) % K < depths)
            terms.append(head.log_softmax(-1).gather(1, indices[:, j:j + 1]).squeeze(-1) * active)
        return torch.stack(terms).sum(0), value


def static_action():
    return Action(60, K, (4,) * K, (4,) * K)
