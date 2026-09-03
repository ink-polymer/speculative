from __future__ import annotations

from dataclasses import dataclass
import heapq

import torch


@dataclass
class Tree:
    tokens: list[int]  # Excludes the root anchor; node i has tokens[i-1].
    parents: list[int]  # Includes root 0, whose parent is -1.
    depths: list[int]
    path_nodes: list[list[int]]

    def visibility(self, device=None):
        n = len(self.parents)
        visible = torch.zeros((n, n), dtype=torch.bool)
        visible[0, 0] = True
        for i in range(1, n):
            visible[i] = visible[self.parents[i]]
            visible[i, i] = True
        return visible.to(device)

    def mask(self, prefix_length: int, dtype, device):
        n = len(self.parents)
        mask = torch.full((n, prefix_length + n), float("-inf"), dtype=dtype, device=device)
        mask[:, :prefix_length] = 0
        mask[:, prefix_length:].masked_fill_(self.visibility(device), 0)
        return mask[None, None]


def sampled_tree(paths, share=True):
    tokens, parents, depths, path_nodes = [], [-1], [0], []
    children = {}
    for path in paths.tolist():
        parent, nodes = 0, []
        for depth, token in enumerate(path, 1):
            child = children.get((parent, token)) if share else None
            if child is None:
                child = len(parents)
                children[(parent, token)] = child
                tokens.append(token)
                parents.append(parent)
                depths.append(depth)
            nodes.append(child)
            parent = child
        path_nodes.append(nodes)
    return Tree(tokens, parents, depths, path_nodes)


def probability_tree(q, budget: int):
    """Best-first product-probability prefix tree, as in DDTree."""
    length, vocab = q.shape
    values, indices = torch.topk(q, min(budget, vocab), dim=-1, sorted=True)
    logs, ids = values.log().tolist(), indices.tolist()
    tokens, parents, depths = [], [-1], [0]
    # Sibling alternatives are expanded lazily; descendants retain parent mass.
    heap = [(-logs[0][0], 0, 0, 0, 0.0)]
    while heap and len(tokens) < budget:
        neg_score, parent, depth, rank, parent_log = heapq.heappop(heap)
        node = len(parents)
        tokens.append(ids[depth][rank])
        parents.append(parent)
        depths.append(depth + 1)
        if rank + 1 < len(ids[depth]):
            heapq.heappush(heap, (-(parent_log + logs[depth][rank + 1]),
                                 parent, depth, rank + 1, parent_log))
        if depth + 1 < length:
            heapq.heappush(heap, (neg_score - logs[depth + 1][0],
                                 node, depth + 1, 0, -neg_score))
    return Tree(tokens, parents, depths, [])


def compact_cache(cache, prefix_length: int, rows: list[int], device):
    keep = torch.tensor(rows, device=device) + prefix_length
    if hasattr(cache, "layers"):
        pairs = [(layer.keys, layer.values) for layer in cache.layers]
    elif hasattr(cache, "key_cache"):
        pairs = list(zip(cache.key_cache, cache.value_cache))
    else:
        raise TypeError("Expected a DynamicCache-compatible key/value cache")
    for keys, values in pairs:
        if keys is None:
            continue
        selected_keys = keys.index_select(-2, keep)
        selected_values = values.index_select(-2, keep)
        keys[..., prefix_length:prefix_length + len(rows), :].copy_(selected_keys)
        values[..., prefix_length:prefix_length + len(rows), :].copy_(selected_values)
    cache.crop(prefix_length + len(rows))
    if cache.get_seq_length() != prefix_length + len(rows):
        raise RuntimeError("Cache length invariant failed")
