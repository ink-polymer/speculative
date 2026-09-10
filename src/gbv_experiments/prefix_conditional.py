"""Lightweight prefix-conditional proposal head for one-forward DFlash.

The frozen DFlash transformer runs once.  This head then recurrently conditions
the proposal at slot ``d`` on the tokens selected at slots ``< d``.  It scores
only a fixed top-R support from the original DFlash logits, so it adds no second
vocabulary projection.  Exact block verification consumes the actual
conditional probabilities returned by :meth:`propose`.
"""
from __future__ import annotations

from collections.abc import Mapping
from collections import defaultdict
import heapq
import math
from pathlib import Path
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from .diffusion_tree_bv import DiffusionBlockLaw, DiffusionProposal
from .tree import Tree


class PrefixConditionalHead(nn.Module):
    """Low-rank recurrent correction over frozen DFlash candidate supports."""

    def __init__(self, hidden_size: int, rank: int = 64,
                 support_size: int = 32, max_length: int = 15):
        super().__init__()
        if min(hidden_size, rank, support_size, max_length) < 1:
            raise ValueError("Invalid prefix-conditional head dimensions")
        self.hidden_size = int(hidden_size)
        self.rank = int(rank)
        self.support_size = int(support_size)
        self.max_length = int(max_length)
        self.hidden_norm = nn.RMSNorm(hidden_size)
        self.token_norm = nn.RMSNorm(hidden_size)
        self.hidden_down = nn.Linear(hidden_size, rank, bias=False)
        self.prefix_down = nn.Linear(hidden_size, rank, bias=False)
        self.recurrent = nn.GRU(rank, rank, batch_first=True)
        self.query = nn.Linear(rank, rank, bias=False)
        self.key = nn.Linear(hidden_size, rank, bias=False)
        self.depth_bias = nn.Parameter(torch.zeros(max_length, support_size))
        # Start exactly at the frozen DFlash distribution while preserving a
        # full-strength gradient into the final correction projection.
        nn.init.zeros_(self.query.weight)
        self.correction_scale = nn.Parameter(torch.ones(()))

    def _candidate_scores(self, states: torch.Tensor,
                          draft_logits: torch.Tensor,
                          token_embeddings: torch.Tensor,
                          support_size: int | None = None):
        if (states.ndim != 3 or draft_logits.ndim != 3
                or states.shape[:2] != draft_logits.shape[:2]
                or states.shape[-1] != self.rank
                or draft_logits.shape[1] > self.max_length):
            raise ValueError("Prefix states and Draft logits disagree")
        width = self.support_size if support_size is None else int(support_size)
        if not 1 <= width <= self.support_size or width > draft_logits.shape[-1]:
            raise ValueError("Prefix proposal support exceeds its trained cap")
        values, candidate_ids = draft_logits.topk(width, dim=-1, sorted=True)
        dtype = self.hidden_down.weight.dtype
        selected = F.embedding(candidate_ids, token_embeddings).to(dtype)
        keys = self.key(self.token_norm(selected))
        queries = self.query(states.to(dtype))[:, :, None]
        correction = (queries * keys).sum(-1) / self.rank ** 0.5
        correction = correction + self.depth_bias[
            :states.shape[1], :width
        ][None]
        scores = values.to(dtype) + self.correction_scale * correction
        return scores, candidate_ids

    def teacher_forced_scores(self, hidden: torch.Tensor,
                              draft_logits: torch.Tensor,
                              token_embeddings: torch.Tensor,
                              labels: torch.Tensor,
                              support_size: int | None = None, *,
                              prefix_embeddings: torch.Tensor | None = None):
        """Score every slot while conditioning only on earlier labels."""
        if (hidden.ndim != 3 or hidden.shape[-1] != self.hidden_size
                or draft_logits.shape[:2] != hidden.shape[:2]
                or labels.shape != hidden.shape[:2]):
            raise ValueError("Teacher-forced prefix batch has inconsistent shapes")
        dtype = self.hidden_down.weight.dtype
        base = self.hidden_down(self.hidden_norm(hidden.to(dtype)))
        previous = torch.zeros_like(hidden)
        prefix_embeddings = (
            token_embeddings
            if prefix_embeddings is None else prefix_embeddings
        )
        if labels.shape[1] > 1:
            previous[:, 1:] = F.embedding(
                labels[:, :-1], prefix_embeddings.to(hidden.dtype),
            )
        recurrent_input = base + self.prefix_down(
            self.token_norm(previous.to(dtype))
        )
        states, _ = self.recurrent(recurrent_input)
        return self._candidate_scores(
            states, draft_logits, token_embeddings, support_size,
        )

    @staticmethod
    def _sparse_probability_tree(tokens: torch.Tensor,
                                 probabilities: torch.Tensor,
                                 budget: int) -> Tree:
        """DDTree best-first pool restricted to an explicit top-R support."""
        if (tokens.ndim != 2 or probabilities.shape != tokens.shape
                or tokens.dtype != torch.long or budget < 1):
            raise ValueError("Invalid sparse candidate pool")
        logs = probabilities.log().tolist()
        ids = tokens.tolist()
        result_tokens, parents, depths = [], [-1], [0]
        heap = [(-logs[0][0], 0, 0, 0, 0.)]
        while heap and len(result_tokens) < budget:
            negative, parent, depth, rank, parent_log = heapq.heappop(heap)
            node = len(parents)
            result_tokens.append(ids[depth][rank])
            parents.append(parent)
            depths.append(depth + 1)
            if rank + 1 < len(ids[depth]):
                heapq.heappush(heap, (
                    -(parent_log + logs[depth][rank + 1]),
                    parent, depth, rank + 1, parent_log,
                ))
            if depth + 1 < tokens.shape[0]:
                heapq.heappush(heap, (
                    negative - logs[depth + 1][0], node, depth + 1,
                    0, -negative,
                ))
        return Tree(result_tokens, parents, depths, [])

    @torch.inference_mode()
    def build_tree(self, hidden: torch.Tensor, draft_logits: torch.Tensor,
                   token_embeddings: torch.Tensor,
                   position_probabilities: torch.Tensor, budget: int,
                   temperature: float, pool_factor: int = 4, *,
                   prefix_embeddings: torch.Tensor | None = None,
                   support_size: int | None = None,
                   strength: float = 1.) -> Tree:
        """Rerank a broad DFlash prefix pool with prefix-conditional scores."""
        if (hidden.ndim != 3 or hidden.shape[0] != 1
                or hidden.shape[-1] != self.hidden_size
                or draft_logits.ndim != 2
                or draft_logits.shape[0] != hidden.shape[1]
                or position_probabilities.shape != draft_logits.shape
                or not 1 <= hidden.shape[1] <= self.max_length
                or budget < hidden.shape[1] or pool_factor < 1
                or not math.isfinite(temperature) or temperature <= 0
                or not math.isfinite(strength) or not 0 <= strength <= 2):
            raise ValueError("Invalid prefix-conditioned tree inputs")
        length = hidden.shape[1]
        prefix_embeddings = (
            token_embeddings
            if prefix_embeddings is None else prefix_embeddings
        )
        width = self.support_size if support_size is None else int(support_size)
        if not 1 <= width <= min(self.support_size, draft_logits.shape[-1]):
            raise ValueError("Prefix tree support exceeds its trained cap")
        q_values, candidate_ids = position_probabilities.topk(
            width, dim=-1, sorted=True,
        )
        q_values = q_values / q_values.sum(-1, keepdim=True)
        pool = self._sparse_probability_tree(
            candidate_ids, q_values, budget * pool_factor,
        )
        dtype = self.hidden_down.weight.dtype
        base = self.hidden_down(self.hidden_norm(hidden.to(dtype)))[0]
        candidate_embeddings = F.embedding(candidate_ids, token_embeddings)
        candidate_keys = self.key(
            self.token_norm(candidate_embeddings.to(dtype))
        )
        draft_values = draft_logits.gather(1, candidate_ids).to(dtype)
        rank_by_token = [
            {token: rank for rank, token in enumerate(row)}
            for row in candidate_ids.tolist()
        ]
        pool_children: dict[int, list[int]] = defaultdict(list)
        for node in range(1, len(pool.parents)):
            pool_children[pool.parents[node]].append(node)

        cumulative = torch.zeros(
            len(pool.parents), dtype=torch.float64, device=hidden.device,
        )
        states: dict[int, torch.Tensor] = {}
        for depth in range(length):
            parents_at_depth = [
                parent for parent in pool_children
                if pool.depths[parent] == depth
            ]
            if not parents_at_depth:
                continue
            if depth == 0:
                previous = hidden.new_zeros(
                    (len(parents_at_depth), self.hidden_size), dtype=dtype,
                )
                prior = hidden.new_zeros(
                    (1, len(parents_at_depth), self.rank), dtype=dtype,
                )
            else:
                parent_tokens = torch.tensor(
                    [pool.tokens[parent - 1] for parent in parents_at_depth],
                    dtype=torch.long, device=hidden.device,
                )
                previous = F.embedding(
                    parent_tokens, prefix_embeddings,
                ).to(dtype)
                prior = torch.stack(
                    [states[parent] for parent in parents_at_depth], dim=0,
                )[None]
            inputs = base[depth][None].expand(len(parents_at_depth), -1)
            inputs = inputs + self.prefix_down(self.token_norm(previous))
            outputs, state = self.recurrent(inputs[:, None], prior)
            queries = self.query(outputs[:, 0])[:, None]
            corrections = (
                queries * candidate_keys[depth][None]
            ).sum(-1) / self.rank ** .5
            residual = (
                self.correction_scale * corrections
                + self.depth_bias[depth, :width][None]
            )
            scores = draft_values[depth][None] + float(strength) * residual
            log_probabilities = torch.log_softmax(
                scores.double() / float(temperature), dim=-1,
            )
            child_nodes, parent_nodes, parent_rows, child_ranks = [], [], [], []
            for row, parent in enumerate(parents_at_depth):
                for child in pool_children[parent]:
                    token = pool.tokens[child - 1]
                    child_nodes.append(child)
                    parent_nodes.append(parent)
                    parent_rows.append(row)
                    child_ranks.append(rank_by_token[depth][token])
                    states[child] = state[0, row]
            child_tensor = torch.tensor(
                child_nodes, dtype=torch.long, device=hidden.device,
            )
            parent_tensor = torch.tensor(
                parent_nodes, dtype=torch.long, device=hidden.device,
            )
            row_tensor = torch.tensor(
                parent_rows, dtype=torch.long, device=hidden.device,
            )
            rank_tensor = torch.tensor(
                child_ranks, dtype=torch.long, device=hidden.device,
            )
            cumulative[child_tensor] = (
                cumulative[parent_tensor]
                + log_probabilities[row_tensor, rank_tensor]
            )

        selected_tokens, selected_parents, selected_depths = [], [-1], [0]
        source_to_selected = {0: 0}
        heap: list[tuple[float, int]] = []
        cumulative_values = cumulative.tolist()
        for child in pool_children[0]:
            heapq.heappush(heap, (-cumulative_values[child], child))
        while heap and len(selected_tokens) < budget:
            _, source = heapq.heappop(heap)
            parent = source_to_selected[pool.parents[source]]
            source_to_selected[source] = len(selected_parents)
            selected_tokens.append(pool.tokens[source - 1])
            selected_parents.append(parent)
            selected_depths.append(pool.depths[source])
            for child in pool_children.get(source, ()):
                heapq.heappush(heap, (-cumulative_values[child], child))
        if len(selected_tokens) != budget:
            raise RuntimeError("Conditioned candidate pool did not fill B")
        return Tree(
            selected_tokens, selected_parents, selected_depths, [],
        )

    @torch.inference_mode()
    def build_beam_tree(self, hidden: torch.Tensor, draft_logits: torch.Tensor,
                        token_embeddings: torch.Tensor, budget: int,
                        temperature: float, *,
                        prefix_embeddings: torch.Tensor | None = None,
                        support_size: int | None = None,
                        strength: float = 1.) -> Tree:
        """Build the B highest-mass prefixes under the conditional head.

        Each depth expands at most ``budget`` live prefixes in one recurrent
        batch.  Since every child probability is at most its parent's, the
        global top-B prefix set is ancestor closed.  Unlike :meth:`build_tree`,
        this search is not restricted to an unconditional DDTree candidate
        pool.
        """
        if (hidden.ndim != 3 or hidden.shape[0] != 1
                or hidden.shape[-1] != self.hidden_size
                or draft_logits.ndim != 2
                or draft_logits.shape[0] != hidden.shape[1]
                or not 1 <= hidden.shape[1] <= self.max_length
                or budget < hidden.shape[1]
                or not math.isfinite(temperature) or temperature <= 0
                or not math.isfinite(strength) or not 0 <= strength <= 2):
            raise ValueError("Invalid conditional beam-tree inputs")
        width = self.support_size if support_size is None else int(support_size)
        if not 1 <= width <= min(self.support_size, draft_logits.shape[-1]):
            raise ValueError("Prefix tree support exceeds its trained cap")
        dtype = self.hidden_down.weight.dtype
        prefix_embeddings = (
            token_embeddings
            if prefix_embeddings is None else prefix_embeddings
        )
        base = self.hidden_down(self.hidden_norm(hidden.to(dtype)))[0]
        values, candidate_ids = draft_logits.topk(
            width, dim=-1, sorted=True,
        )
        candidate_embeddings = F.embedding(candidate_ids, token_embeddings)
        candidate_keys = self.key(
            self.token_norm(candidate_embeddings.to(dtype))
        )

        # Pool indices are one based; zero is the clean anchor.
        pool_tokens: list[int] = []
        pool_parents: list[int] = []
        pool_depths: list[int] = []
        pool_scores: list[float] = []
        frontier_pool = torch.zeros(1, dtype=torch.long, device=hidden.device)
        frontier_scores = torch.zeros(
            1, dtype=torch.float64, device=hidden.device,
        )
        frontier_tokens = torch.zeros(
            1, dtype=torch.long, device=hidden.device,
        )
        frontier_state: torch.Tensor | None = None

        for depth in range(hidden.shape[1]):
            count = frontier_pool.numel()
            if depth == 0:
                previous = hidden.new_zeros(
                    (count, self.hidden_size), dtype=dtype,
                )
                prior = None
            else:
                previous = F.embedding(
                    frontier_tokens, prefix_embeddings,
                ).to(dtype)
                if frontier_state is None:
                    raise RuntimeError("Conditional beam lost its recurrent state")
                prior = frontier_state[None]
            inputs = base[depth][None].expand(count, -1)
            inputs = inputs + self.prefix_down(self.token_norm(previous))
            outputs, state = self.recurrent(inputs[:, None], prior)
            queries = self.query(outputs[:, 0])[:, None]
            correction = (
                queries * candidate_keys[depth][None]
            ).sum(-1) / self.rank ** .5
            residual = (
                self.correction_scale * correction
                + self.depth_bias[depth, :width][None]
            )
            scores = values[depth][None].to(dtype) + float(strength) * residual
            log_probabilities = torch.log_softmax(
                scores.double() / float(temperature), dim=-1,
            )
            cumulative = frontier_scores[:, None] + log_probabilities
            flat = cumulative.flatten()
            keep = min(budget, flat.numel())
            kept_scores, kept = flat.topk(keep, sorted=True)
            parent_rows = torch.div(kept, width, rounding_mode="floor")
            token_ranks = kept.remainder(width)
            next_tokens = candidate_ids[depth].index_select(0, token_ranks)
            next_parents = frontier_pool.index_select(0, parent_rows)
            next_states = state[0].index_select(0, parent_rows)

            start = len(pool_tokens) + 1
            pool_tokens.extend(next_tokens.tolist())
            pool_parents.extend(next_parents.tolist())
            pool_depths.extend([depth + 1] * keep)
            pool_scores.extend(kept_scores.tolist())
            frontier_pool = torch.arange(
                start, start + keep, dtype=torch.long, device=hidden.device,
            )
            frontier_scores = kept_scores
            frontier_tokens = next_tokens
            frontier_state = next_states

        ranked = sorted(
            range(len(pool_tokens)),
            key=lambda index: (
                -pool_scores[index], pool_depths[index], index,
            ),
        )[:budget]
        chosen = set(ranked)
        selected_tokens: list[int] = []
        selected_parents = [-1]
        selected_depths = [0]
        pool_to_selected = {0: 0}
        for index, (token, parent, depth) in enumerate(zip(
                pool_tokens, pool_parents, pool_depths)):
            if index not in chosen:
                continue
            if parent not in pool_to_selected:
                raise RuntimeError("Conditional top-B tree is not ancestor closed")
            pool_to_selected[index + 1] = len(selected_parents)
            selected_tokens.append(token)
            selected_parents.append(pool_to_selected[parent])
            selected_depths.append(depth)
        if len(selected_tokens) != budget:
            raise RuntimeError("Conditional beam tree did not fill B")
        return Tree(
            selected_tokens, selected_parents, selected_depths, [],
        )

    @torch.inference_mode()
    def propose(self, hidden: torch.Tensor, draft_logits: torch.Tensor,
                token_embeddings: torch.Tensor, noise_ids: torch.Tensor,
                length: int, temperature: float,
                generator: torch.Generator | None = None, *,
                greedy: bool = False,
                prefix_embeddings: torch.Tensor | None = None,
                support_size: int | None = None,
                strength: float = 1.,
                ) -> DiffusionProposal:
        """Build one autoregressive top-R proposal without another Draft call.

        ``greedy=False`` samples from the returned law for exact block
        correction.  ``greedy=True`` is reserved for a deterministic tree
        scaffold, whose ancestral Target verifier is exact for any fixed tree.
        """
        if (hidden.ndim != 3 or hidden.shape[0] != 1
                or hidden.shape[-1] != self.hidden_size
                or draft_logits.ndim != 2
                or draft_logits.shape[0] != hidden.shape[1]
                or not 1 <= length <= min(hidden.shape[1], self.max_length)
                or not math.isfinite(temperature) or temperature <= 0
                or not math.isfinite(strength) or not 0 <= strength <= 2):
            raise ValueError("Invalid prefix-conditional proposal inputs")
        if (noise_ids.ndim != 1 or noise_ids.numel() <= length):
            raise ValueError("Prefix proposal requires anchor plus masked slots")
        dtype = self.hidden_down.weight.dtype
        width = self.support_size if support_size is None else int(support_size)
        if not 1 <= width <= self.support_size:
            raise ValueError("Prefix proposal support exceeds its trained cap")
        prefix_embeddings = (
            token_embeddings
            if prefix_embeddings is None else prefix_embeddings
        )
        state = None
        previous = hidden.new_zeros((1, self.hidden_size), dtype=dtype)
        base = self.hidden_down(self.hidden_norm(hidden[:, :length].to(dtype)))
        values, candidate_ids = draft_logits[:length].topk(
            width, dim=-1, sorted=True,
        )
        candidate_embeddings = F.embedding(candidate_ids, token_embeddings)
        candidate_keys = self.key(
            self.token_norm(candidate_embeddings.to(dtype))
        )
        support_tokens = []
        support_weights = []
        draws = []
        for depth in range(length):
            step_input = base[:, depth] + self.prefix_down(
                self.token_norm(previous)
            )
            output, state = self.recurrent(step_input[:, None], state)
            query = self.query(output[:, 0])[:, None]
            correction = (
                query * candidate_keys[depth][None]
            ).sum(-1) / self.rank ** 0.5
            residual = (
                self.correction_scale * correction
                + self.depth_bias[depth, :width][None]
            )
            scores = values[depth][None].to(dtype) + float(strength) * residual
            weights = torch.softmax(
                scores[0].double() / float(temperature), dim=-1,
            )
            draw = (
                weights.argmax()
                if greedy else torch.multinomial(
                    weights, 1, generator=generator,
                )[0]
            )
            selected = candidate_ids[depth].index_select(
                0, draw.reshape(1),
            )[0]
            previous = F.embedding(
                selected.reshape(1), prefix_embeddings,
            ).to(dtype)
            support_tokens.append(candidate_ids[depth])
            support_weights.append(weights)
            draws.append(draw)
        tokens = torch.stack(support_tokens)
        weights = torch.stack(support_weights)
        draw_tensor = torch.stack(draws)
        slots = torch.arange(
            weights.shape[1], dtype=torch.long, device=weights.device,
        )[None, None].expand(1, length, -1)
        law = DiffusionBlockLaw(
            tokens=tokens,
            weights=weights,
            retained_mass=weights.new_ones(length),
            noise_ids=noise_ids[:length + 1].clone(),
            draft_temperature=float(temperature),
            mask_token_id=int(noise_ids[1]),
        )
        return DiffusionProposal(law, slots, weights, draw_tensor)


def load_prefix_conditional_head(
        checkpoint: str | Path, hidden_size: int, device: torch.device,
        expected_metadata: Mapping[str, Any],
        dtype: torch.dtype = torch.float32) -> PrefixConditionalHead:
    payload = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or set(payload) < {"state_dict", "metadata"}:
        raise ValueError("Prefix head checkpoint requires state_dict and metadata")
    metadata = payload["metadata"]
    required = {
        "architecture": "prefix_conditional_topr_v2_split_embeddings",
        "hidden_size": int(hidden_size),
        **dict(expected_metadata),
    }
    mismatched = {
        key: (metadata.get(key), value)
        for key, value in required.items() if metadata.get(key) != value
    }
    if mismatched or int(metadata.get("updates", 0)) < 1:
        raise ValueError(
            f"Prefix head checkpoint contract failed: mismatched={mismatched}, "
            f"updates={metadata.get('updates')}"
        )
    model = PrefixConditionalHead(
        hidden_size=hidden_size,
        rank=int(metadata["rank"]),
        support_size=int(metadata["support_size"]),
        max_length=int(metadata["max_length"]),
    )
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.to(device=device, dtype=dtype).eval()
