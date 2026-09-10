"""Parallel causal slot mixer for DFlash proposal calibration.

This module changes the proposal architecture, not the verifier law.  It mixes
the already-computed DFlash slot states once in parallel and never conditions
on a sampled tree branch.  Target and DFlash parameters remain frozen.
"""
from __future__ import annotations

from collections.abc import Mapping
import heapq
import math
from pathlib import Path
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from .tree import Tree


class CausalSlotMixer(nn.Module):
    """Low-rank, strictly-left-context residual mixer over draft slots."""

    def __init__(self, hidden_size: int, rank: int = 64, kernel_size: int = 3):
        super().__init__()
        if hidden_size < 1 or rank < 1 or kernel_size < 2:
            raise ValueError("Invalid causal slot mixer dimensions")
        self.hidden_size = int(hidden_size)
        self.rank = int(rank)
        self.kernel_size = int(kernel_size)
        self.norm = nn.RMSNorm(hidden_size)
        self.down = nn.Linear(hidden_size, rank, bias=False)
        self.causal = nn.Conv1d(
            rank, rank, kernel_size, groups=rank, bias=False
        )
        self.up = nn.Linear(rank, hidden_size, bias=False)
        # The checkpoint starts as an exact identity proposal.  Training first
        # learns the residual readout, then gradients reach the mixer trunk.
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.ndim != 3 or hidden.shape[-1] != self.hidden_size:
            raise ValueError("Slot mixer expects [batch, slots, hidden_size]")
        parameter_dtype = self.down.weight.dtype
        working = hidden.to(parameter_dtype)
        projected = self.down(self.norm(working)).transpose(1, 2)
        # Pad only on the left and remove the current-slot contribution.  Thus
        # output slot i depends on draft slots < i, never i+1 or a sampled path.
        shifted = F.pad(projected, (self.kernel_size, 0))[..., :-1]
        mixed = self.causal(shifted).transpose(1, 2)
        return hidden + self.up(F.silu(mixed)).to(hidden.dtype)


class MarkovBranchHead(nn.Module):
    """Cheap parent-token-conditioned scorer for a DDTree topology.

    DFlash still runs exactly once. The head scores a fixed top-R support at
    every depth for every possible parent in the preceding support. This makes
    the B-node tree branch conditional without a serial model call or a change
    to the exact Target verifier.
    """

    def __init__(self, hidden_size: int, rank: int = 32,
                 support_size: int = 16, max_length: int = 15,
                 start_depth: int = 1):
        super().__init__()
        if min(hidden_size, rank, support_size, max_length) < 1:
            raise ValueError("Invalid Markov branch-head dimensions")
        self.hidden_size = int(hidden_size)
        self.rank = int(rank)
        self.support_size = int(support_size)
        self.runtime_support_size = int(support_size)
        self.max_length = int(max_length)
        self.start_depth = int(start_depth)
        if not 0 <= self.start_depth < self.max_length:
            raise ValueError("Markov correction start depth is out of range")
        self.hidden_norm = nn.RMSNorm(hidden_size)
        self.token_norm = nn.RMSNorm(hidden_size)
        self.hidden_down = nn.Linear(hidden_size, rank, bias=False)
        self.parent_down = nn.Linear(hidden_size, rank, bias=False)
        self.child_down = nn.Linear(hidden_size, rank, bias=False)
        self.depth_bias = nn.Parameter(torch.zeros(max_length, support_size))
        self.log_scale = nn.Parameter(torch.full((), -2.0))

    def _supports(self, draft_logits: torch.Tensor):
        width = min(self.support_size, draft_logits.shape[-1])
        return draft_logits.topk(width, dim=-1, sorted=True)

    def teacher_forced_scores(self, hidden: torch.Tensor,
                              draft_logits: torch.Tensor,
                              token_embeddings: torch.Tensor,
                              labels: torch.Tensor):
        if (hidden.ndim != 3 or draft_logits.ndim != 3
                or hidden.shape[:2] != draft_logits.shape[:2]
                or labels.shape != hidden.shape[:2]
                or hidden.shape[-1] != self.hidden_size
                or hidden.shape[1] > self.max_length):
            raise ValueError("Markov teacher-forced inputs disagree")
        values, child_ids = self._supports(draft_logits)
        dtype = self.hidden_down.weight.dtype
        base = self.hidden_down(self.hidden_norm(hidden.to(dtype)))
        parent = torch.zeros_like(hidden, dtype=dtype)
        if hidden.shape[1] > 1:
            parent[:, 1:] = F.embedding(
                labels[:, :-1], token_embeddings
            ).to(dtype)
        query = F.silu(base + self.parent_down(self.token_norm(parent)))
        child = F.embedding(child_ids, token_embeddings).to(dtype)
        keys = self.child_down(self.token_norm(child))
        correction = (query[:, :, None] * keys).sum(-1) / math.sqrt(self.rank)
        scores = values.to(dtype) + self.log_scale.exp() * correction
        scores = scores + self.depth_bias[:hidden.shape[1]][None]
        if self.start_depth:
            scores[:, :self.start_depth] = values[
                :, :self.start_depth
            ].to(dtype)
        return scores, child_ids

    @torch.inference_mode()
    def build_tree(self, hidden: torch.Tensor,
                   position_probabilities: torch.Tensor,
                   token_embeddings: torch.Tensor, budget: int,
                   temperature: float) -> Tree:
        if (hidden.ndim != 3 or hidden.shape[0] != 1
                or hidden.shape[-1] != self.hidden_size
                or position_probabilities.ndim != 2
                or position_probabilities.shape[0] != hidden.shape[1]
                or not 1 <= hidden.shape[1] <= self.max_length
                or budget < 1 or not math.isfinite(temperature)
                or temperature <= 0):
            raise ValueError("Invalid Markov branch-tree inputs")
        width = min(
            self.runtime_support_size, position_probabilities.shape[-1]
        )
        values, child_ids = position_probabilities.topk(
            width, dim=-1, sorted=True,
        )
        width = child_ids.shape[-1]
        dtype = self.hidden_down.weight.dtype
        base = self.hidden_down(self.hidden_norm(hidden.to(dtype)))[0]
        child = F.embedding(child_ids, token_embeddings).to(dtype)
        keys = self.child_down(self.token_norm(child))
        scale = self.log_scale.exp()

        base_log = values.double().log()
        tail_log = (1.0 - values.double().sum(-1)).clamp_min(
            torch.finfo(torch.float64).tiny
        ).log()

        def normalize(adjusted: torch.Tensor,
                      log_tail: torch.Tensor) -> torch.Tensor:
            denominator = torch.logaddexp(
                log_tail, torch.logsumexp(adjusted, dim=-1)
            )
            return adjusted - denominator[..., None]

        root_query = F.silu(base[0])
        root_correction = torch.zeros_like(values[0], dtype=dtype)
        if self.start_depth == 0:
            root_correction = scale * (
                root_query[None] * keys[0]
            ).sum(-1) / math.sqrt(self.rank)
            root_correction = root_correction + self.depth_bias[0, :width]
        root_correction = root_correction.double() / temperature
        root_log = normalize(base_log[0] + root_correction, tail_log[0])
        if hidden.shape[1] > 1:
            parent_states = self.parent_down(self.token_norm(child[:-1]))
            queries = F.silu(base[1:, None] + parent_states)
            corrections = torch.einsum("dpr,dcr->dpc", queries, keys[1:])
            corrections = scale * corrections / math.sqrt(self.rank)
            corrections = corrections + self.depth_bias[
                1:hidden.shape[1], :width
            ][:, None]
            if self.start_depth > 1:
                corrections[:self.start_depth - 1] = 0
            adjusted = base_log[1:, None] + corrections.double() / temperature
            later_log = normalize(adjusted, tail_log[1:, None])
            log_rows = [root_log.tolist(), *later_log.tolist()]
        else:
            log_rows = [root_log.tolist()]

        ids = child_ids.tolist()
        logs = log_rows
        result_tokens, parents_out, depths = [], [-1], [0]
        heap = [(-logs[0][rank], 0, 0, rank) for rank in range(width)]
        heapq.heapify(heap)
        while heap and len(result_tokens) < budget:
            negative, parent_node, depth, child_rank = heapq.heappop(heap)
            node = len(parents_out)
            result_tokens.append(ids[depth][child_rank])
            parents_out.append(parent_node)
            depths.append(depth + 1)
            next_depth = depth + 1
            if next_depth < hidden.shape[1]:
                row = logs[next_depth][child_rank]
                parent_log = -negative
                for next_rank, log_probability in enumerate(row):
                    heapq.heappush(heap, (
                        -(parent_log + log_probability), node,
                        next_depth, next_rank,
                    ))
        return Tree(result_tokens, parents_out, depths, [])


class CandidateRatioTransportHead(nn.Module):
    """Correct only tree-eligible logits with a hidden/token ratio score.

    The head never materializes a second vocabulary projection. A shared
    bilinear score transports target-versus-draft log-density information into
    only the tokens that can enter the fixed-budget verification tree.
    """

    def __init__(self, hidden_size: int, rank: int = 32, candidates: int = 45):
        super().__init__()
        if hidden_size < 1 or rank < 1 or candidates < 1:
            raise ValueError("Invalid candidate-ratio head dimensions")
        self.hidden_size = int(hidden_size)
        self.rank = int(rank)
        self.candidates = int(candidates)
        self.hidden_norm = nn.RMSNorm(hidden_size)
        self.token_norm = nn.RMSNorm(hidden_size)
        self.hidden_down = nn.Linear(hidden_size, rank, bias=False)
        self.token_down = nn.Linear(hidden_size, rank, bias=False)
        self.depth_bias = nn.Parameter(torch.zeros(15))
        self.rank_bias = nn.Parameter(torch.zeros(candidates))
        self.log_scale = nn.Parameter(torch.zeros(()))

    def candidate_corrections(self, hidden: torch.Tensor,
                              token_embeddings: torch.Tensor,
                              candidate_ids: torch.Tensor) -> torch.Tensor:
        if hidden.ndim != 3 or hidden.shape[-1] != self.hidden_size:
            raise ValueError("Ratio head expects [batch, slots, hidden_size]")
        if candidate_ids.shape[:2] != hidden.shape[:2]:
            raise ValueError("Candidate ids disagree with hidden slots")
        width = candidate_ids.shape[-1]
        if width > self.candidates or hidden.shape[1] > self.depth_bias.numel():
            raise ValueError("Candidate-ratio architectural cap exceeded")
        dtype = self.hidden_down.weight.dtype
        h = self.hidden_down(self.hidden_norm(hidden.to(dtype)))
        selected = F.embedding(candidate_ids, token_embeddings.to(dtype))
        e = self.token_down(self.token_norm(selected))
        score = (h[:, :, None, :] * e).sum(-1) / self.rank ** 0.5
        score = score + self.depth_bias[:hidden.shape[1]][None, :, None]
        score = score + self.rank_bias[:width][None, None, :]
        return score * self.log_scale.exp()

    def correct_logits(self, hidden: torch.Tensor, logits: torch.Tensor,
                       token_embeddings: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 3 or logits.shape[:2] != hidden.shape[:2]:
            raise ValueError("Ratio-head logits disagree with hidden slots")
        width = min(self.candidates, logits.shape[-1])
        ids = torch.topk(logits, width, dim=-1, sorted=True).indices
        corrections = self.candidate_corrections(hidden, token_embeddings, ids)
        result = logits.clone()
        result.scatter_add_(2, ids, corrections.to(result.dtype))
        return result


def load_slot_mixer(
    checkpoint: str | Path,
    hidden_size: int,
    device: torch.device,
    expected_metadata: Mapping[str, Any],
    dtype: torch.dtype | None = None,
) -> CausalSlotMixer:
    payload = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or set(payload) < {"state_dict", "metadata"}:
        raise ValueError("Slot mixer checkpoint requires state_dict and metadata")
    metadata = payload["metadata"]
    if not isinstance(metadata, dict):
        raise ValueError("Slot mixer metadata must be a dictionary")
    required = {
        "architecture": "parallel_causal_slot_mixer_v1",
        "hidden_size": int(hidden_size),
        **dict(expected_metadata),
    }
    missing = sorted(set(required) - set(metadata))
    mismatched = {
        key: (metadata.get(key), value)
        for key, value in required.items()
        if metadata.get(key) != value
    }
    if missing or mismatched or int(metadata.get("updates", 0)) < 1:
        raise ValueError(
            f"Slot mixer checkpoint contract failed: missing={missing}, "
            f"mismatched={mismatched}, updates={metadata.get('updates')}"
        )
    model = CausalSlotMixer(
        hidden_size=hidden_size,
        rank=int(metadata["rank"]),
        kernel_size=int(metadata["kernel_size"]),
    )
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.to(device=device, dtype=dtype).eval()


def load_markov_branch_head(
    checkpoint: str | Path,
    hidden_size: int,
    device: torch.device,
    expected_metadata: Mapping[str, Any],
    dtype: torch.dtype | None = None,
) -> MarkovBranchHead:
    payload = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or set(payload) < {"state_dict", "metadata"}:
        raise ValueError("Markov branch checkpoint requires state_dict and metadata")
    metadata = payload["metadata"]
    required = {
        "architecture": "markov_branch_topr_v1",
        "hidden_size": int(hidden_size),
        **dict(expected_metadata),
    }
    mismatched = {
        key: (metadata.get(key), value)
        for key, value in required.items() if metadata.get(key) != value
    }
    if mismatched or int(metadata.get("updates", 0)) < 1:
        raise ValueError(
            f"Markov branch checkpoint contract failed: mismatched={mismatched}, "
            f"updates={metadata.get('updates')}"
        )
    model = MarkovBranchHead(
        hidden_size=hidden_size,
        rank=int(metadata["rank"]),
        support_size=int(metadata["support_size"]),
        max_length=int(metadata["max_length"]),
        start_depth=int(metadata.get("start_depth", 1)),
    )
    model.load_state_dict(payload["state_dict"], strict=True)
    runtime_support = int(metadata.get("runtime_support_size", model.support_size))
    if not 1 <= runtime_support <= model.support_size:
        raise ValueError("Markov runtime support is out of trained range")
    model.runtime_support_size = runtime_support
    return model.to(device=device, dtype=dtype).eval()


def load_ratio_transport_head(
    checkpoint: str | Path,
    hidden_size: int,
    device: torch.device,
    expected_metadata: Mapping[str, Any],
    dtype: torch.dtype | None = None,
) -> CandidateRatioTransportHead:
    payload = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or set(payload) < {"state_dict", "metadata"}:
        raise ValueError("Ratio-head checkpoint requires state_dict and metadata")
    metadata = payload["metadata"]
    required = {
        "architecture": "candidate_ratio_transport_v1",
        "hidden_size": int(hidden_size),
        **dict(expected_metadata),
    }
    mismatched = {
        key: (metadata.get(key), value)
        for key, value in required.items() if metadata.get(key) != value
    }
    if mismatched or int(metadata.get("updates", 0)) < 1:
        raise ValueError(
            f"Ratio-head checkpoint contract failed: mismatched={mismatched}, "
            f"updates={metadata.get('updates')}"
        )
    model = CandidateRatioTransportHead(
        hidden_size=hidden_size,
        rank=int(metadata["rank"]),
        candidates=int(metadata["candidates"]),
    )
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.to(device=device, dtype=dtype).eval()
