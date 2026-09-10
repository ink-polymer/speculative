"""Parallel causal slot mixer for DFlash proposal calibration.

This module changes the proposal architecture, not the verifier law.  It mixes
the already-computed DFlash slot states once in parallel and never conditions
on a sampled tree branch.  Target and DFlash parameters remain frozen.
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


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


class ResidualSlotMLP(nn.Module):
    """Position-wise SwiGLU residual adapter for DFlash slot states."""

    def __init__(self, hidden_size: int, bottleneck: int = 512):
        super().__init__()
        if hidden_size < 1 or bottleneck < 1:
            raise ValueError("Invalid residual slot MLP dimensions")
        self.hidden_size = int(hidden_size)
        self.bottleneck = int(bottleneck)
        self.norm = nn.RMSNorm(hidden_size)
        self.in_proj = nn.Linear(hidden_size, 2 * bottleneck, bias=False)
        self.out_proj = nn.Linear(bottleneck, hidden_size, bias=False)
        nn.init.zeros_(self.out_proj.weight)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.ndim != 3 or hidden.shape[-1] != self.hidden_size:
            raise ValueError("Slot MLP expects [batch, slots, hidden_size]")
        dtype = self.in_proj.weight.dtype
        gate, value = self.in_proj(self.norm(hidden.to(dtype))).chunk(2, dim=-1)
        residual = self.out_proj(F.silu(gate) * value)
        return hidden + residual.to(hidden.dtype)


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
