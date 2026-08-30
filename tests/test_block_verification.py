from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


OFFICIAL = Path(__file__).resolve().parents[1] / "third_party" / "ddtree_official"
sys.path.insert(0, str(OFFICIAL))

from block_verification import (  # noqa: E402
    block_rejection_sample,
    sampling_probs,
    token_rejection_sample,
)


def test_sampling_probs_requires_positive_temperature() -> None:
    with pytest.raises(ValueError, match="temperature > 0"):
        sampling_probs(torch.zeros(1, 2), 0.0)


def test_block_verification_accepts_full_block_when_p_equals_q() -> None:
    drafts = torch.tensor([0, 1])
    q = torch.tensor([[0.8, 0.2], [0.3, 0.7]])
    p = torch.cat((q, torch.tensor([[0.4, 0.6]])), dim=0)
    for seed in range(20):
        torch.manual_seed(seed)
        accepted, bonus = block_rejection_sample(drafts, p, q)
        assert accepted == 2
        assert int(bonus) in {0, 1}


def test_block_verification_rejects_disjoint_first_token() -> None:
    drafts = torch.tensor([0])
    q = torch.tensor([[1.0, 0.0]])
    p = torch.tensor([[0.0, 1.0], [0.25, 0.75]])
    for seed in range(20):
        torch.manual_seed(seed)
        accepted, bonus = block_rejection_sample(drafts, p, q)
        assert accepted == 0
        assert int(bonus) == 1


def test_token_and_block_verification_validate_shapes() -> None:
    drafts = torch.tensor([0, 1])
    q = torch.full((2, 3), 1 / 3)
    with pytest.raises(ValueError, match=r"gamma \+ 1"):
        block_rejection_sample(drafts, torch.full((2, 3), 1 / 3), q)
    accepted, bonus = token_rejection_sample(
        drafts, torch.full((3, 3), 1 / 3), q
    )
    assert 0 <= accepted <= 2
    assert 0 <= int(bonus) < 3
