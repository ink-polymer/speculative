from pathlib import Path

import pytest
import torch

from gbv_experiments.slot_mixer import (CandidateRatioTransportHead,
                                        CausalSlotMixer, load_slot_mixer)


def test_slot_mixer_is_identity_at_initialization():
    model = CausalSlotMixer(16, rank=4, kernel_size=3)
    hidden = torch.randn(2, 7, 16)
    torch.testing.assert_close(model(hidden), hidden)


def test_slot_mixer_is_strictly_causal_after_training_like_update():
    model = CausalSlotMixer(12, rank=3, kernel_size=3)
    torch.nn.init.normal_(model.up.weight, std=0.1)
    left = torch.randn(1, 6, 12)
    right = left.clone()
    right[:, 4:] = torch.randn_like(right[:, 4:])
    actual_left = model(left)
    actual_right = model(right)
    torch.testing.assert_close(actual_left[:, :4], actual_right[:, :4])


def test_slot_mixer_checkpoint_contract(tmp_path: Path):
    model = CausalSlotMixer(8, rank=2, kernel_size=2)
    checkpoint = tmp_path / "mixer.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "metadata": {
            "architecture": "parallel_causal_slot_mixer_v1",
            "hidden_size": 8,
            "rank": 2,
            "kernel_size": 2,
            "updates": 1,
            "target_revision": "target-rev",
            "draft_revision": "draft-rev",
        },
    }, checkpoint)
    loaded = load_slot_mixer(
        checkpoint, 8, torch.device("cpu"),
        {"target_revision": "target-rev", "draft_revision": "draft-rev"},
    )
    hidden = torch.randn(1, 3, 8)
    torch.testing.assert_close(loaded(hidden), model(hidden), rtol=0, atol=0)
    with pytest.raises(ValueError, match="contract failed"):
        load_slot_mixer(
            checkpoint, 8, torch.device("cpu"),
            {"target_revision": "different", "draft_revision": "draft-rev"},
        )


def test_candidate_ratio_head_only_changes_selected_logits():
    torch.manual_seed(4)
    head = CandidateRatioTransportHead(8, rank=4, candidates=3)
    hidden = torch.randn(1, 2, 8)
    logits = torch.randn(1, 2, 11)
    embeddings = torch.randn(11, 8)
    selected = logits.topk(3, dim=-1).indices
    result = head.correct_logits(hidden, logits, embeddings)
    untouched = torch.ones_like(logits, dtype=torch.bool)
    untouched.scatter_(2, selected, False)
    assert torch.equal(result[untouched], logits[untouched])
    assert torch.isfinite(result).all()


def test_candidate_ratio_head_has_no_cross_slot_information_flow():
    torch.manual_seed(5)
    head = CandidateRatioTransportHead(8, rank=4, candidates=3)
    embeddings = torch.randn(13, 8)
    ids = torch.tensor([[[1, 2, 3], [4, 5, 6]]])
    hidden = torch.randn(1, 2, 8)
    before = head.candidate_corrections(hidden, embeddings, ids)
    changed = hidden.clone()
    changed[:, 1] += 100
    after = head.candidate_corrections(changed, embeddings, ids)
    assert torch.equal(before[:, 0], after[:, 0])


def test_ratio_head_trains_against_copied_frozen_teacher_values():
    head = CandidateRatioTransportHead(8, rank=4, candidates=3)
    embeddings = torch.randn(13, 8)
    with torch.inference_mode():
        frozen_hidden = torch.randn(1, 2, 8)
        frozen_ids = torch.tensor([[[1, 2, 3], [4, 5, 6]]])
        frozen_desired = torch.randn(1, 2, 3)
    hidden = frozen_hidden.clone()
    ids = frozen_ids.clone()
    desired = frozen_desired.clone()
    predicted = head.candidate_corrections(hidden, embeddings, ids)
    torch.nn.functional.smooth_l1_loss(predicted, desired).backward()
    assert all(parameter.grad is not None for parameter in head.parameters())
