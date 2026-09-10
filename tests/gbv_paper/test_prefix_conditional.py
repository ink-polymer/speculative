"""Prefix-conditional proposal architecture and checkpoint contracts."""
from pathlib import Path

import pytest
import torch

from gbv_experiments.prefix_conditional import (
    PrefixConditionalHead,
    RankCalibratedHead,
    load_prefix_conditional_head,
)
from gbv_experiments.train_prefix_conditional import (
    coverage_distillation_loss, distillation_loss,
)


def test_remaining_kl_weights_early_prefix_errors_more():
    teacher = torch.tensor([[[.8, .2], [.8, .2], [.8, .2]]])
    good = teacher.log()
    early_bad = good.clone()
    early_bad[:, 0] = torch.tensor([.2, .8]).log()
    late_bad = good.clone()
    late_bad[:, 2] = torch.tensor([.2, .8]).log()

    early = distillation_loss(early_bad, teacher, "remaining_kl")
    late = distillation_loss(late_bad, teacher, "remaining_kl")
    assert early > late
    torch.testing.assert_close(
        distillation_loss(early_bad, teacher, "mean_kl"),
        distillation_loss(late_bad, teacher, "mean_kl"),
    )


def test_coverage_kl_is_zero_for_unchanged_target_distribution():
    torch.manual_seed(10)
    logits = torch.randn(1, 3, 17)
    candidates = logits.topk(5, dim=-1).indices
    scores = logits.gather(2, candidates)
    loss = coverage_distillation_loss(
        scores, logits, logits.clone(), candidates,
    )
    torch.testing.assert_close(loss, torch.zeros_like(loss), atol=2e-7, rtol=0)


def test_teacher_forcing_is_strictly_prefix_causal():
    torch.manual_seed(11)
    head = PrefixConditionalHead(12, rank=5, support_size=4, max_length=5)
    hidden = torch.randn(1, 5, 12)
    logits = torch.randn(1, 5, 17)
    embeddings = torch.randn(17, 12)
    labels = torch.tensor([[1, 2, 3, 4, 5]])
    before, ids = head.teacher_forced_scores(
        hidden, logits, embeddings, labels,
    )
    changed = labels.clone()
    changed[:, 3:] = torch.tensor([[8, 9]])
    after, changed_ids = head.teacher_forced_scores(
        hidden, logits, embeddings, changed,
    )

    assert torch.equal(ids, changed_ids)
    torch.testing.assert_close(before[:, :4], after[:, :4])


def test_zero_initialized_residual_starts_at_dflash_scores():
    torch.manual_seed(13)
    head = PrefixConditionalHead(12, rank=5, support_size=4, max_length=5)
    hidden = torch.randn(1, 5, 12)
    logits = torch.randn(1, 5, 17)
    embeddings = torch.randn(17, 12)
    labels = torch.tensor([[1, 2, 3, 4, 5]])
    scores, candidate_ids = head.teacher_forced_scores(
        hidden, logits, embeddings, labels,
    )

    torch.testing.assert_close(scores, logits.gather(2, candidate_ids))


def test_sampled_proposal_retains_its_actual_conditional_rows():
    torch.manual_seed(12)
    head = PrefixConditionalHead(10, rank=4, support_size=3, max_length=4)
    hidden = torch.randn(1, 4, 10)
    logits = torch.randn(4, 13)
    embeddings = torch.randn(13, 10)
    noise_ids = torch.tensor([7, 12, 12, 12, 12])
    proposal = head.propose(
        hidden, logits, embeddings, noise_ids, length=3, temperature=.7,
        generator=torch.Generator().manual_seed(4),
    )

    proposal.validate(13)
    assert proposal.paths().shape == (1, 3)
    assert proposal.law.tokens.shape == (3, 3)
    assert proposal.law.weights.dtype == torch.float64
    assert torch.equal(proposal.source, proposal.law.weights)
    assert torch.equal(
        proposal.slots[0], torch.arange(3)[None].expand(3, -1),
    )


def test_inference_support_can_be_narrower_than_training_support():
    torch.manual_seed(16)
    head = PrefixConditionalHead(10, rank=4, support_size=6, max_length=4)
    proposal = head.propose(
        torch.randn(1, 4, 10), torch.randn(4, 13),
        torch.randn(13, 10), torch.tensor([7, 12, 12, 12, 12]),
        length=3, temperature=1., support_size=2,
    )

    assert proposal.source.shape == (3, 2)
    assert proposal.law.tokens.shape == (3, 2)


def test_zero_prefix_strength_recovers_truncated_dflash_law():
    torch.manual_seed(17)
    head = PrefixConditionalHead(10, rank=4, support_size=4, max_length=4)
    logits = torch.randn(4, 13)
    proposal = head.propose(
        torch.randn(1, 4, 10), logits, torch.randn(13, 10),
        torch.tensor([7, 12, 12, 12, 12]), length=3,
        temperature=.8, strength=0.,
    )
    expected = torch.softmax(logits[:3].topk(4, -1).values.double() / .8, -1)

    torch.testing.assert_close(proposal.law.weights, expected)


def test_greedy_proposal_selects_each_conditional_argmax():
    torch.manual_seed(14)
    head = PrefixConditionalHead(10, rank=4, support_size=3, max_length=4)
    hidden = torch.randn(1, 4, 10)
    logits = torch.randn(4, 13)
    embeddings = torch.randn(13, 10)
    noise_ids = torch.tensor([7, 12, 12, 12, 12])
    proposal = head.propose(
        hidden, logits, embeddings, noise_ids, length=3, temperature=.7,
        generator=torch.Generator().manual_seed(4), greedy=True,
    )

    assert torch.equal(proposal.draws, proposal.source.argmax(-1))


def test_rescored_tree_is_ancestry_closed_and_budgeted():
    torch.manual_seed(15)
    head = PrefixConditionalHead(10, rank=4, support_size=5, max_length=4)
    hidden = torch.randn(1, 4, 10)
    logits = torch.randn(4, 13)
    embeddings = torch.randn(13, 10)
    q = torch.softmax(logits.double(), -1)
    tree = head.build_tree(
        hidden, logits, embeddings, q, budget=9, temperature=.7,
    )

    assert len(tree.tokens) == 9
    assert len(tree.parents) == 10
    assert all(
        0 <= parent < node and tree.depths[node] == tree.depths[parent] + 1
        for node, parent in enumerate(tree.parents[1:], 1)
    )
    assert len(set(zip(tree.parents[1:], tree.tokens))) == 9


def test_zero_residual_tree_preserves_actual_topr_mass():
    torch.manual_seed(19)
    head = PrefixConditionalHead(10, rank=4, support_size=5, max_length=4)
    hidden = torch.randn(1, 4, 10)
    logits = torch.randn(4, 13)
    embeddings = torch.randn(13, 10)
    q = torch.softmax(logits.double() / .7, -1)
    values, tokens = q.topk(5, dim=-1, sorted=True)
    actual = head.build_tree(
        hidden, logits, embeddings, q, budget=9, temperature=.7,
    )
    expected = head._sparse_probability_tree(tokens, values, budget=9)

    assert actual.tokens == expected.tokens
    assert actual.parents == expected.parents
    assert actual.depths == expected.depths


def test_conditional_beam_tree_is_ancestry_closed_and_budgeted():
    torch.manual_seed(18)
    head = PrefixConditionalHead(10, rank=4, support_size=5, max_length=4)
    tree = head.build_beam_tree(
        torch.randn(1, 4, 10), torch.randn(4, 13),
        torch.randn(13, 10), budget=9, temperature=.7,
        support_size=4,
    )

    assert len(tree.tokens) == 9
    assert len(tree.parents) == 10
    assert all(
        0 <= parent < node and tree.depths[node] == tree.depths[parent] + 1
        for node, parent in enumerate(tree.parents[1:], 1)
    )
    assert len(set(zip(tree.parents[1:], tree.tokens))) == 9


def test_prefix_head_checkpoint_contract(tmp_path: Path):
    head = PrefixConditionalHead(8, rank=3, support_size=4, max_length=5)
    checkpoint = tmp_path / "prefix.pt"
    torch.save({
        "state_dict": head.state_dict(),
        "metadata": {
            "architecture": "prefix_conditional_topr_v2_split_embeddings",
            "hidden_size": 8,
            "rank": 3,
            "support_size": 4,
            "max_length": 5,
            "updates": 2,
            "target_revision": "target-rev",
            "draft_revision": "draft-rev",
        },
    }, checkpoint)
    loaded = load_prefix_conditional_head(
        checkpoint, 8, torch.device("cpu"), {
            "target_revision": "target-rev",
            "draft_revision": "draft-rev",
        },
    )
    assert isinstance(loaded, PrefixConditionalHead)
    with pytest.raises(ValueError, match="contract failed"):
        load_prefix_conditional_head(
            checkpoint, 8, torch.device("cpu"), {
                "target_revision": "different",
                "draft_revision": "draft-rev",
            },
        )


@pytest.mark.parametrize("method", [
    "prefix_core_spur_bv", "prefix_core_spur_tree",
    "prefix_sampled_spur_tree", "prefix_rescored_tree", "prefix_beam_tree",
])
@pytest.mark.parametrize("temperature", [.3, 1.])
def test_prefix_core_spur_engine_uses_one_draft_and_target_forward(
        tiny_engine, temperature, method):
    from gbv_experiments.config import Variant
    from gbv_experiments.diffusion_preflight import probe

    head = PrefixConditionalHead(
        int(tiny_engine.draft.config.hidden_size),
        rank=4, support_size=8, max_length=3,
    )
    tiny_engine.proposal_adapter = head
    variant = Variant(
        name="prefix_core_spur", method=method,
        paths=1, length=3, diffusion_spur_length=2,
        diffusion_support_size=8, tree_budget=9,
        temperature=temperature, draft_temperature=temperature,
    )
    ids = torch.tensor([[1, 3, 5]])
    try:
        result = tiny_engine.generate(ids, variant, 18, [], seed=42)
        repeat = tiny_engine.generate(ids, variant, 18, [], seed=42)
        assert result["generated_token_ids"] == repeat["generated_token_ids"]
        assert result["target_forward_calls"] == 1 + result["draft_forward_calls"]
        assert result["draft_forward_calls"] == len(result["rounds"])
        assert all(
            round_["prefix_conditioned_proposal"]
            and round_["tree_nodes"] <= 9
            and round_["core_spur_length"] == (
                None if method in {"prefix_rescored_tree", "prefix_beam_tree"}
                else 2
            )
            and round_["joint_block_verification"] == (
                method == "prefix_core_spur_bv"
            )
            for round_ in result["rounds"]
        )
        assert probe(
            tiny_engine, ids, variant, tokens=12, seed=42,
            tv_limit=1e-6,
        )["passed"]
    finally:
        tiny_engine.proposal_adapter = None


def test_parallel_rank_calibrator_starts_at_sparse_draft_tree():
    torch.manual_seed(17)
    head = RankCalibratedHead(12, rank=5, support_size=4, max_length=5)
    hidden = torch.randn(1, 5, 12)
    logits = torch.randn(1, 5, 17)
    scores, tokens = head.scores(hidden, logits)
    values, expected_tokens = logits.topk(4, dim=-1, sorted=True)
    torch.testing.assert_close(scores, values)
    assert torch.equal(tokens, expected_tokens)
    tree = head.build_tree(hidden, logits[0], budget=8, temperature=1.0)
    assert len(tree.tokens) == 8
