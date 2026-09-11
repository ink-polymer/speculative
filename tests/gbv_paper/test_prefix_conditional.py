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
    tree_mass_importance_weights,
)
from scripts.train_prefix_tree_selector import selection_loss


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


def test_tree_mass_weights_are_prefix_causal_and_clipped():
    target = torch.full((1, 3, 5), -20.)
    labels = torch.tensor([[1, 2, 3]])
    target[0, 0, 1] = 0.
    target[0, 1, 2] = 0.
    target[0, 2, 3] = 0.
    proposal = torch.tensor([[.5, .25, .125]])

    weights = tree_mass_importance_weights(
        target, labels, proposal, clip=3.,
    )

    torch.testing.assert_close(weights[:, :1], torch.ones(1, 1))
    assert weights[0, 1] == pytest.approx(2.)
    assert weights[0, 2] == pytest.approx(3.)


def test_weighted_coverage_kl_focuses_selected_depths():
    torch.manual_seed(20)
    draft = torch.randn(1, 3, 13)
    target = torch.randn(1, 3, 13)
    candidates = draft.topk(4, -1).indices
    scores = draft.gather(2, candidates)
    all_rows = coverage_distillation_loss(
        scores, draft, target, candidates,
    )
    first_only = coverage_distillation_loss(
        scores, draft, target, candidates,
        depth_weights=torch.tensor([[1., 0., 0.]]),
    )

    assert all_rows > 0
    assert first_only > 0
    assert not torch.isclose(all_rows, first_only)


def test_selector_loss_prefers_correct_top_budget_order():
    target = torch.tensor([0., 5., 4., 3., 2., 1.])
    correct = target.clone()
    reversed_scores = torch.cat((target[:1], target[1:].flip(0)))

    assert selection_loss(correct, target, 2) < selection_loss(
        reversed_scores, target, 2,
    )


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


def test_runtime_proposal_matches_teacher_forced_scaled_scores():
    torch.manual_seed(141)
    head = PrefixConditionalHead(10, rank=4, support_size=3, max_length=4)
    torch.nn.init.normal_(head.query.weight)
    torch.nn.init.normal_(head.depth_bias)
    head.correction_scale.data.fill_(0.37)
    hidden = torch.randn(1, 4, 10)
    logits = torch.randn(4, 13)
    embeddings = torch.randn(13, 10)
    proposal = head.propose(
        hidden, logits, embeddings, torch.tensor([7, 12, 12, 12, 12]),
        length=3, temperature=.7, greedy=True,
    )
    labels = proposal.law.tokens.gather(
        1, proposal.draws[:, None],
    )[:, 0][None]
    scores, candidate_ids = head.teacher_forced_scores(
        hidden[:, :3], logits[None, :3], embeddings, labels,
    )

    assert torch.equal(candidate_ids[0], proposal.law.tokens)
    torch.testing.assert_close(
        torch.softmax(scores[0].double() / .7, -1),
        proposal.law.weights,
    )


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


def test_hybrid_tree_preserves_the_declared_ddtree_core():
    torch.manual_seed(191)
    head = PrefixConditionalHead(10, rank=4, support_size=5, max_length=4)
    torch.nn.init.normal_(head.query.weight)
    hidden = torch.randn(1, 4, 10)
    logits = torch.randn(4, 13)
    embeddings = torch.randn(13, 10)
    q = torch.softmax(logits.double(), -1)
    values, candidate_ids = q.topk(5, dim=-1, sorted=True)
    core = head._sparse_probability_tree(candidate_ids, values, budget=4)
    hybrid = head.build_tree(
        hidden, logits, embeddings, q, budget=9, temperature=1.,
        pool_factor=3, core_budget=4,
    )

    assert hybrid.tokens[:4] == core.tokens
    assert hybrid.parents[:5] == core.parents
    assert hybrid.depths[:5] == core.depths


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


def test_embedded_prefix_proposal_preserves_the_exact_ddtree_nodes():
    from gbv_experiments.tree import (
        embedded_prefix_proposal, probability_tree,
    )

    q = torch.tensor([
        [.55, .30, .10, .05],
        [.50, .25, .15, .10],
        [.45, .30, .15, .10],
    ], dtype=torch.float64)
    tree = probability_tree(q, budget=9)
    before = (tree.tokens.copy(), tree.parents.copy(), tree.depths.copy())
    proposal = embedded_prefix_proposal(q, tree, length=2)

    assert (tree.tokens, tree.parents, tree.depths) == before
    assert proposal.paths.shape[1] == 2
    assert proposal.leaf_probabilities.sum() == pytest.approx(1.)
    for path, nodes in zip(proposal.paths.tolist(), tree.path_nodes):
        assert [tree.tokens[node - 1] for node in nodes] == path
        assert all(tree.parents[node] == (0 if i == 0 else nodes[i - 1])
                   for i, node in enumerate(nodes))


@pytest.mark.parametrize("method", [
    "prefix_core_spur_bv", "prefix_core_spur_bv_lazy",
    "prefix_core_spur_tree",
    "prefix_sampled_spur_tree", "prefix_rescored_tree",
    "prefix_rescored_tree_fused_scan", "prefix_beam_tree",
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
                None if method in {
                    "prefix_rescored_tree", "prefix_rescored_tree_fused_scan",
                    "prefix_beam_tree",
                }
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
