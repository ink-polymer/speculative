"""Contextual-bandit DDTree policy tests."""

from __future__ import annotations

import json

import torch

from dflash_specblock.bandit_builder import ContextualBanditDDTreeBuilder
from dflash_specblock.ddtree_builder import DDTreeBuilder


def _peaked_logits(depth: int = 4, vocabulary: int = 64) -> torch.Tensor:
    logits = torch.full((depth, vocabulary), -12.0)
    for row in range(depth):
        logits[row, row * 4 + 1] = 8.0
        logits[row, row * 4 + 2] = 4.0
        logits[row, row * 4 + 3] = 2.0
    return logits


def _topology(tree):
    return [
        (node.token_id, node.parent, node.depth, node.cumulative_log_probability)
        for node in tree.nodes
    ]


def _builder(*, learning_enabled: bool = True, alpha: float = 0.0):
    return ContextualBanditDDTreeBuilder(
        block_size=4,
        tree_budget=12,
        budget_candidates=(4, 8, 12),
        initial_budget=8,
        exploration_alpha=alpha,
        ridge=1.0,
        warmup_rounds_per_budget=1,
        context_length_scale=64,
        learning_enabled=learning_enabled,
    )


def test_bandit_tree_is_an_exact_prefix_of_max_budget_ddtree() -> None:
    logits = _peaked_logits()
    builder = _builder()
    selected = builder.build_from_logits(logits)
    full = DDTreeBuilder(block_size=4, tree_budget=12).build_from_logits(logits)

    assert len(selected) == 8
    assert _topology(selected) == _topology(full)[:8]
    selected.validate(8)


def test_bandit_warms_arms_then_learns_direct_round_throughput() -> None:
    logits = _peaked_logits()
    builder = _builder()
    expected_order = (8, 4, 12)
    feedback = {
        4: dict(draft_ms=10.0, verify_ms=90.0, accepted_draft_tokens=1),
        8: dict(draft_ms=5.0, verify_ms=25.0, accepted_draft_tokens=2),
        12: dict(draft_ms=1.0, verify_ms=4.0, accepted_draft_tokens=4),
    }

    for budget in expected_order:
        tree = builder.build_from_logits(logits)
        assert len(tree) == budget
        builder.observe(tree_nodes=budget, **feedback[budget])

    chosen = builder.build_from_logits(logits)
    assert len(chosen) == 12
    assert builder.last_decision is not None
    assert builder.last_decision.budget == 12
    assert builder.last_decision.forced_exploration is False
    assert builder.policy_diagnostics()["total_updates"] == 3


def test_context_contains_current_target_kv_length() -> None:
    logits = _peaked_logits()
    builder = _builder(learning_enabled=False)
    builder.set_runtime_context(prefix_length=0)
    builder.build_from_logits(logits)
    assert builder.last_decision is not None
    short_context = builder.last_decision.context_features

    builder.set_runtime_context(prefix_length=64)
    builder.build_from_logits(logits)
    assert builder.last_decision is not None
    long_context = builder.last_decision.context_features

    assert short_context[:-1] == long_context[:-1]
    assert short_context[-1] == 0.0
    assert long_context[-1] == 1.0


def test_policy_checkpoint_round_trip_and_frozen_inference() -> None:
    logits = _peaked_logits()
    trained = _builder()
    for budget in (8, 4, 12):
        tree = trained.build_from_logits(logits)
        assert len(tree) == budget
        trained.observe(
            tree_nodes=budget,
            draft_ms=1.0,
            verify_ms=float(20 - budget),
            accepted_draft_tokens=budget // 4,
        )

    payload = json.loads(json.dumps(trained.policy_state()))
    assert payload["algorithm"] == "disjoint_linucb"
    assert payload["total_updates"] == 3

    frozen = _builder(learning_enabled=False)
    frozen.load_policy_state(payload)
    before = frozen.policy_diagnostics()["total_updates"]
    tree = frozen.build_from_logits(logits)
    frozen.observe(
        tree_nodes=len(tree),
        draft_ms=1.0,
        verify_ms=1.0,
        accepted_draft_tokens=4,
    )
    assert frozen.policy_diagnostics()["total_updates"] == before


def test_policy_rejects_incompatible_budget_checkpoint() -> None:
    state = _builder().policy_state()
    state["tree_budget"] = 99
    try:
        _builder().load_policy_state(state)
    except ValueError as error:
        assert "不匹配" in str(error)
    else:
        raise AssertionError("不兼容的 bandit checkpoint 被接受")
