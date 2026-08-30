"""Latency-aware DDTree 的嵌套树与在线预算策略测试。"""

from __future__ import annotations

import torch

from dflash_specblock.ddtree_builder import DDTreeBuilder, LatencyAwareDDTreeBuilder


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


def test_adaptive_tree_is_an_exact_prefix_of_max_budget_ddtree() -> None:
    logits = _peaked_logits()
    adaptive = LatencyAwareDDTreeBuilder(
        block_size=4,
        tree_budget=12,
        budget_candidates=(4, 8, 12),
        initial_budget=8,
        warmup_rounds_per_budget=1,
        exploration_interval=0,
    )
    selected = adaptive.build_from_logits(logits)
    full = DDTreeBuilder(block_size=4, tree_budget=12).build_from_logits(logits)

    assert len(selected) == 8
    assert _topology(selected) == _topology(full)[:8]
    selected.validate(8)


def test_policy_warms_each_budget_then_uses_measured_throughput() -> None:
    logits = _peaked_logits()
    builder = LatencyAwareDDTreeBuilder(
        block_size=4,
        tree_budget=12,
        budget_candidates=(4, 8, 12),
        initial_budget=8,
        warmup_rounds_per_budget=1,
        ewma_alpha=1.0,
        exploration_interval=0,
    )

    expected_order = (8, 4, 12)
    verify_latency = {4: 20.0, 8: 10.0, 12: 1.0}
    accepted = {4: 1, 8: 2, 12: 4}
    for budget in expected_order:
        tree = builder.build_from_logits(logits)
        assert len(tree) == budget
        builder.observe(
            tree_nodes=budget,
            draft_ms=1.0,
            verify_ms=verify_latency[budget],
            accepted_draft_tokens=accepted[budget],
        )

    # 12-node 树的实测 round cost 显著最低，warmup 后应继续选择它。
    chosen = builder.build_from_logits(logits)
    assert len(chosen) == 12
    assert builder.last_decision is not None
    assert builder.last_decision.budget == 12
    assert builder.last_decision.predicted_tokens_per_ms is not None


def test_policy_rejects_invalid_candidate_ranges() -> None:
    try:
        LatencyAwareDDTreeBuilder(
            block_size=4,
            tree_budget=12,
            budget_candidates=(3, 8),
            initial_budget=8,
        )
    except ValueError as error:
        assert "block_size" in str(error)
    else:
        raise AssertionError("低于 block_size 的候选预算被接受")

    try:
        LatencyAwareDDTreeBuilder(
            block_size=4,
            tree_budget=12,
            budget_candidates=(4, 16),
            initial_budget=4,
        )
    except ValueError as error:
        assert "tree_budget" in str(error)
    else:
        raise AssertionError("超过 tree_budget 的候选预算被接受")
