"""Contextual-bandit DDTree policy tests (model-free)."""

from __future__ import annotations

import json

import torch

from dflash_specblock.bandit_builder import ContextualBanditDDTreeBuilder
from dflash_specblock.ddtree_builder import DDTreeBuilder


def _logits() -> torch.Tensor:
    logits = torch.full((4, 64), -12.0)
    for row in range(4):
        logits[row, row * 4 + 1] = 8.0
        logits[row, row * 4 + 2] = 4.0
        logits[row, row * 4 + 3] = 2.0
    return logits


def _builder(*, learning: bool = True) -> ContextualBanditDDTreeBuilder:
    return ContextualBanditDDTreeBuilder(
        block_size=4,
        tree_budget=12,
        budget_candidates=(4, 8, 12),
        initial_budget=8,
        exploration_alpha=0.0,
        warmup_rounds_per_budget=1,
        context_length_scale=64,
        learning_enabled=learning,
    )


def _topology(tree):
    return [(n.token_id, n.parent, n.depth) for n in tree.nodes]


def test_policy_keeps_an_exact_best_first_prefix() -> None:
    selected = _builder().build_from_logits(_logits())
    full = DDTreeBuilder(4, 12).build_from_logits(_logits())
    assert len(selected) == 8
    assert _topology(selected) == _topology(full)[:8]


def test_policy_learns_direct_round_throughput() -> None:
    builder = _builder()
    feedback = {
        8: (5.0, 25.0, 2),
        4: (10.0, 90.0, 1),
        12: (1.0, 4.0, 4),
    }
    for expected in (8, 4, 12):
        tree = builder.build_from_logits(_logits())
        assert len(tree) == expected
        draft_ms, verify_ms, accepted = feedback[expected]
        builder.observe(
            tree_nodes=expected,
            draft_ms=draft_ms,
            verify_ms=verify_ms,
            accepted_draft_tokens=accepted,
        )
    assert len(builder.build_from_logits(_logits())) == 12


def test_checkpoint_round_trip_freezes_updates() -> None:
    trained = _builder()
    tree = trained.build_from_logits(_logits())
    trained.observe(
        tree_nodes=len(tree), draft_ms=1.0, verify_ms=2.0, accepted_draft_tokens=2
    )
    state = json.loads(json.dumps(trained.policy_state()))
    frozen = _builder(learning=False)
    frozen.load_policy_state(state)
    before = frozen.policy_diagnostics()["total_updates"]
    selected = frozen.build_from_logits(_logits())
    frozen.observe(
        tree_nodes=len(selected), draft_ms=1.0, verify_ms=1.0, accepted_draft_tokens=3
    )
    assert frozen.policy_diagnostics()["total_updates"] == before


def test_context_tracks_target_kv_length() -> None:
    builder = _builder(learning=False)
    builder.set_runtime_context(prefix_length=0)
    builder.build_from_logits(_logits())
    short = builder.last_decision.context_features
    builder.set_runtime_context(prefix_length=64)
    builder.build_from_logits(_logits())
    long = builder.last_decision.context_features
    assert short[:-1] == long[:-1]
    assert short[-1] == 0.0
    assert long[-1] == 1.0
