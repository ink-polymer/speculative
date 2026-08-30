"""Model-free tests for the discrete PPO DDTree budget policy."""

from __future__ import annotations

import torch

from dflash_specblock.ddtree_builder import DDTreeBuilder
from dflash_specblock.ppo_builder import PPODDTreeBuilder


def _logits() -> torch.Tensor:
    logits = torch.full((4, 64), -12.0)
    for row in range(4):
        logits[row, row * 4 + 1] = 8.0
        logits[row, row * 4 + 2] = 4.0
        logits[row, row * 4 + 3] = 2.0
    return logits


def _builder(*, learning: bool = True, candidates: tuple[int, ...] = (4, 8, 12)):
    torch.manual_seed(7)
    return PPODDTreeBuilder(
        block_size=4,
        tree_budget=max(candidates),
        budget_candidates=candidates,
        initial_budget=8 if 8 in candidates else candidates[0],
        hidden_size=16,
        learning_rate=1e-3,
        rollout_steps=4,
        update_epochs=2,
        minibatch_size=4,
        tree_build_cost_weight=2.0,
        context_length_scale=64,
        learning_enabled=learning,
        policy_metadata={"test": "ppo"},
    )


def _topology(tree):
    return [(node.token_id, node.parent, node.depth) for node in tree.nodes]


def _one_round(builder: PPODDTreeBuilder, *, tree_build_ms: float = 2.0):
    builder.begin_episode()
    tree = builder.build_from_logits(_logits())
    builder.observe(
        tree_nodes=len(tree),
        draft_ms=3.0,
        tree_build_ms=tree_build_ms,
        verify_ms=5.0,
        accepted_draft_tokens=3,
    )
    builder.end_episode()
    return tree


def test_ppo_keeps_an_exact_best_first_prefix() -> None:
    builder = _builder()
    selected_tree = _one_round(builder)
    selected = len(selected_tree)
    full = DDTreeBuilder(4, 12).build_from_logits(_logits())
    assert selected in builder.budget_candidates
    assert _topology(selected_tree) == _topology(full)[:selected]


def test_ppo_updates_after_complete_episode_rollouts() -> None:
    builder = _builder()
    for _ in range(4):
        _one_round(builder)
    diagnostics = builder.policy_diagnostics()
    assert diagnostics["total_transitions"] == 4
    assert diagnostics["ppo_updates"] == 1
    assert diagnostics["pending_rollout_steps"] == 0


def test_tree_build_latency_is_weighted_in_reward() -> None:
    fast = _builder(candidates=(8,))
    slow = _builder(candidates=(8,))
    _one_round(fast, tree_build_ms=1.0)
    _one_round(slow, tree_build_ms=10.0)
    fast_reward = fast.policy_diagnostics()["reward_ewma_tokens_per_ms"]["8"]
    slow_reward = slow.policy_diagnostics()["reward_ewma_tokens_per_ms"]["8"]
    assert fast_reward > slow_reward
    assert fast.policy_diagnostics()["reward_latency"]["tree_build_cost_weight"] == 2.0


def test_checkpoint_round_trip_freezes_policy_updates() -> None:
    trained = _builder()
    for _ in range(4):
        _one_round(trained)
    state = trained.policy_state()
    frozen = _builder(learning=False)
    frozen.load_policy_state(state)
    before = frozen.policy_diagnostics()["total_transitions"]
    selected = _one_round(frozen)
    assert len(selected) in frozen.budget_candidates
    assert frozen.policy_diagnostics()["total_transitions"] == before


def test_context_tracks_target_kv_length() -> None:
    builder = _builder(learning=False)
    builder.set_runtime_context(prefix_length=0)
    builder.build_from_logits(_logits())
    short = builder.last_decision.action_features
    builder.set_runtime_context(prefix_length=64)
    builder.build_from_logits(_logits())
    long = builder.last_decision.action_features
    assert short[:-1] == long[:-1]
    assert short[-1] == 0.0
    assert long[-1] == 1.0
