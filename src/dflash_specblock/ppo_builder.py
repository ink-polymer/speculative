"""Discrete PPO policy for adaptive DDTree node budgets.

The policy selects only a prefix length from one nested best-first DDTree.  It
never changes target sampling, the ancestor visibility mask, or KV compaction.
The per-round reward is committed tokens divided by draft-plus-verify latency.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from .ddtree_builder import DDTreeBuilder


@dataclass(frozen=True, slots=True)
class PPODecision:
    budget: int
    action_probability: float
    value_estimate: float
    entropy: float
    deterministic: bool
    action_features: tuple[float, ...]


@dataclass(slots=True)
class _Transition:
    action_contexts: np.ndarray
    available_mask: np.ndarray
    state_features: np.ndarray
    action: int
    old_log_probability: float
    old_value: float
    reward: float
    done: bool = False


class _Actor(nn.Module):
    def __init__(self, feature_size: int, hidden_size: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        linear_layers = [module for module in self.modules() if isinstance(module, nn.Linear)]
        for layer in linear_layers:
            nn.init.orthogonal_(layer.weight, gain=math.sqrt(2.0))
            nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(linear_layers[-1].weight, gain=0.01)

    def forward(self, action_contexts: torch.Tensor) -> torch.Tensor:
        return self.network(action_contexts).squeeze(-1)


class _Critic(nn.Module):
    def __init__(self, state_size: int, hidden_size: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )
        linear_layers = [module for module in self.modules() if isinstance(module, nn.Linear)]
        for layer in linear_layers:
            nn.init.orthogonal_(layer.weight, gain=math.sqrt(2.0))
            nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(linear_layers[-1].weight, gain=1.0)

    def forward(self, state_features: torch.Tensor) -> torch.Tensor:
        return self.network(state_features).squeeze(-1)


class PPODDTreeBuilder(DDTreeBuilder):
    """Select a nested DDTree node budget with clipped discrete-action PPO."""

    manages_budget = True
    POLICY_VERSION = 1
    ACTION_FEATURE_NAMES = (
        "bias",
        "budget_fraction",
        "log_budget_fraction",
        "proposal_mass",
        "marginal_mass",
        "captured_mass_fraction",
        "top_prefix_probability",
        "context_length",
    )

    def __init__(
        self,
        block_size: int,
        tree_budget: int,
        budget_candidates: tuple[int, ...],
        initial_budget: int,
        *,
        hidden_size: int = 64,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        value_coefficient: float = 0.5,
        entropy_coefficient: float = 0.01,
        rollout_steps: int = 256,
        update_epochs: int = 4,
        minibatch_size: int = 64,
        max_grad_norm: float = 0.5,
        tree_build_cost_weight: float = 2.0,
        context_length_scale: int = 4096,
        learning_enabled: bool = True,
        policy_metadata: dict[str, str] | None = None,
    ) -> None:
        super().__init__(block_size, tree_budget, reserve_greedy_chain=False)
        candidates = tuple(sorted({int(value) for value in budget_candidates}))
        if not candidates:
            raise ValueError("budget_candidates cannot be empty")
        if candidates[0] < self.block_size or candidates[-1] > self.tree_budget:
            raise ValueError("budget_candidates must lie in [block_size, tree_budget]")
        if int(initial_budget) not in candidates:
            raise ValueError("initial_budget must be one of budget_candidates")
        if hidden_size < 1 or learning_rate <= 0:
            raise ValueError("hidden_size and learning_rate must be positive")
        if not 0.0 <= gamma <= 1.0 or not 0.0 <= gae_lambda <= 1.0:
            raise ValueError("gamma and gae_lambda must lie in [0, 1]")
        if clip_range <= 0 or value_coefficient < 0 or entropy_coefficient < 0:
            raise ValueError("invalid PPO loss coefficient")
        if rollout_steps < 1 or update_epochs < 1 or minibatch_size < 1:
            raise ValueError("rollout/update/minibatch settings must be positive")
        if max_grad_norm <= 0 or tree_build_cost_weight <= 0 or context_length_scale < 1:
            raise ValueError(
                "max_grad_norm, tree_build_cost_weight, and context_length_scale must be positive"
            )

        self.budget_candidates = candidates
        self.initial_budget = int(initial_budget)
        self.hidden_size = int(hidden_size)
        self.learning_rate = float(learning_rate)
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.clip_range = float(clip_range)
        self.value_coefficient = float(value_coefficient)
        self.entropy_coefficient = float(entropy_coefficient)
        self.rollout_steps = int(rollout_steps)
        self.update_epochs = int(update_epochs)
        self.minibatch_size = int(minibatch_size)
        self.max_grad_norm = float(max_grad_norm)
        self.tree_build_cost_weight = float(tree_build_cost_weight)
        self.context_length_scale = int(context_length_scale)
        self.learning_enabled = bool(learning_enabled)
        self.policy_metadata = dict(policy_metadata or {})

        self._action_count = len(candidates)
        self._action_feature_size = len(self.ACTION_FEATURE_NAMES)
        self._state_size = self._action_count * (self._action_feature_size + 1)
        self.actor = _Actor(self._action_feature_size, self.hidden_size).cpu()
        self.critic = _Critic(self._state_size, self.hidden_size).cpu()
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=self.learning_rate,
            eps=1e-5,
        )
        self.actor.eval()
        self.critic.eval()

        self._context_length = 0
        self._pending: dict[str, Any] | None = None
        self._rollout: list[_Transition] = []
        self._episode_open = False
        self._episode_start = 0
        self._total_transitions = 0
        self._ppo_updates = 0
        self._optimizer_steps = 0
        self._selection_counts = {budget: 0 for budget in candidates}
        self._reward_ewma: dict[int, float] = {}
        self._last_update_metrics: dict[str, float] = {}
        self.last_decision: PPODecision | None = None

    def set_runtime_context(self, *, prefix_length: int) -> None:
        if int(prefix_length) < 0:
            raise ValueError("prefix_length must be nonnegative")
        self._context_length = int(prefix_length)

    def begin_episode(self) -> None:
        if self._episode_open:
            self.end_episode()
        self._episode_open = True
        self._episode_start = len(self._rollout)
        self._pending = None

    def end_episode(self) -> None:
        if not self._episode_open:
            return
        if self.learning_enabled and len(self._rollout) > self._episode_start:
            self._rollout[-1].done = True
        self._episode_open = False
        self._pending = None
        if self.learning_enabled and len(self._rollout) >= self.rollout_steps:
            self._update_policy()

    def finalize_training(self) -> None:
        if self._episode_open:
            self.end_episode()
        if self.learning_enabled and self._rollout:
            self._rollout[-1].done = True
            self._update_policy()

    def _features(
        self,
        budget: int,
        previous_budget: int | None,
        mass_by_budget: dict[int, float],
        maximum_mass: float,
        top_prefix_probability: float,
    ) -> np.ndarray:
        mass = mass_by_budget[budget]
        previous_mass = 0.0 if previous_budget is None else mass_by_budget[previous_budget]
        return np.asarray(
            (
                1.0,
                budget / max(float(self.tree_budget), 1.0),
                math.log1p(float(budget)) / math.log1p(float(self.tree_budget)),
                min(1.0, mass / max(float(self.block_size), 1.0)),
                min(
                    1.0,
                    max(0.0, mass - previous_mass) / max(float(self.block_size), 1.0),
                ),
                mass / max(maximum_mass, 1e-12),
                max(0.0, min(1.0, top_prefix_probability)),
                min(
                    2.0,
                    math.log1p(float(self._context_length))
                    / math.log1p(float(self.context_length_scale)),
                ),
            ),
            dtype=np.float32,
        )

    def _policy_inputs(
        self, node_scores: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        node_count = int(node_scores.shape[0])
        action_contexts = np.zeros(
            (self._action_count, self._action_feature_size), dtype=np.float32
        )
        available_mask = np.zeros(self._action_count, dtype=np.bool_)
        probability_mass = np.exp(np.clip(node_scores.astype(np.float64), -745.0, 0.0))
        cumulative_mass = np.cumsum(probability_mass)
        available = [budget for budget in self.budget_candidates if budget <= node_count]
        if not available:
            return action_contexts, available_mask, np.concatenate(
                (action_contexts.reshape(-1), available_mask.astype(np.float32))
            )
        mass_by_budget = {budget: float(cumulative_mass[budget - 1]) for budget in available}
        maximum_mass = max(mass_by_budget.values())
        previous: int | None = None
        for index, budget in enumerate(self.budget_candidates):
            if budget not in mass_by_budget:
                continue
            available_mask[index] = True
            action_contexts[index] = self._features(
                budget,
                previous,
                mass_by_budget,
                maximum_mass,
                float(probability_mass[0]),
            )
            previous = budget
        state_features = np.concatenate(
            (action_contexts.reshape(-1), available_mask.astype(np.float32))
        ).astype(np.float32, copy=False)
        return action_contexts, available_mask, state_features

    def _distribution(
        self, action_contexts: torch.Tensor, available_mask: torch.Tensor
    ) -> Categorical:
        logits = self.actor(action_contexts)
        logits = logits.masked_fill(~available_mask, torch.finfo(logits.dtype).min)
        return Categorical(logits=logits)

    def _select_node_count(self, node_scores: np.ndarray) -> int:
        node_count = int(node_scores.shape[0])
        action_contexts, available_mask, state_features = self._policy_inputs(node_scores)
        if not bool(available_mask.any()):
            return node_count

        contexts_tensor = torch.from_numpy(action_contexts)
        mask_tensor = torch.from_numpy(available_mask)
        state_tensor = torch.from_numpy(state_features)
        with torch.no_grad():
            distribution = self._distribution(contexts_tensor, mask_tensor)
            if self.learning_enabled:
                action_tensor = distribution.sample()
            else:
                action_tensor = distribution.logits.argmax()
            value_tensor = self.critic(state_tensor)
            log_probability = distribution.log_prob(action_tensor)
            probability = distribution.probs[action_tensor]
            entropy = distribution.entropy()

        action = int(action_tensor.item())
        budget = self.budget_candidates[action]
        self._selection_counts[budget] += 1
        self._pending = {
            "action_contexts": action_contexts.copy(),
            "available_mask": available_mask.copy(),
            "state_features": state_features.copy(),
            "action": action,
            "old_log_probability": float(log_probability.item()),
            "old_value": float(value_tensor.item()),
            "budget": budget,
        }
        self.last_decision = PPODecision(
            budget=budget,
            action_probability=float(probability.item()),
            value_estimate=float(value_tensor.item()),
            entropy=float(entropy.item()),
            deterministic=not self.learning_enabled,
            action_features=tuple(float(value) for value in action_contexts[action]),
        )
        return budget

    def observe(
        self,
        *,
        tree_nodes: int,
        draft_ms: float,
        tree_build_ms: float,
        verify_ms: float,
        accepted_draft_tokens: int,
    ) -> None:
        pending = self._pending
        self._pending = None
        if pending is None:
            return
        budget = int(pending["budget"])
        if int(tree_nodes) != budget:
            raise ValueError("observed tree size does not match the latest PPO decision")
        if not self.learning_enabled:
            return
        weighted_latency_ms = (
            float(draft_ms)
            + float(verify_ms)
            + self.tree_build_cost_weight * float(tree_build_ms)
        )
        reward = (1.0 + max(int(accepted_draft_tokens), 0)) / max(
            weighted_latency_ms, 1e-6
        )
        self._rollout.append(
            _Transition(
                action_contexts=pending["action_contexts"],
                available_mask=pending["available_mask"],
                state_features=pending["state_features"],
                action=int(pending["action"]),
                old_log_probability=float(pending["old_log_probability"]),
                old_value=float(pending["old_value"]),
                reward=float(reward),
            )
        )
        self._total_transitions += 1
        previous = self._reward_ewma.get(budget)
        self._reward_ewma[budget] = reward if previous is None else 0.9 * previous + 0.1 * reward

    def _advantages_and_returns(self) -> tuple[np.ndarray, np.ndarray]:
        rewards = np.asarray([item.reward for item in self._rollout], dtype=np.float32)
        values = np.asarray([item.old_value for item in self._rollout], dtype=np.float32)
        done = np.asarray([item.done for item in self._rollout], dtype=np.bool_)
        advantages = np.zeros_like(rewards)
        gae = 0.0
        for index in range(len(self._rollout) - 1, -1, -1):
            nonterminal = 0.0 if done[index] else 1.0
            next_value = 0.0 if index + 1 == len(values) else float(values[index + 1])
            delta = float(rewards[index]) + self.gamma * next_value * nonterminal - float(values[index])
            gae = delta + self.gamma * self.gae_lambda * nonterminal * gae
            advantages[index] = gae
        returns = advantages + values
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages, returns

    def _update_policy(self) -> None:
        if not self._rollout:
            return
        advantages_np, returns_np = self._advantages_and_returns()
        with torch.inference_mode(False), torch.enable_grad():
            action_contexts = torch.from_numpy(
                np.stack([item.action_contexts for item in self._rollout])
            )
            available_masks = torch.from_numpy(
                np.stack([item.available_mask for item in self._rollout])
            )
            state_features = torch.from_numpy(
                np.stack([item.state_features for item in self._rollout])
            )
            actions = torch.tensor([item.action for item in self._rollout], dtype=torch.long)
            old_log_probabilities = torch.tensor(
                [item.old_log_probability for item in self._rollout], dtype=torch.float32
            )
            advantages = torch.from_numpy(advantages_np)
            returns = torch.from_numpy(returns_np)
            batch_size = len(self._rollout)
            minibatch_size = min(self.minibatch_size, batch_size)
            policy_losses: list[float] = []
            value_losses: list[float] = []
            entropies: list[float] = []
            approximate_kls: list[float] = []
            clip_fractions: list[float] = []

            self.actor.train()
            self.critic.train()
            for _ in range(self.update_epochs):
                order = torch.randperm(batch_size)
                for start in range(0, batch_size, minibatch_size):
                    batch = order[start : start + minibatch_size]
                    distribution = self._distribution(
                        action_contexts[batch], available_masks[batch]
                    )
                    new_log_probability = distribution.log_prob(actions[batch])
                    entropy = distribution.entropy().mean()
                    ratio = (new_log_probability - old_log_probabilities[batch]).exp()
                    unclipped = ratio * advantages[batch]
                    clipped = ratio.clamp(
                        1.0 - self.clip_range, 1.0 + self.clip_range
                    ) * advantages[batch]
                    policy_loss = -torch.minimum(unclipped, clipped).mean()
                    value = self.critic(state_features[batch])
                    value_loss = 0.5 * (value - returns[batch]).square().mean()
                    loss = (
                        policy_loss
                        + self.value_coefficient * value_loss
                        - self.entropy_coefficient * entropy
                    )

                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        list(self.actor.parameters()) + list(self.critic.parameters()),
                        self.max_grad_norm,
                    )
                    self.optimizer.step()
                    self._optimizer_steps += 1

                    with torch.no_grad():
                        approximate_kl = (
                            old_log_probabilities[batch] - new_log_probability
                        ).mean()
                        clip_fraction = (
                            (ratio - 1.0).abs() > self.clip_range
                        ).float().mean()
                    policy_losses.append(float(policy_loss.item()))
                    value_losses.append(float(value_loss.item()))
                    entropies.append(float(entropy.item()))
                    approximate_kls.append(float(approximate_kl.item()))
                    clip_fractions.append(float(clip_fraction.item()))
            self.actor.eval()
            self.critic.eval()

        self._ppo_updates += 1
        self._last_update_metrics = {
            "policy_loss": float(np.mean(policy_losses)),
            "value_loss": float(np.mean(value_losses)),
            "entropy": float(np.mean(entropies)),
            "approximate_kl": float(np.mean(approximate_kls)),
            "clip_fraction": float(np.mean(clip_fractions)),
            "rollout_size": float(len(self._rollout)),
        }
        self._rollout.clear()

    def _configuration(self) -> dict[str, Any]:
        return {
            "policy_version": self.POLICY_VERSION,
            "algorithm": "ppo_discrete",
            "action_feature_names": list(self.ACTION_FEATURE_NAMES),
            "block_size": self.block_size,
            "tree_budget": self.tree_budget,
            "budget_candidates": list(self.budget_candidates),
            "initial_budget": self.initial_budget,
            "hidden_size": self.hidden_size,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "clip_range": self.clip_range,
            "value_coefficient": self.value_coefficient,
            "entropy_coefficient": self.entropy_coefficient,
            "rollout_steps": self.rollout_steps,
            "update_epochs": self.update_epochs,
            "minibatch_size": self.minibatch_size,
            "max_grad_norm": self.max_grad_norm,
            "tree_build_cost_weight": self.tree_build_cost_weight,
            "context_length_scale": self.context_length_scale,
            "policy_metadata": self.policy_metadata,
        }

    def policy_state(self) -> dict[str, Any]:
        return {
            **self._configuration(),
            "actor_state": self.actor.state_dict(),
            "critic_state": self.critic.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "rollout": [
                {
                    "action_contexts": torch.from_numpy(item.action_contexts.copy()),
                    "available_mask": torch.from_numpy(item.available_mask.copy()),
                    "state_features": torch.from_numpy(item.state_features.copy()),
                    "action": item.action,
                    "old_log_probability": item.old_log_probability,
                    "old_value": item.old_value,
                    "reward": item.reward,
                    "done": item.done,
                }
                for item in self._rollout
            ],
            "total_transitions": self._total_transitions,
            "ppo_updates": self._ppo_updates,
            "optimizer_steps": self._optimizer_steps,
            "selection_counts": {str(k): value for k, value in self._selection_counts.items()},
            "reward_ewma": {str(k): value for k, value in self._reward_ewma.items()},
            "last_update_metrics": self._last_update_metrics,
        }

    def load_policy_state(self, state: dict[str, Any]) -> None:
        expected = self._configuration()
        mismatches = {
            key: (state.get(key), value)
            for key, value in expected.items()
            if state.get(key) != value
        }
        if mismatches:
            raise ValueError(f"PPO checkpoint is incompatible: {mismatches}")
        actor_state = state.get("actor_state")
        critic_state = state.get("critic_state")
        if not isinstance(actor_state, dict) or not isinstance(critic_state, dict):
            raise ValueError("PPO checkpoint is missing actor/critic state")
        self.actor.load_state_dict(actor_state)
        self.critic.load_state_dict(critic_state)
        optimizer_state = state.get("optimizer_state")
        if isinstance(optimizer_state, dict):
            self.optimizer.load_state_dict(optimizer_state)
        self._rollout = []
        for raw in state.get("rollout", []):
            self._rollout.append(
                _Transition(
                    action_contexts=raw["action_contexts"].cpu().numpy().copy(),
                    available_mask=raw["available_mask"].cpu().numpy().astype(np.bool_, copy=True),
                    state_features=raw["state_features"].cpu().numpy().copy(),
                    action=int(raw["action"]),
                    old_log_probability=float(raw["old_log_probability"]),
                    old_value=float(raw["old_value"]),
                    reward=float(raw["reward"]),
                    done=bool(raw["done"]),
                )
            )
        self._total_transitions = int(state.get("total_transitions", 0))
        self._ppo_updates = int(state.get("ppo_updates", 0))
        self._optimizer_steps = int(state.get("optimizer_steps", 0))
        raw_counts = state.get("selection_counts", {})
        self._selection_counts = {
            budget: int(raw_counts.get(str(budget), 0)) for budget in self.budget_candidates
        }
        raw_reward = state.get("reward_ewma", {})
        self._reward_ewma = {
            budget: float(raw_reward[str(budget)])
            for budget in self.budget_candidates
            if str(budget) in raw_reward
        }
        raw_metrics = state.get("last_update_metrics", {})
        self._last_update_metrics = {
            str(key): float(value) for key, value in raw_metrics.items()
        }

    def save_policy(self, path: str | Path) -> None:
        output = Path(path).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        torch.save(self.policy_state(), temporary)
        temporary.replace(output)

    def load_policy(self, path: str | Path) -> None:
        checkpoint = Path(path).expanduser()
        try:
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(checkpoint, map_location="cpu")
        if not isinstance(state, dict):
            raise ValueError("PPO checkpoint root must be a dictionary")
        self.load_policy_state(state)

    def policy_diagnostics(self) -> dict[str, Any]:
        last_decision = None
        if self.last_decision is not None:
            last_decision = {
                "budget": self.last_decision.budget,
                "action_probability": self.last_decision.action_probability,
                "value_estimate": self.last_decision.value_estimate,
                "entropy": self.last_decision.entropy,
                "deterministic": self.last_decision.deterministic,
            }
        return {
            "algorithm": "ppo_discrete",
            "learning_enabled": self.learning_enabled,
            "total_transitions": self._total_transitions,
            "ppo_updates": self._ppo_updates,
            "optimizer_steps": self._optimizer_steps,
            "pending_rollout_steps": len(self._rollout),
            "selection_counts": {
                str(k): value for k, value in self._selection_counts.items()
            },
            "reward_ewma_tokens_per_ms": {
                str(k): value for k, value in self._reward_ewma.items()
            },
            "reward_latency": {
                "formula": "draft_ms + verify_ms + tree_build_cost_weight * tree_build_ms",
                "tree_build_cost_weight": self.tree_build_cost_weight,
                "tree_build_ms_includes": ["best_first_tree_build", "tree_mask_compile"],
            },
            "last_decision": last_decision,
            "last_update": self._last_update_metrics,
        }
