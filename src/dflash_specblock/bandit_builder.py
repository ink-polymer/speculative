"""Contextual-bandit budget selection for DDTree construction.

This module changes only the truncation point of the nested best-first draft
tree.  Target verification, sampling, and KV-cache compaction stay outside the
policy and therefore keep their existing semantics.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .ddtree_builder import DDTreeBuilder


@dataclass(frozen=True, slots=True)
class BanditDecision:
    budget: int
    predicted_tokens_per_ms: float
    uncertainty: float
    ucb_score: float
    forced_exploration: bool
    context_features: tuple[float, ...]


class ContextualBanditDDTreeBuilder(DDTreeBuilder):
    """Choose a DDTree node budget with disjoint LinUCB.

    Each candidate budget is an arm.  The immediate reward is committed target
    tokens divided by draft-plus-verify latency, matching the online feedback
    available after every speculative decoding round.
    """

    manages_budget = True
    POLICY_VERSION = 1
    FEATURE_NAMES = (
        "bias",
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
        exploration_alpha: float = 0.15,
        ridge: float = 1.0,
        warmup_rounds_per_budget: int = 2,
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
        if exploration_alpha < 0 or ridge <= 0:
            raise ValueError("exploration_alpha must be nonnegative and ridge positive")
        if warmup_rounds_per_budget < 0 or context_length_scale < 1:
            raise ValueError("invalid warmup/context-length setting")

        self.budget_candidates = candidates
        self.initial_budget = int(initial_budget)
        self.exploration_alpha = float(exploration_alpha)
        self.ridge = float(ridge)
        self.warmup_rounds_per_budget = int(warmup_rounds_per_budget)
        self.context_length_scale = int(context_length_scale)
        self.learning_enabled = bool(learning_enabled)
        self.policy_metadata = dict(policy_metadata or {})
        self._warmup_order = tuple(
            sorted(
                candidates,
                key=lambda value: (
                    value != self.initial_budget,
                    abs(value - self.initial_budget),
                    value,
                ),
            )
        )
        dimension = len(self.FEATURE_NAMES)
        self._a = {
            budget: np.eye(dimension, dtype=np.float64) * self.ridge
            for budget in candidates
        }
        self._b = {
            budget: np.zeros(dimension, dtype=np.float64) for budget in candidates
        }
        self._observations = {budget: 0 for budget in candidates}
        self._reward_ewma: dict[int, float] = {}
        self._context_length = 0
        self._last_selected_budget: int | None = None
        self._last_context: np.ndarray | None = None
        self._total_updates = 0
        self.last_decision: BanditDecision | None = None

    def set_runtime_context(self, *, prefix_length: int) -> None:
        if int(prefix_length) < 0:
            raise ValueError("prefix_length must be nonnegative")
        self._context_length = int(prefix_length)

    def _next_warmup_budget(self, available: tuple[int, ...]) -> int | None:
        if not self.learning_enabled:
            return None
        for budget in self._warmup_order:
            if (
                budget in available
                and self._observations[budget] < self.warmup_rounds_per_budget
            ):
                return budget
        return None

    def _features(
        self,
        budget: int,
        previous_budget: int | None,
        mass_by_budget: dict[int, float],
        top_prefix_probability: float,
    ) -> np.ndarray:
        mass = mass_by_budget[budget]
        previous_mass = 0.0 if previous_budget is None else mass_by_budget[previous_budget]
        maximum_mass = max(mass_by_budget.values())
        return np.asarray(
            (
                1.0,
                min(1.0, mass / max(float(self.block_size), 1.0)),
                min(1.0, max(0.0, mass - previous_mass) / max(float(self.block_size), 1.0)),
                mass / max(maximum_mass, 1e-12),
                max(0.0, min(1.0, top_prefix_probability)),
                min(
                    2.0,
                    np.log1p(float(self._context_length))
                    / np.log1p(float(self.context_length_scale)),
                ),
            ),
            dtype=np.float64,
        )

    def _arm_score(self, budget: int, context: np.ndarray) -> tuple[float, float, float]:
        inverse = np.linalg.inv(self._a[budget])
        predicted = float((inverse @ self._b[budget]) @ context)
        uncertainty = float(np.sqrt(max(0.0, context @ inverse @ context)))
        bonus = self.exploration_alpha * uncertainty if self.learning_enabled else 0.0
        return predicted, uncertainty, predicted + bonus

    def _select_node_count(self, node_scores: np.ndarray) -> int:
        node_count = int(node_scores.shape[0])
        available = tuple(b for b in self.budget_candidates if b <= node_count)
        if not available:
            return node_count
        probability_mass = np.exp(np.clip(node_scores.astype(np.float64), -745.0, 0.0))
        cumulative_mass = np.cumsum(probability_mass)
        mass_by_budget = {b: float(cumulative_mass[b - 1]) for b in available}
        contexts: dict[int, np.ndarray] = {}
        previous: int | None = None
        for budget in available:
            contexts[budget] = self._features(
                budget,
                previous,
                mass_by_budget,
                float(probability_mass[0]),
            )
            previous = budget

        selected = self._next_warmup_budget(available)
        forced = selected is not None
        if selected is None:
            if not self.learning_enabled and self._total_updates == 0:
                selected = self.initial_budget if self.initial_budget in available else available[0]
            else:
                selected = max(
                    available,
                    key=lambda arm: (self._arm_score(arm, contexts[arm])[2], -arm),
                )
        predicted, uncertainty, score = self._arm_score(selected, contexts[selected])
        self._last_selected_budget = selected
        self._last_context = contexts[selected].copy()
        self.last_decision = BanditDecision(
            selected,
            predicted,
            uncertainty,
            score,
            forced,
            tuple(float(value) for value in contexts[selected]),
        )
        return selected

    def observe(
        self,
        *,
        tree_nodes: int,
        draft_ms: float,
        verify_ms: float,
        accepted_draft_tokens: int,
        tree_build_ms: float = 0.0,
    ) -> None:
        budget, context = self._last_selected_budget, self._last_context
        if budget is None or context is None:
            return
        if int(tree_nodes) != budget:
            raise ValueError("observed tree size does not match the latest bandit decision")
        if not self.learning_enabled:
            return
        reward = (1.0 + max(int(accepted_draft_tokens), 0)) / max(
            float(draft_ms) + float(tree_build_ms) + float(verify_ms), 1e-6
        )
        self._a[budget] += np.outer(context, context)
        self._b[budget] += reward * context
        self._observations[budget] += 1
        previous = self._reward_ewma.get(budget)
        self._reward_ewma[budget] = reward if previous is None else 0.9 * previous + 0.1 * reward
        self._total_updates += 1

    def policy_state(self) -> dict[str, object]:
        return {
            "policy_version": self.POLICY_VERSION,
            "algorithm": "disjoint_linucb",
            "feature_names": list(self.FEATURE_NAMES),
            "block_size": self.block_size,
            "tree_budget": self.tree_budget,
            "budget_candidates": list(self.budget_candidates),
            "initial_budget": self.initial_budget,
            "exploration_alpha": self.exploration_alpha,
            "ridge": self.ridge,
            "context_length_scale": self.context_length_scale,
            "policy_metadata": self.policy_metadata,
            "a": {str(k): value.tolist() for k, value in self._a.items()},
            "b": {str(k): value.tolist() for k, value in self._b.items()},
            "observations": {str(k): value for k, value in self._observations.items()},
            "reward_ewma": {str(k): value for k, value in self._reward_ewma.items()},
            "total_updates": self._total_updates,
        }

    def load_policy_state(self, state: dict[str, object]) -> None:
        expected = {
            "policy_version": self.POLICY_VERSION,
            "algorithm": "disjoint_linucb",
            "feature_names": list(self.FEATURE_NAMES),
            "block_size": self.block_size,
            "tree_budget": self.tree_budget,
            "budget_candidates": list(self.budget_candidates),
            "context_length_scale": self.context_length_scale,
            "policy_metadata": self.policy_metadata,
        }
        mismatches = {
            key: (state.get(key), value)
            for key, value in expected.items()
            if state.get(key) != value
        }
        if mismatches:
            raise ValueError(f"bandit checkpoint is incompatible: {mismatches}")
        dimension = len(self.FEATURE_NAMES)
        raw_a, raw_b = state.get("a"), state.get("b")
        raw_observations = state.get("observations")
        if not isinstance(raw_a, dict) or not isinstance(raw_b, dict) or not isinstance(raw_observations, dict):
            raise ValueError("bandit checkpoint is missing arm state")
        loaded_a: dict[int, np.ndarray] = {}
        loaded_b: dict[int, np.ndarray] = {}
        for budget in self.budget_candidates:
            key = str(budget)
            matrix = np.asarray(raw_a.get(key), dtype=np.float64)
            vector = np.asarray(raw_b.get(key), dtype=np.float64)
            if matrix.shape != (dimension, dimension) or vector.shape != (dimension,):
                raise ValueError(f"invalid checkpoint tensor shape for budget={budget}")
            loaded_a[budget], loaded_b[budget] = matrix, vector
        self._a, self._b = loaded_a, loaded_b
        self._observations = {b: int(raw_observations[str(b)]) for b in self.budget_candidates}
        raw_reward = state.get("reward_ewma", {})
        self._reward_ewma = (
            {b: float(raw_reward[str(b)]) for b in self.budget_candidates if str(b) in raw_reward}
            if isinstance(raw_reward, dict)
            else {}
        )
        self._total_updates = int(state.get("total_updates", 0))

    def save_policy(self, path: str | Path) -> None:
        output = Path(path).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(json.dumps(self.policy_state(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(output)

    def load_policy(self, path: str | Path) -> None:
        state = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
        if not isinstance(state, dict):
            raise ValueError("bandit checkpoint root must be a JSON object")
        self.load_policy_state(state)

    def policy_diagnostics(self) -> dict[str, object]:
        return {
            "algorithm": "disjoint_linucb",
            "learning_enabled": self.learning_enabled,
            "total_updates": self._total_updates,
            "observations": {str(k): value for k, value in self._observations.items()},
            "reward_ewma_tokens_per_ms": {str(k): value for k, value in self._reward_ewma.items()},
        }
