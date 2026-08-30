"""Episode-level contextual bandit for speculative-decoding topology actions."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


_CODE_PATTERN = re.compile(
    r"\b(def|class|function|return|import|public|private|int|str|list|array)\b|[{};]",
    re.IGNORECASE,
)
_MATH_PATTERN = re.compile(
    r"\\(?:frac|sqrt|sum|int)|\$|\b(?:prove|equation|integer|probability|geometry)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class TopologyDecision:
    action: str
    predicted_tokens_per_ms: float
    uncertainty: float
    forced_exploration: bool
    deterministic: bool
    context_features: tuple[float, ...]


def prompt_context_features(prompt: str, prompt_tokens: int) -> np.ndarray:
    """Production-safe prompt features; no benchmark label is consumed."""
    text = str(prompt)
    characters = max(len(text), 1)
    token_count = max(int(prompt_tokens), 1)
    length_feature = min(1.5, math.log1p(token_count) / math.log1p(4096.0))
    digit_fraction = sum(character.isdigit() for character in text) / characters
    newline_fraction = text.count("\n") / characters
    ascii_fraction = sum(ord(character) < 128 for character in text) / characters
    code_signal = 1.0 if _CODE_PATTERN.search(text) else 0.0
    math_signal = 1.0 if _MATH_PATTERN.search(text) else 0.0
    return np.asarray(
        (
            1.0,
            length_feature,
            min(2.0, characters / token_count / 4.0),
            min(1.0, 8.0 * digit_fraction),
            min(1.0, 32.0 * newline_fraction),
            ascii_fraction,
            code_signal,
            math_signal,
            length_feature * length_feature,
        ),
        dtype=np.float64,
    )


class TopologyRatioBandit:
    """Bayesian contextual bandit over joint ``topology:capacity`` actions.

    Training begins with randomized, exactly balanced logging.  It then uses
    Thompson sampling.  Frozen inference is deterministic and chooses the arm
    with the largest posterior expected token throughput.
    """

    POLICY_VERSION = 1
    FEATURE_NAMES = (
        "bias",
        "log_prompt_tokens",
        "characters_per_token",
        "digit_fraction",
        "newline_fraction",
        "ascii_fraction",
        "code_signal",
        "math_signal",
        "length_squared",
    )

    def __init__(
        self,
        actions: Sequence[str],
        initial_action: str,
        *,
        ridge: float = 2.0,
        exploration_scale: float = 0.25,
        warmup_episodes_per_action: int = 12,
        reward_noise_scale: float = 0.12,
        learning_enabled: bool = True,
        random_seed: int = 42,
        policy_metadata: dict[str, str] | None = None,
    ) -> None:
        normalized = tuple(dict.fromkeys(str(action).strip() for action in actions))
        if not normalized or any(not action for action in normalized):
            raise ValueError("actions must contain at least one non-empty action")
        if initial_action not in normalized:
            raise ValueError("initial_action must belong to actions")
        if ridge <= 0.0 or exploration_scale < 0.0 or reward_noise_scale <= 0.0:
            raise ValueError("invalid ridge/exploration/noise configuration")
        if warmup_episodes_per_action < 0:
            raise ValueError("warmup_episodes_per_action cannot be negative")
        self.actions = normalized
        self.initial_action = str(initial_action)
        self.ridge = float(ridge)
        self.exploration_scale = float(exploration_scale)
        self.warmup_episodes_per_action = int(warmup_episodes_per_action)
        self.reward_noise_scale = float(reward_noise_scale)
        self.learning_enabled = bool(learning_enabled)
        self.random_seed = int(random_seed)
        self.policy_metadata = dict(policy_metadata or {})

        dimension = len(self.FEATURE_NAMES)
        self._a = {
            action: np.eye(dimension, dtype=np.float64) * self.ridge
            for action in self.actions
        }
        self._b = {
            action: np.zeros(dimension, dtype=np.float64)
            for action in self.actions
        }
        self._observations = {action: 0 for action in self.actions}
        self._selection_counts = {action: 0 for action in self.actions}
        self._reward_ewma: dict[str, float] = {}
        self._rng = np.random.default_rng(self.random_seed)
        self._total_updates = 0
        self._pending: tuple[str, np.ndarray] | None = None
        self.last_decision: TopologyDecision | None = None

    @staticmethod
    def _posterior(
        matrix: np.ndarray, vector: np.ndarray, context: np.ndarray
    ) -> tuple[float, float]:
        inverse_context = np.linalg.solve(matrix, context)
        theta = np.linalg.solve(matrix, vector)
        mean = float(theta @ context)
        uncertainty = float(math.sqrt(max(context @ inverse_context, 0.0)))
        return mean, uncertainty

    def _warmup_action(self) -> str | None:
        if not self.learning_enabled:
            return None
        minimum = min(self._observations.values())
        if minimum >= self.warmup_episodes_per_action:
            return None
        eligible = [
            action for action in self.actions if self._observations[action] == minimum
        ]
        if self._total_updates == 0 and self.initial_action in eligible:
            return self.initial_action
        return str(self._rng.choice(np.asarray(eligible, dtype=object)))

    def select(self, context: np.ndarray) -> str:
        features = np.asarray(context, dtype=np.float64)
        if features.shape != (len(self.FEATURE_NAMES),) or not np.isfinite(features).all():
            raise ValueError("context has an invalid shape or non-finite values")
        selected = self._warmup_action()
        forced = selected is not None
        predictions: dict[str, tuple[float, float]] = {}
        for action in self.actions:
            mean, uncertainty = self._posterior(
                self._a[action], self._b[action], features
            )
            sampled_mean = mean
            if self.learning_enabled and not forced and self.exploration_scale > 0.0:
                sampled_mean += float(
                    self._rng.normal(
                        0.0,
                        self.exploration_scale
                        * self.reward_noise_scale
                        * uncertainty,
                    )
                )
            predictions[action] = (sampled_mean, uncertainty)
        if selected is None:
            if not self.learning_enabled and self._total_updates == 0:
                selected = self.initial_action
            else:
                selected = max(
                    self.actions,
                    key=lambda action: (predictions[action][0], -self.actions.index(action)),
                )
        sampled_log_reward, uncertainty = predictions[selected]
        self._selection_counts[selected] += 1
        self._pending = (selected, features.copy())
        self.last_decision = TopologyDecision(
            action=selected,
            predicted_tokens_per_ms=float(math.exp(np.clip(sampled_log_reward, -20, 20))),
            uncertainty=uncertainty,
            forced_exploration=forced,
            deterministic=not self.learning_enabled,
            context_features=tuple(float(value) for value in features),
        )
        return selected

    def observe(self, *, committed_tokens: int, decode_ms: float) -> None:
        pending = self._pending
        self._pending = None
        if pending is None:
            raise RuntimeError("observe() requires a preceding select()")
        action, context = pending
        throughput = max(int(committed_tokens), 1) / max(float(decode_ms), 1e-6)
        if not self.learning_enabled:
            return
        target = math.log(max(throughput, 1e-12))
        self._a[action] += np.outer(context, context)
        self._b[action] += target * context
        self._observations[action] += 1
        self._total_updates += 1
        previous = self._reward_ewma.get(action)
        self._reward_ewma[action] = (
            throughput if previous is None else 0.95 * previous + 0.05 * throughput
        )

    def _configuration(self) -> dict[str, Any]:
        return {
            "policy_version": self.POLICY_VERSION,
            "algorithm": "disjoint_bayesian_topology_ratio_thompson",
            "reward": "log(generated_tokens / decode_ms)",
            "exploration_assignment": "balanced_randomized_min_count_then_thompson",
            "feature_names": list(self.FEATURE_NAMES),
            "actions": list(self.actions),
            "initial_action": self.initial_action,
            "ridge": self.ridge,
            "exploration_scale": self.exploration_scale,
            "warmup_episodes_per_action": self.warmup_episodes_per_action,
            "reward_noise_scale": self.reward_noise_scale,
            "random_seed": self.random_seed,
            "policy_metadata": self.policy_metadata,
        }

    def policy_state(self) -> dict[str, Any]:
        return {
            **self._configuration(),
            "a": {action: matrix.tolist() for action, matrix in self._a.items()},
            "b": {action: vector.tolist() for action, vector in self._b.items()},
            "observations": dict(self._observations),
            "selection_counts": dict(self._selection_counts),
            "reward_ewma": dict(self._reward_ewma),
            "total_updates": self._total_updates,
        }

    def load_policy_state(self, state: dict[str, Any]) -> None:
        expected = self._configuration()
        mismatches = {
            key: (state.get(key), value)
            for key, value in expected.items()
            if state.get(key) != value
        }
        if mismatches:
            raise ValueError(f"topology-bandit checkpoint mismatch: {mismatches}")
        dimension = len(self.FEATURE_NAMES)
        for name, matrix in (("a", True), ("b", False)):
            raw = state.get(name)
            if not isinstance(raw, dict):
                raise ValueError(f"checkpoint is missing {name}")
            expected_shape = (dimension, dimension) if matrix else (dimension,)
            loaded = {}
            for action in self.actions:
                value = np.asarray(raw.get(action), dtype=np.float64)
                if value.shape != expected_shape or not np.isfinite(value).all():
                    raise ValueError(f"checkpoint {name}[{action}] is invalid")
                loaded[action] = value
            setattr(self, f"_{name}", loaded)
        self._observations = {
            action: int(state["observations"][action]) for action in self.actions
        }
        self._selection_counts = {
            action: int(state.get("selection_counts", {}).get(action, 0))
            for action in self.actions
        }
        self._reward_ewma = {
            action: float(value)
            for action, value in state.get("reward_ewma", {}).items()
            if action in self.actions
        }
        self._total_updates = int(state.get("total_updates", 0))

    def save_policy(self, path: str | Path) -> None:
        output = Path(path).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.policy_state(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output)

    def load_policy(self, path: str | Path) -> None:
        state = json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))
        if not isinstance(state, dict):
            raise ValueError("topology-bandit checkpoint must be a JSON object")
        self.load_policy_state(state)

    def diagnostics(self) -> dict[str, Any]:
        return {
            "algorithm": self._configuration()["algorithm"],
            "learning_enabled": self.learning_enabled,
            "total_updates": self._total_updates,
            "observations": dict(self._observations),
            "selection_counts": dict(self._selection_counts),
            "reward_ewma_tokens_per_ms": dict(self._reward_ewma),
        }
