"""Original controller plus explicitly named, non-learning ablations.

The full method delegates every decision/update to the pinned original class.
Serialization preserves causal state across prompts and process restarts.
"""
from __future__ import annotations

from dataclasses import asdict
import math

from ..ddtree_builder import BudgetDecision, DDTreeBuilder, LatencyAwareDDTreeBuilder
from .common import BASELINES, K, VARIANTS, digest


class FixedBudgetBuilder(DDTreeBuilder):
    # Bypass the legacy engine's separate previous-acceptance budget interpolation.
    manages_budget = True


class PaperAdaptiveBuilder(LatencyAwareDDTreeBuilder):
    def __init__(self, cfg, variant="adaptive"):
        if variant not in VARIANTS:
            raise ValueError(f"Unknown controller variant: {variant}")
        super().__init__(K, max(cfg["budget_candidates"]), tuple(cfg["budget_candidates"]),
                         cfg["initial_budget"], cfg["warmup_rounds_per_budget"],
                         cfg["ewma_alpha"], 0 if variant == "no_exploration" else cfg["exploration_interval"])
        self.variant = variant
        self.identity = digest({"variant": variant, "budgets": self.budget_candidates,
            "initial": self.initial_budget, "warmup": self.warmup_rounds_per_budget,
            "alpha": self.ewma_alpha, "explore": self.exploration_interval})
        self.trace = []

    def _select_node_count(self, scores):
        if self.variant != "no_latency":
            return super()._select_node_count(scores)
        # Remove cost discrimination, but preserve warmup/exploration and observations.
        fixed, verify = self._fixed_ms, self._verify_ms
        self._fixed_ms = 1. if fixed is not None else None
        self._verify_ms = {b: 0. for b in verify}
        try:
            return super()._select_node_count(scores)
        finally:
            self._fixed_ms, self._verify_ms = fixed, verify

    def observe(self, **kwargs):
        frozen = self.variant == "frozen_after_warmup" and all(
            n >= self.warmup_rounds_per_budget for n in self._observations.values())
        previous = self._fixed_ms, self._verify_ms.copy(), self._acceptance_scale
        super().observe(**kwargs)
        if frozen:
            self._fixed_ms, self._verify_ms, self._acceptance_scale = previous
        if self.variant == "no_acceptance_calibration":
            self._acceptance_scale = 1.
        self.trace.append({"decision": asdict(self.last_decision) if self.last_decision else None,
                           **kwargs})

    def state_dict(self):
        return {"version": 1, "identity": self.identity,
                "observations": {str(k): v for k, v in self._observations.items()},
                "verify_ms": {str(k): v for k, v in self._verify_ms.items()},
                "fixed_ms": self._fixed_ms, "acceptance_scale": self._acceptance_scale,
                "decision_count": self._decision_count,
                "last_mass_by_budget": {str(k): v for k, v in self._last_mass_by_budget.items()},
                "last_selected_budget": self._last_selected_budget,
                "last_expected_draft_tokens": self._last_expected_draft_tokens,
                "last_decision": asdict(self.last_decision) if self.last_decision else None}

    def load_state_dict(self, state):
        if set(state) != set(self.state_dict()) or state["version"] != 1 or state["identity"] != self.identity:
            raise ValueError("Controller state identity/schema mismatch")
        observations = {int(k): v for k, v in state["observations"].items()}
        verify = {int(k): v for k, v in state["verify_ms"].items()}
        masses = {int(k): v for k, v in state["last_mass_by_budget"].items()}
        if set(observations) != set(self.budget_candidates) or set(verify) - set(observations) or set(masses) - set(observations):
            raise ValueError("Invalid controller budgets")
        if any(type(v) is not int or v < 0 for v in observations.values()):
            raise ValueError("Invalid observation counts")
        if type(state["decision_count"]) is not int or state["decision_count"] < 0:
            raise ValueError("Invalid decision count")
        values = list(verify.values()) + list(masses.values()) + [state["acceptance_scale"]]
        values += [v for v in (state["fixed_ms"], state["last_expected_draft_tokens"]) if v is not None]
        if any(not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in values):
            raise ValueError("Invalid controller numeric state")
        if state["acceptance_scale"] > 2 or state["last_selected_budget"] not in (None, *self.budget_candidates):
            raise ValueError("Invalid controller calibration/last budget")
        self._observations, self._verify_ms, self._last_mass_by_budget = observations, verify, masses
        self._fixed_ms, self._acceptance_scale = state["fixed_ms"], state["acceptance_scale"]
        self._decision_count = state["decision_count"]
        self._last_selected_budget = state["last_selected_budget"]
        self._last_expected_draft_tokens = state["last_expected_draft_tokens"]
        self.last_decision = BudgetDecision(**state["last_decision"]) if state["last_decision"] else None
        self.trace = []


def make_builder(cfg, method):
    if method in VARIANTS:
        return PaperAdaptiveBuilder(cfg, method)
    if method == "ddtree":
        return FixedBudgetBuilder(K, cfg["baseline_budget"])
    if method in BASELINES and method.startswith("fixed_"):
        return FixedBudgetBuilder(K, int(method.split("_")[1]))
    raise ValueError(f"Not a tree method: {method}")
