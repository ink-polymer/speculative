"""Original controller plus explicitly named, non-learning ablations.

The full method delegates every decision/update to the pinned original class.
Serialization preserves causal state across prompts and process restarts.
"""
from __future__ import annotations

from dataclasses import asdict
import math

from ..ddtree_builder import BudgetDecision, DDTreeBuilder, LatencyAwareDDTreeBuilder
from .common import BASELINES, K, OFFICIAL_VARIANTS, VARIANTS, digest

TIMING_PARTITIONS = ("legacy", "budget_aware")
# Canonical formal method names.  The primary name is defined in common.py;
# B256 is deliberately named as an ablation rather than owning a generic key.
B256_ABLATION_VARIANT = "adaptive_b256"
LEGACY_ADAPTIVE_VARIANT = "adaptive_legacy"
B128_ABLATION_VARIANT = "adaptive_b128"
LEGACY_COST_ATTRIBUTION_ABLATION_VARIANT = "adaptive_legacy_cost_attribution"
EXPLORATION_ABLATION_VARIANT = "adaptive_with_exploration"
NO_ACCEPTANCE_ABLATION_VARIANT = "adaptive_no_acceptance_calibration"
NO_LATENCY_ABLATION_VARIANT = "adaptive_no_latency"
FROZEN_ABLATION_VARIANT = "adaptive_frozen_after_warmup"

# Read-only/reproduction aliases emitted by the pre-migration diagnostic
# runner.  They remain constructible so old artifacts are intelligible, but
# no new official method list emits them.
COST_ATTRIBUTED_VARIANT = "cost_attributed_no_exploration"
EXTENDED_BUDGET_VARIANT = "cost_attributed_no_exploration_b256"
LEGACY_BUDGETS = (30, 45, 60, 80, 100, 128)
EXTENDED_BUDGETS = (30, 45, 60, 80, 100, 128, 160, 192, 256)
DIAGNOSTIC_VARIANTS = (COST_ATTRIBUTED_VARIANT, EXTENDED_BUDGET_VARIANT)

# This registry is deliberately independent of the factory below.  It is the
# immutable, auditable contract for every controller allowed in a new formal
# artifact; neither a changed factory nor a forged artifact can redefine the
# expected method semantics at validation time.
OFFICIAL_CONTROLLER_REGISTRY = {
    B256_ABLATION_VARIANT: {
        "budget_candidates": EXTENDED_BUDGETS,
        "maximum_draft_nodes": 256,
        "timing_partition": "budget_aware",
        "controller_variant": "no_exploration",
        "exploration_interval": 0,
    },
    LEGACY_ADAPTIVE_VARIANT: {
        "budget_candidates": LEGACY_BUDGETS,
        "maximum_draft_nodes": 128,
        "timing_partition": "legacy",
        "controller_variant": "adaptive",
        "exploration_interval": 64,
    },
    B128_ABLATION_VARIANT: {
        "budget_candidates": LEGACY_BUDGETS,
        "maximum_draft_nodes": 128,
        "timing_partition": "budget_aware",
        "controller_variant": "no_exploration",
        "exploration_interval": 0,
    },
    LEGACY_COST_ATTRIBUTION_ABLATION_VARIANT: {
        "budget_candidates": LEGACY_BUDGETS,
        "maximum_draft_nodes": 128,
        "timing_partition": "legacy",
        "controller_variant": "no_exploration",
        "exploration_interval": 0,
    },
    EXPLORATION_ABLATION_VARIANT: {
        "budget_candidates": LEGACY_BUDGETS,
        "maximum_draft_nodes": 128,
        "timing_partition": "budget_aware",
        "controller_variant": "adaptive",
        "exploration_interval": 64,
    },
    NO_ACCEPTANCE_ABLATION_VARIANT: {
        "budget_candidates": LEGACY_BUDGETS,
        "maximum_draft_nodes": 128,
        "timing_partition": "budget_aware",
        "controller_variant": "no_acceptance_calibration",
        "exploration_interval": 0,
    },
    NO_LATENCY_ABLATION_VARIANT: {
        "budget_candidates": LEGACY_BUDGETS,
        "maximum_draft_nodes": 128,
        "timing_partition": "budget_aware",
        "controller_variant": "no_latency",
        "exploration_interval": 0,
    },
    FROZEN_ABLATION_VARIANT: {
        "budget_candidates": LEGACY_BUDGETS,
        "maximum_draft_nodes": 128,
        "timing_partition": "budget_aware",
        "controller_variant": "frozen_after_warmup",
        "exploration_interval": 0,
    },
}


def controller_config(builder):
    """Return the complete public contract recorded in a run artifact."""
    return {
        "budget_candidates": list(builder.budget_candidates),
        "maximum_draft_nodes": builder.tree_budget,
        "timing_partition": builder.timing_partition,
        "controller_variant": builder.variant,
        "exploration_interval": builder.exploration_interval,
    }


def expected_official_controller_configs(methods=OFFICIAL_VARIANTS):
    """Serialize registered method contracts without consulting live builders."""
    names = tuple(methods)
    if len(names) != len(set(names)) or set(names) - set(OFFICIAL_CONTROLLER_REGISTRY):
        raise ValueError("Unknown or duplicate official AdaptiveTree controller")
    return {
        name: {
            **OFFICIAL_CONTROLLER_REGISTRY[name],
            "budget_candidates": list(
                OFFICIAL_CONTROLLER_REGISTRY[name]["budget_candidates"]
            ),
        }
        for name in names
    }


class FixedBudgetBuilder(DDTreeBuilder):
    # Bypass the legacy engine's separate previous-acceptance budget interpolation.
    manages_budget = True


class PaperAdaptiveBuilder(LatencyAwareDDTreeBuilder):
    def __init__(self, cfg, variant="adaptive", timing_partition="legacy"):
        if variant not in VARIANTS:
            raise ValueError(f"Unknown controller variant: {variant}")
        if timing_partition not in TIMING_PARTITIONS:
            raise ValueError(f"Unknown timing partition: {timing_partition}")
        super().__init__(K, max(cfg["budget_candidates"]), tuple(cfg["budget_candidates"]),
                         cfg["initial_budget"], cfg["warmup_rounds_per_budget"],
                         cfg["ewma_alpha"], 0 if variant == "no_exploration" else cfg["exploration_interval"])
        self.variant = variant
        self.timing_partition = timing_partition
        identity = {"variant": variant, "budgets": self.budget_candidates,
            "initial": self.initial_budget, "warmup": self.warmup_rounds_per_budget,
            "alpha": self.ewma_alpha, "explore": self.exploration_interval}
        # Preserve all existing official controller identities and resume files.
        # The corrected experimental controller must never load legacy state.
        if timing_partition != "legacy":
            identity["timing_partition"] = timing_partition
        self.identity = digest(identity)
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

    def observe_stages(self, *, tree_nodes, draft_ms, tree_build_ms,
                       tree_compile_ms, target_verify_ms, commit_ms,
                       accepted_draft_tokens):
        """Attribute measured stages without changing the frozen default.

        Tree construction varies materially with the selected node budget.  The
        legacy paper protocol counted it as fixed proposal cost, which biases
        the controller toward large budgets.  ``budget_aware`` moves only that
        observed stage into the per-budget latency EWMA.  The timed generation
        loop, tree, outputs and total latency are unchanged.
        """
        stages = {
            "draft": float(draft_ms),
            "tree_build": float(tree_build_ms),
            "tree_compile": float(tree_compile_ms),
            "target_verify": float(target_verify_ms),
            "commit": float(commit_ms),
        }
        if any(not math.isfinite(value) or value < 0 for value in stages.values()):
            raise ValueError("Controller stage timings must be finite and nonnegative")
        if self.timing_partition == "legacy":
            fixed_ms = stages["draft"] + stages["tree_build"]
            budget_ms = stages["tree_compile"] + stages["target_verify"] + stages["commit"]
        else:
            fixed_ms = stages["draft"]
            budget_ms = (stages["tree_build"] + stages["tree_compile"]
                         + stages["target_verify"] + stages["commit"])
        self.observe(tree_nodes=tree_nodes, draft_ms=fixed_ms,
                     verify_ms=budget_ms,
                     accepted_draft_tokens=accepted_draft_tokens)
        if self.trace and self.timing_partition != "legacy":
            self.trace[-1]["raw_stage_ms"] = stages
            self.trace[-1]["timing_partition"] = self.timing_partition

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


def make_paper_builder(cfg, method):
    """Build a canonical official controller or a historical reproduction.

    The formal primary ``adaptive_b128`` charges tree construction to the
    selected budget, disables periodic exploration, and uses the original
    B<=128 candidates.  ``adaptive_b256`` changes only the candidate ceiling.
    The exact pre-migration method remains available as ``adaptive_legacy``.
    """
    configured = tuple(cfg["budget_candidates"])
    if configured not in (LEGACY_BUDGETS, EXTENDED_BUDGETS):
        raise ValueError("AdaptiveTree requires the registered B=128 or B=256 candidates")

    def with_budgets(budgets, *, exploration=True):
        result = {**cfg, "budget_candidates": list(budgets)}
        if not exploration:
            result["exploration_interval"] = 0
        return result

    if method in OFFICIAL_CONTROLLER_REGISTRY:
        spec = OFFICIAL_CONTROLLER_REGISTRY[method]
        builder_cfg = with_budgets(spec["budget_candidates"])
        builder_cfg["exploration_interval"] = spec["exploration_interval"]
        builder = PaperAdaptiveBuilder(
            builder_cfg, spec["controller_variant"],
            timing_partition=spec["timing_partition"],
        )
        if controller_config(builder) != expected_official_controller_configs((method,))[method]:
            raise RuntimeError(f"Registered controller construction drifted: {method}")
        return builder

    # Historical official-v4 names reproduce their original behavior.  This
    # compatibility branch is deliberately outside OFFICIAL_VARIANTS.
    if method in VARIANTS:
        return PaperAdaptiveBuilder(with_budgets(LEGACY_BUDGETS), method)
    if method == COST_ATTRIBUTED_VARIANT:
        return PaperAdaptiveBuilder(
            with_budgets(LEGACY_BUDGETS, exploration=False),
            "no_exploration", timing_partition="budget_aware",
        )
    if method == EXTENDED_BUDGET_VARIANT:
        return PaperAdaptiveBuilder(
            with_budgets(EXTENDED_BUDGETS, exploration=False),
            "no_exploration", timing_partition="budget_aware",
        )
    raise ValueError(f"Unknown paper controller: {method}")


def selected_diagnostic_variants(*, cost_attribution=False,
                                 extended_budgets=False):
    """Compatibility shim for pre-migration CLI feature flags.

    Both experiments have been promoted into the canonical matrix as
    ``adaptive_b128`` and ``adaptive_b256``.  The old flags are accepted so launch
    scripts do not fail, but must not duplicate an identical timed method.
    """
    del cost_attribution, extended_budgets
    return ()


def deprecated_experiment_flags(*, cost_attribution=False,
                                extended_budgets=False):
    """Describe legacy CLI flags in immutable contracts without adding work."""
    aliases = []
    if cost_attribution:
        aliases.append("experimental_cost_attribution_is_adaptive_b128")
    if extended_budgets:
        aliases.append("experimental_extended_budgets_is_adaptive_b256")
    return tuple(aliases)


assert set(OFFICIAL_VARIANTS) == {
    B256_ABLATION_VARIANT,
    LEGACY_ADAPTIVE_VARIANT,
    B128_ABLATION_VARIANT,
    LEGACY_COST_ATTRIBUTION_ABLATION_VARIANT,
    EXPLORATION_ABLATION_VARIANT,
    NO_ACCEPTANCE_ABLATION_VARIANT,
    NO_LATENCY_ABLATION_VARIANT,
    FROZEN_ABLATION_VARIANT,
}
