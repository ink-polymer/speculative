"""Original controller plus explicitly named, non-learning ablations.

The full method delegates every decision/update to the pinned original class.
Serialization preserves causal state across prompts and process restarts.
"""
from __future__ import annotations

from dataclasses import asdict
import heapq
import math
import statistics

import numpy as np
import torch

from ..ddtree_builder import BudgetDecision, DDTreeBuilder, LatencyAwareDDTreeBuilder
from .common import BASELINES, K, OFFICIAL_VARIANTS, VARIANTS, digest
from .cpp_raw_tree import load_cpp_raw_tree_module

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
CONTEXTUAL_V8_VARIANT = "adaptive_contextual_v8"

# Read-only/reproduction aliases emitted by the pre-migration diagnostic
# runner.  They remain constructible so old artifacts are intelligible, but
# no new official method list emits them.
COST_ATTRIBUTED_VARIANT = "cost_attributed_no_exploration"
EXTENDED_BUDGET_VARIANT = "cost_attributed_no_exploration_b256"
LEGACY_BUDGETS = (30, 45, 60, 80, 100, 128)
EXTENDED_BUDGETS = (30, 45, 60, 80, 100, 128, 160, 192, 256)
DIAGNOSTIC_VARIANTS = (
    COST_ATTRIBUTED_VARIANT,
    EXTENDED_BUDGET_VARIANT,
    CONTEXTUAL_V8_VARIANT,
)

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
        "controller_variant": "guarded_raw_prefix",
        "exploration_interval": 0,
        "architecture": "guarded_raw_prefix_v7",
        "initial_latency_samples": 1,
        "minimum_latency_samples": 3,
        "latency_window": 9,
        "probe_latency_saving_ratio": 0.15,
        "minimum_latency_saving_ratio": 0.08,
        "minimum_utility_gain_ratio": 0.03,
        "reevaluation_interval": 32,
        "proposal_temperature": 1.0,
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
    result = {
        "budget_candidates": list(builder.budget_candidates),
        "maximum_draft_nodes": builder.tree_budget,
        "timing_partition": builder.timing_partition,
        "controller_variant": builder.variant,
        "exploration_interval": builder.exploration_interval,
    }
    if builder.variant == "guarded_raw_prefix":
        result.update({
            "architecture": "guarded_raw_prefix_v7",
            "initial_latency_samples": builder.initial_latency_samples,
            "minimum_latency_samples": builder.minimum_latency_samples,
            "latency_window": builder.latency_window,
            "probe_latency_saving_ratio": builder.probe_latency_saving_ratio,
            "minimum_latency_saving_ratio": builder.minimum_latency_saving_ratio,
            "minimum_utility_gain_ratio": builder.minimum_utility_gain_ratio,
            "reevaluation_interval": builder.reevaluation_interval,
            "proposal_temperature": builder.proposal_temperature,
        })
    elif builder.variant == "contextual_prefix_v8":
        result.update({
            "architecture": "contextual_prefix_guard_v8",
            "contextual_warmup_rounds": builder.contextual_warmup_rounds,
            "contextual_refresh_interval": builder.contextual_refresh_interval,
            "contextual_history_window": builder.contextual_history_window,
            "contextual_minimum_support": builder.contextual_minimum_support,
            "contextual_mass_retention_ratio": (
                builder.contextual_mass_retention_ratio
            ),
            "contextual_floor_budget": builder.contextual_floor_budget,
            "contextual_fallback_rounds": builder.contextual_fallback_rounds,
            "proposal_temperature": builder.proposal_temperature,
        })
    return result


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
        if variant not in (*VARIANTS, "guarded_raw_prefix", "contextual_prefix_v8"):
            raise ValueError(f"Unknown controller variant: {variant}")
        if timing_partition not in TIMING_PARTITIONS:
            raise ValueError(f"Unknown timing partition: {timing_partition}")
        super().__init__(K, max(cfg["budget_candidates"]), tuple(cfg["budget_candidates"]),
                         cfg["initial_budget"], cfg["warmup_rounds_per_budget"],
                         cfg["ewma_alpha"], 0 if variant == "no_exploration" else cfg["exploration_interval"])
        self.variant = variant
        self.timing_partition = timing_partition
        if variant in {"guarded_raw_prefix", "contextual_prefix_v8"}:
            # The production path materializes the official verifier tensors
            # directly.  Avoiding DraftTree construction followed by an
            # immediate conversion removes work that fixed DDTree never pays.
            self.initial_latency_samples = 1
            self.minimum_latency_samples = 3
            self.latency_window = 9
            self.probe_latency_saving_ratio = .15
            self.minimum_latency_saving_ratio = .08
            self.minimum_utility_gain_ratio = .03
            self.reevaluation_interval = 32
            self.proposal_temperature = 1.
            self._warmup_order = tuple(reversed(self.budget_candidates))
            self._latency_samples = {budget: [] for budget in self.budget_candidates}
            self._acceptance_scale_by_budget = {
                budget: 1. for budget in self.budget_candidates
            }
            self._acceptance_observations = {
                budget: 0 for budget in self.budget_candidates
            }
            self._guard_diagnostics = None
            self._evaluated_last_decision = False
            self._raw_node_token_ids = np.empty(self.tree_budget, dtype=np.int64)
            self._raw_node_depths = np.empty(self.tree_budget, dtype=np.int64)
            self._raw_node_scores = np.empty(self.tree_budget, dtype=np.float64)
            self._raw_parents = np.empty(self.tree_budget + 1, dtype=np.int32)
            self._raw_visibility = np.empty(
                (self.tree_budget + 1, self.tree_budget + 1), dtype=np.bool_)
            self._raw_child_maps = [{} for _ in range(self.tree_budget + 1)]
            self._raw_tree_backend = "python"
            self._raw_topk_width = self.tree_budget
            self._raw_compiled_nodes = None
            self._raw_compiled_parents = None
            self._raw_posterior_host = None
            self._cpp_node_tokens = torch.empty(self.tree_budget, dtype=torch.long)
            self._cpp_node_depths = torch.empty(self.tree_budget, dtype=torch.long)
            self._cpp_node_scores = torch.empty(self.tree_budget, dtype=torch.float64)
            self._cpp_parents = torch.empty(self.tree_budget + 1, dtype=torch.long)
            self._cpp_visibility = torch.empty(
                (self.tree_budget + 1, self.tree_budget + 1), dtype=torch.bool)
        if variant == "contextual_prefix_v8":
            # The current proposal mass is an exact expectation under the
            # factorized draft distribution.  B128 verification additionally
            # reveals, for free, the smallest nested prefix that would have
            # retained the realized accepted path.  Combine both signals so a
            # smaller verifier batch is used only in high-retention contexts.
            self.contextual_warmup_rounds = 3
            self.contextual_refresh_interval = 4
            self.contextual_history_window = 12
            self.contextual_minimum_support = .90
            self.contextual_mass_retention_ratio = .995
            self.contextual_floor_budget = 100
            self.contextual_fallback_rounds = 2
            self._safe_required_budgets = []
            self._rounds_since_safe = 0
            self._force_safe_rounds = 0
            self._safe_acceptance_ewma = None
        identity = {"variant": variant, "budgets": self.budget_candidates,
            "initial": self.initial_budget, "warmup": self.warmup_rounds_per_budget,
            "alpha": self.ewma_alpha, "explore": self.exploration_interval}
        # Preserve all existing official controller identities and resume files.
        # The corrected experimental controller must never load legacy state.
        if timing_partition != "legacy":
            identity["timing_partition"] = timing_partition
        if variant == "guarded_raw_prefix":
            identity.update({
                "architecture": "guarded_raw_prefix_v7",
                "initial_latency_samples": self.initial_latency_samples,
                "minimum_latency_samples": self.minimum_latency_samples,
                "latency_window": self.latency_window,
                "probe_latency_saving_ratio": self.probe_latency_saving_ratio,
                "minimum_latency_saving_ratio": self.minimum_latency_saving_ratio,
                "minimum_utility_gain_ratio": self.minimum_utility_gain_ratio,
                "reevaluation_interval": self.reevaluation_interval,
                "proposal_temperature": self.proposal_temperature,
            })
        elif variant == "contextual_prefix_v8":
            identity.update({
                "architecture": "contextual_prefix_guard_v8",
                "contextual_warmup_rounds": self.contextual_warmup_rounds,
                "contextual_refresh_interval": self.contextual_refresh_interval,
                "contextual_history_window": self.contextual_history_window,
                "contextual_minimum_support": self.contextual_minimum_support,
                "contextual_mass_retention_ratio": (
                    self.contextual_mass_retention_ratio
                ),
                "contextual_floor_budget": self.contextual_floor_budget,
                "contextual_fallback_rounds": self.contextual_fallback_rounds,
                "proposal_temperature": self.proposal_temperature,
            })
        self.identity = digest(identity)
        self.trace = []

    def _raw_topk_to_host(self, top_log_probs, top_token_ids):
        """Private tensor-preserving D2H path; the frozen DDTree stays untouched."""
        tensors = (top_log_probs, top_token_ids)
        if top_log_probs.device.type == "cuda":
            if top_token_ids.device != top_log_probs.device:
                raise ValueError("AdaptiveTree top-k tensors must share one device")
            signature = tuple((tuple(t.shape), t.dtype) for t in tensors)
            if self._host_buffers is None or signature != self._host_signature:
                self._host_buffers = tuple(
                    torch.empty(t.shape, dtype=t.dtype, device="cpu",
                                pin_memory=True)
                    for t in tensors
                )
                self._host_signature = signature
            for host, source in zip(self._host_buffers, tensors):
                host.copy_(source, non_blocking=True)
            torch.cuda.current_stream(top_log_probs.device).synchronize()
            return self._host_buffers
        return tuple(t.detach().cpu() for t in tensors)

    def build_official_tree_from_logits(self, draft_logits):
        """Build one max-budget official tree and materialize only its prefix.

        The heap order is byte-for-byte the pinned DDTree rule.  Unlike the
        generic adapter, this path does not allocate DraftNode objects and then
        reconstruct verifier tensors from them.  All candidate budgets remain
        nested prefixes of the one max-budget heap enumeration.
        """
        if self.variant not in {"guarded_raw_prefix", "contextual_prefix_v8"}:
            raise ValueError("Raw official-tree path is reserved for guarded_raw_prefix")
        if draft_logits.ndim != 2:
            raise ValueError("draft_logits must be [K, V]")
        if (not math.isfinite(self.proposal_temperature)
                or self.proposal_temperature <= 0):
            raise ValueError("proposal_temperature must be finite and positive")
        budget = self.tree_budget
        depth_limit = min(int(draft_logits.shape[0]), self.block_size)
        maximum_topk = min(budget, int(draft_logits.shape[-1]))
        logits = draft_logits.float() / self.proposal_temperature
        log_z = torch.logsumexp(logits, dim=-1, keepdim=True)

        # CUDA experiments use a compiled CPU enumerator after the unavoidable
        # top-k metadata transfer.  It preserves Python heap tuple ordering but
        # removes per-node interpreter traffic from the timed hot path.  CPU
        # tests and toolchains without an extension compiler retain the exact
        # reference implementation below.
        cpp_module = (load_cpp_raw_tree_module()
                      if draft_logits.device.type == "cuda" else None)
        if cpp_module is not None:
            topk = maximum_topk
            top_values, top_token_ids = torch.topk(logits, k=topk, dim=-1)
            top_log_probs_host, top_token_ids_host = self._raw_topk_to_host(
                top_values - log_z, top_token_ids.to(torch.int64))
            (node_token_tensor, node_depth_tensor, node_score_tensor,
             parent_tensor, visibility_tensor) = cpp_module.build_raw_prefix_tree(
                 top_log_probs_host, top_token_ids_host, budget,
                 self._cpp_node_tokens, self._cpp_node_depths,
                 self._cpp_node_scores, self._cpp_parents,
                 self._cpp_visibility)
            node_scores = node_score_tensor.numpy()
            selected = self._select_node_count(node_scores)
            if not 0 <= selected <= budget:
                raise AssertionError("Adaptive budget is outside the enumerated tree")
            length = selected + 1
            visibility = visibility_tensor[:length, :length]
            if length != budget + 1:
                visibility = visibility.contiguous()
            self._raw_tree_backend = "cpp"
            self._raw_topk_width = topk
            self._raw_compiled_nodes = node_token_tensor[:selected]
            self._raw_compiled_parents = parent_tensor[:length]
            return (node_token_tensor[:selected], node_depth_tensor[:selected],
                    parent_tensor[:length], None, visibility, {})

        topk = maximum_topk
        top_values, top_token_ids = torch.topk(logits, k=topk, dim=-1)
        top_log_probs_host, top_token_ids_host = self._raw_topk_to_host(
            top_values - log_z, top_token_ids.to(torch.int64))
        top_log_probs = top_log_probs_host.numpy()
        top_token_ids = top_token_ids_host.numpy()

        node_token_ids = self._raw_node_token_ids
        node_depths = self._raw_node_depths
        node_scores = self._raw_node_scores
        parents = self._raw_parents
        parents[0] = -1
        first = float(top_log_probs[0, 0])
        heap = [(-first, (0,), 0, 1, 0, first)]
        node_count = 0
        while heap and node_count < budget:
            _, ranks, parent, depth, rank, logw = heapq.heappop(heap)
            current = node_count + 1
            node_token_ids[node_count] = int(top_token_ids[depth - 1, rank])
            node_depths[node_count] = depth
            node_scores[node_count] = logw
            parents[current] = parent
            node_count += 1
            if rank + 1 < topk:
                sibling_logw = (logw - float(top_log_probs[depth - 1, rank])
                                + float(top_log_probs[depth - 1, rank + 1]))
                heapq.heappush(heap, (-sibling_logw,
                    ranks[:-1] + (rank + 1,), parent, depth, rank + 1,
                    sibling_logw))
            if depth < depth_limit:
                child_logw = logw + float(top_log_probs[depth, 0])
                heapq.heappush(heap, (-child_logw, ranks + (0,), current,
                                      depth + 1, 0, child_logw))

        selected = self._select_node_count(node_scores[:node_count])
        if not 0 <= selected <= node_count:
            raise AssertionError("Adaptive budget is outside the enumerated tree")
        length = selected + 1
        child_maps = self._raw_child_maps[:length]
        for mapping in child_maps:
            mapping.clear()
        for index in range(1, length):
            child_maps[int(parents[index])][int(node_token_ids[index - 1])] = index
        visibility = self._raw_visibility[:length, :length]
        visibility.fill(False)
        visibility[0, 0] = True
        for index in range(1, length):
            parent = int(parents[index])
            visibility[index, :index] = visibility[parent, :index]
            visibility[index, index] = True
        return (torch.from_numpy(node_token_ids[:selected]),
                torch.from_numpy(node_depths[:selected]),
                parents[:length].tolist(), child_maps,
                torch.from_numpy(visibility), {})

    def follow_compiled_tree(self, posterior):
        """Follow a compiled raw tree without materializing Python child maps."""
        if self._raw_compiled_nodes is None or self._raw_compiled_parents is None:
            raise RuntimeError("No compiled raw tree is available")
        source = posterior[0]
        length = int(source.numel())
        if (self._raw_posterior_host is None
                or self._raw_posterior_host.numel() < length):
            self._raw_posterior_host = torch.empty(
                length, dtype=torch.long, device="cpu", pin_memory=True)
        host = self._raw_posterior_host[:length]
        host.copy_(source, non_blocking=True)
        torch.cuda.current_stream(source.device).synchronize()
        module = load_cpp_raw_tree_module()
        if module is None:
            raise RuntimeError("Compiled raw-tree backend disappeared")
        followed = module.follow_raw_tree(
            self._raw_compiled_nodes, self._raw_compiled_parents, host,
            int(self._raw_compiled_nodes.numel()))
        return followed[1:], followed[0]

    def _select_node_count(self, scores):
        if self.variant == "guarded_raw_prefix":
            return self._select_guarded_node_count(scores)
        if self.variant == "contextual_prefix_v8":
            return self._select_contextual_node_count(scores)
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

    @staticmethod
    def _isotonic_latency(budgets, values, weights):
        """Weighted PAVA fit enforcing nondecreasing latency with node count."""
        blocks = []
        for budget, value, weight in zip(budgets, values, weights):
            blocks.append([budget, budget, float(value) * weight, float(weight)])
            while len(blocks) > 1:
                left, right = blocks[-2], blocks[-1]
                if left[2] / left[3] <= right[2] / right[3]:
                    break
                blocks[-2:] = [[left[0], right[1], left[2] + right[2],
                                left[3] + right[3]]]
        fitted = {}
        for first, last, weighted_sum, total_weight in blocks:
            value = weighted_sum / total_weight
            for budget in budgets:
                if first <= budget <= last:
                    fitted[budget] = value
        return fitted

    def _robust_latency_by_budget(self, available):
        measured = [budget for budget in available if self._latency_samples[budget]]
        if not measured:
            return {}
        medians = [statistics.median(self._latency_samples[budget])
                   for budget in measured]
        weights = [len(self._latency_samples[budget]) for budget in measured]
        return self._isotonic_latency(measured, medians, weights)

    def _select_guarded_node_count(self, node_scores):
        """Choose a challenger only when it robustly dominates the B128 arm.

        A single noisy latency observation must never demote the equal-capacity
        reference.  Three observations per arm, a rolling median, and a
        monotone latency projection make the comparison robust.  The controller
        still reacts to the current proposal through prefix probability mass.
        """
        node_count = int(node_scores.shape[0])
        available = tuple(value for value in self.budget_candidates
                          if value <= node_count)
        if not available:
            return node_count

        safe_budget = available[-1]
        # Calibrate the equal-capacity arm first. Smaller arms receive free
        # counterfactual acceptance observations from the accepted B128 path;
        # they are not blindly sampled at startup.
        calibration = ([safe_budget]
                       if self._observations[safe_budget]
                       < self.initial_latency_samples else [])
        if (not calibration and self._decision_count % self.reevaluation_interval):
            # Most rounds take the equal-capacity safe arm without computing
            # proposal mass or refitting latency.  This keeps the online policy
            # off the hot path while still allowing periodic contextual checks.
            previous = self.last_decision
            self._last_mass_by_budget = {}
            self._last_selected_budget = safe_budget
            self._last_expected_draft_tokens = None
            self._decision_count += 1
            self.last_decision = BudgetDecision(
                budget=safe_budget,
                expected_draft_tokens=(previous.expected_draft_tokens
                                       if previous is not None else 0.),
                predicted_round_ms=None,
                predicted_tokens_per_ms=None,
            )
            self._guard_diagnostics = {
                "safe_budget": safe_budget,
                "reason": "cached_safe_arm",
            }
            self._evaluated_last_decision = False
            return safe_budget

        probability_mass = np.exp(np.clip(
            node_scores.astype(np.float64), -745., 0.))
        cumulative_mass = np.cumsum(probability_mass)
        mass_by_budget = {
            budget: float(cumulative_mass[budget - 1]) for budget in available
        }
        self._last_mass_by_budget = mass_by_budget

        selected = (min(calibration, key=lambda budget: (
            self._observations[budget], self._warmup_order.index(budget)))
            if calibration else None)
        robust_latency = self._robust_latency_by_budget(available)
        utilities = {}
        round_ms = {}
        if self._fixed_ms is not None:
            for budget, latency in robust_latency.items():
                expected = min(
                    float(self.block_size),
                    self._acceptance_scale_by_budget[budget]
                    * mass_by_budget[budget],
                )
                round_ms[budget] = self._fixed_ms + latency
                utilities[budget] = ((1. + expected)
                                     / max(round_ms[budget], 1e-6))

        guard = {
            "safe_budget": safe_budget,
            "reason": "latency_calibration" if selected is not None else "safe_fallback",
            "robust_budget_ms": robust_latency,
        }
        if selected is None and safe_budget in utilities:
            safe_utility = utilities[safe_budget]
            safe_ms = round_ms[safe_budget]
            probes = []
            if self._observations[safe_budget] >= self.minimum_latency_samples:
                for budget in available[:-1]:
                    if (self.initial_latency_samples
                            <= self._observations[budget]
                            < self.minimum_latency_samples
                            and budget in round_ms):
                        saving = ((safe_ms - round_ms[budget])
                                  / max(safe_ms, 1e-6))
                        if saving >= self.probe_latency_saving_ratio:
                            probes.append((saving, budget))
            if probes:
                _, selected = max(probes)
                guard.update({"reason": "targeted_latency_probe",
                              "latency_saving_ratio": max(probes)[0]})
            measured_challengers = [
                budget for budget in available[:-1]
                if self._observations[budget] > 0
            ]
            if not probes and not measured_challengers:
                # One conservative pilot is allowed only if it can beat the
                # safe arm even after charging the fixed draft cost. The
                # remaining variable cost is scaled linearly with node count,
                # an optimistic bound; failure under it rules the arm out
                # without executing it.
                optimistic = []
                variable_safe_ms = max(safe_ms - self._fixed_ms, 0.)
                for budget in available[:-1]:
                    if self._acceptance_observations[budget] < 1:
                        continue
                    expected_tokens = 1. + min(
                        float(self.block_size),
                        self._acceptance_scale_by_budget[budget]
                        * mass_by_budget[budget],
                    )
                    estimated_ms = (self._fixed_ms + variable_safe_ms
                                    * budget / safe_budget)
                    estimated_utility = expected_tokens / max(estimated_ms, 1e-6)
                    gain = ((estimated_utility - safe_utility)
                            / max(safe_utility, 1e-9))
                    if gain >= self.minimum_utility_gain_ratio:
                        optimistic.append((estimated_utility, budget, gain))
                if optimistic:
                    _, selected, gain = max(optimistic)
                    guard.update({"reason": "counterfactual_pilot",
                                  "optimistic_utility_gain_ratio": gain})
            qualified = []
            if selected is None:
                for budget in available[:-1]:
                    if (self._observations[budget] < self.minimum_latency_samples
                            or budget not in utilities):
                        continue
                    latency_saving = ((safe_ms - round_ms[budget])
                                      / max(safe_ms, 1e-6))
                    utility_gain = ((utilities[budget] - safe_utility)
                                    / max(safe_utility, 1e-9))
                    if (latency_saving >= self.minimum_latency_saving_ratio
                            and utility_gain >= self.minimum_utility_gain_ratio):
                        qualified.append((utilities[budget], budget,
                                          latency_saving, utility_gain))
                if qualified:
                    _, selected, latency_saving, utility_gain = max(qualified)
                    guard.update({"reason": "qualified_challenger",
                                  "latency_saving_ratio": latency_saving,
                                  "utility_gain_ratio": utility_gain})
                else:
                    selected = safe_budget
        elif selected is None:
            selected = safe_budget

        expected = min(
            float(self.block_size),
            self._acceptance_scale_by_budget[selected] * mass_by_budget[selected],
        )
        predicted_ms = round_ms.get(selected)
        utility = utilities.get(selected)
        self._last_selected_budget = selected
        self._last_expected_draft_tokens = mass_by_budget[selected]
        self._decision_count += 1
        self.last_decision = BudgetDecision(
            budget=selected,
            expected_draft_tokens=expected,
            predicted_round_ms=predicted_ms,
            predicted_tokens_per_ms=utility,
        )
        self._guard_diagnostics = guard
        self._evaluated_last_decision = True
        return selected

    def _select_contextual_node_count(self, node_scores):
        """Select B80/B100 only when present and historical evidence agree.

        The budgets are nested prefixes of the exact official B128 heap.  A
        safe B128 round therefore labels all smaller arms without executing
        them: the maximum accepted node index is the minimum prefix required
        to preserve that realized path.  Periodic B128 refreshes prevent a
        stale context from keeping the controller on a small arm indefinitely.
        """
        node_count = int(node_scores.shape[0])
        available = tuple(value for value in self.budget_candidates
                          if value <= node_count)
        if not available:
            return node_count
        safe_budget = available[-1]
        probability_mass = np.exp(np.clip(
            node_scores.astype(np.float64), -745., 0.))
        cumulative_mass = np.cumsum(probability_mass)
        mass_by_budget = {
            budget: float(cumulative_mass[budget - 1]) for budget in available
        }
        self._last_mass_by_budget = mass_by_budget
        safe_observations = self._observations[safe_budget]
        reason = "contextual_safe_fallback"
        selected = safe_budget
        candidates = []

        if safe_observations < self.contextual_warmup_rounds:
            reason = "contextual_safe_warmup"
        elif self._force_safe_rounds > 0:
            reason = "contextual_acceptance_fallback"
            self._force_safe_rounds -= 1
        elif self._rounds_since_safe >= self.contextual_refresh_interval - 1:
            reason = "contextual_periodic_refresh"
        else:
            history = self._safe_required_budgets[-self.contextual_history_window:]
            safe_expected = 1. + min(
                float(self.block_size),
                self._acceptance_scale_by_budget[safe_budget]
                * mass_by_budget[safe_budget],
            )
            for budget in available[:-1]:
                if budget < self.contextual_floor_budget or not history:
                    continue
                support = sum(required <= budget for required in history) / len(history)
                expected = 1. + min(
                    float(self.block_size),
                    self._acceptance_scale_by_budget[budget]
                    * mass_by_budget[budget],
                )
                retention = expected / max(safe_expected, 1e-9)
                if (support >= self.contextual_minimum_support
                        and retention >= self.contextual_mass_retention_ratio):
                    candidates.append((budget, support, retention))
            if candidates:
                selected, _, _ = min(candidates)
                reason = "contextual_high_retention_prefix"

        expected = min(
            float(self.block_size),
            self._acceptance_scale_by_budget[selected] * mass_by_budget[selected],
        )
        robust_latency = self._robust_latency_by_budget(available)
        predicted_ms = None
        utility = None
        if self._fixed_ms is not None and selected in robust_latency:
            predicted_ms = self._fixed_ms + robust_latency[selected]
            utility = (1. + expected) / max(predicted_ms, 1e-6)
        support = None
        retention = None
        for budget, candidate_support, candidate_retention in candidates:
            if budget == selected:
                support = candidate_support
                retention = candidate_retention
                break
        self._last_selected_budget = selected
        self._last_expected_draft_tokens = mass_by_budget[selected]
        self._decision_count += 1
        self.last_decision = BudgetDecision(
            budget=selected,
            expected_draft_tokens=expected,
            predicted_round_ms=predicted_ms,
            predicted_tokens_per_ms=utility,
        )
        self._guard_diagnostics = {
            "safe_budget": safe_budget,
            "reason": reason,
            "mass_retention_ratio": retention,
            "historical_support": support,
            "safe_history_size": len(self._safe_required_budgets),
            "rounds_since_safe": self._rounds_since_safe,
        }
        self._evaluated_last_decision = True
        return selected

    def observe(self, **kwargs):
        accepted_node_indices = kwargs.pop("accepted_node_indices", None)
        frozen = self.variant == "frozen_after_warmup" and all(
            n >= self.warmup_rounds_per_budget for n in self._observations.values())
        previous = self._fixed_ms, self._verify_ms.copy(), self._acceptance_scale
        if self.variant in {"guarded_raw_prefix", "contextual_prefix_v8"}:
            selected = self._last_selected_budget
            if selected is not None:
                self._observations[selected] += 1
                self._fixed_ms = self._ewma(
                    self._fixed_ms, max(float(kwargs["draft_ms"]), 0.),
                    self.ewma_alpha)
                self._verify_ms[selected] = self._ewma(
                    self._verify_ms.get(selected),
                    max(float(kwargs["verify_ms"]), 0.), self.ewma_alpha)
        else:
            super().observe(**kwargs)
        if frozen:
            self._fixed_ms, self._verify_ms, self._acceptance_scale = previous
        if self.variant == "no_acceptance_calibration":
            self._acceptance_scale = 1.
        if self.variant in {"guarded_raw_prefix", "contextual_prefix_v8"}:
            selected = self._last_selected_budget
            if selected is not None:
                samples = self._latency_samples[selected]
                samples.append(float(kwargs["verify_ms"]))
                del samples[:-self.latency_window]
            observed_budgets = ()
            if (self._evaluated_last_decision
                    and accepted_node_indices is not None and selected is not None):
                indices = tuple(int(index) for index in accepted_node_indices
                                if int(index) > 0)
                observed_budgets = tuple(
                    budget for budget in self.budget_candidates
                    if budget <= selected and budget in self._last_mass_by_budget
                )
                for budget in observed_budgets:
                    accepted = sum(index <= budget for index in indices)
                    mass = self._last_mass_by_budget[budget]
                    if mass > 1e-9:
                        ratio = max(0., min(2., accepted / mass))
                        self._acceptance_scale_by_budget[budget] = self._ewma(
                            self._acceptance_scale_by_budget[budget], ratio,
                            self.ewma_alpha)
                        self._acceptance_observations[budget] += 1
            elif self._evaluated_last_decision and selected is not None:
                mass = self._last_mass_by_budget.get(selected, 0.)
                if mass > 1e-9:
                    ratio = max(0., min(
                        2., float(kwargs["accepted_draft_tokens"]) / mass))
                    self._acceptance_scale_by_budget[selected] = self._ewma(
                        self._acceptance_scale_by_budget[selected], ratio,
                        self.ewma_alpha)
                    self._acceptance_observations[selected] += 1
                    observed_budgets = (selected,)
            if self.variant == "contextual_prefix_v8" and selected is not None:
                safe_budget = self.budget_candidates[-1]
                accepted = max(float(kwargs["accepted_draft_tokens"]), 0.)
                if selected == safe_budget:
                    indices = tuple(int(index) for index in
                                    (accepted_node_indices or ())
                                    if int(index) > 0)
                    required = max(indices, default=0)
                    self._safe_required_budgets.append(required)
                    del self._safe_required_budgets[:-self.contextual_history_window]
                    self._rounds_since_safe = 0
                    self._safe_acceptance_ewma = self._ewma(
                        self._safe_acceptance_ewma, accepted, self.ewma_alpha)
                else:
                    self._rounds_since_safe += 1
                    if (self._safe_acceptance_ewma is not None
                            and accepted + 1. < self._safe_acceptance_ewma):
                        self._force_safe_rounds = self.contextual_fallback_rounds
        trace = {"decision": asdict(self.last_decision) if self.last_decision else None,
                 **kwargs}
        if self.variant in {"guarded_raw_prefix", "contextual_prefix_v8"}:
            trace["guard"] = self._guard_diagnostics
            trace["tree_backend"] = self._raw_tree_backend
            trace["topk_width"] = self._raw_topk_width
            trace["counterfactual_acceptance_budgets"] = list(observed_budgets)
        self.trace.append(trace)

    def observe_stages(self, *, tree_nodes, draft_ms, tree_build_ms,
                       tree_compile_ms, target_verify_ms, commit_ms,
                       accepted_draft_tokens, accepted_node_indices=None):
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
                     accepted_draft_tokens=accepted_draft_tokens,
                     accepted_node_indices=accepted_node_indices)
        if self.trace and self.timing_partition != "legacy":
            self.trace[-1]["raw_stage_ms"] = stages
            self.trace[-1]["timing_partition"] = self.timing_partition

    def state_dict(self):
        state = {"version": 1, "identity": self.identity,
                "observations": {str(k): v for k, v in self._observations.items()},
                "verify_ms": {str(k): v for k, v in self._verify_ms.items()},
                "fixed_ms": self._fixed_ms, "acceptance_scale": self._acceptance_scale,
                "decision_count": self._decision_count,
                "last_mass_by_budget": {str(k): v for k, v in self._last_mass_by_budget.items()},
                "last_selected_budget": self._last_selected_budget,
                "last_expected_draft_tokens": self._last_expected_draft_tokens,
                "last_decision": asdict(self.last_decision) if self.last_decision else None}
        if self.variant in {"guarded_raw_prefix", "contextual_prefix_v8"}:
            state.update({
                "version": 2,
                "latency_samples": {str(k): list(v)
                                    for k, v in self._latency_samples.items()},
                "acceptance_scale_by_budget": {
                    str(k): v for k, v in self._acceptance_scale_by_budget.items()
                },
                "acceptance_observations": {
                    str(k): v for k, v in self._acceptance_observations.items()
                },
                "evaluated_last_decision": self._evaluated_last_decision,
            })
        if self.variant == "contextual_prefix_v8":
            state.update({
                "version": 3,
                "safe_required_budgets": list(self._safe_required_budgets),
                "rounds_since_safe": self._rounds_since_safe,
                "force_safe_rounds": self._force_safe_rounds,
                "safe_acceptance_ewma": self._safe_acceptance_ewma,
            })
        return state

    def load_state_dict(self, state):
        expected_version = (3 if self.variant == "contextual_prefix_v8" else
                            2 if self.variant == "guarded_raw_prefix" else 1)
        if (set(state) != set(self.state_dict())
                or state["version"] != expected_version
                or state["identity"] != self.identity):
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
        if self.variant in {"guarded_raw_prefix", "contextual_prefix_v8"}:
            latency_samples = {int(k): list(v)
                               for k, v in state["latency_samples"].items()}
            scales = {int(k): v for k, v in
                      state["acceptance_scale_by_budget"].items()}
            acceptance_observations = {int(k): v for k, v in
                                       state["acceptance_observations"].items()}
            if (set(latency_samples) != set(self.budget_candidates)
                    or set(scales) != set(self.budget_candidates)
                    or set(acceptance_observations) != set(self.budget_candidates)):
                raise ValueError("Invalid guarded controller budgets")
            flat_samples = [value for values in latency_samples.values()
                            for value in values]
            if (any(len(values) > self.latency_window
                    for values in latency_samples.values())
                    or any(not isinstance(value, (int, float))
                           or not math.isfinite(value) or value < 0
                           for value in flat_samples + list(scales.values()))
                    or any(type(value) is not int or value < 0
                           for value in acceptance_observations.values())):
                raise ValueError("Invalid guarded controller state")
            self._latency_samples = latency_samples
            self._acceptance_scale_by_budget = scales
            self._acceptance_observations = acceptance_observations
            self._guard_diagnostics = None
            if type(state["evaluated_last_decision"]) is not bool:
                raise ValueError("Invalid guarded controller state")
            self._evaluated_last_decision = state["evaluated_last_decision"]
        if self.variant == "contextual_prefix_v8":
            required = list(state["safe_required_budgets"])
            counters = (state["rounds_since_safe"], state["force_safe_rounds"])
            safe_acceptance = state["safe_acceptance_ewma"]
            if (len(required) > self.contextual_history_window
                    or any(type(value) is not int or value < 0
                           or value > self.tree_budget for value in required)
                    or any(type(value) is not int or value < 0 for value in counters)
                    or (safe_acceptance is not None
                        and (not isinstance(safe_acceptance, (int, float))
                             or not math.isfinite(safe_acceptance)
                             or safe_acceptance < 0))):
                raise ValueError("Invalid contextual controller state")
            self._safe_required_budgets = required
            self._rounds_since_safe, self._force_safe_rounds = counters
            self._safe_acceptance_ewma = safe_acceptance
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

    The formal primary ``adaptive_b128`` is the guarded raw-prefix
    architecture.  It charges tree construction to the selected budget,
    uses B128 as a safe arm, and only admits a smaller challenger after robust
    latency calibration.  ``adaptive_b256`` remains the historical
    budget-extension ablation.  The exact pre-migration method remains
    available as ``adaptive_legacy``.
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
    if method == CONTEXTUAL_V8_VARIANT:
        return PaperAdaptiveBuilder(
            with_budgets(LEGACY_BUDGETS, exploration=False),
            "contextual_prefix_v8", timing_partition="budget_aware",
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
