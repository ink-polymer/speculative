"""Fail-closed checks for architecture-only DDTree comparisons."""
from __future__ import annotations

from dataclasses import asdict
from typing import Mapping

from .config import Variant


# ``method`` and ``paths`` describe the candidate architecture.  Every other
# Variant field affects the experimental workload or numerical protocol and is
# frozen for an architecture-only comparison.
ARCHITECTURE_FIELDS = frozenset({"name", "method", "paths"})
FROZEN_MODEL_SETTINGS = {
    "dtype": "bfloat16",
    "target_attention": "sdpa",
    "draft_attention": "sdpa",
    "enable_thinking": False,
    "allow_tf32": False,
}


def assert_architecture_only_pair(
    baseline: Variant,
    candidate: Variant,
    model: Mapping[str, object],
    *, positive_temperature_study: bool = False,
) -> dict[str, object]:
    """Validate and return an auditable single-variable comparison record.

    Both methods must run in the same process with the same loaded model
    objects; callers are responsible for interleaving their execution.  This
    check prevents scripts from changing attention kernels, precision, length,
    budget, temperatures, cache policy, or feature conditioning per method.
    """
    baseline.validate()
    candidate.validate()
    if baseline.method != "ddtree":
        raise ValueError("The frozen baseline must be DDTree")

    left, right = asdict(baseline), asdict(candidate)
    mismatches = {
        field: (left[field], right[field])
        for field in left
        if field not in ARCHITECTURE_FIELDS and left[field] != right[field]
    }
    if mismatches:
        raise ValueError(
            "Architecture-only comparison changed frozen controls: "
            f"{mismatches}"
        )

    actual_model = {
        key: model.get(key, default)
        for key, default in FROZEN_MODEL_SETTINGS.items()
    }
    if actual_model != FROZEN_MODEL_SETTINGS:
        raise ValueError(
            "Architecture-only model execution settings are not frozen: "
            f"expected={FROZEN_MODEL_SETTINGS}, actual={actual_model}"
        )
    if positive_temperature_study:
        if baseline.temperature <= 0 or baseline.draft_temperature != baseline.temperature:
            raise ValueError("Diffusion study requires matched positive target/draft temperatures")
    elif baseline.temperature != 1.0 or baseline.draft_temperature != 1.0:
        raise ValueError("Formal comparison requires T_target=T_draft=1.0")
    if baseline.probability_dtype != "float64":
        raise ValueError("Formal comparison requires FP64 sampling probabilities")
    if baseline.length != 15 or baseline.tree_budget != 45:
        raise ValueError("Formal comparison requires L=15 and tree budget=45")

    return {
        "comparison": "architecture_only",
        "architecture_delta": {
            "baseline": {field: left[field] for field in sorted(ARCHITECTURE_FIELDS)},
            "candidate": {field: right[field] for field in sorted(ARCHITECTURE_FIELDS)},
        },
        "frozen_variant_controls": {
            field: left[field] for field in left if field not in ARCHITECTURE_FIELDS
        },
        "frozen_model_controls": actual_model,
    }
