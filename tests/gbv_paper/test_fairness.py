from dataclasses import replace

import pytest

from gbv_experiments.config import Variant
from gbv_experiments.fairness import assert_architecture_only_pair


MODEL = {
    "dtype": "bfloat16",
    "target_attention": "sdpa",
    "draft_attention": "sdpa",
    "enable_thinking": False,
    "allow_tf32": False,
}


def pair():
    baseline = Variant(
        name="ddtree", method="ddtree", paths=1, length=15,
        temperature=1.0, draft_temperature=1.0, tree_budget=45,
        probability_dtype="float64",
    )
    candidate = replace(
        baseline, name="candidate", method="tree_gbv_prefix_recycle", paths=3
    )
    return baseline, candidate


def test_architecture_only_pair_emits_frozen_manifest():
    baseline, candidate = pair()
    manifest = assert_architecture_only_pair(baseline, candidate, MODEL)
    assert manifest["comparison"] == "architecture_only"
    assert manifest["frozen_variant_controls"]["length"] == 15
    assert manifest["frozen_variant_controls"]["tree_budget"] == 45
    assert manifest["frozen_model_controls"]["target_attention"] == "sdpa"


@pytest.mark.parametrize(
    "field,value",
    [
        ("length", 14),
        ("tree_budget", 32),
        ("temperature", 0.8),
        ("draft_temperature", 0.8),
        ("probability_dtype", "float32"),
        ("reuse_draft_cache", False),
        ("draft_attention", "causal"),
        ("condition_features", "zero"),
    ],
)
def test_architecture_only_pair_rejects_control_drift(field, value):
    baseline, candidate = pair()
    with pytest.raises(ValueError, match="changed frozen controls"):
        assert_architecture_only_pair(
            baseline, replace(candidate, **{field: value}), MODEL
        )


def test_architecture_only_pair_rejects_backend_drift():
    baseline, candidate = pair()
    with pytest.raises(ValueError, match="model execution settings"):
        assert_architecture_only_pair(
            baseline, candidate, {**MODEL, "target_attention": "eager"}
        )
