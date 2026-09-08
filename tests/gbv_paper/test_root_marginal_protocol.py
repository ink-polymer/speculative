from dataclasses import replace

import pytest

from gbv_experiments.common import ROOT
from gbv_experiments.config import ROOT_MARGINAL_METHODS, SHARED_SUFFIX_METHODS, Variant, build_variants, load_config
from gbv_experiments.fairness import assert_architecture_only_pair
from gbv_experiments.runner import make_plan


def test_rm_config_preserves_original_models_datasets_and_frozen_controls():
    cfg = load_config(ROOT / "configs/root_marginal_bv_qwen3_4b.json")
    previous = load_config(ROOT / "configs/tree_bv_qwen3_4b.json")
    for field in ("model", "datasets", "seeds", "max_new_tokens", "evaluation", "scoring"):
        assert cfg[field] == previous[field]
    variants = {v["variant"]["name"]: Variant(**v["variant"]) for v in build_variants(cfg)}
    assert len(variants) == 10
    assert {v.method for v in variants.values()} >= ROOT_MARGINAL_METHODS | {"target", "dflash", "ddtree", "bv", "gbv"}
    for candidate in variants.values():
        assert_architecture_only_pair(variants["ddtree"], candidate, cfg["model"])
    plan = make_plan(cfg)
    assert plan["expected_records"] == 23580
    assert plan["expected_generations"] == 25980
    assert make_plan(cfg, groups=["main"])["variant_count"] == 4
    # The shared runner retains AR when selecting any experimental group.
    assert make_plan(cfg, groups=["ablations"])["variant_count"] == 5


@pytest.mark.parametrize("method", sorted(SHARED_SUFFIX_METHODS))
def test_rm_variants_fail_closed_on_precision_or_node_budget(method):
    valid = Variant(name=method, method=method, paths=3, length=15, tree_budget=45)
    valid.validate()
    with pytest.raises(ValueError, match="tree_budget"):
        replace(valid, tree_budget=44).validate()
    with pytest.raises(ValueError, match="FP64"):
        replace(valid, probability_dtype="float32").validate()
