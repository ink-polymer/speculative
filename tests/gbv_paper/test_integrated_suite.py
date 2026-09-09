import json

import pytest

from gbv_experiments.common import ROOT, write_json
from gbv_experiments.config import build_variants
from gbv_experiments.integrated_suite import (
    audit_integrated_suite,
    load_integrated_suite,
    plan_integrated_suite,
    validate_block_order,
)
from gbv_experiments.runner import scheduled_variants


SUITE = ROOT / "configs/adaptive_tree_block_suite.json"


def test_integrated_plan_keeps_protocol_families_separate_and_complete():
    plan = plan_integrated_suite(SUITE)
    assert plan["models"] == ["qwen3_4b", "qwen3_8b"]
    assert plan["adaptive_t0"]["generation_calls"] == 41472
    assert plan["stochastic_block"]["expected_records"] == 56592
    assert plan["stochastic_block"]["expected_generations"] == 62352
    assert plan["total_generation_calls"] == 103824
    assert plan["cross_protocol_aggregate"] is None
    assert plan["fairness_audit"]["passed"]
    assert not plan["fairness_audit"]["claim_boundaries"][
        "cross_protocol_speedup_pooling_allowed"
    ]


def test_models_revisions_backends_and_block_controls_are_exactly_matched():
    suite = load_integrated_suite(SUITE)
    assert [model["adaptive_model_index"] for model in suite["models"]] == [0, 1]
    for model in suite["models"]:
        cfg = model["config"]
        assert cfg["method_order"] == {"policy":"balanced_rotation", "seed":20260909}
        assert cfg["model"]["target_attention"] == "sdpa"
        assert cfg["model"]["draft_attention"] == "sdpa"
        variants = {entry["variant"]["name"]:entry["variant"]
                    for entry in build_variants(cfg)}
        assert len(variants) == 12
        for tag in ("t0p3", "t0p6", "t1p0"):
            ddtree = variants[f"ddtree_{tag}"]
            lazy = variants[f"lazy_projection_{tag}"]
            assert all(ddtree[key] == lazy[key] for key in ddtree
                       if key not in {"name", "method"})


def test_balanced_rotation_is_reproducible_and_exactly_position_balanced():
    entries = [{"variant":{"name":name}} for name in "abcd"]
    policy = {"policy":"balanced_rotation", "seed":20260909}
    orders = [scheduled_variants(entries, policy, ordinal, 999)
              for ordinal in range(12)]
    assert orders[0] == scheduled_variants(entries, policy, 0, 1)
    for name in "abcd":
        positions = [next(i for i, entry in enumerate(order)
                          if entry["variant"]["name"] == name) for order in orders]
        assert [positions.count(position) for position in range(4)] == [3, 3, 3, 3]


def _portable_suite(tmp_path):
    spec = json.loads(SUITE.read_text())
    spec["adaptive"]["project"] = str(ROOT / "adaptivetree_paper")
    spec["adaptive"]["config"] = str(
        ROOT / "adaptivetree_paper/configs/paper_t0_full.json"
    )
    for model in spec["models"]:
        original = ROOT / "configs" / model["block_config"]
        copied = tmp_path / model["block_config"]
        copied.write_text(original.read_text())
        model["block_config"] = copied.name
    path = tmp_path / "suite.json"
    write_json(path, spec)
    return path


def test_integrated_suite_fails_closed_on_revision_or_method_drift(tmp_path):
    path = _portable_suite(tmp_path)
    spec = json.loads(path.read_text())
    spec["models"][0]["target_revision"] = "0" * 40
    write_json(path, spec)
    with pytest.raises(ValueError, match="revisions differ"):
        load_integrated_suite(path)

    path = _portable_suite(tmp_path)
    spec = json.loads(path.read_text())
    config_path = tmp_path / spec["models"][0]["block_config"]
    config = json.loads(config_path.read_text())
    lazy = next(item for item in config["explicit_variants"]
                if item["variant"]["method"] == "ddtree_lazy_projection")
    lazy["variant"]["tree_budget"] = 46
    write_json(config_path, config)
    with pytest.raises(ValueError, match="Unfair or invalid block control"):
        load_integrated_suite(path)


def test_source_and_baseline_audit_is_fail_closed():
    result = audit_integrated_suite(SUITE)
    assert result["passed"] and all(result["checks"].values())
    assert result["pinned_ddtree_commit"] == "c96427a185677bf4133ed865dd1626a5041aef9b"
    assert any(path.endswith("official_worker.py") for path in result["source_sha256"])


def test_post_run_block_order_audit_requires_complete_rotations(tmp_path):
    names = ["a", "b", "c"]
    write_json(tmp_path / "run_manifest.json", {
        "method_order":{"policy":"balanced_rotation", "seed":20260909},
        "variants":[{"variant":{"name":name}} for name in names],
    })
    rows = []
    for ordinal in range(6):
        for position, name in enumerate(names[ordinal % 3:] + names[:ordinal % 3]):
            rows.append({"variant":name, "dataset":"d", "source_id":str(ordinal),
                         "seed":17, "method_order_ordinal":ordinal,
                         "method_execution_position":position})
    (tmp_path / "results.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    assert validate_block_order(tmp_path) == {"passed":True, "prompts":6, "methods":3}
    rows.pop()
    (tmp_path / "results.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    with pytest.raises(ValueError, match="complete balanced"):
        validate_block_order(tmp_path)
