from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/audit_formal_experiment_matrix.py"


def load_auditor():
    spec = importlib.util.spec_from_file_location("formal_matrix_audit", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_registered_formal_matrix_is_complete_and_fail_closed():
    report = load_auditor().audit(ROOT / "configs/formal_experiment_matrix.json")
    assert report["passed"], report["errors"]
    scope = report["registered_scope"]
    assert scope == {
        "models": ["qwen3_4b", "qwen3_8b"],
        "adaptive_t0_datasets": 10,
        "adaptive_t0_turns_per_method_model": 1152,
        "adaptive_t0_generation_calls": 43776,
        "stochastic_t1_datasets": 8,
        "stochastic_t1_seeds": 3,
        "stochastic_t1_generation_calls": 21504,
        "total_generation_calls": 65280,
    }
    assert all(value is False for value in report["claim_boundaries"].values())
    assert "tree_block_verification" not in report["deferred"]
    assert report["deferred"]["qwen3_coder_30b"]["run_in_this_matrix"] is False


def test_t1_configs_are_valid_for_the_runtime_loader():
    from gbv_experiments.config import build_variants, load_config
    from gbv_experiments.data import evaluation_policy

    for name in ("adaptive_block_qwen3_4b.json", "adaptive_block_qwen3_8b.json"):
        cfg = load_config(ROOT / "configs" / name)
        assert evaluation_policy(cfg["datasets"], cfg["evaluation"])["counts"] == {
            "gsm8k": 128,
            "math500": 128,
            "aime24": 30,
            "aime25": 30,
            "humaneval": 164,
            "mbpp_sanitized": 128,
            "livecodebench": 128,
            "mt-bench": 80,
        }
        variants = build_variants(cfg)
        assert [entry["variant"]["method"] for entry in variants] == [
            "target",
            "dflash",
            "ddtree",
            "ddtree_fused_scan",
        ]
        assert {entry["variant"]["temperature"] for entry in variants} == {1.0}
        by_method = {entry["variant"]["method"]: entry["variant"] for entry in variants}
        assert by_method["dflash"]["draft_temperature"] is None
        assert by_method["ddtree"]["draft_temperature"] == 1.0
        assert by_method["ddtree_fused_scan"]["draft_temperature"] == 1.0


def test_contract_registers_only_audited_tree_block_and_no_adaptive_t1_claim():
    matrix = json.loads((ROOT / "configs/formal_experiment_matrix.json").read_text())
    assert matrix["stochastic_t1_baselines"]["methods"] == [
        "target", "dflash", "ddtree", "tree_block_verification",
    ]
    assert matrix["stochastic_t1_baselines"]["greedy_exact_match_applicable"] is False
    assert matrix["claim_policy"]["claim_adaptivetree_at_t1"] is False
    assert matrix["claim_policy"]["claim_tree_block_results_from_pilots"] is False
    assert matrix["claim_policy"][
        "claim_tree_block_results_before_completed_validated_full_matrix"
    ] is False
    assert matrix["claim_policy"][
        "claim_tree_block_superiority_when_ci_includes_one"
    ] is False
    assert matrix["tree_block_verification"]["implementation"] == "ddtree_fused_scan"
    assert matrix["tree_block_verification"][
        "publication_claim_requires_all_integrity_and_fairness_gates"
    ] is True
    assert matrix["tree_block_verification"][
        "same_tree_runtime_witness_required"
    ] is True
