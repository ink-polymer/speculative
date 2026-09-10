#!/usr/bin/env python3
"""Fail-closed static audit for the registered T=0/T=1 experiment matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_T0_DATASETS = {
    "gsm8k": 128,
    "math500": 128,
    "aime24": 30,
    "aime25": 30,
    "humaneval": 164,
    "mbpp": 128,
    "livecodebench": 128,
    "swe-bench": 128,
    "mt-bench": 80,
    "alpaca": 128,
}
EXPECTED_T1_DATASETS = {
    "gsm8k": 128,
    "math500": 128,
    "aime24": 30,
    "aime25": 30,
    "humaneval": 164,
    "mbpp_sanitized": 128,
    "livecodebench": 128,
    "mt-bench": 80,
}
EXPECTED_T0_VARIANTS = {
    "adaptive_b128",
    "adaptive_b256",
    "adaptive_legacy",
    "adaptive_legacy_cost_attribution",
    "adaptive_with_exploration",
    "adaptive_no_acceptance_calibration",
    "adaptive_no_latency",
    "adaptive_frozen_after_warmup",
}
EXPECTED_T1_METHODS = {"target", "dflash", "ddtree"}
EXPECTED_T1_SEEDS = [17, 29, 43]
EXPECTED_T1_SCORING = {
    "code_backend": "docker",
    "workers": 4,
    "timeout_seconds": 10,
    "lcb_timeout_seconds": 6,
}
EXPECTED_T1_SCORING_BACKEND_POLICY = {
    "config_default": "docker",
    "allowed_runtime_backends": ["docker", "process"],
    "runtime_identity_must_match_doctor_and_scoring_manifest": True,
    "process_requires_dedicated_non_root_path_when_orchestrator_is_root": True,
    "process_identity_fields": [
        "venv_invocation_path",
        "resolved_binary_path_and_sha256",
        "pyvenv_cfg_sha256",
        "python_numpy_sympy_runtime_versions",
    ],
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def audit(matrix_path: Path) -> dict:
    matrix_path = matrix_path.resolve()
    matrix = load_json(matrix_path)
    errors: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    require(matrix.get("schema") == 1, "matrix schema must be 1")
    require(
        matrix.get("study") == "adaptivetree_ddtree_dflash_complete_formal_matrix",
        "unexpected study name",
    )
    require(len(matrix.get("models", [])) == 2, "formal matrix must contain exactly 4B and 8B")
    require([m.get("id") for m in matrix.get("models", [])] == ["qwen3_4b", "qwen3_8b"],
            "formal model order/IDs must be qwen3_4b, qwen3_8b")

    suite_path = matrix_path.parent / "adaptive_tree_block_suite.json"
    require(suite_path.is_file(), f"missing integrated suite: {suite_path}")
    if suite_path.is_file():
        suite = load_json(suite_path)
        require(suite.get("study") == "adaptive_tree_ddtree_dflash_t0_t1",
                "integrated suite study ID drifted")
        require(suite.get("adaptive", {}).get("config") ==
                matrix.get("adaptive_t0", {}).get("config"),
                "formal matrix and integrated suite select different T=0 configs")
        require(suite.get("adaptive", {}).get("primary_method") == "adaptive_b128"
                and suite.get("adaptive", {}).get("method_schema_version") == 3,
                "integrated suite must register dynamic adaptive_b128 under method schema v3")
        require(suite.get("positive_temperature") == {
            "temperatures":[1.0], "methods":["target", "dflash", "ddtree"],
            "length":15, "tree_budget":45, "probability_dtype":"float64",
            "method_order_policy":"balanced_rotation",
            "tree_block_verification":"deferred_not_run",
        }, "formal matrix and integrated suite select different T=1 methods/settings")
        suite_models = suite.get("models", [])
        matrix_models = matrix.get("models", [])
        require(len(suite_models) == len(matrix_models) == 2,
                "integrated suite must contain both registered models")
        for suite_model, matrix_model in zip(suite_models, matrix_models):
            comparable = ("id", "adaptive_model_index", "target", "target_revision",
                          "draft", "draft_revision")
            require(all(suite_model.get(key) == matrix_model.get(key)
                        for key in comparable)
                    and suite_model.get("block_config") == matrix_model.get("t1_config"),
                    f"integrated suite model differs from formal matrix: {matrix_model.get('id')}")

    t0 = matrix.get("adaptive_t0", {})
    require(t0.get("temperature") == 0.0, "AdaptiveTree is registered only at T=0")
    require(t0.get("datasets") == EXPECTED_T0_DATASETS, "T=0 must contain all ten official datasets/counts")
    require(t0.get("target_backends") == ["sdpa", "flash_attention_2"],
            "T=0 must retain both official target backends")
    require(t0.get("primary_comparison_backend") == "sdpa", "T=0 architecture table must be same-backend SDPA")
    require(t0.get("draft_backend") == "flash_attention_2", "T=0 draft backend must remain official FA2")
    require(t0.get("baselines") == ["baseline", "dflash"], "T=0 Target/DFlash baselines changed")
    require(t0.get("ddtree_budgets") == [16, 32, 64, 128, 256, 512, 1024],
            "T=0 fixed DDTree budget scan is incomplete")
    registered_variants = {
        t0.get("adaptive_primary"),
        *t0.get("adaptive_controls", []),
        *t0.get("adaptive_ablations", []),
    }
    require(registered_variants == EXPECTED_T0_VARIANTS, "T=0 canonical AdaptiveTree/ablation keys are incomplete")
    require(t0.get("adaptive_primary") == "adaptive_b128",
            "corrected dynamic B128 must be the formal primary")
    require(t0.get("sample_seed") == 0 and t0.get("generation_seed") == 0,
            "T=0 official selection/generation seed must be zero")
    require(t0.get("max_new_tokens") == 2048, "T=0 max_new_tokens must be 2048")
    require(t0.get("method_order_policy") == "balanced-rotation", "T=0 method positions must be balanced")
    require(t0.get("greedy_audit_policy") == "record-bf16-mismatches",
            "T=0 must retain rather than censor BF16 divergences")
    repeats = t0.get("timing_repeats", {})
    require(repeats.get("count") == 1 and repeats.get("unit") == "complete_model_dataset_backend_run",
            "T=0 primary pass must remain one complete pinned-official run")
    require("no confidence interval" in repeats.get("reason", ""),
            "T=0 matrix must not claim an unimplemented confidence interval")

    t0_path = (matrix_path.parent / t0.get("config", "")).resolve()
    require(t0_path.is_file(), f"missing T=0 config: {t0_path}")
    if t0_path.is_file():
        cfg = load_json(t0_path)
        require(cfg.get("temperature") == 0, "underlying AdaptiveTree config is not T=0")
        require(cfg.get("sample_limits") == EXPECTED_T0_DATASETS, "underlying T=0 dataset matrix drifted")
        require(cfg.get("tree_budgets") == t0.get("ddtree_budgets"), "underlying DDTree budgets drifted")
        require(set(cfg.get("variants", [])) == EXPECTED_T0_VARIANTS,
                "underlying T=0 config does not use all canonical AdaptiveTree keys")
        require(cfg.get("version") == 6
                and cfg.get("primary_adaptive_method") == "adaptive_b128"
                and cfg.get("method_schema_version") == 3,
                "underlying T=0 config must register dynamic B128 schema v3")
        require(cfg.get("adaptive", {}).get("budget_candidates") == [30, 45, 60, 80, 100, 128],
                "dynamic B128 primary must use the original six candidates")
        selected_pairs = [cfg.get("models", [])[m["adaptive_model_index"]] for m in matrix.get("models", [])
                          if isinstance(m.get("adaptive_model_index"), int)
                          and m["adaptive_model_index"] < len(cfg.get("models", []))]
        expected_pairs = [[m["target"], m["draft"]] for m in matrix.get("models", [])]
        require(selected_pairs == expected_pairs, "T=0 model indices do not select the registered 4B/8B pairs")

    t1 = matrix.get("stochastic_t1_baselines", {})
    require(t1.get("temperature") == 1.0, "positive-temperature baseline is registered only at T=1")
    require(t1.get("datasets") == EXPECTED_T1_DATASETS, "T=1 must contain all eight supported datasets/counts")
    require(set(t1.get("methods", [])) == EXPECTED_T1_METHODS, "T=1 is limited to Target/DFlash/DDTree")
    require(t1.get("seeds") == EXPECTED_T1_SEEDS, "T=1 requires three frozen sampling seeds")
    require(t1.get("max_new_tokens") == 2048, "T=1 max_new_tokens must be 2048")
    require(t1.get("target_backend") == "sdpa" and t1.get("draft_backend") == "sdpa",
            "T=1 methods must share the SDPA/SDPA engine")
    require(t1.get("draft_proposal_policy") == {
        "dflash": "greedy_argmax",
        "ddtree_temperature": 1.0,
    }, "T=1 DFlash/DDTree draft proposal policies drifted")
    require(t1.get("method_order_policy") == {"policy": "balanced_rotation", "seed": 20260909},
            "T=1 method positions must use the frozen balanced rotation")
    require(t1.get("bootstrap") == {
        "samples": 10000,
        "cluster": "source_id_with_all_seeds",
        "paired_baseline": "same_model_dataset_source_id_seed",
    }, "T=1 paired clustered-bootstrap contract drifted")
    require(t1.get("scoring_backend_policy") == EXPECTED_T1_SCORING_BACKEND_POLICY,
            "T=1 runtime scoring backend policy drifted")
    require(t1.get("correctness_audit") == "distributional_and_task_quality",
            "T=1 requires distributional/task-quality auditing")
    require(t1.get("greedy_exact_match_applicable") is False,
            "greedy exact-match must not gate sampled T=1 outputs")

    t1_configs = []
    for model in matrix.get("models", []):
        path = (matrix_path.parent / model.get("t1_config", "")).resolve()
        require(path.is_file(), f"missing T=1 config for {model.get('id')}: {path}")
        if not path.is_file():
            continue
        cfg = load_json(path)
        t1_configs.append(cfg)
        expected_identity = {key: model[key] for key in ("target", "target_revision", "draft", "draft_revision")}
        require(all(cfg.get("model", {}).get(key) == value for key, value in expected_identity.items()),
                f"{model.get('id')} T=1 model/revision mismatch")
        require({key:cfg.get("model", {}).get(key) for key in (
            "dtype", "target_attention", "draft_attention", "enable_thinking", "allow_tf32"
        )} == {
            "dtype":"bfloat16", "target_attention":"sdpa", "draft_attention":"sdpa",
            "enable_thinking":False, "allow_tf32":False,
        }, f"{model.get('id')} T=1 numerical/backend controls drifted")
        require(cfg.get("datasets") == list(EXPECTED_T1_DATASETS),
                f"{model.get('id')} T=1 dataset order/count contract drifted")
        require(cfg.get("evaluation") == {
            "protocol": "ddtree_counts",
            "sample_seed": 0,
            "counts": EXPECTED_T1_DATASETS,
        }, f"{model.get('id')} T=1 sample selection is not frozen")
        require(cfg.get("seeds") == EXPECTED_T1_SEEDS, f"{model.get('id')} T=1 seeds drifted")
        require(cfg.get("max_new_tokens") == 2048 and cfg.get("warmup_tokens") == 32,
                f"{model.get('id')} T=1 generation/warmup limits drifted")
        require(cfg.get("method_order") == t1.get("method_order_policy"),
                f"{model.get('id')} T=1 order policy drifted")
        require(cfg.get("bootstrap_samples") == 10000, f"{model.get('id')} T=1 bootstrap count drifted")
        require(cfg.get("scoring") == EXPECTED_T1_SCORING,
                f"{model.get('id')} T=1 scoring settings drifted")
        variants = [entry.get("variant", {}) for entry in cfg.get("explicit_variants", [])]
        require({v.get("method") for v in variants} == EXPECTED_T1_METHODS and len(variants) == 3,
                f"{model.get('id')} T=1 must contain exactly Target/DFlash/DDTree")
        require(all(v.get("temperature") == 1.0 and v.get("paths") == 1
                    and v.get("length") == 15 and v.get("tree_budget") == 45
                    and v.get("probability_dtype") == "float64" for v in variants),
                f"{model.get('id')} T=1 controls are not matched")
        by_method = {v.get("method"): v for v in variants}
        require(by_method.get("dflash", {}).get("draft_temperature") is None,
                f"{model.get('id')} T=1 DFlash must keep its greedy argmax draft")
        require(by_method.get("ddtree", {}).get("draft_temperature") == 1.0,
                f"{model.get('id')} T=1 DDTree draft temperature drifted")
        require(not any("adaptive" in str(v.get("name", "")) or "tree_block" in str(v.get("name", ""))
                        or "lazy_projection" in str(v.get("name", "")) for v in variants),
                f"{model.get('id')} T=1 contains an unregistered AdaptiveTree/tree-block claim")
    if len(t1_configs) == 2:
        fields = ("datasets", "seeds", "max_new_tokens", "warmup_tokens", "method_order",
                  "explicit_variants", "bootstrap_samples", "scoring", "evaluation")
        require(all(t1_configs[0].get(field) == t1_configs[1].get(field) for field in fields),
                "4B and 8B T=1 experiment controls differ")

    deferred = matrix.get("deferred", {})
    tree = deferred.get("tree_block_verification", {})
    require(tree.get("run_in_this_matrix") is False and tree.get("publication_claim_allowed") is False,
            "tree-block verification must stay explicitly deferred")
    require("ddtree_lazy_projection" in tree.get("reason", ""),
            "tree-block deferral must state that lazy projection is not tree-block verification")
    coder = deferred.get("qwen3_coder_30b", {})
    require(coder.get("run_in_this_matrix") is False and coder.get("publication_claim_allowed") is False,
            "unverified 30B pair must stay outside the formal H20 matrix")
    require(all(value is False for value in matrix.get("claim_policy", {}).values()),
            "all forbidden cross-protocol/deferred claims must remain false")

    t0_turns = sum(EXPECTED_T0_DATASETS.values()) + EXPECTED_T0_DATASETS["mt-bench"]
    t0_sdpa_methods = 2 + len(t0["ddtree_budgets"]) + len(registered_variants)
    t0_generations = len(matrix.get("models", [])) * t0_turns * (t0_sdpa_methods + 2)
    t1_cases = sum(EXPECTED_T1_DATASETS.values())
    t1_turns = t1_cases + EXPECTED_T1_DATASETS["mt-bench"]
    t1_generations = (len(matrix.get("models", [])) * t1_turns
                      * len(EXPECTED_T1_SEEDS) * len(EXPECTED_T1_METHODS))
    return {
        "passed": not errors,
        "errors": errors,
        "registered_scope": {
            "models": [m.get("id") for m in matrix.get("models", [])],
            "adaptive_t0_datasets": len(EXPECTED_T0_DATASETS),
            "adaptive_t0_turns_per_method_model": t0_turns,
            "adaptive_t0_generation_calls": t0_generations,
            "stochastic_t1_datasets": len(EXPECTED_T1_DATASETS),
            "stochastic_t1_seeds": len(EXPECTED_T1_SEEDS),
            "stochastic_t1_generation_calls": t1_generations,
            "total_generation_calls": t0_generations + t1_generations,
        },
        "claim_boundaries": matrix.get("claim_policy", {}),
        "deferred": matrix.get("deferred", {}),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--matrix",
        type=Path,
        default=ROOT / "configs/formal_experiment_matrix.json",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit(args.matrix)
    payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
