import json
import inspect
import hashlib
import os
import subprocess
import sys
import time
import venv
from dataclasses import replace

import pytest

from gbv_experiments.common import ROOT, digest, file_hash, source_hashes, write_json
from gbv_experiments.config import Variant, build_variants
from gbv_experiments.data import DATASETS
from gbv_experiments.integrated_suite import (
    ADAPTIVE_METHOD_SCHEMA_VERSION,
    ADAPTIVE_PRIMARY_METHOD,
    DISTRIBUTION_LAW_TESTS,
    T1_DOCTOR_TESTS,
    T1_IMPLEMENTATION_TESTS,
    _adaptive_code_identity,
    _adaptive_contract_digest,
    _literal_assignment,
    _run_distribution_law_audit,
    _validate_adaptive_summary,
    _validate_sampling_manifest,
    _validate_scoring_manifest,
    _validate_sampling_summary,
    _validate_t1_preflight,
    audit_integrated_suite,
    load_integrated_suite,
    plan_integrated_suite,
    run_integrated_suite,
    validate_block_order,
)
from gbv_experiments.runner import dataset_local_schedule, make_plan, scheduled_variants


SUITE = ROOT / "configs/adaptive_tree_block_suite.json"


@pytest.mark.skipif(
    os.name != "posix" or (hasattr(os, "geteuid") and os.geteuid() == 0),
    reason="symlink semantics test requires a non-root POSIX interpreter",
)
def test_process_scorer_identity_preserves_venv_invocation_symlink(tmp_path, monkeypatch):
    from gbv_experiments.scoring import process_python_identity

    environment = tmp_path / "evaluator-venv"
    venv.EnvBuilder(with_pip=False, system_site_packages=True, symlinks=True).create(
        environment
    )
    invocation = environment / "bin" / "python"
    assert invocation.is_symlink()
    monkeypatch.setenv("GBV_PROCESS_PYTHON", str(invocation))
    identity = process_python_identity()

    assert identity["process_python"] == str(invocation)
    assert identity["process_python_resolved"] == str(invocation.resolve())
    assert identity["process_python"] != identity["process_python_resolved"]
    assert identity["process_python_runtime"]["prefix"] == str(environment)
    assert identity["process_python_runtime"]["numpy"]
    assert identity["process_python_runtime"]["sympy"]
    assert identity["process_pyvenv_cfg_sha256"] == file_hash(
        environment / "pyvenv.cfg"
    )


def test_integrated_plan_keeps_protocol_families_separate_and_complete():
    plan = plan_integrated_suite(SUITE)
    assert plan["models"] == ["qwen3_4b", "qwen3_8b"]
    assert plan["adaptive_t0"]["generation_calls"] == 43776
    assert plan["adaptive_t0"]["primary_method"] == ADAPTIVE_PRIMARY_METHOD
    assert plan["adaptive_t0"]["method_schema_version"] == ADAPTIVE_METHOD_SCHEMA_VERSION
    assert plan["adaptive_t0"]["greedy_audit_policy"] == "record-bf16-mismatches"
    assert not plan["adaptive_t0"]["strict_lossless_claim_allowed"]
    assert plan["positive_temperature_t1"]["temperatures"] == [1.0]
    assert plan["positive_temperature_t1"]["methods"] == ["target", "dflash", "ddtree"]
    assert plan["positive_temperature_t1"]["expected_records"] == 14688
    assert plan["positive_temperature_t1"]["expected_generations"] == 16128
    assert plan["tree_block_verification"] == "deferred_not_run"
    assert plan["total_generation_calls"] == 59904
    assert plan["cross_protocol_aggregate"] is None
    assert plan["fairness_audit"]["passed"]
    assert not plan["fairness_audit"]["claim_boundaries"][
        "cross_protocol_speedup_pooling_allowed"
    ]
    assert not plan["fairness_audit"]["claim_boundaries"][
        "adaptive_strict_lossless_claim_allowed"
    ]


def test_models_revisions_backends_and_t1_controls_are_exactly_matched():
    suite = load_integrated_suite(SUITE)
    assert suite["spec"]["adaptive"]["greedy_audit_policy"] == (
        "record-bf16-mismatches"
    )
    assert suite["spec"]["adaptive"]["primary_method"] == ADAPTIVE_PRIMARY_METHOD
    assert [model["adaptive_model_index"] for model in suite["models"]] == [0, 1]
    for model in suite["models"]:
        cfg = model["config"]
        assert cfg["method_order"] == {"policy":"balanced_rotation", "seed":20260909}
        assert cfg["model"]["target_attention"] == "sdpa"
        assert cfg["model"]["draft_attention"] == "sdpa"
        variants = {entry["variant"]["name"]:entry["variant"]
                    for entry in build_variants(cfg)}
        assert set(variants) == {"target_t1p0", "dflash_t1p0", "ddtree_t1p0"}
        assert {variant["temperature"] for variant in variants.values()} == {1.0}
        assert {variant["method"] for variant in variants.values()} == {
            "target", "dflash", "ddtree"
        }
        assert variants["dflash_t1p0"]["draft_temperature"] is None
        assert variants["ddtree_t1p0"]["draft_temperature"] == 1.0
        assert cfg["datasets"] == [
            "gsm8k", "math500", "aime24", "aime25", "humaneval", "mbpp_sanitized",
            "livecodebench", "mt-bench",
        ]


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


def test_dataset_local_ordinals_restart_per_dataset_and_continue_across_seeds():
    rows = [{"dataset":"a"}, {"dataset":"b"}, {"dataset":"a"},
            {"dataset":"b"}, {"dataset":"b"}]
    ordinals, counts = dataset_local_schedule(rows)
    assert ordinals == [0, 0, 1, 1, 2]
    assert counts == {"a":2, "b":3}
    assert [[seed * counts[row["dataset"]] + ordinals[index]
             for seed in range(2) for index, row in enumerate(rows)
             if row["dataset"] == dataset]
            for dataset in ("a", "b")] == [list(range(4)), list(range(6))]


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
    ddtree = next(item for item in config["explicit_variants"]
                  if item["variant"]["method"] == "ddtree")
    ddtree["variant"]["method"] = "ddtree_lazy_projection"
    write_json(config_path, config)
    with pytest.raises(ValueError, match="Unfair or invalid T=1 control"):
        load_integrated_suite(path)

    path = _portable_suite(tmp_path)
    spec = json.loads(path.read_text())
    spec["positive_temperature"]["tree_block_verification"] = "enabled"
    write_json(path, spec)
    with pytest.raises(ValueError, match="tree-block deferral"):
        load_integrated_suite(path)

    path = _portable_suite(tmp_path)
    spec = json.loads(path.read_text())
    spec["models"].pop()
    write_json(path, spec)
    with pytest.raises(ValueError, match="exactly Qwen3-4B and Qwen3-8B"):
        load_integrated_suite(path)


def test_source_and_baseline_audit_is_fail_closed():
    result = audit_integrated_suite(SUITE)
    assert result["passed"] and all(result["checks"].values())
    assert result["claim_boundaries"]["tree_block_verification"] == "deferred_not_run"
    assert result["pinned_ddtree_commit"] == "c96427a185677bf4133ed865dd1626a5041aef9b"
    assert any(path.endswith("official_worker.py") for path in result["source_sha256"])


def test_post_run_sampling_order_audit_requires_complete_rotations(tmp_path):
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


def test_post_run_order_rejects_global_balance_that_is_biased_by_dataset(tmp_path):
    names = ["a", "b", "c"]
    write_json(tmp_path / "run_manifest.json", {
        "method_order":{"policy":"balanced_rotation", "seed":20260909},
        "variants":[{"variant":{"name":name}} for name in names],
    })
    rows = []
    # Across both datasets every offset occurs twice (globally exact), while
    # each dataset has offset counts 2/1/0 and is therefore not fair locally.
    for dataset, offsets in {"d0":[0, 0, 1], "d1":[1, 2, 2]}.items():
        for prompt, offset in enumerate(offsets):
            for position, name in enumerate(names[offset:] + names[:offset]):
                rows.append({"variant":name, "dataset":dataset,
                             "source_id":str(prompt), "seed":17,
                             "method_order_ordinal":offset,
                             "method_execution_position":position})
    (tmp_path / "results.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    with pytest.raises(ValueError, match="within dataset"):
        validate_block_order(tmp_path)


def test_doctor_distribution_gate_executes_both_exact_law_tests():
    evidence = _run_distribution_law_audit()
    assert evidence["passed"]
    assert tuple(evidence["pytest_node_ids"]) == T1_DOCTOR_TESTS
    assert tuple(evidence["distribution_law_node_ids"]) == DISTRIBUTION_LAW_TESTS
    assert tuple(evidence["implementation_contract_node_ids"]) == T1_IMPLEMENTATION_TESTS
    assert evidence["exact_finite_enumeration"]


def test_t1_summary_contract_checks_identity_coverage_and_quality():
    cfg = load_integrated_suite(SUITE, ["qwen3_4b"])["models"][0]["config"]
    variants = build_variants(cfg)
    manifest = {"run_id":"run"}
    counts = cfg["evaluation"]["counts"]
    rows = []
    for dataset in cfg["datasets"]:
        for entry in variants:
            variant = entry["variant"]
            samples = counts[dataset] * len(cfg["seeds"])
            rows.append({"dataset":dataset, "variant":variant["name"],
                         "method":variant["method"],
                         "temperature":variant["temperature"], "samples":samples,
                         "unique_prompts":counts[dataset],
                         "matched_baseline_samples":samples,
                         "mean_generation_length":64.0,
                         "accepted_per_verify":None if variant["method"] == "target" else 3.0,
                         "acceptance_rate":None if variant["method"] == "target" else .5,
                         "quality":None if dataset == "mt-bench" else .75,
                         "quality_status":"external_judge_report_separate"
                         if dataset == "mt-bench" else "scored",
                         "scored_samples":0 if dataset == "mt-bench" else samples})
    summary = {"run_id":"run", "bootstrap_clusters":"source_id_with_all_seeds",
               "bootstrap_samples":cfg["bootstrap_samples"],
               "performance_only":False, "coverage":{"complete":True}, "rows":rows}
    assert _validate_sampling_summary(summary, manifest, cfg) == rows
    broken = json.loads(json.dumps(summary))
    broken["rows"][-2] = broken["rows"][-1]
    with pytest.raises(ValueError, match="coverage"):
        _validate_sampling_summary(broken, manifest, cfg)
    broken = json.loads(json.dumps(summary))
    broken["rows"][0]["scored_samples"] -= 1
    with pytest.raises(ValueError, match="quality"):
        _validate_sampling_summary(broken, manifest, cfg)
    broken = json.loads(json.dumps(summary))
    broken["bootstrap_clusters"] = "seed_rows"
    with pytest.raises(ValueError, match="bootstrap"):
        _validate_sampling_summary(broken, manifest, cfg)


def test_t1_manifest_and_scoring_contract_reject_old_or_mismatched_runs():
    cfg = load_integrated_suite(SUITE, ["qwen3_4b"])["models"][0]["config"]
    plan = make_plan(cfg)
    doctor = {
        "python":"3.11.16", "cuda":"12.8", "gpu":"NVIDIA H20",
        "gpu_uuid":"GPU-formal", "gpu_memory_bytes":105087467520,
        "torch":"2.8.0+cu128",
        "code_backend":"process",
        "docker_image":None,
        "code_evaluator_python":"/opt/gbv-code-eval/bin/python",
        "code_evaluator_python_resolved":"/usr/local/bin/python3.11",
        "code_evaluator_python_sha256":"evaluator-python-sha256",
        "code_evaluator_python_runtime":{
            "executable":"/opt/gbv-code-eval/bin/python",
            "prefix":"/opt/gbv-code-eval", "base_prefix":"/usr/local",
            "implementation":"CPython", "python":"3.11.16",
            "numpy":"2.2.6", "sympy":"1.14.0",
        },
        "code_evaluator_pyvenv_cfg_sha256":"pyvenv-sha256",
        "packages":{
            "transformers":"4.57.1", "datasets":"3.6.0",
            "huggingface-hub":"0.36.0", "numpy":"2.2.6",
            "math-verify":"0.8.0",
        },
    }
    manifest = {
        "run_id":"formal-run", "schema":3, "model":cfg["model"],
        "variants":build_variants(cfg),
        "seeds":cfg["seeds"], "max_new_tokens":cfg["max_new_tokens"],
        "method_order":cfg["method_order"], "coverage":plan["coverage"],
        "evaluation":plan["evaluation"], "profile":False,
        "scoring":cfg["scoring"], "bootstrap_samples":cfg["bootstrap_samples"],
        "dataset_names":cfg["datasets"],
        "dataset_turn_counts":{name:DATASETS[name].turns for name in cfg["datasets"]},
        "expected_records":plan["expected_records"],
        "expected_generations":plan["expected_generations"],
        "source_hashes":source_hashes(), "python":doctor["python"],
        "cuda":doctor["cuda"], "gpu":doctor["gpu"],
        "gpu_uuid":doctor["gpu_uuid"],
        "gpu_memory_bytes":doctor["gpu_memory_bytes"],
        "versions":{
            "torch":doctor["torch"], "transformers":"4.57.1",
            "datasets":"3.6.0", "huggingface-hub":"0.36.0", "numpy":"2.2.6",
        },
    }
    _validate_sampling_manifest(manifest, cfg, doctor)
    for field, bad_value in (("seeds", [17, 29, 99]),
                             ("max_new_tokens", 1024),
                             ("bootstrap_samples", 1000)):
        broken = json.loads(json.dumps(manifest))
        broken[field] = bad_value
        with pytest.raises(ValueError, match="frozen experiment contract"):
            _validate_sampling_manifest(broken, cfg, doctor)

    scoring_id = digest({
        "run_id":manifest["run_id"], "backend":"process", "image_id":None,
        "process_python":doctor["code_evaluator_python"],
        "process_python_resolved":doctor["code_evaluator_python_resolved"],
        "process_python_sha256":doctor["code_evaluator_python_sha256"],
        "process_python_runtime":doctor["code_evaluator_python_runtime"],
        "process_pyvenv_cfg_sha256":doctor["code_evaluator_pyvenv_cfg_sha256"],
        "math_verify":"0.8.0", "timeout":10, "lcb_timeout_per_test":6,
        "scorer_sources":source_hashes(),
    })
    scoring = {"backend":"process", "image_id":None,
               "process_python":doctor["code_evaluator_python"],
               "process_python_resolved":doctor["code_evaluator_python_resolved"],
               "process_python_sha256":doctor["code_evaluator_python_sha256"],
               "process_python_runtime":doctor["code_evaluator_python_runtime"],
               "process_pyvenv_cfg_sha256":doctor["code_evaluator_pyvenv_cfg_sha256"],
               "math_verify":"0.8.0", "timeout_seconds":10,
               "lcb_timeout_seconds_per_test":6, "scoring_id":scoring_id}
    scores = [{"scoring_id":scoring_id}]
    _validate_scoring_manifest(scoring, scores, cfg, doctor, manifest)
    scores[0]["scoring_id"] = "old-contract"
    with pytest.raises(ValueError, match="scoring contract"):
        _validate_scoring_manifest(scoring, scores, cfg, doctor, manifest)
    scores[0]["scoring_id"] = scoring_id
    scoring["process_python_sha256"] = "changed-interpreter"
    with pytest.raises(ValueError, match="scoring contract"):
        _validate_scoring_manifest(scoring, scores, cfg, doctor, manifest)


def test_t1_gpu_preflight_is_bound_to_model_source_backend_and_gpu():
    cfg = load_integrated_suite(SUITE, ["qwen3_4b"])["models"][0]["config"]
    doctor = {
        "python":"3.11.16", "cuda":"12.8", "gpu":"NVIDIA H20",
        "gpu_uuid":"GPU-formal", "gpu_memory_bytes":105087467520,
        "torch":"2.8.0+cu128", "code_backend":"process",
        "docker_image":None,
        "code_evaluator_python":"/opt/gbv-code-eval/bin/python",
        "code_evaluator_python_resolved":"/usr/local/bin/python3.11",
        "code_evaluator_python_sha256":"evaluator-python-sha256",
        "code_evaluator_python_runtime":{
            "executable":"/opt/gbv-code-eval/bin/python",
            "prefix":"/opt/gbv-code-eval", "base_prefix":"/usr/local",
            "implementation":"CPython", "python":"3.11.16",
            "numpy":"2.2.6", "sympy":"1.14.0",
        },
        "code_evaluator_pyvenv_cfg_sha256":"pyvenv-sha256",
        "packages":{
            "transformers":"4.57.1", "huggingface-hub":"0.36.0",
            "datasets":"3.6.0", "numpy":"2.2.6", "math-verify":"0.8.0",
        },
    }
    variants = [
        replace(Variant(**entry["variant"]), temperature=0.0,
                draft_temperature=1.0).to_dict()
        for entry in build_variants(cfg)
    ]
    prompts = [
        "Compute 19 + 23. Give a brief explanation.",
        "Write a Python function that reverses a list.",
        "Explain why the sum of two even integers is even.",
    ]
    checks = [
        {"prompt_sha256":digest(prompt), "variant":variant,
         "greedy_equal":True, "gate_passed":True}
        for prompt in prompts for variant in variants
    ]
    checks += [
        {"check":"tree_argmax", "passed":True,
         "max_absolute_logit_error":0.01}
        for _ in range(2 * len(prompts))
    ]
    checks += [
        {"check":"compacted_cache_argmax", "passed":True,
         "max_absolute_logit_error":0.01}
        for _ in prompts
    ]
    preflight = {
        "passed":True, "greedy_exact_passed":True, "numerical_ambiguities":0,
        "numerical_policy":{
            "dtype":"bfloat16_only", "max_absolute_logit_error":0.5,
            "required":"stable repeated outputs and top-1 margins <= 2 * measured error",
        },
        "scope":"checkpoint structural and bounded-numerical smoke gate, not full benchmark results",
        "model":cfg["model"], "stop_token_ids":[151645],
        "source_hashes":source_hashes(), "checks":checks,
        "environment":{
            "versions":{
                "torch":doctor["torch"], "transformers":"4.57.1",
                "huggingface-hub":"0.36.0", "datasets":"3.6.0",
                "numpy":"2.2.6", "math-verify":"0.8.0",
            },
            "python":doctor["python"], "cuda":doctor["cuda"],
            "gpu":doctor["gpu"], "gpu_uuid":doctor["gpu_uuid"],
            "total_memory_bytes":doctor["gpu_memory_bytes"],
            "code_backend":"process", "code_image_id":None,
            "process_python":doctor["code_evaluator_python"],
            "process_python_resolved":doctor["code_evaluator_python_resolved"],
            "process_python_sha256":doctor["code_evaluator_python_sha256"],
            "process_python_runtime":doctor["code_evaluator_python_runtime"],
            "process_pyvenv_cfg_sha256":doctor["code_evaluator_pyvenv_cfg_sha256"],
            "lcb_evaluator_checks":[
                {"functional":functional, "expected_pass":expected,
                 "result":{"passed":expected}}
                for functional in (False, True) for expected in (False, True)
            ],
        },
    }
    _validate_t1_preflight(preflight, cfg, doctor)
    broken = json.loads(json.dumps(preflight))
    broken["environment"]["gpu_uuid"] = "GPU-other"
    with pytest.raises(ValueError, match="preflight environment"):
        _validate_t1_preflight(broken, cfg, doctor)


def test_adaptive_summary_contract_requires_complete_registered_matrix(tmp_path):
    suite = load_integrated_suite(SUITE, ["qwen3_4b"])
    model = suite["models"][0]
    cfg = suite["adaptive_config"]
    doctor = {
        "python":"3.11.16", "cuda":"12.8", "gpu":"NVIDIA H20",
        "gpu_uuid":"GPU-formal", "gpu_memory_bytes":105087467520,
        "torch":"2.8.0+cu128", "flash_attn":"2.8.3",
        "packages":{
            "transformers":"4.57.1", "datasets":"3.6.0",
            "huggingface-hub":"0.36.0", "accelerate":"1.10.1",
            "numpy":"2.2.6",
        },
    }
    environment = {
        "python":doctor["python"], "cuda":doctor["cuda"], "gpu":doctor["gpu"],
        "gpu_uuid":doctor["gpu_uuid"], "gpu_memory_bytes":doctor["gpu_memory_bytes"],
        "packages":{
            "torch":doctor["torch"],
            **{name:doctor["packages"][name] for name in (
                "transformers", "datasets", "huggingface-hub", "accelerate", "numpy"
            )},
        },
        "flash_attn":doctor["flash_attn"], "nproc_per_node":1,
        "benchmark_gpus":[{"rank":0, "gpu":doctor["gpu"],
                           "uuid":doctor["gpu_uuid"]}],
    }
    write_json(tmp_path / "environment.json", environment)
    official_spec = (
        suite["adaptive_root"] / "src/dflash_specblock/paper/official_spec.py"
    )
    sources = _literal_assignment(official_spec, "SOURCES")
    declared_models = _literal_assignment(official_spec, "MODELS")
    pinned = _literal_assignment(official_spec, "PINNED_MODEL_REVISIONS")
    source_lock = {
        "official_commit":cfg["official_commit"],
        "datasets":{entry[0]:"1" * 40 for entry in sources.values()},
        "models":{name:"2" * 40 for pair in declared_models for name in pair},
    }
    source_lock["models"].update(pinned)
    serialized_lock = (
        json.dumps(source_lock, ensure_ascii=False, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode()
    source_manifest = json.loads((
        suite["adaptive_root"] / "third_party/ddtree_pinned/SOURCE_SHA256.json"
    ).read_text())
    dataset_manifest = {
        "version":4, "official_commit":cfg["official_commit"],
        "sampling_seed":0, "sample_limits":cfg["sample_limits"],
        "training":False, "full_split":False,
        "source_lock_sha256":hashlib.sha256(serialized_lock).hexdigest(),
        "official_source_manifest":source_manifest,
        "files":{
            f"{dataset}.json":{
                "rows":count, "full_source_rows":count, "sha256":"3" * 64,
            }
            for dataset, count in cfg["sample_limits"].items()
        },
    }
    metadata = {
        "version":6, "config":cfg, "nproc_per_node":1,
        "source_manifest":source_manifest,
        "dataset_manifest":dataset_manifest,
        "code_identity":_adaptive_code_identity(suite["adaptive_root"]),
        "model_indices":[model["adaptive_model_index"]], "datasets":cfg["datasets"],
        "smoke_count":0, "max_new_tokens":2048,
        "method_schema_version":ADAPTIVE_METHOD_SCHEMA_VERSION,
        "primary_adaptive_method":ADAPTIVE_PRIMARY_METHOD,
        "greedy_audit_policy":"record-bf16-mismatches",
        "method_order_policy":"balanced-rotation",
    }
    identity = _adaptive_contract_digest(metadata)
    write_json(tmp_path / "contract.json", {"identity":identity, "metadata":metadata})
    budgets = [f"ddtree_tb{budget}" for budget in cfg["tree_budgets"]]
    input_artifacts = []
    for dataset in cfg["datasets"]:
        turns = cfg["sample_limits"][dataset] * (2 if dataset == "mt-bench" else 1)
        for backend, suffix in (("sdpa", "sdpa"),
                                ("flash_attention_2", "flash_attn")):
            stem = (
                f"{dataset}__model{model['adaptive_model_index']}"
                f"__temp0.0__{suffix}"
            )
            artifact_path = tmp_path / f"{stem}.pt"
            artifact_path.write_text(f"synthetic {dataset} {backend}")
            marker_path = tmp_path / f"{stem}.complete.json"
            marker = {
                "identity":identity, "sha256":file_hash(artifact_path),
                "turns":turns, "cases":cfg["sample_limits"][dataset],
                "methods":["baseline", "dflash"]
                + ([*budgets, *cfg["variants"]] if backend == "sdpa" else []),
                "smoke":False,
                "greedy_audit_policy":"record-bf16-mismatches",
            }
            write_json(marker_path, marker)
            input_artifacts.append({
                "dataset":dataset,
                "model_index":model["adaptive_model_index"],
                "model":model["target"],
                "backend":backend,
                "artifact":artifact_path.name,
                "artifact_sha256":marker["sha256"],
                "completion_marker":marker_path.name,
                "completion_marker_sha256":file_hash(marker_path),
                "protocol_identity":identity,
            })
    methods = [("DFlash", "dflash"), ("DDTree-best", budgets[0])]
    methods += [(name, name) for name in (*budgets, *cfg["variants"])]
    controlled = []
    for dataset in cfg["datasets"]:
        turns = cfg["sample_limits"][dataset] * (2 if dataset == "mt-bench" else 1)
        for method, selected in methods:
            role = ("primary" if selected == ADAPTIVE_PRIMARY_METHOD else
                    "historical_control" if selected == "adaptive_legacy" else
                    "ablation" if selected.startswith("adaptive_") else "baseline")
            controlled.append({
                "dataset":dataset, "model":model["target"],
                "cases":cfg["sample_limits"][dataset], "turns":turns,
                "method":method, "selected_key":selected, "method_role":role,
                "mean_decode_tpot_seconds":1.0, "speedup_vs_target":1.0,
                "speedup_vs_best_ddtree":1.0, "mean_acceptance_length":1.0,
                "responses":turns, "exact_output_responses":turns,
                "output_divergence_responses":0, "exact_output_rate":1.0,
                "target_baseline_backend":"sdpa", "method_backend":"sdpa",
            })
    exact_methods = [("DFlash", "dflash")] + [
        (name, name) for name in (*budgets, *cfg["variants"])
    ]
    exact_rows = []
    for dataset in cfg["datasets"]:
        turns = cfg["sample_limits"][dataset] * (2 if dataset == "mt-bench" else 1)
        for method, selected in exact_methods:
            exact_rows.append({
                "dataset":dataset, "model":model["target"],
                "cases":cfg["sample_limits"][dataset], "turns":turns,
                "method":method, "selected_key":selected, "responses":turns,
                "target_baseline_backend":"sdpa", "method_backend":"sdpa",
                "selection_bias_warning":True,
            })
    usage = [{"dataset":dataset, "model":model["target"], "method":method}
             for dataset in cfg["datasets"] for method in cfg["variants"]]
    auxiliary = [
        {**row, "comparison_scope":"upstream_best_backend_auxiliary",
         "cross_backend_inputs_identical":True}
        for row in controlled
    ]
    expected_responses = 2 * sum(
        count * (2 if dataset == "mt-bench" else 1)
        for dataset, count in cfg["sample_limits"].items()
    )
    summary = {
        "protocol":"ddtree_official_t0", "training":False,
        "official_samples":True,
        "method_schema_version":ADAPTIVE_METHOD_SCHEMA_VERSION,
        "primary_adaptive_method":ADAPTIVE_PRIMARY_METHOD,
        "greedy_audit_policy":"record-bf16-mismatches",
        "method_order_policy":"balanced-rotation",
        "controlled_protocol_gate_passed":True,
        "strict_lossless_claim_eligible":False,
        "protocol_identity":identity,
        "environment_sha256":file_hash(tmp_path / "environment.json"),
        "dataset_manifest":dataset_manifest,
        "source_lock":source_lock,
        "input_artifacts":input_artifacts,
        "controlled_same_backend_rows":controlled,
        "controlled_exact_output_subset_rows":exact_rows,
        "rows":auxiliary,
        "primary_rows":[row for row in controlled if row["method_role"] == "primary"],
        "ablation_rows":[row for row in controlled if row["method_role"] in {
            "ablation", "historical_control"
        }],
        "adaptive_budget_usage":usage,
        "numerical_audit":{
            "responses":expected_responses, "exact_responses":expected_responses,
            "mismatching_responses":0, "cross_backend_input_mismatches":0,
        },
        "cross_backend_auxiliary_table_warning":None,
    }
    assert _validate_adaptive_summary(summary, tmp_path, model, suite, doctor) == (
        controlled, exact_rows, auxiliary
    )
    broken = json.loads(json.dumps(summary))
    broken["controlled_same_backend_rows"].pop()
    with pytest.raises(ValueError, match="coverage"):
        _validate_adaptive_summary(broken, tmp_path, model, suite, doctor)
    (tmp_path / input_artifacts[0]["artifact"]).write_text("changed raw timing")
    with pytest.raises(ValueError, match="raw artifact lineage"):
        _validate_adaptive_summary(summary, tmp_path, model, suite, doctor)


def test_fresh_server_launcher_keeps_selected_python_helpers_on_path():
    launcher = (ROOT / "scripts/run_integrated_fresh_server.sh").read_text()
    assert 'PYTHON_RESOLVED="$(command -v "$PYTHON_BIN")"' in launcher
    assert 'export PATH="$(dirname "$PYTHON_RESOLVED"):$PATH"' in launcher
    assert 'PYTHON_BIN="$PYTHON_RESOLVED"' in launcher
    assert '--sampling-data-dir "$SAMPLING_DATA_DIR"' in launcher
    assert 'audit_formal_experiment_matrix.py' in launcher
    assert 'FORMAL_MATRIX' not in launcher
    assert '--matrix "$REPOSITORY_ROOT/configs/formal_experiment_matrix.json"' in launcher
    assert "fcntl.LOCK_EX | fcntl.LOCK_NB" in launcher
    assert "acquire_lifecycle_lock_and_reexec _start_locked" in launcher
    assert "acquire_lifecycle_lock_and_reexec _resume_locked" in launcher
    assert 'if [[ "$current_pid" == "$$" ]]' in launcher
    assert 'write_atomic_line "$PID_FILE" "$worker_pid"' in launcher
    assert "ddtree_lazy_projection" not in launcher


def _launcher_fixture(tmp_path, *, worker_seconds="0.8"):
    repository = tmp_path / "repository"
    scripts = repository / "scripts"
    configs = repository / "configs"
    scripts.mkdir(parents=True)
    configs.mkdir()
    launcher = scripts / "run_integrated_fresh_server.sh"
    launcher.write_text(
        (ROOT / "scripts/run_integrated_fresh_server.sh").read_text()
    )
    launcher.chmod(0o755)
    (configs / "formal_experiment_matrix.json").write_text("{}\n")
    (configs / "suite.json").write_text("{}\n")
    fake_python = repository / "fake-python"
    fake_python.write_text("""#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == "-c" ]]; then
  exec "$REAL_PYTHON" "$@"
fi
printf '%s\\n' "$*" >> "$FAKE_PYTHON_LOG"
if [[ "$*" == *"gbv_experiments run-integrated-suite"* ]]; then
  sleep "$FAKE_WORKER_SECONDS"
fi
""")
    fake_python.chmod(0o755)
    run_dir = tmp_path / "outputs" / "formal-run"
    command_log = tmp_path / "fake-python.log"
    environment = os.environ.copy()
    environment.pop("INTEGRATED_LIFECYCLE_LOCK_FD", None)
    environment.update({
        "INTEGRATED_PYTHON":str(fake_python),
        "INTEGRATED_SUITE":str(configs / "suite.json"),
        "INTEGRATED_RUN_DIR":str(run_dir),
        "ADAPTIVE_DATA_DIR":str(tmp_path / "adaptive-data"),
        "SAMPLING_DATA_DIR":str(tmp_path / "sampling-data"),
        "REAL_PYTHON":sys.executable,
        "FAKE_PYTHON_LOG":str(command_log),
        "FAKE_WORKER_SECONDS":worker_seconds,
    })
    return launcher, run_dir, command_log, environment


def _wait_until(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("timed out waiting for launcher lifecycle transition")


def test_launcher_lock_serializes_resume_and_pid_cleanup_is_owner_guarded(tmp_path):
    launcher, run_dir, command_log, environment = _launcher_fixture(tmp_path)
    run_dir.mkdir(parents=True)
    direct_worker = subprocess.run(
        [str(launcher), "_worker"], env=environment,
        capture_output=True, text=True, timeout=5,
    )
    assert direct_worker.returncode == 2
    assert "requires an inherited lifecycle lock" in direct_worker.stderr
    first = subprocess.run(
        [str(launcher), "resume"], env=environment,
        capture_output=True, text=True, timeout=5,
    )
    assert first.returncode == 0, first.stderr
    pid_file = run_dir.with_suffix(".pid")
    first_pid = pid_file.read_text().strip()

    overlapping = subprocess.run(
        [str(launcher), "resume"], env=environment,
        capture_output=True, text=True, timeout=5,
    )
    assert overlapping.returncode == 2
    assert "lifecycle is already locked" in overlapping.stderr
    assert pid_file.read_text().strip() == first_pid
    _wait_until(lambda: not pid_file.exists())

    # Model a PID file already transferred to a newer owner.  The exiting old
    # worker must not unlink a value it does not own.
    status_file = run_dir.with_suffix(".status")
    if status_file.exists():
        status_file.unlink()
    resumed = subprocess.run(
        [str(launcher), "resume"], env=environment,
        capture_output=True, text=True, timeout=5,
    )
    assert resumed.returncode == 0, resumed.stderr
    _wait_until(
        lambda: command_log.exists()
        and command_log.read_text().count("gbv_experiments run-integrated-suite") >= 2
    )
    pid_file.write_text("424242\n")
    _wait_until(status_file.exists)
    assert pid_file.read_text() == "424242\n"


def test_launcher_ignores_formal_matrix_environment_override(tmp_path):
    launcher, run_dir, command_log, environment = _launcher_fixture(
        tmp_path, worker_seconds="0.2"
    )
    hostile_matrix = tmp_path / "unbound-matrix.json"
    hostile_matrix.write_text("{}\n")
    environment["FORMAL_MATRIX"] = str(hostile_matrix)
    started = subprocess.run(
        [str(launcher), "start"], env=environment,
        capture_output=True, text=True, timeout=5,
    )
    assert started.returncode == 0, started.stderr
    invocation_log = command_log.read_text()
    canonical = run_dir.parents[1] / "repository" / "configs" / "formal_experiment_matrix.json"
    assert f"--matrix {canonical}" in invocation_log
    assert str(hostile_matrix) not in invocation_log
    _wait_until(lambda: not run_dir.with_suffix(".pid").exists())


def test_long_run_frontloads_t1_data_scoring_and_model_gates_before_t0_timing():
    source = inspect.getsource(run_integrated_suite)
    t0_timing = source.index('adaptive_base + ["evaluate"')
    assert source.index("prepare(first[\"datasets\"]") < t0_timing
    assert source.index("validate_gold(block_data_dir") < t0_timing
    assert source.index("check_model(") < t0_timing
    assert source.index("run(cfg, block_data_dir") > t0_timing
