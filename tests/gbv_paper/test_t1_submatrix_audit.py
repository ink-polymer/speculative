from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from gbv_experiments.common import digest
from gbv_experiments.t1_submatrix_audit import (
    CONFIG_PATH,
    EVIDENCE_FILES,
    EXPECTED_GENERATION_TURNS,
    EXPECTED_METHODS,
    EXPECTED_OBJECTIVE_SCORES,
    EXPECTED_ORDER_PROMPTS,
    EXPECTED_PAIRWISE_ROWS,
    EXPECTED_PREPARED_ROWS,
    EXPECTED_RECORDS,
    EXPECTED_SCORES,
    EXPECTED_SUMMARY_ROWS,
    MODEL_ID,
    PINNED_DATA_MANIFEST_FILE_SHA256,
    PINNED_H20_MEMORY_BYTES,
    SCOPE_LABEL,
    _claim_status,
    _doctor_from_preflight,
    _load_evidence,
    _validate_fixed_cardinality,
    _validate_process_evaluator_self_test,
    _validate_registered_data_snapshot,
    _validate_run_runtime_gate,
    audit_t1_submatrix,
)


DATASET_COUNTS = {
    "gsm8k": 128,
    "math500": 128,
    "aime24": 30,
    "aime25": 30,
    "humaneval": 164,
    "mbpp_sanitized": 128,
    "livecodebench": 128,
    "mt-bench": 80,
}
METHODS = (
    "target_t1p0",
    "dflash_t1p0",
    "ddtree_t1p0",
    "tree_block_verification_t1p0",
)


def _registered_artifacts():
    prepared = []
    for dataset, count in DATASET_COUNTS.items():
        for index in range(count):
            prepared.append({
                "dataset": dataset,
                "source_id": str(index),
                "prompt_sha256": digest([dataset, index]),
            })
    records = []
    scores = []
    for row in prepared:
        for seed in (17, 29, 43):
            for method in METHODS:
                records.append({
                    "dataset": row["dataset"],
                    "turn_count": 2 if row["dataset"] == "mt-bench" else 1,
                })
                scores.append({
                    "dataset": row["dataset"],
                    "metric": (
                        "not_scored"
                        if row["dataset"] == "mt-bench"
                        else "accuracy"
                        if row["dataset"] in {
                            "gsm8k", "math500", "aime24", "aime25",
                        }
                        else "pass@1"
                    ),
                    "passed": None if row["dataset"] == "mt-bench" else True,
                })
    summary_rows = []
    for dataset, count in DATASET_COUNTS.items():
        for method in METHODS:
            summary_rows.append({
                "dataset": dataset,
                "variant": method,
                "samples": count * 3,
                "unique_prompts": count,
                "generation_turns": count * 3 * (2 if dataset == "mt-bench" else 1),
                "generated_tokens": count * 3,
                "matched_baseline_samples": count * 3,
                "target_forward_calls": 0,
                "draft_forward_calls": 0,
                "scored_samples": 0 if dataset == "mt-bench" else count * 3,
            })
    pairwise = []
    for dataset, count in DATASET_COUNTS.items():
        for baseline in ("ddtree", "dflash"):
            pairwise.append({
                "dataset": dataset,
                "candidate": "tree_block_verification",
                "baseline": baseline,
                "speedup": 1.01,
                "speedup_ci_low": 1.001,
                "speedup_ci_high": 1.03,
                "paired_samples": count * 3,
                "source_clusters": count,
                "seeds_per_source": 3,
                "bootstrap_samples": 10_000,
            })
    manifest = {
        "expected_records": EXPECTED_RECORDS,
        "expected_generations": EXPECTED_GENERATION_TURNS,
        "dataset_names": list(DATASET_COUNTS),
        "evaluation": {
            "protocol": "ddtree_counts",
            "sample_seed": 0,
            "counts": DATASET_COUNTS,
        },
        "coverage": "fixed_evaluation_subset",
        "seeds": [17, 29, 43],
    }
    order = {
        "passed": True,
        "prompts": EXPECTED_ORDER_PROMPTS,
        "methods": EXPECTED_METHODS,
    }
    return manifest, records, scores, summary_rows, order, pairwise, prepared


def _process_preflight() -> dict:
    return {
        "environment": {
            "versions": {
                "torch": "2.8.0+cu128",
                "transformers": "4.57.1",
                "huggingface-hub": "0.36.0",
                "datasets": "3.6.0",
                "numpy": "2.2.6",
                "math-verify": "0.8.0",
            },
            "python": "3.11.16",
            "cuda": "12.8",
            "gpu": "NVIDIA H20",
            "gpu_uuid": "GPU-fixed",
            "total_memory_bytes": PINNED_H20_MEMORY_BYTES,
            "code_backend": "process",
            "code_image_id": None,
            "process_python": "/opt/gbv-code-eval/bin/python",
            "process_python_resolved": "/opt/gbv-code-eval/bin/python3.11",
            "process_python_sha256": (
                "a609b6535c67b341f3a040c1ae8141eb8f18c3cf798a134ba5da7d186d1b5372"
            ),
            "process_python_runtime": {
                "executable": "/opt/gbv-code-eval/bin/python",
                "prefix": "/opt/gbv-code-eval",
                "base_prefix": "/usr/local",
                "implementation": "CPython",
                "python": "3.11.16",
                "numpy": "2.2.6",
                "sympy": "1.14.0",
            },
            "process_pyvenv_cfg_sha256": None,
            "lcb_evaluator_checks": [],
        }
    }


def _process_self_test(preflight: dict) -> dict:
    environment = preflight["environment"]
    identity = {
        "process_python": environment["process_python"],
        "process_python_resolved": environment["process_python_resolved"],
        "process_python_sha256": environment["process_python_sha256"],
        "process_python_runtime": environment["process_python_runtime"],
        "process_pyvenv_cfg_sha256": environment["process_pyvenv_cfg_sha256"],
    }
    return {
        "schema": 1,
        "passed": True,
        "backend": "process",
        "expected_process_uid": 65_534,
        "result": {"passed": True, "reason": "passed"},
        "process_identity_before": identity,
        "process_identity_after": identity,
        "identities_match": True,
    }


def test_registered_cardinality_is_exact_and_keeps_mtbench_separate():
    artifacts = _registered_artifacts()
    counts = _validate_fixed_cardinality(*artifacts)
    assert counts == {
        "records": EXPECTED_RECORDS,
        "generation_turns": EXPECTED_GENERATION_TURNS,
        "scores": EXPECTED_SCORES,
        "objective_scores": EXPECTED_OBJECTIVE_SCORES,
        "mt_bench_not_scored": 960,
        "summary_rows": EXPECTED_SUMMARY_ROWS,
        "order_prompts": EXPECTED_ORDER_PROMPTS,
        "methods": EXPECTED_METHODS,
        "pairwise_rows": EXPECTED_PAIRWISE_ROWS,
        "prepared_rows": EXPECTED_PREPARED_ROWS,
    }


@pytest.mark.parametrize(
    ("component", "mutate"),
    [
        ("manifest", lambda value: value.__setitem__("expected_records", 9792.0)),
        ("records", lambda value: value.pop()),
        ("records", lambda value: value[0].__setitem__("turn_count", True)),
        ("scores", lambda value: value.pop()),
        ("scores", lambda value: value[0].__setitem__("passed", None)),
        ("summary", lambda value: value.pop()),
        ("order", lambda value: value.__setitem__("prompts", 2447)),
        ("pairwise", lambda value: value[0].__setitem__("speedup_ci_low", None)),
        ("pairwise", lambda value: value[0].__setitem__("paired_samples", 384.0)),
        ("prepared", lambda value: value.pop()),
    ],
)
def test_registered_cardinality_fails_closed(component, mutate):
    names = ("manifest", "records", "scores", "summary", "order", "pairwise", "prepared")
    artifacts = dict(zip(names, copy.deepcopy(_registered_artifacts())))
    mutate(artifacts[component])
    with pytest.raises(ValueError):
        _validate_fixed_cardinality(*(artifacts[name] for name in names))


def test_preflight_environment_requires_h20_and_full_process_identity():
    preflight = _process_preflight()
    doctor = _doctor_from_preflight(preflight)
    assert doctor["gpu"] == "NVIDIA H20"
    assert doctor["code_backend"] == "process"
    assert doctor["code_evaluator_pyvenv_cfg_sha256"] is None

    wrong_gpu = copy.deepcopy(preflight)
    wrong_gpu["environment"]["gpu"] = "NVIDIA A100"
    with pytest.raises(ValueError, match="H20"):
        _doctor_from_preflight(wrong_gpu)

    float_memory = copy.deepcopy(preflight)
    float_memory["environment"]["total_memory_bytes"] = float(
        float_memory["environment"]["total_memory_bytes"]
    )
    with pytest.raises(ValueError, match="H20"):
        _doctor_from_preflight(float_memory)

    wrong_memory = copy.deepcopy(preflight)
    wrong_memory["environment"]["total_memory_bytes"] -= 1
    with pytest.raises(ValueError, match="H20"):
        _doctor_from_preflight(wrong_memory)

    missing_identity = copy.deepcopy(preflight)
    missing_identity["environment"]["process_python_sha256"] = None
    with pytest.raises(ValueError, match="process evaluator"):
        _doctor_from_preflight(missing_identity)

    fake_pyvenv = copy.deepcopy(preflight)
    fake_pyvenv["environment"]["process_pyvenv_cfg_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="process evaluator"):
        _doctor_from_preflight(fake_pyvenv)


def test_claim_status_never_promotes_one_submatrix():
    assert _claim_status() == {
        "submatrix_complete": True,
        "formal_complete": False,
        "whole_formal_matrix_complete": False,
        "publication_claim_eligible": False,
        "tree_block_superiority_claim_eligible": False,
    }


def test_process_evaluator_self_test_binds_uid_and_preflight_identity():
    preflight = _process_preflight()
    doctor = _doctor_from_preflight(preflight)
    evidence = _process_self_test(preflight)
    _validate_process_evaluator_self_test(evidence, doctor)

    wrong_uid = copy.deepcopy(evidence)
    wrong_uid["expected_process_uid"] = 0
    with pytest.raises(ValueError, match="uid/identity"):
        _validate_process_evaluator_self_test(wrong_uid, doctor)

    changed_after = copy.deepcopy(evidence)
    changed_after["process_identity_after"]["process_python_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="uid/identity"):
        _validate_process_evaluator_self_test(changed_after, doctor)


def test_registered_data_snapshot_requires_exact_file_sha(tmp_path, monkeypatch):
    import gbv_experiments.t1_submatrix_audit as module

    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    monkeypatch.setattr(
        module,
        "file_hash",
        lambda path: PINNED_DATA_MANIFEST_FILE_SHA256,
    )
    assert _validate_registered_data_snapshot(tmp_path) == manifest
    monkeypatch.setattr(module, "file_hash", lambda path: "0" * 64)
    with pytest.raises(ValueError, match="registered H20 snapshot"):
        _validate_registered_data_snapshot(tmp_path)


def test_run_runtime_gate_reuses_ready_evidence_validator(monkeypatch):
    import gbv_experiments.t1_submatrix_audit as module

    cfg = json.loads(CONFIG_PATH.read_text())
    identity = {"schema": 2, "identity": "fixed"}
    expected_holder = {}

    def validate(parameters, *, run_id, expected, require_ready):
        expected_holder.update(expected)
        assert parameters["schema"] == 1
        assert run_id == "run-fixed"
        assert require_ready is True
        return True

    monkeypatch.setattr(module, "validate_model_parameters_evidence", validate)
    parameters = {
        "schema": 1,
        "draft_block_size": 16,
        "runtime_model_gate": {
            "identity_after_load": identity,
            "identity_before_timing": identity,
        },
    }
    preflight = {
        "runtime_model_gate": {
            "expectations": None,
            "identity_after_load": identity,
            "identity_after_preflight": identity,
        }
    }
    # The expected object is constructed inside the function; capture it on the
    # first fail-closed call, then bind the independently persisted preflight.
    with pytest.raises(ValueError, match="differs"):
        _validate_run_runtime_gate(parameters, {"run_id": "run-fixed"}, preflight, cfg)
    preflight["runtime_model_gate"]["expectations"] = dict(expected_holder)
    _validate_run_runtime_gate(parameters, {"run_id": "run-fixed"}, preflight, cfg)
    assert expected_holder["strict_same_tree_t1"] is True
    assert expected_holder["parameter_device"] == "cuda:0"


def test_missing_audit_evidence_fails_before_output(tmp_path):
    with pytest.raises(ValueError, match="Missing required audit evidence"):
        _load_evidence(tmp_path)


def test_post_run_audit_emits_only_submatrix_completion(tmp_path, monkeypatch):
    import gbv_experiments.t1_submatrix_audit as module

    manifest, records, scores, summary_rows, order, pairwise, prepared = (
        _registered_artifacts()
    )
    cfg = {
        "model": {"target": "Qwen/Qwen3-4B"},
        "datasets": list(DATASET_COUNTS),
        "seeds": [17, 29, 43],
        "evaluation": manifest["evaluation"],
        "bootstrap_samples": 10_000,
    }
    plan = {
        "expected_records": EXPECTED_RECORDS,
        "expected_generations": EXPECTED_GENERATION_TURNS,
    }
    data_manifest = {"schema": 2, "datasets": {}}
    prompt_ids = [
        [row["dataset"], row["source_id"], row["prompt_sha256"]]
        for row in prepared
    ]
    manifest.update({"data_manifest": data_manifest, "prompt_ids": prompt_ids})
    manifest["run_id"] = digest(manifest)
    coverage = {
        "expected": EXPECTED_RECORDS,
        "actual": EXPECTED_RECORDS,
        "missing": 0,
        "coverage": manifest["coverage"],
        "evaluation": manifest["evaluation"],
        "complete": True,
    }

    run_dir = tmp_path / "run"
    report_dir = run_dir / "report"
    evidence_dir = tmp_path / "evidence"
    data_dir = tmp_path / "data"
    report_dir.mkdir(parents=True)
    evidence_dir.mkdir()
    data_dir.mkdir()
    (data_dir / "manifest.json").write_text("{}")
    preflight = _process_preflight()
    preflight_path = tmp_path / "preflight.json"
    preflight_path.write_text(json.dumps(preflight))
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest))
    (run_dir / "results.jsonl").write_text("")
    (run_dir / "scores.jsonl").write_text("")
    (run_dir / "scoring_manifest.json").write_text("{}")
    (report_dir / "summary.json").write_text(json.dumps({
        "coverage": coverage,
        "rows": summary_rows,
    }))
    (run_dir / "completed.json").write_text(json.dumps({
        "run_id": manifest["run_id"],
        "records": EXPECTED_RECORDS,
        "generations": EXPECTED_GENERATION_TURNS,
    }))
    (run_dir / "model_parameters.json").write_text("{}")

    matrix_audit = {
        "report": {
            "registered_scope": {
                "stochastic_t1_generation_calls": 21_504,
                "total_generation_calls": 65_280,
            }
        }
    }
    fairness_audit = {"passed": True, "model_ids": [MODEL_ID]}
    evidence_values = {
        "plan": plan,
        "formal_matrix_audit": matrix_audit,
        "fairness_audit": fairness_audit,
        "distribution_law_audit": {},
        "data_audit": {},
        "gold_audit": {},
        "process_evaluator_self_test": _process_self_test(preflight),
        "gpu_allocation_audit": {
            "passed": True,
            "gpu_uuid": "GPU-fixed",
            "allowed_pid": 123,
            "observed_compute_pids": [123],
            "foreign_compute_pids": [],
        },
    }
    for name, filename in EVIDENCE_FILES.items():
        (evidence_dir / filename).write_text(json.dumps(evidence_values[name]))

    monkeypatch.setattr(module, "load_integrated_suite", lambda *args, **kwargs: {
        "models": [{
            "id": MODEL_ID,
            "config": cfg,
            "config_path": CONFIG_PATH.resolve(),
        }]
    })
    monkeypatch.setattr(module, "make_plan", lambda value: plan)
    monkeypatch.setattr(module, "_formal_matrix_audit", lambda: matrix_audit)
    monkeypatch.setattr(
        module, "audit_integrated_suite", lambda *args, **kwargs: fairness_audit
    )
    monkeypatch.setattr(module, "_validate_distribution_law_evidence", lambda value: None)
    monkeypatch.setattr(module, "_validate_t1_preflight", lambda *args: None)
    monkeypatch.setattr(module, "_validate_sampling_manifest", lambda *args: None)
    monkeypatch.setattr(module, "_validate_sampling_data_audits", lambda *args: None)
    monkeypatch.setattr(module, "_validate_scoring_manifest", lambda *args: None)
    monkeypatch.setattr(module, "_validate_sampling_summary", lambda *args: summary_rows)
    monkeypatch.setattr(module, "_validate_run_runtime_gate", lambda *args: None)
    monkeypatch.setattr(module, "load_prepared", lambda *args: (data_manifest, prepared))
    monkeypatch.setattr(
        module, "_validate_registered_data_snapshot",
        lambda path: path / "manifest.json",
    )
    monkeypatch.setattr(module, "validate_results", lambda *args: coverage)
    monkeypatch.setattr(module, "summarize_results", lambda *args: summary_rows)
    monkeypatch.setattr(module, "validate_block_order", lambda path: order)
    monkeypatch.setattr(module, "_tree_block_pairwise_rows", lambda *args: pairwise)
    monkeypatch.setattr(
        module,
        "read_jsonl",
        lambda path: scores if Path(path).name == "scores.jsonl" else records,
    )

    output = tmp_path / "submatrix_audit.json"
    result = audit_t1_submatrix(
        run_dir, data_dir, preflight_path, evidence_dir, output
    )
    assert json.loads(output.read_text()) == result
    assert result["scope_label"] == SCOPE_LABEL
    assert result["submatrix_complete"] is True
    assert result["formal_complete"] is False
    assert result["whole_formal_matrix_complete"] is False
    assert result["publication_claim_eligible"] is False
    assert result["tree_block_superiority_claim_eligible"] is False
    assert result["counts"]["records"] == EXPECTED_RECORDS
    assert len(result["lineage"]["auditor_sha256"]) == 64
    assert len(result["lineage"]["cli_sha256"]) == 64
    assert len(result["lineage"]["process_evaluator_probe_cli_sha256"]) == 64

    float_coverage = {**coverage, "expected": float(EXPECTED_RECORDS)}
    monkeypatch.setattr(module, "validate_results", lambda *args: float_coverage)
    with pytest.raises(ValueError, match="coverage"):
        audit_t1_submatrix(
            run_dir,
            data_dir,
            preflight_path,
            evidence_dir,
            tmp_path / "float_coverage_audit.json",
        )
    monkeypatch.setattr(module, "validate_results", lambda *args: coverage)

    summary_rows[0]["samples"] = float(summary_rows[0]["samples"])
    (report_dir / "summary.json").write_text(json.dumps({
        "coverage": coverage,
        "rows": summary_rows,
    }))
    with pytest.raises(ValueError, match="non-integral"):
        audit_t1_submatrix(
            run_dir,
            data_dir,
            preflight_path,
            evidence_dir,
            tmp_path / "float_summary_audit.json",
        )
    summary_rows[0]["samples"] = int(summary_rows[0]["samples"])
    (report_dir / "summary.json").write_text(json.dumps({
        "coverage": coverage,
        "rows": summary_rows,
    }))

    original_manifest = (run_dir / "run_manifest.json").read_text()
    with pytest.raises(ValueError, match="must not overwrite"):
        audit_t1_submatrix(
            run_dir,
            data_dir,
            preflight_path,
            evidence_dir,
            run_dir / "run_manifest.json",
        )
    assert (run_dir / "run_manifest.json").read_text() == original_manifest
