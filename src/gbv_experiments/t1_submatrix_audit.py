"""Fail-closed post-run audit for the registered Qwen3-4B T=1 H20 submatrix."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

from .common import ROOT, digest, file_hash, read_jsonl, write_json
from .data import DATASETS, load_prepared
from .integrated_suite import (
    _formal_matrix_audit,
    _tree_block_pairwise_rows,
    _validate_distribution_law_evidence,
    _validate_sampling_data_audits,
    _validate_sampling_manifest,
    _validate_sampling_summary,
    _validate_scoring_manifest,
    _validate_t1_preflight,
    audit_integrated_suite,
    load_integrated_suite,
    validate_block_order,
)
from .report import summarize as summarize_results
from .report import validate_results
from .runner import make_plan
from .runtime_model import (
    runtime_model_expectations,
    validate_model_parameters_evidence,
)


SCOPE_LABEL = "Qwen3-4B / T=1 / H20 登记子矩阵"
MODEL_ID = "qwen3_4b"
EXPECTED_RECORDS = 9_792
EXPECTED_GENERATION_TURNS = 10_752
EXPECTED_SCORES = 9_792
EXPECTED_OBJECTIVE_SCORES = 8_832
EXPECTED_MT_BENCH_NOT_SCORED = 960
EXPECTED_SUMMARY_ROWS = 32
EXPECTED_ORDER_PROMPTS = 2_448
EXPECTED_METHODS = 4
EXPECTED_PAIRWISE_ROWS = 16
EXPECTED_PREPARED_ROWS = 816
EXPECTED_PREPARED_TURNS = 896
SUITE_PATH = ROOT / "configs/adaptive_tree_block_suite.json"
CONFIG_PATH = ROOT / "configs/adaptive_block_qwen3_4b.json"
EVIDENCE_FILES = {
    "plan": "plan.json",
    "formal_matrix_audit": "formal_matrix_audit.json",
    "fairness_audit": "fairness_audit_qwen3_4b.json",
    "distribution_law_audit": "distribution_law_audit.json",
    "data_audit": "sampling_data_audit.json",
    "gold_audit": "sampling_gold_audit.json",
    "process_evaluator_self_test": "process_evaluator_self_test.json",
    "gpu_allocation_audit": "gpu_allocation_before_timing.json",
}
PINNED_PREFLIGHT_VERSIONS = {
    "torch": "2.8.0+cu128",
    "transformers": "4.57.1",
    "huggingface-hub": "0.36.0",
    "datasets": "3.6.0",
    "numpy": "2.2.6",
    "math-verify": "0.8.0",
}
PINNED_PROCESS_PYTHON = "/opt/gbv-code-eval/bin/python"
PINNED_PROCESS_PYTHON_RESOLVED = "/opt/gbv-code-eval/bin/python3.11"
PINNED_PROCESS_PYTHON_SHA256 = (
    "a609b6535c67b341f3a040c1ae8141eb8f18c3cf798a134ba5da7d186d1b5372"
)
PINNED_H20_MEMORY_BYTES = 102_085_623_808
PINNED_DATA_MANIFEST_FILE_SHA256 = (
    "3c46d4f25a3d4009ed9784c0b09aa2a870546ac2fed912204be6d8d3ce267faf"
)


def _claim_status() -> dict[str, bool]:
    """Keep a completed 4B submatrix from being promoted to a full-matrix claim."""
    return {
        "submatrix_complete": True,
        "formal_complete": False,
        "whole_formal_matrix_complete": False,
        "publication_claim_eligible": False,
        "tree_block_superiority_claim_eligible": False,
    }


def _load_json(path: Path) -> dict:
    import json

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _plain_int(value: Any, *, minimum: int | None = None) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and (minimum is None or value >= minimum)
    )


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _load_evidence(directory: Path) -> tuple[dict[str, dict], dict[str, Path]]:
    if not directory.is_dir():
        raise ValueError(f"Audit evidence directory does not exist: {directory}")
    paths = {name: directory / filename for name, filename in EVIDENCE_FILES.items()}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise ValueError(f"Missing required audit evidence: {missing}")
    return {name: _load_json(path) for name, path in paths.items()}, paths


def _doctor_from_preflight(preflight: Mapping[str, Any]) -> dict:
    """Project the independently executed preflight environment into validators."""
    environment = preflight.get("environment")
    expected_keys = {
        "versions", "python", "cuda", "gpu", "gpu_uuid", "total_memory_bytes",
        "code_backend", "code_image_id", "process_python",
        "process_python_resolved", "process_python_sha256",
        "process_python_runtime", "process_pyvenv_cfg_sha256",
        "lcb_evaluator_checks",
    }
    if not isinstance(environment, Mapping) or set(environment) != expected_keys:
        raise ValueError("Preflight environment has an invalid schema")
    versions = environment.get("versions")
    expected_versions = {
        "torch", "transformers", "huggingface-hub", "datasets", "numpy",
        "math-verify",
    }
    if (
        not isinstance(versions, Mapping)
        or set(versions) != expected_versions
        or any(not isinstance(value, str) or not value for value in versions.values())
    ):
        raise ValueError("Preflight package identity is incomplete")
    if (
        not isinstance(environment.get("python"), str)
        or not environment["python"]
        or not isinstance(environment.get("cuda"), str)
        or not environment["cuda"]
        or not isinstance(environment.get("gpu"), str)
        or "h20" not in environment["gpu"].casefold()
        or not isinstance(environment.get("gpu_uuid"), str)
        or environment["gpu_uuid"].casefold() in {"", "none", "unknown", "unavailable"}
        or not _plain_int(environment.get("total_memory_bytes"), minimum=1)
        or environment.get("total_memory_bytes") != PINNED_H20_MEMORY_BYTES
        or environment.get("code_backend") != "process"
        or not isinstance(environment.get("lcb_evaluator_checks"), list)
        or environment.get("python") != "3.11.16"
        or environment.get("cuda") != "12.8"
    ):
        raise ValueError("Preflight does not identify the registered H20 environment")

    required_process_fields = (
        "process_python", "process_python_resolved", "process_python_sha256",
        "process_python_runtime",
    )
    runtime = environment.get("process_python_runtime")
    runtime_keys = {
        "executable", "prefix", "base_prefix", "implementation", "python",
        "numpy", "sympy",
    }
    if (
        versions != PINNED_PREFLIGHT_VERSIONS
        or any(environment.get(field) is None for field in required_process_fields)
        or environment.get("process_python") != PINNED_PROCESS_PYTHON
        or environment.get("process_python_resolved")
        != PINNED_PROCESS_PYTHON_RESOLVED
        or environment.get("process_python_sha256")
        != PINNED_PROCESS_PYTHON_SHA256
        or environment.get("process_pyvenv_cfg_sha256") is not None
        or not isinstance(runtime, Mapping)
        or set(runtime) != runtime_keys
        or any(not isinstance(value, str) or not value for value in runtime.values())
        or runtime.get("executable") != PINNED_PROCESS_PYTHON
        or runtime.get("prefix") != "/opt/gbv-code-eval"
        or runtime.get("implementation") != "CPython"
        or runtime.get("python") != "3.11.16"
        or runtime.get("numpy") != "2.2.6"
        or runtime.get("sympy") != "1.14.0"
        or environment.get("code_image_id") is not None
    ):
        raise ValueError("Registered process evaluator identity is incomplete")

    return {
        "python": environment["python"],
        "cuda": environment["cuda"],
        "gpu": environment["gpu"],
        "gpu_uuid": environment["gpu_uuid"],
        "gpu_memory_bytes": environment["total_memory_bytes"],
        "torch": versions["torch"],
        "packages": {
            name: versions[name]
            for name in (
                "transformers", "huggingface-hub", "datasets", "numpy",
                "math-verify",
            )
        },
        "code_backend": environment["code_backend"],
        "docker_image": environment["code_image_id"],
        "code_evaluator_python": environment["process_python"],
        "code_evaluator_python_resolved": environment["process_python_resolved"],
        "code_evaluator_python_sha256": environment["process_python_sha256"],
        "code_evaluator_python_runtime": environment["process_python_runtime"],
        "code_evaluator_pyvenv_cfg_sha256": environment[
            "process_pyvenv_cfg_sha256"
        ],
    }


def _validate_registered_data_snapshot(data_dir: Path) -> Path:
    manifest_path = data_dir / "manifest.json"
    if (
        not manifest_path.is_file()
        or file_hash(manifest_path) != PINNED_DATA_MANIFEST_FILE_SHA256
    ):
        raise ValueError("Prepared data manifest is not the registered H20 snapshot")
    return manifest_path


def capture_process_evaluator_self_test(output: Path) -> dict:
    """Exercise production process scoring and prove root workers drop to nobody."""
    from .scoring import evaluate_code, process_python_identity

    identity_before = process_python_identity()
    candidate = "import os\ndef evaluator_uid():\n    return os.geteuid()"
    result = evaluate_code(
        candidate,
        {
            "kind": "humaneval",
            "entry_point": "evaluator_uid",
            "prompt": "",
            "test": "def check(candidate):\n    assert candidate() == 65534",
        },
        backend="process",
        timeout=3,
    )
    identity_after = process_python_identity()
    identities_match = identity_before == identity_after
    passed = result == {"passed": True, "reason": "passed"} and identities_match
    evidence = {
        "schema": 1,
        "passed": passed,
        "backend": "process",
        "expected_process_uid": 65_534,
        "result": result,
        "process_identity_before": identity_before,
        "process_identity_after": identity_after,
        "identities_match": identities_match,
    }
    write_json(output, evidence)
    if not passed:
        raise RuntimeError(
            "Production process evaluator did not execute as uid 65534 with a stable identity"
        )
    return evidence


def _validate_process_evaluator_self_test(
    evidence: Mapping[str, Any], doctor: Mapping[str, Any]
) -> None:
    expected_keys = {
        "schema", "passed", "backend", "expected_process_uid", "result",
        "process_identity_before", "process_identity_after", "identities_match",
    }
    expected_identity = {
        "process_python": doctor["code_evaluator_python"],
        "process_python_resolved": doctor["code_evaluator_python_resolved"],
        "process_python_sha256": doctor["code_evaluator_python_sha256"],
        "process_python_runtime": doctor["code_evaluator_python_runtime"],
        "process_pyvenv_cfg_sha256": doctor[
            "code_evaluator_pyvenv_cfg_sha256"
        ],
    }
    if (
        not isinstance(evidence, Mapping)
        or set(evidence) != expected_keys
        or not _plain_int(evidence.get("schema"))
        or evidence["schema"] != 1
        or evidence.get("passed") is not True
        or evidence.get("backend") != "process"
        or not _plain_int(evidence.get("expected_process_uid"))
        or evidence["expected_process_uid"] != 65_534
        or not isinstance(evidence.get("result"), Mapping)
        or set(evidence["result"]) != {"passed", "reason"}
        or evidence["result"].get("passed") is not True
        or evidence["result"].get("reason") != "passed"
        or evidence.get("process_identity_before") != expected_identity
        or evidence.get("process_identity_after") != expected_identity
        or evidence.get("identities_match") is not True
    ):
        raise ValueError("Production process evaluator uid/identity evidence is invalid")


def _validate_run_runtime_gate(
    parameters: Mapping[str, Any],
    manifest: Mapping[str, Any],
    preflight: Mapping[str, Any],
    cfg: Mapping[str, Any],
) -> None:
    expected = runtime_model_expectations(
        cfg["model"], strict_same_tree_t1=True, device="cuda:0"
    )
    try:
        ready = validate_model_parameters_evidence(
            parameters,
            run_id=manifest["run_id"],
            expected=expected,
            require_ready=True,
        )
    except (KeyError, RuntimeError, TypeError, ValueError) as exc:
        raise ValueError("Formal run runtime model evidence is invalid") from exc
    if ready is not True:
        raise ValueError("Formal run is not ready for registered DFlash-b16 timing")

    gate = parameters["runtime_model_gate"]
    preflight_gate = preflight.get("runtime_model_gate")
    if (
        not isinstance(preflight_gate, Mapping)
        or digest(preflight_gate.get("expectations")) != digest(expected)
        or gate["identity_after_load"] != preflight_gate.get("identity_after_load")
        or gate["identity_before_timing"]
        != preflight_gate.get("identity_after_preflight")
    ):
        raise ValueError("Formal timing runtime differs from the registered preflight")


def _validate_gpu_allocation(evidence: Mapping[str, Any], doctor: Mapping[str, Any]) -> None:
    expected_keys = {
        "passed", "gpu_uuid", "allowed_pid", "observed_compute_pids",
        "foreign_compute_pids",
    }
    observed = evidence.get("observed_compute_pids")
    if (
        not isinstance(evidence, Mapping)
        or set(evidence) != expected_keys
        or evidence.get("passed") is not True
        or evidence.get("gpu_uuid") != doctor["gpu_uuid"]
        or not _plain_int(evidence.get("allowed_pid"), minimum=1)
        or not isinstance(observed, list)
        or any(not _plain_int(pid, minimum=1) for pid in observed)
        or len(observed) != len(set(observed))
        or evidence.get("foreign_compute_pids") != []
    ):
        raise ValueError("Uncontended-H20 allocation evidence is invalid")


def _validate_fixed_cardinality(
    manifest: Mapping[str, Any],
    records: list[dict],
    scores: list[dict],
    summary_rows: list[dict],
    order_audit: Mapping[str, Any],
    pairwise_rows: list[dict],
    prepared_rows: list[dict],
) -> dict:
    if (
        not _plain_int(manifest.get("expected_records"))
        or manifest["expected_records"] != EXPECTED_RECORDS
        or not _plain_int(manifest.get("expected_generations"))
        or manifest["expected_generations"] != EXPECTED_GENERATION_TURNS
        or len(records) != EXPECTED_RECORDS
        or any(not _plain_int(row.get("turn_count"), minimum=1) for row in records)
        or sum(row["turn_count"] for row in records) != EXPECTED_GENERATION_TURNS
        or len(scores) != EXPECTED_SCORES
        or len(summary_rows) != EXPECTED_SUMMARY_ROWS
        or not isinstance(order_audit, Mapping)
        or set(order_audit) != {"passed", "prompts", "methods"}
        or order_audit.get("passed") is not True
        or not _plain_int(order_audit.get("prompts"))
        or order_audit["prompts"] != EXPECTED_ORDER_PROMPTS
        or not _plain_int(order_audit.get("methods"))
        or order_audit["methods"] != EXPECTED_METHODS
        or len(pairwise_rows) != EXPECTED_PAIRWISE_ROWS
        or len(prepared_rows) != EXPECTED_PREPARED_ROWS
        or sum(DATASETS[row["dataset"]].turns for row in prepared_rows)
        != EXPECTED_PREPARED_TURNS
    ):
        raise ValueError("Qwen3-4B T=1 registered submatrix cardinality is incomplete")

    objective_scores = [score for score in scores if score.get("dataset") != "mt-bench"]
    mt_bench_scores = [score for score in scores if score.get("dataset") == "mt-bench"]
    if (
        len(objective_scores) != EXPECTED_OBJECTIVE_SCORES
        or len(mt_bench_scores) != EXPECTED_MT_BENCH_NOT_SCORED
        or any(
            not isinstance(score.get("passed"), bool)
            or score.get("metric")
            != (
                "accuracy"
                if score.get("dataset") in {"gsm8k", "math500", "aime24", "aime25"}
                else "pass@1"
            )
            for score in objective_scores
        )
        or any(
            score.get("metric") != "not_scored" or score.get("passed") is not None
            for score in mt_bench_scores
        )
    ):
        raise ValueError("Objective/MT-Bench scoring cardinality is incomplete")

    expected_pairs = {
        (dataset, baseline)
        for dataset in manifest["dataset_names"]
        for baseline in ("ddtree", "dflash")
    }
    actual_pairs = [(row.get("dataset"), row.get("baseline")) for row in pairwise_rows]
    counts = manifest["evaluation"]["counts"]
    if len(actual_pairs) != len(set(actual_pairs)) or set(actual_pairs) != expected_pairs:
        raise ValueError("Tree-block pairwise table is incomplete")
    for row in pairwise_rows:
        expected_samples = counts[row["dataset"]] * len(manifest["seeds"])
        numeric = (
            row.get("speedup"), row.get("speedup_ci_low"),
            row.get("speedup_ci_high"),
        )
        if (
            row.get("candidate") != "tree_block_verification"
            or not _plain_int(row.get("paired_samples"))
            or row["paired_samples"] != expected_samples
            or not _plain_int(row.get("source_clusters"))
            or row["source_clusters"] != counts[row["dataset"]]
            or not _plain_int(row.get("seeds_per_source"))
            or row["seeds_per_source"] != 3
            or not _plain_int(row.get("bootstrap_samples"))
            or row["bootstrap_samples"] != 10_000
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
                for value in numeric
            )
            or row["speedup_ci_low"] > row["speedup_ci_high"]
        ):
            raise ValueError("Tree-block pairwise row is malformed or incomplete")
    return {
        "records": len(records),
        "generation_turns": sum(row["turn_count"] for row in records),
        "scores": len(scores),
        "objective_scores": len(objective_scores),
        "mt_bench_not_scored": len(mt_bench_scores),
        "summary_rows": len(summary_rows),
        "order_prompts": order_audit["prompts"],
        "methods": order_audit["methods"],
        "pairwise_rows": len(pairwise_rows),
        "prepared_rows": len(prepared_rows),
    }


def audit_t1_submatrix(
    run_dir: Path,
    data_dir: Path,
    preflight_path: Path,
    audit_evidence_dir: Path,
    output: Path,
) -> dict:
    """Validate raw artifacts and emit a deliberately submatrix-only claim record."""
    run_dir = run_dir.resolve()
    data_dir = data_dir.resolve()
    preflight_path = preflight_path.resolve()
    audit_evidence_dir = audit_evidence_dir.resolve()
    output = output.resolve()
    if not run_dir.is_dir() or not data_dir.is_dir() or not preflight_path.is_file():
        raise ValueError("Run, prepared data, or preflight input is missing")

    suite = load_integrated_suite(SUITE_PATH, model_ids=[MODEL_ID])
    if len(suite["models"]) != 1 or suite["models"][0]["id"] != MODEL_ID:
        raise ValueError("Integrated suite did not resolve the registered Qwen3-4B model")
    model = suite["models"][0]
    cfg = model["config"]
    if model["config_path"] != CONFIG_PATH.resolve():
        raise ValueError("Qwen3-4B suite points at an unexpected T=1 configuration")
    expected_plan = make_plan(cfg)
    if (
        expected_plan.get("expected_records") != EXPECTED_RECORDS
        or expected_plan.get("expected_generations") != EXPECTED_GENERATION_TURNS
    ):
        raise ValueError("Registered Qwen3-4B T=1 plan cardinality drifted")

    evidence, evidence_paths = _load_evidence(audit_evidence_dir)
    if evidence["plan"] != expected_plan:
        raise ValueError("Saved T=1 plan differs from the current registered plan")
    current_matrix_audit = _formal_matrix_audit()
    if evidence["formal_matrix_audit"] != current_matrix_audit:
        raise ValueError("Formal matrix evidence is missing, stale, or changed")
    scope = current_matrix_audit.get("report", {}).get("registered_scope", {})
    if (
        scope.get("stochastic_t1_generation_calls") != 21_504
        or scope.get("total_generation_calls") != 65_280
    ):
        raise ValueError("Whole registered formal matrix cardinality drifted")
    current_fairness_audit = audit_integrated_suite(
        SUITE_PATH, model_ids=[MODEL_ID]
    )
    if evidence["fairness_audit"] != current_fairness_audit:
        raise ValueError("Qwen3-4B fairness/source audit is missing, stale, or changed")
    _validate_distribution_law_evidence(evidence["distribution_law_audit"])

    preflight = _load_json(preflight_path)
    doctor = _doctor_from_preflight(preflight)
    _validate_t1_preflight(preflight, cfg, doctor)
    _validate_process_evaluator_self_test(
        evidence["process_evaluator_self_test"], doctor
    )
    _validate_gpu_allocation(evidence["gpu_allocation_audit"], doctor)

    manifest_path = run_dir / "run_manifest.json"
    records_path = run_dir / "results.jsonl"
    scores_path = run_dir / "scores.jsonl"
    scoring_manifest_path = run_dir / "scoring_manifest.json"
    summary_path = run_dir / "report/summary.json"
    completed_path = run_dir / "completed.json"
    parameters_path = run_dir / "model_parameters.json"
    run_paths = {
        "manifest": manifest_path,
        "records": records_path,
        "scores": scores_path,
        "scoring_manifest": scoring_manifest_path,
        "summary": summary_path,
        "completed": completed_path,
        "model_parameters": parameters_path,
    }
    missing = [str(path) for path in run_paths.values() if not path.is_file()]
    if missing:
        raise ValueError(f"Missing completed formal run artifacts: {missing}")

    manifest = _load_json(manifest_path)
    if (
        not isinstance(manifest.get("run_id"), str)
        or manifest["run_id"] != digest({
            key: value for key, value in manifest.items() if key != "run_id"
        })
    ):
        raise ValueError("Run manifest ID does not bind its complete specification")
    _validate_sampling_manifest(manifest, cfg, doctor)

    data_manifest_path = _validate_registered_data_snapshot(data_dir)
    prepared_manifest, prepared_rows = load_prepared(
        data_dir, cfg["datasets"], cfg.get("evaluation")
    )
    data_paths = {data_manifest_path}
    data_paths.update(
        data_dir / entry["file"]
        for entry in prepared_manifest.get("datasets", {}).values()
        if isinstance(entry, Mapping) and isinstance(entry.get("file"), str)
    )
    protected_inputs = {
        CONFIG_PATH.resolve(), SUITE_PATH.resolve(), preflight_path,
        *run_paths.values(), *evidence_paths.values(),
        *(path.resolve() for path in data_paths),
    }
    if output in protected_inputs:
        raise ValueError("Audit output must not overwrite an input artifact")
    expected_prompt_ids = [
        [row["dataset"], row["source_id"], row["prompt_sha256"]]
        for row in prepared_rows
    ]
    if (
        manifest.get("data_manifest") != prepared_manifest
        or manifest.get("prompt_ids") != expected_prompt_ids
        or len(expected_prompt_ids) != len({tuple(row) for row in expected_prompt_ids})
    ):
        raise ValueError("Formal run is not bound to the supplied prepared data")

    _validate_sampling_data_audits(
        evidence["data_audit"], evidence["gold_audit"], manifest, cfg, doctor
    )
    records = read_jsonl(records_path)
    scores = read_jsonl(scores_path)
    coverage = validate_results(manifest, records, scores)
    if (
        not isinstance(coverage, Mapping)
        or set(coverage)
        != {"expected", "actual", "missing", "coverage", "evaluation", "complete"}
        or any(not _plain_int(coverage.get(field)) for field in (
            "expected", "actual", "missing",
        ))
        or coverage["expected"] != EXPECTED_RECORDS
        or coverage["actual"] != EXPECTED_RECORDS
        or coverage["missing"] != 0
        or coverage.get("coverage") != manifest["coverage"]
        or coverage.get("evaluation") != manifest["evaluation"]
        or coverage.get("complete") is not True
    ):
        raise ValueError("Raw T=1 result coverage is not exactly complete")

    scoring_manifest = _load_json(scoring_manifest_path)
    _validate_scoring_manifest(scoring_manifest, scores, cfg, doctor, manifest)
    summary = _load_json(summary_path)
    if summary.get("coverage") != coverage:
        raise ValueError("Saved summary coverage differs from raw artifacts")
    recomputed_summary_rows = summarize_results(
        manifest, records, scores, cfg["bootstrap_samples"]
    )
    if summary.get("rows") != recomputed_summary_rows:
        raise ValueError("Saved summary differs from recomputed raw results")
    summary_rows = _validate_sampling_summary(summary, manifest, cfg)
    count_fields = {
        "samples", "unique_prompts", "generation_turns", "generated_tokens",
        "matched_baseline_samples", "target_forward_calls", "draft_forward_calls",
        "scored_samples",
    }
    if any(
        not _plain_int(row.get(field), minimum=0)
        for row in summary_rows
        for field in count_fields
    ):
        raise ValueError("T=1 summary contains non-integral cardinalities")
    order_audit = validate_block_order(run_dir)
    pairwise_rows = _tree_block_pairwise_rows(records, cfg)

    completed = _load_json(completed_path)
    if (
        set(completed) != {"run_id", "records", "generations"}
        or completed.get("run_id") != manifest["run_id"]
        or not _plain_int(completed.get("records"))
        or completed["records"] != EXPECTED_RECORDS
        or not _plain_int(completed.get("generations"))
        or completed["generations"] != EXPECTED_GENERATION_TURNS
    ):
        raise ValueError("Runner completion marker is missing or incomplete")
    _validate_run_runtime_gate(
        _load_json(parameters_path), manifest, preflight, cfg
    )
    counts = _validate_fixed_cardinality(
        manifest, records, scores, summary_rows, order_audit, pairwise_rows,
        prepared_rows,
    )

    artifacts = {
        **{f"run/{name}": path for name, path in run_paths.items()},
        "preflight": preflight_path,
        **{f"evidence/{name}": path for name, path in evidence_paths.items()},
    }
    result = {
        "schema": 1,
        "study": "adaptive_tree_ddtree_dflash_t0_t1",
        "scope_label": SCOPE_LABEL,
        "model_id": MODEL_ID,
        "temperature": 1.0,
        "hardware": doctor["gpu"],
        "passed": True,
        **_claim_status(),
        "counts": counts,
        "gates": {
            "registered_config_and_plan": True,
            "whole_matrix_registration_audited": True,
            "source_and_fairness_audit": True,
            "distribution_law_audit": True,
            "real_checkpoint_h20_preflight": True,
            "same_tree_runtime_witness": True,
            "uncontended_gpu_gate": True,
            "prepared_data_and_gold_audits": True,
            "production_process_evaluator_uid_and_identity": True,
            "runtime_model_state_before_timing": True,
            "complete_raw_results_and_scores": True,
            "balanced_method_order": True,
            "summary_recomputed_from_raw": True,
            "pairwise_table_recomputed_from_raw": True,
        },
        "pairwise_results": pairwise_rows,
        "claim_boundary": (
            "This artifact validates only the registered Qwen3-4B/T=1/H20 "
            "submatrix. Full-matrix publication and tree-block superiority claims "
            "remain disabled regardless of these point estimates or intervals."
        ),
        "lineage": {
            "run_id": manifest["run_id"],
            "config": str(CONFIG_PATH.relative_to(ROOT)),
            "config_sha256": file_hash(CONFIG_PATH),
            "suite": str(SUITE_PATH.relative_to(ROOT)),
            "suite_sha256": file_hash(SUITE_PATH),
            "auditor": "src/gbv_experiments/t1_submatrix_audit.py",
            "auditor_sha256": file_hash(Path(__file__).resolve()),
            "cli": "scripts/audit_t1_submatrix.py",
            "cli_sha256": file_hash(ROOT / "scripts/audit_t1_submatrix.py"),
            "process_evaluator_probe_cli": (
                "scripts/capture_process_evaluator_self_test.py"
            ),
            "process_evaluator_probe_cli_sha256": file_hash(
                ROOT / "scripts/capture_process_evaluator_self_test.py"
            ),
            "data_manifest_sha256": digest(prepared_manifest),
            "data_manifest_file_sha256": file_hash(data_manifest_path),
            "artifact_sha256": {
                name: file_hash(path) for name, path in sorted(artifacts.items())
            },
        },
    }
    write_json(output, result)
    return result
