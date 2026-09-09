"""Fair, fail-closed orchestration for AdaptiveTree, DDTree, and DFlash.

This server runs only the registered T=0 AdaptiveTree matrix and the matched
T=1 Target/DFlash/DDTree sampling controls. Tree-block verification is a
separate experiment and is deliberately not scheduled or reported here.
"""
from __future__ import annotations

import ast
import csv
import gc
from dataclasses import replace
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile

from .common import ROOT, digest, file_hash, read_jsonl, source_hashes, write_json
from .config import Variant, build_variants, load_config
from .data import DATASETS, evaluation_policy
from .report import summarize as summarize_results, validate_results
from .runner import make_plan


STUDY = "adaptive_tree_ddtree_dflash_t0_t1"
TEMPERATURES = (1.0,)
SAMPLING_METHODS = ("target", "dflash", "ddtree")
TREE_BLOCK_STATUS = "deferred_not_run"
FORMAL_MODEL_IDS = ("qwen3_4b", "qwen3_8b")
T1_DATASET_COUNTS = {
    "gsm8k":128, "math500":128, "aime24":30, "aime25":30,
    "humaneval":164, "mbpp_sanitized":128, "livecodebench":128,
    "mt-bench":80,
}
T1_SEEDS = [17, 29, 43]
T1_SCORING = {
    "code_backend":"docker", "workers":4, "timeout_seconds":10,
    "lcb_timeout_seconds":6,
}
DISTRIBUTION_LAW_TESTS = (
    "tests/gbv_paper/test_probability_law.py::test_matching_verifier_preserves_target_output_law",
    "tests/gbv_paper/test_terminal_mass.py::test_official_ddtree_batched_posterior_has_exact_ancestral_law",
)
T1_IMPLEMENTATION_TESTS = (
    "tests/gbv_paper/test_engine.py::test_dflash_uses_greedy_draft_at_nonzero_target_temperature",
)
T1_DOCTOR_TESTS = DISTRIBUTION_LAW_TESTS + T1_IMPLEMENTATION_TESTS


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _literal_assignment(path: Path, name: str):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for statement in tree.body:
        if isinstance(statement, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id == name
                   for target in statement.targets):
                return ast.literal_eval(statement.value)
    raise ValueError(f"Missing pinned assignment {name} in {path}")


def _verify_manifest(directory: Path, manifest_path: Path, *, commit: str | None = None) -> dict:
    manifest = _load_json(manifest_path)
    if commit is not None and manifest.get("commit") != commit:
        raise ValueError(f"Wrong source commit in {manifest_path}")
    files = manifest.get("files", manifest)
    if not isinstance(files, dict) or not files:
        raise ValueError(f"Empty source manifest: {manifest_path}")
    for relative, expected in files.items():
        path = directory / relative
        if not path.is_file() or file_hash(path) != expected:
            raise ValueError(f"Source manifest mismatch: {path}")
    return manifest


def _formal_matrix_audit() -> dict:
    matrix_path = ROOT / "configs/formal_experiment_matrix.json"
    auditor_path = ROOT / "scripts/audit_formal_experiment_matrix.py"
    spec = importlib.util.spec_from_file_location(
        "_registered_formal_experiment_matrix_auditor", auditor_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load the formal experiment matrix auditor")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = module.audit(matrix_path)
    if report.get("passed") is not True or report.get("errors") != []:
        raise ValueError(f"Formal experiment matrix audit failed: {report.get('errors')}")
    return {
        "passed":True,
        "matrix":"configs/formal_experiment_matrix.json",
        "matrix_sha256":file_hash(matrix_path),
        "auditor":"scripts/audit_formal_experiment_matrix.py",
        "auditor_sha256":file_hash(auditor_path),
        "report":report,
    }


def _run_distribution_law_audit() -> dict:
    """Execute exact T=1 law checks plus the DFlash proposal-policy check."""
    command = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
               *T1_DOCTOR_TESTS]
    env = os.environ.copy()
    env.update({"PYTHONHASHSEED":"0", "PYTEST_DISABLE_PLUGIN_AUTOLOAD":"1",
                "PYTHONPATH":str(ROOT / "src")})
    completed = subprocess.run(command, cwd=ROOT, env=env, text=True,
                               capture_output=True)
    evidence = {
        "passed":completed.returncode == 0,
        "exact_finite_enumeration":True,
        "deterministic_inputs":True,
        "pytest_node_ids":list(T1_DOCTOR_TESTS),
        "distribution_law_node_ids":list(DISTRIBUTION_LAW_TESTS),
        "implementation_contract_node_ids":list(T1_IMPLEMENTATION_TESTS),
        "pytest_exit_code":completed.returncode,
        "source_sha256":{
            node.split("::", 1)[0]:file_hash(ROOT / node.split("::", 1)[0])
            for node in T1_DOCTOR_TESTS
        },
        "pytest_output":(completed.stdout + completed.stderr).strip()[-4000:],
    }
    if not evidence["passed"]:
        raise RuntimeError(
            "Exact T=1 distribution-law audit failed: " + evidence["pytest_output"]
        )
    return evidence


def _validate_distribution_law_evidence(evidence: dict) -> None:
    expected_hashes = {
        node.split("::", 1)[0]:file_hash(ROOT / node.split("::", 1)[0])
        for node in T1_DOCTOR_TESTS
    }
    if (evidence.get("passed") is not True
            or evidence.get("exact_finite_enumeration") is not True
            or evidence.get("deterministic_inputs") is not True
            or tuple(evidence.get("pytest_node_ids", ())) != T1_DOCTOR_TESTS
            or tuple(evidence.get("distribution_law_node_ids", ())) != DISTRIBUTION_LAW_TESTS
            or tuple(evidence.get("implementation_contract_node_ids", ())) != T1_IMPLEMENTATION_TESTS
            or evidence.get("pytest_exit_code") != 0
            or evidence.get("source_sha256") != expected_hashes):
        raise ValueError("Server doctor lacks valid exact T=1 distribution-law evidence")


def _gpu_allocation_gate(device) -> dict:
    """Refuse detectable foreign compute processes on the measured GPU."""
    import torch
    identity = str(getattr(torch.cuda.get_device_properties(device), "uuid", ""))
    if identity.lower() in {"", "none", "unknown", "unavailable"}:
        raise RuntimeError("Cannot resolve the assigned GPU UUID for contention checks")
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError("GPU process audit unavailable: " + result.stderr.strip()[-1000:])
    normalized = identity.lower().removeprefix("gpu-")
    observed = []
    foreign = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        pieces = [part.strip() for part in line.split(",", 1)]
        if len(pieces) != 2 or not pieces[0].isdigit():
            raise RuntimeError(f"Unparseable GPU process audit row: {line!r}")
        pid, gpu_uuid = int(pieces[0]), pieces[1]
        if gpu_uuid.lower().removeprefix("gpu-") == normalized:
            observed.append(pid)
            if pid != os.getpid():
                foreign.append(pid)
    if foreign:
        raise RuntimeError(f"Concurrent compute on the measured GPU: {foreign}")
    return {
        "passed":True, "gpu_uuid":identity,
        "allowed_pid":os.getpid(), "observed_compute_pids":observed,
        "foreign_compute_pids":foreign,
    }


def _validate_sampling_summary(summary: dict, manifest: dict, cfg: dict) -> list[dict]:
    expected_variants = build_variants(cfg)
    variants = {entry["variant"]["name"]:entry["variant"]
                for entry in expected_variants}
    expected_pairs = {(dataset, name) for dataset in cfg["datasets"] for name in variants}
    rows = summary.get("rows", [])
    actual_pairs = [(row.get("dataset"), row.get("variant")) for row in rows]
    if (summary.get("run_id") != manifest.get("run_id")
            or summary.get("bootstrap_clusters") != "source_id_with_all_seeds"
            or summary.get("bootstrap_samples") != cfg["bootstrap_samples"]
            or summary.get("performance_only") is not False
            or summary.get("coverage", {}).get("complete") is not True
            or len(actual_pairs) != len(set(actual_pairs))
            or set(actual_pairs) != expected_pairs):
        raise ValueError("T=1 summary identity, bootstrap, scoring, or row coverage is invalid")
    counts = evaluation_policy(cfg["datasets"], cfg.get("evaluation"))["counts"]
    for row in rows:
        variant = variants[row["variant"]]
        expected_samples = counts[row["dataset"]] * len(cfg["seeds"])
        if (row.get("method") != variant["method"]
                or row.get("temperature") != variant["temperature"]
                or row.get("samples") != expected_samples
                or row.get("unique_prompts") != counts[row["dataset"]]
                or row.get("matched_baseline_samples") != expected_samples
                or "mean_generation_length" not in row
                or "accepted_per_verify" not in row):
            raise ValueError("T=1 summary row does not match its frozen method contract")
        if row["dataset"] == "mt-bench":
            if (row.get("quality_status") != "external_judge_report_separate"
                    or row.get("scored_samples") != 0
                    or row.get("quality") is not None):
                raise ValueError("MT-Bench must remain explicitly external-judge-only")
        elif (row.get("quality_status") != "scored"
                or row.get("scored_samples") != expected_samples
                or row.get("quality") is None):
            raise ValueError("T=1 objective-task quality scores are incomplete")
    return rows


def _validate_sampling_manifest(manifest: dict, cfg: dict, doctor: dict) -> None:
    """Reject T=1 artifacts generated under any non-registered run contract."""
    plan = make_plan(cfg)
    expected = {
        "schema":3,
        "model":cfg["model"],
        "variants":build_variants(cfg),
        "seeds":cfg["seeds"],
        "max_new_tokens":cfg["max_new_tokens"],
        "method_order":cfg["method_order"],
        "coverage":plan["coverage"],
        "evaluation":plan["evaluation"],
        "profile":False,
        "scoring":cfg["scoring"],
        "bootstrap_samples":cfg["bootstrap_samples"],
        "dataset_names":cfg["datasets"],
        "dataset_turn_counts":{name:DATASETS[name].turns for name in cfg["datasets"]},
        "expected_records":plan["expected_records"],
        "expected_generations":plan["expected_generations"],
        "source_hashes":source_hashes(),
        "python":doctor["python"],
        "cuda":doctor["cuda"],
        "gpu":doctor["gpu"],
        "gpu_uuid":doctor["gpu_uuid"],
        "gpu_memory_bytes":doctor["gpu_memory_bytes"],
    }
    expected_versions = {
        "torch":doctor["torch"],
        **{name:doctor["packages"][name] for name in (
            "transformers", "datasets", "huggingface-hub", "numpy"
        )},
    }
    if any(manifest.get(field) != value for field, value in expected.items()):
        raise ValueError("T=1 run manifest differs from the frozen experiment contract")
    if manifest.get("versions") != expected_versions:
        raise ValueError("T=1 run package versions differ from the server doctor")


def _doctor_process_identity(doctor: dict) -> dict:
    if doctor["code_backend"] != "process":
        return {
            "process_python":None, "process_python_resolved":None,
            "process_python_sha256":None, "process_python_runtime":None,
            "process_pyvenv_cfg_sha256":None,
        }
    return {
        "process_python":doctor["code_evaluator_python"],
        "process_python_resolved":doctor["code_evaluator_python_resolved"],
        "process_python_sha256":doctor["code_evaluator_python_sha256"],
        "process_python_runtime":doctor["code_evaluator_python_runtime"],
        "process_pyvenv_cfg_sha256":doctor["code_evaluator_pyvenv_cfg_sha256"],
    }


def _validate_t1_preflight(preflight: dict, cfg: dict, doctor: dict) -> None:
    """Bind the real-checkpoint/tree/cache gate to the final T=1 report."""
    environment = preflight.get("environment", {})
    expected_versions = {
        "torch":doctor["torch"],
        **{name:doctor["packages"][name] for name in (
            "transformers", "huggingface-hub", "datasets", "numpy", "math-verify"
        )},
    }
    expected_image = doctor["docker_image"] if doctor["code_backend"] == "docker" else None
    expected_process_identity = _doctor_process_identity(doctor)
    evaluator_checks = environment.get("lcb_evaluator_checks", [])
    evaluator_keys = [
        (check.get("functional"), check.get("expected_pass"))
        for check in evaluator_checks
    ]
    if (preflight.get("passed") is not True
            or preflight.get("scope") !=
               "checkpoint structural and bounded-numerical smoke gate, not full benchmark results"
            or preflight.get("model") != cfg["model"]
            or preflight.get("source_hashes") != source_hashes()
            or preflight.get("numerical_policy") != {
                "dtype":"bfloat16_only",
                "max_absolute_logit_error":0.5,
                "required":"stable repeated outputs and top-1 margins <= 2 * measured error",
            }
            or not preflight.get("stop_token_ids")
            or not all(isinstance(token, int) for token in preflight["stop_token_ids"])
            or environment.get("versions") != expected_versions
            or environment.get("python") != doctor["python"]
            or environment.get("cuda") != doctor["cuda"]
            or environment.get("gpu") != doctor["gpu"]
            or environment.get("gpu_uuid") != doctor["gpu_uuid"]
            or environment.get("total_memory_bytes") != doctor["gpu_memory_bytes"]
            or environment.get("code_backend") != doctor["code_backend"]
            or environment.get("code_image_id") != expected_image
            or any(environment.get(field) != value
                   for field, value in expected_process_identity.items())
            or len(evaluator_keys) != len(set(evaluator_keys))
            or set(evaluator_keys) != {
                (False, False), (False, True), (True, False), (True, True)
            }
            or any(check.get("result", {}).get("passed") is not check["expected_pass"]
                   for check in evaluator_checks)):
        raise ValueError("T=1 GPU preflight environment or evaluator evidence is invalid")

    expected_variants = {}
    for entry in build_variants(cfg):
        variant = replace(
            Variant(**entry["variant"]), temperature=0.0, draft_temperature=1.0
        )
        expected_variants[variant.name] = variant.to_dict()
    prompts = [
        "Compute 19 + 23. Give a brief explanation.",
        "Write a Python function that reverses a list.",
        "Explain why the sum of two even integers is even.",
    ]
    expected_generation_keys = {
        (digest(prompt), name) for prompt in prompts for name in expected_variants
    }
    checks = preflight.get("checks", [])
    generation_checks = [check for check in checks if "greedy_equal" in check]
    structural_checks = [check for check in checks if "greedy_equal" not in check]
    generation_keys = [
        (check.get("prompt_sha256"), check.get("variant", {}).get("name"))
        for check in generation_checks
    ]
    if (len(generation_keys) != len(set(generation_keys))
            or set(generation_keys) != expected_generation_keys
            or any(check.get("variant") != expected_variants.get(
                       check.get("variant", {}).get("name")
                   )
                   for check in generation_checks)
            or any(check.get("gate_passed") is not True for check in generation_checks)
            or preflight.get("greedy_exact_passed") is not
               all(check.get("greedy_equal") is True for check in generation_checks)
            or preflight.get("numerical_ambiguities") !=
               sum(check.get("greedy_equal") is not True for check in generation_checks)
            or any(check.get("greedy_equal") is not True
                   and (check.get("numerical_ambiguity") is not True
                        or check.get("mismatch", {}).get(
                            "numerical_diagnosis", {}
                        ).get("gate_passed") is not True)
                   for check in generation_checks)):
        raise ValueError("T=1 GPU preflight generation coverage or numerical gate is invalid")
    structural_names = [check.get("check") for check in structural_checks]
    if (structural_names.count("tree_argmax") != 2 * len(prompts)
            or structural_names.count("compacted_cache_argmax") != len(prompts)
            or len(structural_names) != 3 * len(prompts)
            or any(check.get("passed") is not True
                   or not isinstance(check.get("max_absolute_logit_error"), (int, float))
                   or not math.isfinite(check["max_absolute_logit_error"])
                   or check["max_absolute_logit_error"] < 0
                   for check in structural_checks)):
        raise ValueError("T=1 GPU preflight tree/cache checks are incomplete")


def _validate_sampling_data_audits(data_audit: dict, gold_audit: dict,
                                   manifest: dict, cfg: dict, doctor: dict) -> None:
    """Bind prepared T=1 samples and reference scoring checks to final results."""
    data_manifest = manifest.get("data_manifest", {})
    manifest_hash = digest(data_manifest)
    policy = evaluation_policy(cfg["datasets"], cfg.get("evaluation"))
    counts = policy["counts"]
    prompt_keys = {(dataset, source_id)
                   for dataset, source_id, _ in manifest.get("prompt_ids", [])}
    expected_total = sum(counts.values())
    reference_free = {"livecodebench", "mt-bench"}
    reference_free_count = sum(counts[name] for name in reference_free)
    if (data_audit.get("data_manifest_sha256") != manifest_hash
            or data_audit.get("coverage") != manifest.get("coverage")
            or data_audit.get("evaluation") != policy
            or data_audit.get("evaluation_counts") != counts
            or data_audit.get("training_policy") !=
               "frozen_target_and_frozen_published_draft; no local training or fitting"
            or data_audit.get("locally_trained_components_used") != []
            or data_audit.get("training_files_checked") != []
            or data_audit.get("exact_text_overlaps") != []
            or data_audit.get("training_rows_without_text") != []
            or data_audit.get("unparseable_math_gold") != []
            or data_audit.get("missing_code_reference") != []
            or set(data_audit.get("reference_free_datasets", [])) != reference_free
            or data_audit.get("checks_passed") is not True):
        raise ValueError("T=1 prepared-data audit is missing or differs from the run")
    results = gold_audit.get("results", [])
    result_keys = [(row.get("dataset"), row.get("source_id")) for row in results]
    if (gold_audit.get("data_manifest_sha256") != manifest_hash
            or gold_audit.get("backend") != doctor["code_backend"]
            or gold_audit.get("count") != expected_total
            or gold_audit.get("passed") is not True
            or gold_audit.get("canonical_answers_checked") !=
               expected_total - reference_free_count
            or gold_audit.get("without_canonical_answer") != reference_free_count
            or gold_audit.get("failures") != []
            or len(result_keys) != len(set(result_keys))
            or set(result_keys) != prompt_keys
            or any((row["passed"] is None) != (row["dataset"] in reference_free)
                   or (row["dataset"] not in reference_free
                       and row["passed"] is not True)
                   for row in results)):
        raise ValueError("T=1 gold/reference scoring audit is incomplete or stale")


def _validate_scoring_manifest(scoring: dict, scores: list[dict], cfg: dict,
                               doctor: dict, run_manifest: dict) -> None:
    settings = cfg["scoring"]
    expected_image = doctor["docker_image"] if doctor["code_backend"] == "docker" else None
    expected_process_identity = _doctor_process_identity(doctor)
    expected_scoring_id = digest({
        "run_id":run_manifest["run_id"],
        "backend":doctor["code_backend"],
        "image_id":expected_image,
        **expected_process_identity,
        "math_verify":doctor["packages"]["math-verify"],
        "timeout":settings["timeout_seconds"],
        "lcb_timeout_per_test":settings["lcb_timeout_seconds"],
        "scorer_sources":source_hashes(),
    })
    if (scoring.get("backend") != doctor["code_backend"]
            or scoring.get("image_id") != expected_image
            or any(scoring.get(field) != value
                   for field, value in expected_process_identity.items())
            or scoring.get("math_verify") != doctor["packages"]["math-verify"]
            or scoring.get("timeout_seconds") != settings["timeout_seconds"]
            or scoring.get("lcb_timeout_seconds_per_test") != settings["lcb_timeout_seconds"]
            or scoring.get("scoring_id") != expected_scoring_id
            or any(score.get("scoring_id") != scoring["scoring_id"] for score in scores)):
        raise ValueError("T=1 scores differ from the frozen scoring contract")


def _adaptive_contract_digest(metadata: dict) -> str:
    payload = json.dumps(metadata, sort_keys=True, ensure_ascii=False,
                         allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def _adaptive_code_identity(adaptive_root: Path) -> str:
    paths = sorted((adaptive_root / "src/dflash_specblock").rglob("*.py"))
    if not paths:
        raise ValueError("AdaptiveTree source tree is empty")
    return _adaptive_contract_digest({
        str(path.relative_to(adaptive_root)):file_hash(path) for path in paths
    })


def _is_hex_sha(value, length: int) -> bool:
    return (isinstance(value, str) and len(value) == length
            and all(character in "0123456789abcdef" for character in value))


def _validate_adaptive_data_lineage(metadata: dict, summary: dict, suite: dict) -> None:
    """Validate the immutable data/source snapshot embedded in a T=0 result."""
    config = suite["adaptive_config"]
    adaptive_root = suite["adaptive_root"]
    official_spec = adaptive_root / "src/dflash_specblock/paper/official_spec.py"
    sources = _literal_assignment(official_spec, "SOURCES")
    models = _literal_assignment(official_spec, "MODELS")
    pinned_revisions = _literal_assignment(official_spec, "PINNED_MODEL_REVISIONS")
    expected_source_manifest = _load_json(
        adaptive_root / "third_party/ddtree_pinned/SOURCE_SHA256.json"
    )
    data_manifest = metadata.get("dataset_manifest", {})
    source_lock = summary.get("source_lock", {})
    serialized_lock = (
        json.dumps(source_lock, ensure_ascii=False, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode()
    lock_hash = hashlib.sha256(serialized_lock).hexdigest()
    expected_files = {f"{name}.json" for name in config["datasets"]}
    files = data_manifest.get("files", {})
    dataset_repositories = {entry[0] for entry in sources.values()}
    model_repositories = {name for pair in models for name in pair}
    if (metadata.get("source_manifest") != expected_source_manifest
            or metadata.get("code_identity") != _adaptive_code_identity(adaptive_root)
            or summary.get("dataset_manifest") != data_manifest
            or data_manifest.get("version") != 4
            or data_manifest.get("official_commit") != config["official_commit"]
            or data_manifest.get("sampling_seed") != 0
            or data_manifest.get("sample_limits") != config["sample_limits"]
            or data_manifest.get("training") is not False
            or data_manifest.get("full_split") is not False
            or data_manifest.get("official_source_manifest") != expected_source_manifest
            or data_manifest.get("source_lock_sha256") != lock_hash
            or set(files) != expected_files
            or source_lock.get("official_commit") != config["official_commit"]
            or set(source_lock.get("datasets", {})) != dataset_repositories
            or set(source_lock.get("models", {})) != model_repositories):
        raise ValueError("AdaptiveTree code, pinned source, or dataset lineage is invalid")
    if any(not _is_hex_sha(revision, 40) for revision in (
            *source_lock["datasets"].values(), *source_lock["models"].values())):
        raise ValueError("AdaptiveTree data/model revisions are not immutable commit SHAs")
    if any(source_lock["models"].get(name) != revision
           for name, revision in pinned_revisions.items()):
        raise ValueError("AdaptiveTree pinned 4B/8B model revisions changed")
    for name, count in config["sample_limits"].items():
        info = files.get(f"{name}.json", {})
        if (set(info) != {"rows", "full_source_rows", "sha256"}
                or info.get("rows") != count
                or not isinstance(info.get("full_source_rows"), int)
                or info["full_source_rows"] < count
                or not _is_hex_sha(info.get("sha256"), 64)):
            raise ValueError("AdaptiveTree dataset file identities are incomplete")


def _validate_adaptive_summary(summary: dict, adaptive_run: Path, model: dict,
                               suite: dict, doctor: dict) -> tuple[list[dict], ...]:
    """Validate the complete one-model T=0 artifact, not just nonempty rows."""
    config = suite["adaptive_config"]
    datasets = list(config["datasets"])
    counts = config["sample_limits"]
    budgets = [f"ddtree_tb{budget}" for budget in config["tree_budgets"]]
    variants = list(config["variants"])
    expected_methods = {
        "DFlash":"dflash", "DDTree-best":None,
        **{name:name for name in (*budgets, *variants)},
    }

    contract = _load_json(adaptive_run / "contract.json")
    metadata = contract.get("metadata", {})
    environment_path = adaptive_run / "environment.json"
    environment = _load_json(environment_path)
    _validate_adaptive_data_lineage(metadata, summary, suite)
    if (contract.get("identity") != _adaptive_contract_digest(metadata)
            or summary.get("protocol_identity") != contract.get("identity")
            or summary.get("environment_sha256") != file_hash(environment_path)
            or metadata.get("version") != 5
            or metadata.get("config") != config
            or metadata.get("nproc_per_node") != 1
            or metadata.get("model_indices") != [model["adaptive_model_index"]]
            or metadata.get("datasets") != datasets
            or metadata.get("smoke_count") != 0
            or metadata.get("max_new_tokens") != 2048
            or metadata.get("method_schema_version") != 2
            or metadata.get("primary_adaptive_method") != "adaptive"
            or metadata.get("greedy_audit_policy") != "record-bf16-mismatches"
            or metadata.get("method_order_policy") != "balanced-rotation"
            or "diagnostic_variants" in metadata
            or metadata.get("deprecated_cli_aliases", [])
            or metadata.get("wandb") is not None):
        raise ValueError("AdaptiveTree run identity or frozen parent contract is invalid")
    artifact_entries = summary.get("input_artifacts", [])
    expected_artifacts = {
        (dataset, backend) for dataset in datasets
        for backend in ("sdpa", "flash_attention_2")
    }
    artifact_keys = [
        (entry.get("dataset"), entry.get("backend")) for entry in artifact_entries
    ]
    if (len(artifact_keys) != len(set(artifact_keys))
            or set(artifact_keys) != expected_artifacts):
        raise ValueError("AdaptiveTree summary is not bound to every raw input artifact")
    for entry in artifact_entries:
        dataset, backend = entry["dataset"], entry["backend"]
        suffix = "sdpa" if backend == "sdpa" else "flash_attn"
        stem = (
            f"{dataset}__model{model['adaptive_model_index']}__temp0.0__{suffix}"
        )
        artifact_path = adaptive_run / f"{stem}.pt"
        marker_path = adaptive_run / f"{stem}.complete.json"
        marker = _load_json(marker_path)
        turns = counts[dataset] * (2 if dataset == "mt-bench" else 1)
        expected_methods_for_backend = ["baseline", "dflash"]
        if backend == "sdpa":
            expected_methods_for_backend += [*budgets, *variants]
        if (entry != {
                "dataset":dataset,
                "model_index":model["adaptive_model_index"],
                "model":model["target"],
                "backend":backend,
                "artifact":artifact_path.name,
                "artifact_sha256":marker.get("sha256"),
                "completion_marker":marker_path.name,
                "completion_marker_sha256":file_hash(marker_path),
                "protocol_identity":marker.get("identity"),
            }
                or marker.get("identity") != contract["identity"]
                or marker.get("sha256") != file_hash(artifact_path)
                or marker.get("turns") != turns
                or marker.get("cases") != counts[dataset]
                or marker.get("methods") != expected_methods_for_backend
                or marker.get("smoke") is not False
                or marker.get("greedy_audit_policy") != "record-bf16-mismatches"):
            raise ValueError("AdaptiveTree raw artifact lineage or completion marker changed")
    expected_packages = {
        name:(doctor["torch"] if name == "torch" else doctor["packages"][name])
        for name in ("torch", "transformers", "datasets", "huggingface-hub",
                     "accelerate", "numpy")
    }
    benchmark_gpus = environment.get("benchmark_gpus", [])
    if (environment.get("python") != doctor["python"]
            or environment.get("packages") != expected_packages
            or environment.get("cuda") != doctor["cuda"]
            or environment.get("gpu") != doctor["gpu"]
            or environment.get("gpu_uuid") != doctor["gpu_uuid"]
            or environment.get("gpu_memory_bytes") != doctor["gpu_memory_bytes"]
            or environment.get("flash_attn") != doctor["flash_attn"]
            or environment.get("nproc_per_node") != 1
            or len(benchmark_gpus) != 1
            or benchmark_gpus[0].get("rank") != 0
            or benchmark_gpus[0].get("gpu") != doctor["gpu"]
            or benchmark_gpus[0].get("uuid") != doctor["gpu_uuid"]):
        raise ValueError("AdaptiveTree hardware/software differs from the server doctor")

    controlled = summary.get("controlled_same_backend_rows", [])
    exact_rows = summary.get("controlled_exact_output_subset_rows", [])
    auxiliary = summary.get("rows", [])
    expected_controlled_keys = {(dataset, method) for dataset in datasets
                                for method in expected_methods}
    controlled_keys = [(row.get("dataset"), row.get("method")) for row in controlled]
    exact_methods = {"DFlash", *budgets, *variants}
    expected_exact_keys = {(dataset, method) for dataset in datasets
                           for method in exact_methods}
    exact_keys = [(row.get("dataset"), row.get("method")) for row in exact_rows]
    auxiliary_keys = [(row.get("dataset"), row.get("method")) for row in auxiliary]
    if (summary.get("protocol") != "ddtree_official_t0"
            or summary.get("training") is not False
            or summary.get("official_samples") is not True
            or summary.get("method_schema_version") != 2
            or summary.get("primary_adaptive_method") != "adaptive"
            or summary.get("greedy_audit_policy") != "record-bf16-mismatches"
            or summary.get("method_order_policy") != "balanced-rotation"
            or summary.get("controlled_protocol_gate_passed") is not True
            or summary.get("strict_lossless_claim_eligible") is not False
            or len(controlled_keys) != len(set(controlled_keys))
            or set(controlled_keys) != expected_controlled_keys
            or len(exact_keys) != len(set(exact_keys))
            or set(exact_keys) != expected_exact_keys
            or len(auxiliary_keys) != len(set(auxiliary_keys))
            or set(auxiliary_keys) != expected_controlled_keys):
        raise ValueError("AdaptiveTree summary method/dataset coverage is incomplete")

    for row in controlled:
        dataset, method = row["dataset"], row["method"]
        expected_key = expected_methods[method]
        turns = counts[dataset] * (2 if dataset == "mt-bench" else 1)
        selected = row.get("selected_key")
        if (expected_key is None and selected not in budgets):
            raise ValueError("AdaptiveTree best-DDTree row selected an unregistered budget")
        if (expected_key is not None and selected != expected_key):
            raise ValueError("AdaptiveTree summary method label/key mismatch")
        expected_role = ("primary" if selected == "adaptive" else
                         "historical_control" if selected == "adaptive_legacy" else
                         "ablation" if selected.startswith("adaptive_") else "baseline")
        numeric = (row.get("speedup_vs_target"), row.get("speedup_vs_best_ddtree"),
                   row.get("mean_acceptance_length"), row.get("exact_output_rate"))
        if (row.get("model") != model["target"]
                or row.get("cases") != counts[dataset]
                or row.get("turns") != turns
                or row.get("responses") != turns
                or row.get("method_role") != expected_role
                or row.get("target_baseline_backend") != "sdpa"
                or row.get("method_backend") != "sdpa"
                or row.get("exact_output_responses", -1)
                   + row.get("output_divergence_responses", -1) != turns
                or not all(isinstance(value, (int, float)) and math.isfinite(value)
                           and value >= 0 for value in numeric)
                or not 0 <= row["exact_output_rate"] <= 1
                or not math.isclose(row["exact_output_rate"],
                                    row["exact_output_responses"] / turns)):
            raise ValueError("AdaptiveTree summary row differs from its frozen contract")
    for rows, methods, exact_subset in (
            (auxiliary, expected_methods, False),
            (exact_rows, {"DFlash":"dflash",
                          **{name:name for name in (*budgets, *variants)}}, True)):
        for row in rows:
            dataset, method = row["dataset"], row["method"]
            expected_key = methods[method]
            turns = counts[dataset] * (2 if dataset == "mt-bench" else 1)
            selected = row.get("selected_key")
            if ((expected_key is None and selected not in budgets)
                    or (expected_key is not None and selected != expected_key)
                    or row.get("model") != model["target"]
                    or row.get("cases") != counts[dataset]
                    or row.get("turns") != turns
                    or row.get("responses") != turns):
                raise ValueError("AdaptiveTree auxiliary/diagnostic row contract is invalid")
            if not exact_subset and (
                    row.get("comparison_scope") != "upstream_best_backend_auxiliary"
                    or type(row.get("cross_backend_inputs_identical")) is not bool
                    or (dataset != "mt-bench"
                        and row["cross_backend_inputs_identical"] is not True)):
                raise ValueError("AdaptiveTree cross-backend context label is invalid")
            if exact_subset and (row.get("target_baseline_backend") != "sdpa"
                                 or row.get("method_backend") != "sdpa"
                                 or row.get("selection_bias_warning") is not True):
                raise ValueError("AdaptiveTree exact-output diagnostic lost its bias warning")
    if (summary.get("primary_rows") != [row for row in controlled
                                         if row["method_role"] == "primary"]
            or summary.get("ablation_rows") != [row for row in controlled
                                                 if row["method_role"] in {
                                                     "ablation", "historical_control"
                                                 }]):
        raise ValueError("AdaptiveTree primary/ablation row classification is invalid")
    usage = summary.get("adaptive_budget_usage", [])
    if ({(row.get("dataset"), row.get("method")) for row in usage}
            != {(dataset, method) for dataset in datasets for method in variants}
            or len(usage) != len(datasets) * len(variants)
            or any(row.get("model") != model["target"] for row in usage)):
        raise ValueError("AdaptiveTree budget-usage evidence is incomplete")
    numerical = summary.get("numerical_audit", {})
    expected_responses = 2 * sum(
        count * (2 if dataset == "mt-bench" else 1)
        for dataset, count in counts.items()
    )
    cross_backend_inputs = numerical.get("cross_backend_input_mismatches")
    auxiliary_has_context_mismatch = any(
        row["cross_backend_inputs_identical"] is False for row in auxiliary
    )
    expected_warning = summary.get("cross_backend_auxiliary_table_warning")
    if (numerical.get("responses") != expected_responses
            or numerical.get("exact_responses", -1)
               + numerical.get("mismatching_responses", -1) != expected_responses
            or type(cross_backend_inputs) is not int
            or not 0 <= cross_backend_inputs <= counts.get("mt-bench", 0)
            or auxiliary_has_context_mismatch != (cross_backend_inputs > 0)
            or bool(expected_warning) != (cross_backend_inputs > 0)):
        raise ValueError("AdaptiveTree numerical audit coverage is incomplete")
    return controlled, exact_rows, auxiliary


def _recompute_adaptive_summary(adaptive_run: Path, adaptive_data_dir: Path,
                                model: dict, suite: dict, scratch_parent: Path) -> dict:
    """Rebuild T=0 tables from hash-checked raw artifacts in an isolated copy."""
    scratch_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
            prefix=f".recompute-{model['id']}-", dir=scratch_parent) as temporary:
        scratch = Path(temporary)
        required = [adaptive_run / "contract.json", adaptive_run / "environment.json"]
        for dataset in suite["adaptive_config"]["datasets"]:
            for suffix in ("sdpa", "flash_attn"):
                stem = (
                    f"{dataset}__model{model['adaptive_model_index']}"
                    f"__temp0.0__{suffix}"
                )
                required.extend([
                    adaptive_run / f"{stem}.pt",
                    adaptive_run / f"{stem}.complete.json",
                ])
        for source in required:
            if not source.is_file():
                raise ValueError(f"Missing AdaptiveTree raw artifact: {source}")
            os.link(source, scratch / source.name)
        env = os.environ.copy()
        env["PYTHONPATH"] = str(suite["adaptive_root"] / "src")
        command = [
            sys.executable, "-m", "dflash_specblock.paper", "summarize",
            "--config", str(suite["adaptive_config_path"]),
            "--data-dir", str(adaptive_data_dir),
            "--run-dir", str(scratch),
            "--nproc-per-node", "1",
            "--model-index", str(model["adaptive_model_index"]),
            "--greedy-audit-policy", "record-bf16-mismatches",
            "--method-order-policy", "balanced-rotation",
        ]
        completed = subprocess.run(
            command, cwd=suite["adaptive_root"], env=env, text=True,
            capture_output=True,
        )
        if completed.returncode:
            detail = (completed.stderr or completed.stdout).strip()[-4000:]
            raise ValueError(f"AdaptiveTree raw-artifact recomputation failed: {detail}")
        return _load_json(scratch / "tables.json")


def _validate_adaptive_gpu_preflight(evidence: dict, run_dir: Path,
                                     suite: dict, doctor: dict) -> None:
    if (evidence.get("passed") is not True
            or evidence.get("gpu_uuid") != doctor["gpu_uuid"]
            or evidence.get("nested_code_identity") !=
               _adaptive_code_identity(suite["adaptive_root"])
            or set(evidence.get("models", {})) != set(FORMAL_MODEL_IDS)):
        raise ValueError("AdaptiveTree real-checkpoint GPU preflight is invalid")
    for model in suite["models"]:
        recorded = evidence["models"][model["id"]]
        relative = f"preflight/adaptive_{model['id']}_smoke"
        smoke_dir = run_dir / relative
        summary_path = smoke_dir / "tables.json"
        summary = _load_json(summary_path)
        contract = _load_json(smoke_dir / "contract.json")
        metadata = contract.get("metadata", {})
        environment = _load_json(smoke_dir / "environment.json")
        _validate_adaptive_data_lineage(metadata, summary, suite)
        if (recorded.get("run_dir") != relative
                or recorded.get("tables_sha256") != file_hash(summary_path)
                or recorded.get("protocol_identity") != contract.get("identity")
                or recorded.get("input_artifacts") != summary.get("input_artifacts")
                or contract.get("identity") != _adaptive_contract_digest(metadata)
                or summary.get("protocol_identity") != contract.get("identity")
                or metadata.get("model_indices") != [model["adaptive_model_index"]]
                or metadata.get("datasets") != ["gsm8k"]
                or metadata.get("smoke_count") != 1
                or metadata.get("max_new_tokens") != 32
                or metadata.get("greedy_audit_policy") != "record-bf16-mismatches"
                or metadata.get("method_order_policy") != "balanced-rotation"
                or summary.get("official_samples") is not False
                or len(summary.get("input_artifacts", [])) != 2
                or environment.get("gpu_uuid") != doctor["gpu_uuid"]):
            raise ValueError("AdaptiveTree GPU smoke identity or coverage is invalid")
        for artifact in summary["input_artifacts"]:
            path = smoke_dir / artifact["artifact"]
            marker_path = smoke_dir / artifact["completion_marker"]
            marker = _load_json(marker_path)
            if (artifact.get("artifact_sha256") != file_hash(path)
                    or artifact.get("completion_marker_sha256") != file_hash(marker_path)
                    or marker.get("sha256") != artifact["artifact_sha256"]
                    or marker.get("identity") != contract["identity"]
                    or marker.get("smoke") is not True):
                raise ValueError("AdaptiveTree GPU smoke raw artifacts changed")


def _tag(temperature: float) -> str:
    return {1.0: "t1p0"}[temperature]


def _group_tag(temperature: float) -> str:
    return {1.0: "t10"}[temperature]


def _expected_name(method: str, temperature: float) -> str:
    return f"{method}_{_tag(temperature)}"


def _matched_variant(left: dict, right: dict) -> bool:
    ignored = {"name", "method", "draft_temperature"}
    return all(left[key] == right[key] for key in left if key not in ignored)


def _validate_sampling_config(cfg: dict) -> None:
    if "explicit_variants" not in cfg:
        raise ValueError("Integrated T=1 controls must explicitly freeze every variant")
    model = cfg["model"]
    if (model.get("dtype"), model.get("target_attention"), model.get("draft_attention"),
            model.get("enable_thinking"), model.get("allow_tf32")) != (
            "bfloat16", "sdpa", "sdpa", False, False):
        raise ValueError("T=1 controls must share BF16, SDPA/SDPA, no thinking, and TF32 disabled")
    if cfg.get("method_order") != {"policy": "balanced_rotation", "seed": 20260909}:
        raise ValueError("T=1 controls require the frozen balanced-rotation schedule")
    if (cfg.get("datasets") != list(T1_DATASET_COUNTS)
            or cfg.get("seeds") != T1_SEEDS
            or cfg.get("max_new_tokens") != 2048
            or cfg.get("warmup_tokens") != 32
            or cfg.get("bootstrap_samples") != 10000
            or cfg.get("scoring") != T1_SCORING
            or evaluation_policy(cfg["datasets"], cfg.get("evaluation")) != {
                "protocol":"ddtree_counts", "sample_seed":0,
                "counts":T1_DATASET_COUNTS,
            }):
        raise ValueError("T=1 data, seeds, limits, bootstrap, or scoring contract drifted")
    entries = build_variants(cfg)
    by_name = {entry["variant"]["name"]: entry for entry in entries}
    expected = {_expected_name(method, temperature)
                for temperature in TEMPERATURES for method in SAMPLING_METHODS}
    if set(by_name) != expected:
        raise ValueError(f"Wrong T=1 control matrix; missing={sorted(expected-set(by_name))}, "
                         f"extra={sorted(set(by_name)-expected)}")
    for temperature in TEMPERATURES:
        variants = {
            method:by_name[_expected_name(method, temperature)]["variant"]
            for method in SAMPLING_METHODS
        }
        for method, variant in variants.items():
            if set(by_name[variant["name"]]["groups"]) != {"main", _group_tag(temperature)}:
                raise ValueError("Every T=1 control requires main and temperature groups")
            # Official DFlash remains a greedy argmax draft at sampled Target
            # T=1; only DDTree consumes the stochastic draft distribution.
            expected_draft_temperature = temperature if method == "ddtree" else None
            if (variant["method"] != method or variant["temperature"] != temperature
                    or variant["draft_temperature"] != expected_draft_temperature
                    or variant["paths"] != 1 or variant["length"] != 15
                    or variant["tree_budget"] != 45
                    or variant["probability_dtype"] != "float64"):
                raise ValueError(f"Unfair or invalid T=1 control: {variant['name']}")
        reference = variants["ddtree"]
        for method in ("dflash",):
            if not _matched_variant(reference, variants[method]):
                raise ValueError(f"{method} and DDTree differ in more than implementation")


def load_integrated_suite(path: Path, model_ids=None) -> dict:
    path = path.resolve()
    spec = _load_json(path)
    required = {"schema", "study", "adaptive", "positive_temperature", "models"}
    if set(spec) != required or spec.get("schema") != 1 or spec.get("study") != STUDY:
        raise ValueError("Invalid integrated suite schema or study ID")
    adaptive_spec = spec["adaptive"]
    if set(adaptive_spec) != {"project", "config", "temperature", "primary_method",
                             "method_schema_version", "greedy_audit_policy",
                             "method_order_policy"}:
        raise ValueError("Invalid AdaptiveTree suite section")
    if (adaptive_spec["temperature"] != 0
            or adaptive_spec["primary_method"] != "adaptive"
            or adaptive_spec["method_schema_version"] != 2
            or adaptive_spec["greedy_audit_policy"] != "record-bf16-mismatches"
            or adaptive_spec["method_order_policy"] != "balanced-rotation"):
        raise ValueError(
            "Integrated AdaptiveTree requires canonical adaptive, method schema v2, "
            "BF16 divergence recording, and balanced order"
        )
    sampling_spec = spec["positive_temperature"]
    if (set(sampling_spec) != {"temperatures", "methods", "length", "tree_budget",
                               "probability_dtype", "method_order_policy",
                               "tree_block_verification"}
            or tuple(sampling_spec["temperatures"]) != TEMPERATURES
            or tuple(sampling_spec["methods"]) != SAMPLING_METHODS
            or (sampling_spec["length"], sampling_spec["tree_budget"],
                sampling_spec["probability_dtype"], sampling_spec["method_order_policy"],
                sampling_spec["tree_block_verification"])
               != (15, 45, "float64", "balanced_rotation", TREE_BLOCK_STATUS)):
        raise ValueError("Invalid T=1 sampling controls or tree-block deferral status")

    adaptive_root = (path.parent / adaptive_spec["project"]).resolve()
    adaptive_config_path = (path.parent / adaptive_spec["config"]).resolve()
    adaptive_config = _load_json(adaptive_config_path)
    if (adaptive_config.get("protocol") != "ddtree_official_t0"
            or adaptive_config.get("temperature") != 0
            or adaptive_config.get("seed") != 0
            or adaptive_config.get("max_new_tokens") != 2048
            or adaptive_config.get("official_commit") !=
               "c96427a185677bf4133ed865dd1626a5041aef9b"):
        raise ValueError("AdaptiveTree must use the pinned official T=0 protocol")
    official_spec_path = adaptive_root / "src/dflash_specblock/paper/official_spec.py"
    pinned_revisions = _literal_assignment(official_spec_path, "PINNED_MODEL_REVISIONS")

    if (not isinstance(spec["models"], list)
            or [(entry.get("id"), entry.get("adaptive_model_index"))
                for entry in spec["models"]] != list(zip(FORMAL_MODEL_IDS, (0, 1)))):
        raise ValueError("Formal integrated suite requires exactly Qwen3-4B and Qwen3-8B")
    models = []
    seen = set()
    expected_model_fields = {"id", "adaptive_model_index", "block_config", "target",
                             "target_revision", "draft", "draft_revision"}
    for entry in spec["models"]:
        if set(entry) != expected_model_fields or entry["id"] in seen:
            raise ValueError("Invalid or duplicate integrated model")
        seen.add(entry["id"])
        index = entry["adaptive_model_index"]
        pair = [entry["target"], entry["draft"]]
        if (not isinstance(index, int) or isinstance(index, bool)
                or index < 0 or index >= len(adaptive_config["models"])
                or adaptive_config["models"][index] != pair):
            raise ValueError("AdaptiveTree model index does not match the declared pair")
        if (pinned_revisions.get(entry["target"]) != entry["target_revision"]
                or pinned_revisions.get(entry["draft"]) != entry["draft_revision"]):
            raise ValueError("AdaptiveTree and integrated suite model revisions differ")
        config_path = (path.parent / entry["block_config"]).resolve()
        cfg = load_config(config_path)
        _validate_sampling_config(cfg)
        expected_model = {key:entry[key] for key in (
            "target", "target_revision", "draft", "draft_revision")}
        if any(cfg["model"].get(key) != value for key, value in expected_model.items()):
            raise ValueError("T=1 controls and AdaptiveTree model identities differ")
        models.append({**entry, "config_path":config_path, "config":cfg})

    reference = models[0]["config"]
    for model in models[1:]:
        cfg = model["config"]
        for field in ("datasets", "seeds", "max_new_tokens", "warmup_tokens", "scoring",
                      "evaluation", "method_order"):
            if cfg.get(field) != reference.get(field):
                raise ValueError("T=1 models must share data, seeds, limits, scoring, and order")
        if evaluation_policy(cfg["datasets"], cfg.get("evaluation")) != evaluation_policy(
                reference["datasets"], reference.get("evaluation")):
            raise ValueError("T=1 models must share the same sample selection")

    if model_ids is not None:
        if not model_ids or len(set(model_ids)) != len(model_ids) or set(model_ids) - seen:
            raise ValueError("Unknown or duplicate integrated model selection")
        models = [model for model in models if model["id"] in model_ids]
    return {"path":path, "spec":spec, "adaptive_root":adaptive_root,
            "adaptive_config_path":adaptive_config_path,
            "adaptive_config":adaptive_config, "models":models}


def audit_integrated_suite(path: Path, output: Path | None = None, model_ids=None) -> dict:
    suite = load_integrated_suite(path, model_ids)
    adaptive_root = suite["adaptive_root"]
    commit = suite["adaptive_config"]["official_commit"]
    adaptive_distribution = _verify_manifest(
        adaptive_root, adaptive_root / "SOURCE_SHA256.json"
    )
    adaptive_upstream = _verify_manifest(
        adaptive_root / "third_party/ddtree_pinned",
        adaptive_root / "third_party/ddtree_pinned/SOURCE_SHA256.json",
        commit=commit,
    )
    root_upstream = _verify_manifest(
        ROOT / "third_party/ddtree_pinned",
        ROOT / "third_party/ddtree_pinned/SOURCE_SHA256.json",
        commit=commit,
    )
    if adaptive_upstream != root_upstream:
        raise ValueError("AdaptiveTree and T=1 controls do not share the same pinned DDTree source")
    source_paths = [
        adaptive_root / "src/dflash_specblock/paper/controller.py",
        adaptive_root / "src/dflash_specblock/paper/adaptive_official.py",
        adaptive_root / "src/dflash_specblock/paper/official_worker.py",
        ROOT / "src/gbv_experiments/engine.py",
        ROOT / "src/gbv_experiments/sampling.py",
        ROOT / "src/gbv_experiments/config.py",
        ROOT / "src/gbv_experiments/runner.py",
        ROOT / "src/gbv_experiments/report.py",
        ROOT / "src/gbv_experiments/integrated_suite.py",
        ROOT / "third_party/ddtree_pinned/ddtree.py",
        ROOT / "third_party/ddtree_pinned/dflash.py",
    ]
    result = {
        "study":STUDY,
        "passed":True,
        "model_ids":[model["id"] for model in suite["models"]],
        "checks":{
            "immutable_model_revisions":True,
            "adaptive_official_t0_protocol":True,
            "adaptive_primary_is_canonical_corrected_method":True,
            "adaptive_token_divergence_recorded":True,
            "adaptive_strict_lossless_claim_disabled":True,
            "adaptive_balanced_method_positions":True,
            "adaptive_primary_table_same_target_backend_sdpa":True,
            "t1_same_target_and_draft_backend_sdpa":True,
            "t1_same_data_seeds_limits_and_order":True,
            "t1_only_target_dflash_ddtree_registered":True,
            "t1_official_dflash_greedy_draft":True,
            "t1_fp64_probability_math":True,
            "lazy_projection_excluded":True,
            "tree_block_verification_deferred_not_run":True,
            "pinned_ddtree_sources_identical":True,
        },
        "claim_boundaries":{
            "adaptive_t0_primary":"same-backend SDPA Target/DFlash/DDTree/AdaptiveTree only",
            "adaptive_t0_output_control":(
                "all-response performance includes pairwise exact-output rates; exact-output "
                "subset speed is a selection-biased diagnostic only"
            ),
            "adaptive_strict_lossless_claim_allowed":False,
            "adaptive_best_backend":"auxiliary table only; never mixed into the primary architecture table",
            "positive_temperature":"T=1 Target/DFlash/DDTree within one SDPA/SDPA engine; DFlash draft is greedy",
            "tree_block_verification":TREE_BLOCK_STATUS,
            "cross_protocol_speedup_pooling_allowed":False,
            "old_server_result_reuse_allowed":False,
            "reason":"T=0 AdaptiveTree and T=1 sampling controls use different sampling laws, data matrices, and draft attention backends",
        },
        "source_sha256":{str(source.relative_to(ROOT)):file_hash(source)
                         for source in source_paths},
        "adaptive_distribution_files":len(adaptive_distribution),
        "pinned_ddtree_commit":commit,
    }
    if output is not None:
        write_json(output, result)
    return result


def doctor_integrated_suite(path: Path, output: Path | None = None,
                            device="cuda:0", code_backend="docker") -> dict:
    """Fail before a long run when the fresh server cannot reproduce the suite."""
    import torch

    audit = audit_integrated_suite(path)
    if sys.version_info[:2] not in {(3, 10), (3, 11)}:
        raise RuntimeError("Integrated experiments require Python 3.10 or 3.11")
    expected_versions = {
        "transformers":"4.57.1",
        "huggingface-hub":"0.36.0",
        "datasets":"3.6.0",
        "accelerate":"1.10.1",
        "numpy":"2.2.6",
        "math-verify":"0.8.0",
        "sacrebleu":"2.5.1",
        "einops":"0.8.1",
        "safetensors":"0.6.2",
    }
    versions = {name:importlib.metadata.version(name) for name in expected_versions}
    wrong = {name:{"expected":expected_versions[name], "actual":actual}
             for name, actual in versions.items() if actual != expected_versions[name]}
    if wrong:
        raise RuntimeError(f"Pinned Python dependency mismatch: {wrong}")
    distribution_law_audit = _run_distribution_law_audit()
    selected_device = torch.device(device)
    if (not torch.cuda.is_available() or selected_device.type != "cuda"
            or selected_device.index not in (None, 0)):
        raise RuntimeError("Integrated formal runs require an NVIDIA CUDA GPU")
    if torch.__version__.split("+", 1)[0] != "2.8.0":
        raise RuntimeError(f"Expected torch 2.8.0, found {torch.__version__}")
    properties = torch.cuda.get_device_properties(selected_device)
    uuid = str(getattr(properties, "uuid", "unknown"))
    if uuid.lower() in {"", "none", "unknown", "unavailable"}:
        raise RuntimeError("CUDA build must expose a stable GPU UUID")
    allocation_audit = _gpu_allocation_gate(selected_device)
    flash_attn = importlib.import_module("flash_attn")
    for module in ("ninja", "loguru"):
        importlib.import_module(module)
    ninja_executable = shutil.which("ninja")
    if ninja_executable is None:
        raise RuntimeError(
            "Ninja must be on PATH for the official DDTree C++ cache compaction extension"
        )
    ninja_version = subprocess.check_output(
        [ninja_executable, "--version"], text=True
    ).strip()
    compiler = shutil.which("c++") or shutil.which("g++")
    if compiler is None:
        raise RuntimeError("A C++ compiler is required for official DDTree cache compaction")
    suite = load_integrated_suite(path)
    adaptive_env = os.environ.copy()
    adaptive_env["PYTHONPATH"] = str(suite["adaptive_root"] / "src")
    compact_check = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from dflash_specblock.paper.official_spec import upstream; "
                "assert upstream().ddtree.load_cpp_compact_module() is not None"
            ),
        ],
        cwd=suite["adaptive_root"],
        env=adaptive_env,
        text=True,
        capture_output=True,
    )
    if compact_check.returncode:
        detail = (compact_check.stderr or compact_check.stdout).strip()[-2000:]
        raise RuntimeError(
            f"Official DDTree C++ cache compaction preflight failed: {detail}"
        )
    docker_image = None
    code_evaluator_python = None
    code_evaluator_python_resolved = None
    code_evaluator_python_sha256 = None
    code_evaluator_python_runtime = None
    code_evaluator_pyvenv_cfg_sha256 = None
    if code_backend == "docker":
        docker_image = subprocess.check_output(
            ["docker", "image", "inspect", "gbv-code-eval:py311", "--format", "{{.Id}}"],
            text=True,
        ).strip()
    elif code_backend == "process":
        from .scoring import process_python_identity
        process_identity = process_python_identity()
        code_evaluator_python = process_identity["process_python"]
        code_evaluator_python_resolved = process_identity["process_python_resolved"]
        code_evaluator_python_sha256 = process_identity["process_python_sha256"]
        code_evaluator_python_runtime = process_identity["process_python_runtime"]
        code_evaluator_pyvenv_cfg_sha256 = process_identity[
            "process_pyvenv_cfg_sha256"
        ]
    else:
        raise ValueError("Unknown code scoring backend")
    # Exercise the actual production evaluator.  On root-run rental hosts this
    # also proves that the worker can set no-new-privileges and drop to nobody
    # before candidate code is compiled or executed.
    from .scoring import evaluate_code
    expected_process_uid = (
        65534
        if code_backend == "process" and os.name == "posix" and os.geteuid() == 0
        else os.geteuid()
        if code_backend == "process" and os.name == "posix"
        else None
    )
    if code_backend == "process":
        candidate = "import os\ndef evaluator_uid():\n    return os.geteuid()"
        assertion = f"assert candidate() == {expected_process_uid}"
    else:
        candidate = "def evaluator_uid():\n    return 1"
        assertion = "assert candidate() == 1"
    evaluator_self_test = evaluate_code(
        candidate,
        {
            "kind":"humaneval",
            "entry_point":"evaluator_uid",
            "prompt":"",
            "test":f"def check(candidate):\n    {assertion}",
        },
        backend=code_backend,
        timeout=3,
    )
    if evaluator_self_test != {"passed":True, "reason":"passed"}:
        raise RuntimeError(
            f"Production code evaluator self-test failed: {evaluator_self_test}"
        )
    result = {
        "passed":True,
        "fairness_audit":audit,
        "distribution_law_audit":distribution_law_audit,
        "python":platform.python_version(),
        "packages":versions,
        "torch":torch.__version__,
        "cuda":torch.version.cuda,
        "gpu":torch.cuda.get_device_name(torch.device(device)),
        "gpu_uuid":uuid,
        "gpu_memory_bytes":properties.total_memory,
        "gpu_allocation_audit":allocation_audit,
        "flash_attn":getattr(flash_attn, "__version__", "unknown"),
        "ninja_executable":ninja_executable,
        "ninja_version":ninja_version,
        "ddtree_cpp_compaction":True,
        "compiler":compiler,
        "code_backend":code_backend,
        "code_evaluator_python":code_evaluator_python,
        "code_evaluator_python_resolved":code_evaluator_python_resolved,
        "code_evaluator_python_sha256":code_evaluator_python_sha256,
        "code_evaluator_python_runtime":code_evaluator_python_runtime,
        "code_evaluator_pyvenv_cfg_sha256":code_evaluator_pyvenv_cfg_sha256,
        "code_evaluator_self_test":{
            **evaluator_self_test,
            "backend":code_backend,
            "expected_process_uid":expected_process_uid,
        },
        "code_isolation":(
            "Docker: network disabled, all capabilities dropped, no-new-privileges"
            if code_backend == "docker" else
            "process: sanitized environment and rlimits; root workers irreversibly drop to uid/gid 65534 with no-new-privileges"
        ),
        "docker_image":docker_image,
    }
    if output is not None:
        write_json(output, result)
    return result


def plan_integrated_suite(path: Path, model_ids=None) -> dict:
    suite = load_integrated_suite(path, model_ids)
    sampling_plans = {model["id"]:make_plan(model["config"])
                      for model in suite["models"]}
    config = suite["adaptive_config"]
    counts = config["sample_limits"]
    turns = sum(count * (2 if dataset == "mt-bench" else 1)
                for dataset, count in counts.items())
    adaptive_methods_per_turn = 2 + (2 + len(config["tree_budgets"])
                                      + len(config["variants"]))
    adaptive_calls = turns * len(suite["models"]) * adaptive_methods_per_turn
    sampling_records = sum(plan["expected_records"] for plan in sampling_plans.values())
    sampling_generations = sum(plan["expected_generations"] for plan in sampling_plans.values())
    return {
        "study":STUDY,
        "model_count":len(suite["models"]),
        "models":[model["id"] for model in suite["models"]],
        "fairness_audit":audit_integrated_suite(path, model_ids=model_ids),
        "adaptive_t0":{
            "temperature":0,
            "cases_per_model":sum(counts.values()),
            "turns_per_method_per_model":turns,
            "methods_per_turn_across_backends":adaptive_methods_per_turn,
            "generation_calls":adaptive_calls,
            "registered_variants":list(config["variants"]),
            "primary_method":"adaptive",
            "method_schema_version":2,
            "primary_backend":"sdpa",
            "greedy_audit_policy":suite["spec"]["adaptive"]["greedy_audit_policy"],
            "strict_lossless_claim_allowed":False,
            "all_response_table_includes_pairwise_exact_output_rates":True,
            "exact_output_subset_table_is_selection_biased_diagnostic":True,
            "method_order_policy":"balanced-rotation",
        },
        "positive_temperature_t1":{
            "temperatures":list(TEMPERATURES),
            "methods":list(SAMPLING_METHODS),
            "models":sampling_plans,
            "expected_records":sampling_records,
            "expected_generations":sampling_generations,
            "method_order_policy":"balanced_rotation",
        },
        "tree_block_verification":TREE_BLOCK_STATUS,
        "total_generation_calls":adaptive_calls + sampling_generations,
        "cross_protocol_aggregate":None,
        "old_server_results_imported":False,
        "note":"Use a fresh output directory on new hardware. This server runs T=0 AdaptiveTree and T=1 Target/DFlash/DDTree only; tree-block verification is deferred to a separate server and speedups are never pooled across protocols.",
    }


def _run_command(argv: list[str], *, cwd: Path, env=None) -> None:
    print(f"Running in {cwd}: {' '.join(argv)}", flush=True)
    subprocess.run(argv, cwd=cwd, env=env, check=True)


def run_integrated_suite(path: Path, adaptive_data_dir: Path, block_data_dir: Path,
                         output: Path, device="cuda:0", code_backend="docker",
                         model_ids=None) -> dict:
    import torch
    from .audit import audit
    from .data import prepare
    from .preflight import check_model
    from .report import report
    from .runner import run
    from .scoring import score_run, validate_gold

    suite = load_integrated_suite(path, model_ids)
    if tuple(model["id"] for model in suite["models"]) != FORMAL_MODEL_IDS:
        raise ValueError("Formal integrated runs cannot filter the 4B/8B model matrix")
    doctor = doctor_integrated_suite(path, device=device, code_backend=code_backend)
    properties = torch.cuda.get_device_properties(torch.device(device))
    uuid = str(getattr(properties, "uuid", "unknown"))
    if uuid.lower() in {"", "none", "unknown", "unavailable"}:
        raise RuntimeError("A stable GPU UUID is required for the integrated fairness contract")
    environment = {"device":device, "gpu":torch.cuda.get_device_name(torch.device(device)),
                   "gpu_uuid":uuid, "cuda":torch.version.cuda, "single_gpu":True}
    output.mkdir(parents=True, exist_ok=True)
    run_inputs = {
        "study":STUDY,
        "suite_sha256":file_hash(path.resolve()),
        "adaptive_data_dir":str(adaptive_data_dir.resolve()),
        "sampling_data_dir":str(block_data_dir.resolve()),
        "old_server_results_imported":False,
    }
    inputs_path = output / "run_inputs.json"
    if inputs_path.exists() and _load_json(inputs_path) != run_inputs:
        raise ValueError("Integrated input paths or suite changed; use a new run directory")
    write_json(inputs_path, run_inputs)
    environment_path = output / "environment.json"
    if environment_path.exists() and _load_json(environment_path) != environment:
        raise ValueError("GPU environment changed; use a new integrated run directory")
    write_json(environment_path, environment)
    write_json(output / "server_doctor.json", doctor)
    write_json(output / "plan.json", plan_integrated_suite(path, model_ids))
    write_json(output / "formal_matrix_audit.json", _formal_matrix_audit())
    audit_integrated_suite(path, output / "fairness_audit.json", model_ids)

    adaptive_env = os.environ.copy()
    adaptive_env["PYTHONPATH"] = str(suite["adaptive_root"] / "src")
    adaptive_base = [sys.executable, "-m", "dflash_specblock.paper"]
    _run_command(adaptive_base + ["prepare", "--config", str(suite["adaptive_config_path"]),
                 "--data-dir", str(adaptive_data_dir.resolve()), "--nproc-per-node", "1"],
                 cwd=suite["adaptive_root"], env=adaptive_env)

    # Finish every T=1 data, scoring, and real-checkpoint preflight before the
    # multi-day T=0 timing phase.  Unsupported evaluators, corrupt data, and
    # incompatible SDPA checkpoints therefore fail before formal timing begins.
    if suite["spec"]["positive_temperature"]["tree_block_verification"] != TREE_BLOCK_STATUS:
        raise RuntimeError("Tree-block verification must remain deferred on this server")
    first = suite["models"][0]["config"]
    prepare(first["datasets"], block_data_dir, first.get("evaluation"))
    audit(first, block_data_dir, [], output / "sampling_data_audit.json")
    validate_gold(block_data_dir, first["datasets"], output / "sampling_gold_audit.json",
                  code_backend, first["scoring"]["timeout_seconds"], first.get("evaluation"))
    for model in suite["models"]:
        cfg = model["config"]
        _validate_sampling_config(cfg)
        sampling_run_dir = output / "sampling_t1" / model["id"]
        check_model(
            cfg, sampling_run_dir / "gpu_preflight.json", device, code_backend
        )
        gc.collect()
        torch.cuda.empty_cache()

    # Exercise the nested official T=0 path with real 4B/8B checkpoints, Draft
    # FA2, both Target backends, C++ compaction, and every registered method
    # before collecting any formal timing.
    adaptive_preflight = {
        "passed":True,
        "gpu_uuid":doctor["gpu_uuid"],
        "nested_code_identity":_adaptive_code_identity(suite["adaptive_root"]),
        "models":{},
    }
    for model in suite["models"]:
        _gpu_allocation_gate(torch.device(device))
        smoke_dir = (output / "preflight" / f"adaptive_{model['id']}_smoke").resolve()
        smoke_common = [
            "--config", str(suite["adaptive_config_path"]),
            "--data-dir", str(adaptive_data_dir.resolve()),
            "--run-dir", str(smoke_dir),
            "--nproc-per-node", "1",
            "--model-index", str(model["adaptive_model_index"]),
            "--dataset", "gsm8k",
            "--smoke-count", "1",
            "--greedy-audit-policy", "record-bf16-mismatches",
            "--method-order-policy", "balanced-rotation",
        ]
        _run_command(
            adaptive_base + ["evaluate", *smoke_common],
            cwd=suite["adaptive_root"], env=adaptive_env,
        )
        _run_command(
            adaptive_base + ["summarize", *smoke_common],
            cwd=suite["adaptive_root"], env=adaptive_env,
        )
        smoke_summary = _load_json(smoke_dir / "tables.json")
        if (smoke_summary.get("protocol") != "ddtree_official_t0"
                or smoke_summary.get("official_samples") is not False
                or smoke_summary.get("method_schema_version") != 2
                or smoke_summary.get("primary_adaptive_method") != "adaptive"
                or smoke_summary.get("greedy_audit_policy") !=
                   "record-bf16-mismatches"
                or smoke_summary.get("method_order_policy") != "balanced-rotation"
                or len(smoke_summary.get("input_artifacts", [])) != 2):
            raise RuntimeError(f"Incomplete AdaptiveTree GPU smoke for {model['id']}")
        adaptive_preflight["models"][model["id"]] = {
            "run_dir":str(smoke_dir.relative_to(output.resolve())),
            "tables_sha256":file_hash(smoke_dir / "tables.json"),
            "protocol_identity":smoke_summary["protocol_identity"],
            "input_artifacts":smoke_summary["input_artifacts"],
        }
    write_json(output / "adaptive_gpu_preflight.json", adaptive_preflight)

    for model in suite["models"]:
        _gpu_allocation_gate(torch.device(device))
        run_dir = (output / "adaptive" / model["id"]).resolve()
        common = ["--config", str(suite["adaptive_config_path"]),
                  "--data-dir", str(adaptive_data_dir.resolve()),
                  "--run-dir", str(run_dir), "--nproc-per-node", "1",
                  "--model-index", str(model["adaptive_model_index"]),
                  "--greedy-audit-policy",
                  suite["spec"]["adaptive"]["greedy_audit_policy"],
                  "--method-order-policy", "balanced-rotation"]
        _run_command(adaptive_base + ["evaluate", *common],
                     cwd=suite["adaptive_root"], env=adaptive_env)
        _run_command(adaptive_base + ["summarize", *common],
                     cwd=suite["adaptive_root"], env=adaptive_env)

    for model in suite["models"]:
        _gpu_allocation_gate(torch.device(device))
        cfg = model["config"]
        run_dir = output / "sampling_t1" / model["id"]
        run(cfg, block_data_dir, run_dir, device)
        gc.collect()
        torch.cuda.empty_cache()
        settings = cfg["scoring"]
        score_run(run_dir, block_data_dir, code_backend, settings["workers"],
                  settings["timeout_seconds"], settings["lcb_timeout_seconds"])
        report(run_dir, run_dir / "report", cfg["bootstrap_samples"])

    integrated_report = report_integrated_suite(
        path, output, output / "report", model_ids
    )
    result = {"study":STUDY, "complete":True,
              "model_ids":[model["id"] for model in suite["models"]],
              "fairness_gate_passed":integrated_report["fairness_gate_passed"],
              "strict_lossless_gate_passed":integrated_report[
                  "strict_lossless_gate_passed"
              ],
              "fairness_audit":"fairness_audit.json",
              "adaptive_results":"adaptive/*/tables.json",
              "sampling_t1_results":"sampling_t1/*/report/summary.json",
              "tree_block_verification":TREE_BLOCK_STATUS,
              "integrated_report":"report/integrated_results.md",
              "old_server_results_imported":False,
              "cross_protocol_speedup_pooling_allowed":False}
    write_json(output / "completed.json", result)
    return result


def validate_block_order(run_dir: Path) -> dict:
    """Read-only post-run check for complete balanced variant positions."""
    manifest = _load_json(run_dir / "run_manifest.json")
    rows = read_jsonl(run_dir / "results.jsonl")
    if manifest.get("method_order", {}).get("policy") != "balanced_rotation":
        raise ValueError("Run did not use balanced rotation")
    names = {entry["variant"]["name"] for entry in manifest["variants"]}
    positions = {}
    groups = {}
    for row in rows:
        name = row["variant"]
        position = row.get("method_execution_position")
        ordinal = row.get("method_order_ordinal")
        if name not in names or not isinstance(position, int) or not isinstance(ordinal, int):
            raise ValueError("Missing balanced-order evidence")
        positions.setdefault(row["dataset"], {method:[] for method in names})[
            name
        ].append(position)
        key = row["dataset"], row["source_id"], row["seed"]
        groups.setdefault(key, []).append((name, position, ordinal))
    for group in groups.values():
        if ({name for name, _, _ in group} != names
                or {position for _, position, _ in group} != set(range(len(names)))
                or len({ordinal for _, _, ordinal in group}) != 1):
            raise ValueError("A prompt does not contain one complete balanced method rotation")
    for dataset, by_name in positions.items():
        for values in by_name.values():
            counts = [values.count(position) for position in range(len(names))]
            if max(counts) - min(counts) > 1:
                raise ValueError(
                    f"T=1 method positions are not balanced within dataset {dataset}"
                )
    return {"passed":True, "prompts":len(groups), "methods":len(names)}


def report_integrated_suite(path: Path, run_dir: Path, output: Path,
                            model_ids=None) -> dict:
    """Validate finished artifacts and write separate, non-pooled result tables."""
    suite = load_integrated_suite(path, model_ids)
    if tuple(model["id"] for model in suite["models"]) != FORMAL_MODEL_IDS:
        raise ValueError("Formal integrated reports cannot filter the 4B/8B model matrix")
    run_inputs = _load_json(run_dir / "run_inputs.json")
    if (run_inputs.get("study") != STUDY
            or run_inputs.get("suite_sha256") != file_hash(path.resolve())
            or run_inputs.get("old_server_results_imported") is not False
            or not Path(run_inputs.get("adaptive_data_dir", "")).is_absolute()
            or not Path(run_inputs.get("sampling_data_dir", "")).is_absolute()):
        raise ValueError("Integrated report input lineage is invalid")
    matrix_audit = _load_json(run_dir / "formal_matrix_audit.json")
    if matrix_audit != _formal_matrix_audit():
        raise ValueError("Integrated report is not bound to the current formal matrix audit")
    audit = audit_integrated_suite(path, model_ids=model_ids)
    doctor = _load_json(run_dir / "server_doctor.json")
    doctor_audit = doctor.get("fairness_audit", {})
    if (doctor.get("passed") is not True
            or doctor_audit.get("passed") is not True
            or doctor_audit.get("source_sha256") != audit["source_sha256"]
            or doctor_audit.get("checks") != audit["checks"]
            or doctor_audit.get("claim_boundaries") != audit["claim_boundaries"]
            or doctor_audit.get("pinned_ddtree_commit") != audit["pinned_ddtree_commit"]
            or audit["model_ids"] != doctor_audit.get("model_ids")):
        raise ValueError("Integrated report requires a passing server doctor")
    evaluator_evidence = doctor.get("code_evaluator_self_test", {})
    if (evaluator_evidence.get("passed") is not True
            or evaluator_evidence.get("reason") != "passed"
            or evaluator_evidence.get("backend") != doctor.get("code_backend")
            or (doctor.get("code_backend") == "process"
                and not isinstance(evaluator_evidence.get("expected_process_uid"), int))):
        raise ValueError("Integrated report requires a real code-evaluator self-test")
    allocation_evidence = doctor.get("gpu_allocation_audit", {})
    if (allocation_evidence.get("passed") is not True
            or allocation_evidence.get("gpu_uuid") != doctor.get("gpu_uuid")
            or allocation_evidence.get("foreign_compute_pids") != []):
        raise ValueError("Integrated report requires an uncontended-GPU doctor gate")
    _validate_distribution_law_evidence(doctor.get("distribution_law_audit", {}))
    _validate_adaptive_gpu_preflight(
        _load_json(run_dir / "adaptive_gpu_preflight.json"), run_dir, suite, doctor
    )
    adaptive_rows = []
    adaptive_exact_subset_rows = []
    adaptive_auxiliary_rows = []
    sampling_rows = []
    gates = {}
    adaptive_data_dir = Path(run_inputs["adaptive_data_dir"])
    sampling_data_audit = _load_json(run_dir / "sampling_data_audit.json")
    sampling_gold_audit = _load_json(run_dir / "sampling_gold_audit.json")
    adaptive_lineages = []
    sampling_lineages = []
    for model in suite["models"]:
        model_id = model["id"]
        adaptive_run = run_dir / "adaptive" / model_id
        adaptive_path = adaptive_run / "tables.json"
        adaptive = _load_json(adaptive_path)
        controlled, exact_subset, auxiliary = _validate_adaptive_summary(
            adaptive, adaptive_run, model, suite, doctor
        )
        recomputed_adaptive = _recompute_adaptive_summary(
            adaptive_run, adaptive_data_dir, model, suite, output
        )
        if adaptive != recomputed_adaptive:
            raise ValueError(
                f"AdaptiveTree tables differ from raw artifacts for {model_id}"
            )
        adaptive_lineages.append((
            adaptive.get("dataset_manifest"), adaptive.get("source_lock")
        ))
        numerical = adaptive.get("numerical_audit", {})
        adaptive_rows.extend({"model_id":model_id, **row} for row in controlled)
        adaptive_exact_subset_rows.extend(
            {"model_id":model_id, **row}
            for row in exact_subset
        )
        adaptive_auxiliary_rows.extend(
            {"model_id":model_id, **row} for row in auxiliary
        )

        sampling_run = run_dir / "sampling_t1" / model_id
        preflight = _load_json(sampling_run / "gpu_preflight.json")
        _validate_t1_preflight(preflight, model["config"], doctor)
        order_audit = validate_block_order(sampling_run)
        sampling_report = _load_json(sampling_run / "report/summary.json")
        manifest = _load_json(sampling_run / "run_manifest.json")
        _validate_sampling_manifest(manifest, model["config"], doctor)
        _validate_sampling_data_audits(
            sampling_data_audit, sampling_gold_audit, manifest,
            model["config"], doctor,
        )
        sampling_lineages.append((
            manifest.get("data_manifest"), manifest.get("prompt_ids")
        ))
        records = read_jsonl(sampling_run / "results.jsonl")
        scores = read_jsonl(sampling_run / "scores.jsonl")
        coverage = validate_results(manifest, records, scores)
        if sampling_report.get("coverage") != coverage:
            raise ValueError(f"T=1 summary coverage differs from raw records for {model_id}")
        recomputed_rows = summarize_results(
            manifest, records, scores, model["config"]["bootstrap_samples"]
        )
        if sampling_report.get("rows") != recomputed_rows:
            raise ValueError(f"T=1 summary rows differ from raw records for {model_id}")
        scoring_manifest = _load_json(sampling_run / "scoring_manifest.json")
        _validate_scoring_manifest(
            scoring_manifest, scores, model["config"], doctor, manifest
        )
        rows = _validate_sampling_summary(sampling_report, manifest, model["config"])
        if ({row.get("method") for row in rows} != set(SAMPLING_METHODS)
                or {row.get("temperature") for row in rows} != set(TEMPERATURES)):
            raise ValueError(f"Unregistered method or temperature in T=1 results for {model_id}")
        sampling_rows.extend({"model_id":model_id, **row} for row in rows)
        gates[model_id] = {
            "adaptive_same_backend_balanced_divergence_audited":True,
            "t1_gpu_preflight_passed":True,
            "t1_data_and_gold_audits_passed":True,
            "t1_controls_complete_balanced":order_audit["passed"],
            "tree_block_verification_deferred_not_run":True,
            "adaptive_strict_lossless":(
                adaptive.get("greedy_audit_policy") == "strict"
                and numerical.get("mismatching_responses") == 0
            ),
        }

    if (len({json.dumps(value, sort_keys=True) for value in adaptive_lineages}) != 1
            or len({json.dumps(value, sort_keys=True) for value in sampling_lineages}) != 1):
        raise ValueError("4B and 8B formal results did not use identical data snapshots")

    comparison_keys = (
        "adaptive_same_backend_balanced_divergence_audited",
        "t1_gpu_preflight_passed",
        "t1_data_and_gold_audits_passed",
        "t1_controls_complete_balanced",
        "tree_block_verification_deferred_not_run",
    )
    result = {
        "study":STUDY,
        "fairness_gate_passed":all(
            all(values[key] for key in comparison_keys) for values in gates.values()
        ),
        "strict_lossless_gate_passed":all(
            values["adaptive_strict_lossless"] for values in gates.values()
        ),
        "gates":gates,
        "fairness_audit":audit,
        "server_doctor_distribution_law_audit":doctor["distribution_law_audit"],
        "primary_tables":{
            "adaptive_t0_same_backend_sdpa_with_exact_output_rates":adaptive_rows,
            "sampling_t1_target_dflash_ddtree_sdpa":sampling_rows,
        },
        "auxiliary_tables":{
            "adaptive_t0_pairwise_exact_output_subset_selection_biased":
                adaptive_exact_subset_rows,
            "adaptive_t0_best_available_backend_not_for_architecture_claims":
                adaptive_auxiliary_rows,
        },
        "tree_block_verification":TREE_BLOCK_STATUS,
        "cross_protocol_speedup_pooling_allowed":False,
    }
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "integrated_results.json", result)
    for filename, rows in (
            ("adaptive_t0_controlled_sdpa.csv", adaptive_rows),
            ("adaptive_t0_exact_output_subset_diagnostic.csv", adaptive_exact_subset_rows),
            ("adaptive_t0_best_backend_auxiliary.csv", adaptive_auxiliary_rows),
            ("sampling_t1_target_dflash_ddtree_sdpa.csv", sampling_rows)):
        if not rows:
            raise ValueError(f"No rows for {filename}")
        with (output / filename).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    def fmt(value):
        return "--" if value is None else f"{value:.4f}"
    def sampling_label(row):
        return {
            "target":"Target",
            "dflash":"DFlash (greedy Draft)",
            "ddtree":"DDTree-B45 (L=15)",
        }[row["method"]]

    lines = ["# AdaptiveTree、DDTree 与 DFlash：公平实验汇总", "",
             "所有主表均已通过同 revision、同后端、完整性、均衡顺序和差异审计门禁。",
             "T=0 在 BF16 执行中观察到的输出差异被保留并逐方法报告；"
             "本报告不推断差异成因，也不声称严格无损。",
             "T=0 与 T=1 属于不同协议，禁止汇总成一个跨协议加速比。",
             "树状块验证状态：deferred_not_run；本服务器未运行、未汇报该方法。", ""]
    lines += ["## T=0 主表：统一 SDPA 后端（全样本，非无损声明）", "",
              "| 模型 | 数据集 | 方法 | 相对 Target | 相对最佳 DDTree | 接受长度 | 精确输出率 |",
              "|---|---|---|---:|---:|---:|---:|"]
    lines += [f"| {row['model_id']} | {row['dataset']} | {row['method']} | "
              f"{row['speedup_vs_target']:.4f}× | {row['speedup_vs_best_ddtree']:.4f}× | "
              f"{row['mean_acceptance_length']:.3f} | {row['exact_output_rate']:.2%} |"
              for row in adaptive_rows]
    lines += ["", "## T=0 完全相同输出配对子集（选择偏差诊断）", "",
              "| 模型 | 数据集 | 方法 | 相同输出数/总数 | 子集相对 Target |",
              "|---|---|---|---:|---:|"]
    lines += [f"| {row['model_id']} | {row['dataset']} | {row['method']} | "
              f"{row['exact_output_responses']}/{row['responses']} | "
              f"{fmt(row['exact_subset_speedup_vs_target'])}× |"
              for row in adaptive_exact_subset_rows]
    lines += ["", "## T=1 主表：Target / DFlash / DDTree，统一 SDPA/SDPA", "",
              "DDTree 固定使用 L=15、B=45 和 FP64 概率计算；DFlash 使用贪心 Draft。", "",
              "接受/提议比的分母在 DFlash（路径 token）与 DDTree（树节点）间不同，故不作横向比较；下表统一报告每次 Target 验证接受的 token 数。", "",
              "| 模型 | 数据集 | 温度 | 方法 | Decode tok/s | 相对 Target | 95% CI | 质量 | 平均生成长度 | 每次验证接受 token |",
              "|---|---|---:|---|---:|---:|---:|---:|---:|---:|"]
    lines += [f"| {row['model_id']} | {row['dataset']} | {row['temperature']:.1f} | "
              f"{sampling_label(row)} | {fmt(row['decode_tps'])} | {fmt(row['speedup_vs_ar'])}× | "
              f"[{fmt(row['speedup_ci_low'])}, {fmt(row['speedup_ci_high'])}] | "
              f"{fmt(row['quality'])} | {fmt(row['mean_generation_length'])} | "
              f"{fmt(row['accepted_per_verify'])} |"
              for row in sampling_rows]
    (output / "integrated_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result
