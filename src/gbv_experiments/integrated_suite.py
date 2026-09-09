"""Fair, fail-closed orchestration for AdaptiveTree and stochastic block decoding."""
from __future__ import annotations

import ast
import csv
import gc
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

from .common import ROOT, file_hash, read_jsonl, write_json
from .config import build_variants, load_config
from .data import evaluation_policy
from .runner import make_plan


STUDY = "adaptive_tree_and_stochastic_block_verification"
TEMPERATURES = (0.3, 0.6, 1.0)
BLOCK_METHODS = ("target", "dflash", "ddtree", "ddtree_lazy_projection")
DIAGNOSTICS = (
    "cost_attributed_no_exploration",
    "cost_attributed_no_exploration_b256",
)


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


def _tag(temperature: float) -> str:
    return {0.3: "t0p3", 0.6: "t0p6", 1.0: "t1p0"}[temperature]


def _group_tag(temperature: float) -> str:
    return {0.3: "t03", 0.6: "t06", 1.0: "t10"}[temperature]


def _expected_name(method: str, temperature: float) -> str:
    prefix = "lazy_projection" if method == "ddtree_lazy_projection" else method
    return f"{prefix}_{_tag(temperature)}"


def _matched_variant(left: dict, right: dict) -> bool:
    ignored = {"name", "method"}
    return all(left[key] == right[key] for key in left if key not in ignored)


def _validate_block_config(cfg: dict) -> None:
    if "explicit_variants" not in cfg:
        raise ValueError("Integrated block experiments must explicitly freeze every variant")
    model = cfg["model"]
    if (model.get("dtype"), model.get("target_attention"), model.get("draft_attention"),
            model.get("enable_thinking"), model.get("allow_tf32")) != (
            "bfloat16", "sdpa", "sdpa", False, False):
        raise ValueError("Block methods must share BF16, SDPA/SDPA, no thinking, and TF32 disabled")
    if cfg.get("method_order") != {"policy": "balanced_rotation", "seed": 20260909}:
        raise ValueError("Block methods require the frozen balanced-rotation schedule")
    entries = build_variants(cfg)
    by_name = {entry["variant"]["name"]: entry for entry in entries}
    expected = {_expected_name(method, temperature)
                for temperature in TEMPERATURES for method in BLOCK_METHODS}
    if set(by_name) != expected:
        raise ValueError(f"Wrong block matrix; missing={sorted(expected-set(by_name))}, "
                         f"extra={sorted(set(by_name)-expected)}")
    for temperature in TEMPERATURES:
        variants = {
            method:by_name[_expected_name(method, temperature)]["variant"]
            for method in BLOCK_METHODS
        }
        for method, variant in variants.items():
            if set(by_name[variant["name"]]["groups"]) != {"main", _group_tag(temperature)}:
                raise ValueError("Every block variant requires main and temperature groups")
            expected_draft_temperature = None if method == "target" else temperature
            if (variant["method"] != method or variant["temperature"] != temperature
                    or variant["draft_temperature"] != expected_draft_temperature
                    or variant["paths"] != 1 or variant["length"] != 15
                    or variant["tree_budget"] != 45
                    or variant["probability_dtype"] != "float64"):
                raise ValueError(f"Unfair or invalid block control: {variant['name']}")
        reference = variants["ddtree"]
        for method in ("dflash", "ddtree_lazy_projection"):
            if not _matched_variant(reference, variants[method]):
                raise ValueError(f"{method} and DDTree differ in more than implementation")


def load_integrated_suite(path: Path, model_ids=None) -> dict:
    path = path.resolve()
    spec = _load_json(path)
    required = {"schema", "study", "adaptive", "block", "models"}
    if set(spec) != required or spec.get("schema") != 1 or spec.get("study") != STUDY:
        raise ValueError("Invalid integrated suite schema or study ID")
    adaptive_spec = spec["adaptive"]
    if set(adaptive_spec) != {"project", "config", "temperature", "diagnostic_variants",
                             "greedy_audit_policy", "method_order_policy"}:
        raise ValueError("Invalid AdaptiveTree suite section")
    if (adaptive_spec["temperature"] != 0
            or tuple(adaptive_spec["diagnostic_variants"]) != DIAGNOSTICS
            or adaptive_spec["greedy_audit_policy"] != "strict"
            or adaptive_spec["method_order_policy"] != "balanced-rotation"):
        raise ValueError("Integrated AdaptiveTree requires both diagnostics, strict tokens, and balanced order")
    block_spec = spec["block"]
    if (set(block_spec) != {"temperatures", "methods", "length", "tree_budget",
                            "probability_dtype", "method_order_policy"}
            or tuple(block_spec["temperatures"]) != TEMPERATURES
            or tuple(block_spec["methods"]) != BLOCK_METHODS
            or (block_spec["length"], block_spec["tree_budget"],
                block_spec["probability_dtype"], block_spec["method_order_policy"])
               != (15, 45, "float64", "balanced_rotation")):
        raise ValueError("Invalid stochastic block suite controls")

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

    if not isinstance(spec["models"], list) or not spec["models"]:
        raise ValueError("Integrated suite requires models")
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
        _validate_block_config(cfg)
        expected_model = {key:entry[key] for key in (
            "target", "target_revision", "draft", "draft_revision")}
        if any(cfg["model"].get(key) != value for key, value in expected_model.items()):
            raise ValueError("Block and AdaptiveTree model identities differ")
        models.append({**entry, "config_path":config_path, "config":cfg})

    reference = models[0]["config"]
    for model in models[1:]:
        cfg = model["config"]
        for field in ("datasets", "seeds", "max_new_tokens", "warmup_tokens", "scoring",
                      "evaluation", "method_order"):
            if cfg.get(field) != reference.get(field):
                raise ValueError("Block models must share data, seeds, limits, scoring, and order")
        if evaluation_policy(cfg["datasets"], cfg.get("evaluation")) != evaluation_policy(
                reference["datasets"], reference.get("evaluation")):
            raise ValueError("Block models must share the same sample selection")

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
        raise ValueError("AdaptiveTree and block suites do not share the same pinned DDTree source")
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
            "adaptive_strict_token_gate":True,
            "adaptive_balanced_method_positions":True,
            "adaptive_primary_table_same_target_backend_sdpa":True,
            "block_same_target_and_draft_backend_sdpa":True,
            "block_same_data_seeds_limits_and_order":True,
            "ddtree_and_lazy_projection_matched_except_implementation":True,
            "fp64_probability_math_at_positive_temperature":True,
            "pinned_ddtree_sources_identical":True,
        },
        "claim_boundaries":{
            "adaptive_t0_primary":"same-backend SDPA Target/DFlash/DDTree/AdaptiveTree only",
            "adaptive_best_backend":"auxiliary table only; never mixed into the primary architecture table",
            "stochastic_block":"T=0.3/0.6/1.0 within one SDPA/SDPA engine",
            "cross_protocol_speedup_pooling_allowed":False,
            "old_server_result_reuse_allowed":False,
            "reason":"T=0 AdaptiveTree and T>0 block decoding use different sampling laws, data matrices, and draft attention backends",
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
    if not torch.cuda.is_available() or torch.device(device).type != "cuda":
        raise RuntimeError("Integrated formal runs require an NVIDIA CUDA GPU")
    if torch.__version__.split("+", 1)[0] != "2.8.0":
        raise RuntimeError(f"Expected torch 2.8.0, found {torch.__version__}")
    properties = torch.cuda.get_device_properties(torch.device(device))
    uuid = str(getattr(properties, "uuid", "unknown"))
    if uuid.lower() in {"", "none", "unknown", "unavailable"}:
        raise RuntimeError("CUDA build must expose a stable GPU UUID")
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
    if code_backend == "docker":
        docker_image = subprocess.check_output(
            ["docker", "image", "inspect", "gbv-code-eval:py311", "--format", "{{.Id}}"],
            text=True,
        ).strip()
    elif code_backend == "process":
        code_evaluator_python = str(Path(os.environ.get(
            "GBV_PROCESS_PYTHON", sys.executable
        )).resolve())
        if os.name == "posix" and os.geteuid() == 0:
            resolved = Path(code_evaluator_python)
            if resolved == Path("/root") or Path("/root") in resolved.parents:
                raise RuntimeError(
                    "Root process evaluation requires GBV_PROCESS_PYTHON outside /root"
                )
        subprocess.run(
            [code_evaluator_python, "-I", "-c", "import numpy; print(numpy.__version__)"],
            check=True, stdout=subprocess.DEVNULL,
        )
    else:
        raise ValueError("Unknown code scoring backend")
    result = {
        "passed":True,
        "fairness_audit":audit,
        "python":platform.python_version(),
        "packages":versions,
        "torch":torch.__version__,
        "cuda":torch.version.cuda,
        "gpu":torch.cuda.get_device_name(torch.device(device)),
        "gpu_uuid":uuid,
        "gpu_memory_bytes":properties.total_memory,
        "flash_attn":getattr(flash_attn, "__version__", "unknown"),
        "ninja_executable":ninja_executable,
        "ninja_version":ninja_version,
        "ddtree_cpp_compaction":True,
        "compiler":compiler,
        "code_backend":code_backend,
        "code_evaluator_python":code_evaluator_python,
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
    block_plans = {model["id"]:make_plan(model["config"])
                   for model in suite["models"]}
    config = suite["adaptive_config"]
    counts = config["sample_limits"]
    turns = sum(count * (2 if dataset == "mt-bench" else 1)
                for dataset, count in counts.items())
    adaptive_methods_per_turn = 2 + (2 + len(config["tree_budgets"])
                                      + len(config["variants"]) + len(DIAGNOSTICS))
    adaptive_calls = turns * len(suite["models"]) * adaptive_methods_per_turn
    block_records = sum(plan["expected_records"] for plan in block_plans.values())
    block_generations = sum(plan["expected_generations"] for plan in block_plans.values())
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
            "diagnostics":list(DIAGNOSTICS),
            "primary_backend":"sdpa",
            "method_order_policy":"balanced-rotation",
        },
        "stochastic_block":{
            "temperatures":list(TEMPERATURES),
            "methods":list(BLOCK_METHODS),
            "models":block_plans,
            "expected_records":block_records,
            "expected_generations":block_generations,
            "method_order_policy":"balanced_rotation",
        },
        "total_generation_calls":adaptive_calls + block_generations,
        "cross_protocol_aggregate":None,
        "old_server_results_imported":False,
        "note":"Use a fresh output directory on new hardware. Families are launched and reported together, but speedups are never pooled across T=0 and T>0 protocols.",
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
    doctor = doctor_integrated_suite(path, device=device, code_backend=code_backend)
    properties = torch.cuda.get_device_properties(torch.device(device))
    uuid = str(getattr(properties, "uuid", "unknown"))
    if uuid.lower() in {"", "none", "unknown", "unavailable"}:
        raise RuntimeError("A stable GPU UUID is required for the integrated fairness contract")
    environment = {"device":device, "gpu":torch.cuda.get_device_name(torch.device(device)),
                   "gpu_uuid":uuid, "cuda":torch.version.cuda, "single_gpu":True}
    output.mkdir(parents=True, exist_ok=True)
    environment_path = output / "environment.json"
    if environment_path.exists() and _load_json(environment_path) != environment:
        raise ValueError("GPU environment changed; use a new integrated run directory")
    write_json(environment_path, environment)
    write_json(output / "server_doctor.json", doctor)
    write_json(output / "plan.json", plan_integrated_suite(path, model_ids))
    audit_integrated_suite(path, output / "fairness_audit.json", model_ids)

    adaptive_env = os.environ.copy()
    adaptive_env["PYTHONPATH"] = str(suite["adaptive_root"] / "src")
    adaptive_base = [sys.executable, "-m", "dflash_specblock.paper"]
    _run_command(adaptive_base + ["prepare", "--config", str(suite["adaptive_config_path"]),
                 "--data-dir", str(adaptive_data_dir.resolve()), "--nproc-per-node", "1"],
                 cwd=suite["adaptive_root"], env=adaptive_env)
    for model in suite["models"]:
        run_dir = (output / "adaptive" / f"{model['id']}_diagnostic").resolve()
        common = ["--config", str(suite["adaptive_config_path"]),
                  "--data-dir", str(adaptive_data_dir.resolve()),
                  "--run-dir", str(run_dir), "--nproc-per-node", "1",
                  "--model-index", str(model["adaptive_model_index"]),
                  "--greedy-audit-policy", "strict",
                  "--method-order-policy", "balanced-rotation",
                  "--experimental-cost-attribution", "--experimental-extended-budgets"]
        _run_command(adaptive_base + ["evaluate", *common],
                     cwd=suite["adaptive_root"], env=adaptive_env)
        _run_command(adaptive_base + ["summarize", *common],
                     cwd=suite["adaptive_root"], env=adaptive_env)

    first = suite["models"][0]["config"]
    prepare(first["datasets"], block_data_dir, first.get("evaluation"))
    audit(first, block_data_dir, [], output / "block_data_audit.json")
    validate_gold(block_data_dir, first["datasets"], output / "block_gold_audit.json",
                  code_backend, first["scoring"]["timeout_seconds"], first.get("evaluation"))
    for model in suite["models"]:
        cfg = model["config"]
        run_dir = output / "block" / model["id"]
        check_model(cfg, run_dir / "gpu_preflight.json", device, code_backend)
        gc.collect()
        torch.cuda.empty_cache()
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
              "fairness_audit":"fairness_audit.json",
              "adaptive_results":"adaptive/*_diagnostic/tables.json",
              "block_results":"block/*/report/summary.json",
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
    positions = {name:[] for name in names}
    groups = {}
    for row in rows:
        name = row["variant"]
        position = row.get("method_execution_position")
        ordinal = row.get("method_order_ordinal")
        if name not in names or not isinstance(position, int) or not isinstance(ordinal, int):
            raise ValueError("Missing balanced-order evidence")
        positions[name].append(position)
        key = row["dataset"], row["source_id"], row["seed"]
        groups.setdefault(key, []).append((name, position, ordinal))
    for group in groups.values():
        if ({name for name, _, _ in group} != names
                or {position for _, position, _ in group} != set(range(len(names)))
                or len({ordinal for _, _, ordinal in group}) != 1):
            raise ValueError("A prompt does not contain one complete balanced method rotation")
    for values in positions.values():
        counts = [values.count(position) for position in range(len(names))]
        if max(counts) - min(counts) > 1:
            raise ValueError("Block method positions are not globally balanced")
    return {"passed":True, "prompts":len(groups), "methods":len(names)}


def report_integrated_suite(path: Path, run_dir: Path, output: Path,
                            model_ids=None) -> dict:
    """Validate finished artifacts and write separate, non-pooled result tables."""
    suite = load_integrated_suite(path, model_ids)
    audit = audit_integrated_suite(path, model_ids=model_ids)
    adaptive_rows = []
    adaptive_auxiliary_rows = []
    block_rows = []
    gates = {}
    for model in suite["models"]:
        model_id = model["id"]
        adaptive_path = run_dir / "adaptive" / f"{model_id}_diagnostic" / "tables.json"
        adaptive = _load_json(adaptive_path)
        numerical = adaptive.get("numerical_audit", {})
        controlled = adaptive.get("controlled_same_backend_rows", [])
        adaptive_gate = (
            adaptive.get("controlled_comparison_gate_passed") is True
            and adaptive.get("greedy_audit_policy") == "strict"
            and adaptive.get("method_order_policy") == "balanced-rotation"
            and numerical.get("mismatching_responses") == 0
            and controlled
            and all(row.get("target_baseline_backend") == "sdpa"
                    and row.get("method_backend") == "sdpa" for row in controlled)
        )
        if not adaptive_gate:
            raise ValueError(f"AdaptiveTree fairness gate failed for {model_id}")
        adaptive_rows.extend({"model_id":model_id, **row} for row in controlled)
        adaptive_auxiliary_rows.extend(
            {"model_id":model_id, **row} for row in adaptive.get("rows", [])
        )

        block_run = run_dir / "block" / model_id
        order_audit = validate_block_order(block_run)
        block_report = _load_json(block_run / "report/summary.json")
        manifest = _load_json(block_run / "run_manifest.json")
        expected_variants = build_variants(model["config"])
        if (block_report.get("coverage", {}).get("complete") is not True
                or manifest.get("model") != model["config"]["model"]
                or manifest.get("variants") != expected_variants
                or manifest.get("method_order") != model["config"]["method_order"]):
            raise ValueError(f"Block result contract failed for {model_id}")
        rows = block_report.get("rows", [])
        if len(rows) != len(model["config"]["datasets"]) * len(expected_variants):
            raise ValueError(f"Incomplete block summary for {model_id}")
        block_rows.extend({"model_id":model_id, **row} for row in rows)
        gates[model_id] = {"adaptive_same_backend_strict_balanced":True,
                           "block_complete_balanced":order_audit["passed"]}

    result = {
        "study":STUDY,
        "fairness_gate_passed":all(all(values.values()) for values in gates.values()),
        "gates":gates,
        "fairness_audit":audit,
        "primary_tables":{
            "adaptive_t0_same_backend_sdpa":adaptive_rows,
            "stochastic_block_t_positive_sdpa":block_rows,
        },
        "auxiliary_tables":{
            "adaptive_t0_best_available_backend_not_for_architecture_claims":
                adaptive_auxiliary_rows,
        },
        "cross_protocol_speedup_pooling_allowed":False,
    }
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "integrated_results.json", result)
    for filename, rows in (
            ("adaptive_t0_controlled_sdpa.csv", adaptive_rows),
            ("adaptive_t0_best_backend_auxiliary.csv", adaptive_auxiliary_rows),
            ("stochastic_block_t_positive_sdpa.csv", block_rows)):
        if not rows:
            raise ValueError(f"No rows for {filename}")
        with (output / filename).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    lines = ["# AdaptiveTree 与块解码：公平实验汇总", "",
             "所有主表均已通过同 revision、同后端、完整性、均衡顺序和正确性门禁。",
             "T=0 与 T>0 属于不同协议，禁止汇总成一个跨协议加速比。", ""]
    lines += ["## T=0 主表：统一 SDPA 后端", "",
              "| 模型 | 数据集 | 方法 | 相对 Target | 相对最佳 DDTree | 接受长度 |",
              "|---|---|---|---:|---:|---:|"]
    lines += [f"| {row['model_id']} | {row['dataset']} | {row['method']} | "
              f"{row['speedup_vs_target']:.4f}× | {row['speedup_vs_best_ddtree']:.4f}× | "
              f"{row['mean_acceptance_length']:.3f} |" for row in adaptive_rows]
    lines += ["", "## T>0 主表：统一 SDPA/SDPA", "",
              "| 模型 | 数据集 | 温度 | 方法 | Decode tok/s | 相对 Target | 95% CI |",
              "|---|---|---:|---|---:|---:|---:|"]
    def fmt(value):
        return "--" if value is None else f"{value:.4f}"
    lines += [f"| {row['model_id']} | {row['dataset']} | {row['temperature']:.1f} | "
              f"{row['variant']} | {fmt(row['decode_tps'])} | {fmt(row['speedup_vs_ar'])}× | "
              f"[{fmt(row['speedup_ci_low'])}, {fmt(row['speedup_ci_high'])}] |"
              for row in block_rows]
    (output / "integrated_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result
