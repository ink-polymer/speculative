from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import uuid

from .common import canonical, digest, read_jsonl, source_hashes, write_json
from .data import load_prepared, resolve_evaluation
from .runner import key, output_lock, resume_records


def extract_code(text: str) -> str:
    # Prefer the final Python fence, excluding preceding reasoning fences.
    blocks = re.findall(r"```(?:python|py)\s*\n(.*?)```", text, re.S | re.I)
    if not blocks:
        blocks = re.findall(r"```\s*\n(.*?)```", text, re.S)
    return (blocks[-1] if blocks else text.rsplit("</think>", 1)[-1]).strip("\n")


def make_program(text, evaluation):
    code = extract_code(text)
    if evaluation["kind"] == "humaneval":
        entry = evaluation["entry_point"]
        if not re.search(rf"^\s*(?:async\s+)?def\s+{re.escape(entry)}\s*\(", code, re.M):
            code = evaluation["prompt"] + code
        tests = evaluation["test"] + f"\ncheck({entry})\n"
    else:
        code = evaluation.get("setup", "") + "\n" + code
        tests = "\n".join(evaluation["tests"])
    return code, tests


WORKER = '''import json, resource, sys
from pathlib import Path
resource.setrlimit(resource.RLIMIT_CPU, (int(sys.argv[1]), int(sys.argv[1])+1))
if sys.platform == "linux":
    resource.setrlimit(resource.RLIMIT_AS, (2*1024**3, 2*1024**3))
resource.setrlimit(resource.RLIMIT_FSIZE, (1024**2, 1024**2))
resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
namespace = {"__name__": "__main__"}
candidate = Path("candidate.py").read_text()
tests = Path("tests.py").read_text()
try:
    exec(compile(candidate, "candidate.py", "exec"), namespace)
    exec(compile(tests, "tests.py", "exec"), namespace)
except BaseException as exc:
    result = {"passed": False, "reason": type(exc).__name__}
else:
    result = {"passed": True, "reason": "passed"}
Path("result.json").write_text(json.dumps(result))
'''


def evaluate_code(text, evaluation, backend="docker", timeout=10, image="gbv-code-eval:py311"):
    code, tests = make_program(text, evaluation)
    return run_sandbox({"candidate.py": code, "tests.py": tests, "worker.py": WORKER},
                       [str(math.ceil(timeout))], backend, timeout, image)


def run_sandbox(files, arguments, backend, timeout, image="gbv-code-eval:py311", memory_gb=2):
    with tempfile.TemporaryDirectory(prefix="gbv_eval_") as tmp:
        directory = Path(tmp)
        for name, content in files.items():
            (directory / name).write_text(content)
        if backend == "docker":
            container_name = "gbv-eval-" + uuid.uuid4().hex
            # Container receives only this disposable evaluation directory.
            command = ["docker", "run", "--rm", "--name", container_name, "--network=none", f"--memory={memory_gb}g", "--cpus=1",
                       "--pids-limit=64", "--cap-drop=ALL", "--security-opt=no-new-privileges",
                       "-v", f"{tmp}:/work", "-w", "/work", image,
                       "python", "-I", "worker.py", *arguments]
        elif backend == "process":
            command = [sys.executable, "-I", "worker.py", *arguments]
        else:
            raise ValueError(f"Unknown code backend: {backend}")
        # No model credentials are forwarded to generated code.
        env = {k: v for k, v in os.environ.items() if k in {"PATH", "SYSTEMROOT", "DOCKER_HOST"}}
        env.update({"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"})
        with (directory / "stdout.log").open("wb") as stdout:
            process = subprocess.Popen(command, cwd=directory, env=env,
                                       stdout=stdout, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                process.wait(timeout=timeout + (15 if backend == "docker" else 1))
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                if backend == "docker":
                    subprocess.run(["docker", "rm", "-f", container_name], timeout=15,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return {"passed": False, "reason": "timeout"}
        result = directory / "result.json"
        if not result.exists():
            if process.returncode in (1, 125, 126, 127):
                details = (directory / "stdout.log").read_text(errors="replace")[-2000:]
                raise RuntimeError(f"Code evaluator failed to start: {details}")
            return {"passed": False, "reason": "worker_exit", "returncode": process.returncode}
        return json.loads(result.read_text())


def math_score(text, answer):
    from math_verify import LatexExtractionConfig, parse, verify
    gold = parse("$" + str(answer) + "$", extraction_config=[LatexExtractionConfig()])
    prediction = parse(text.rsplit("</think>", 1)[-1])
    return {"passed": bool(verify(gold, prediction)), "reason": "math_verify"}


def score_run(run_dir: Path, data_dir: Path, backend="docker", workers=4, timeout=10, lcb_timeout=6):
    with output_lock(run_dir):
        return _score_run(run_dir, data_dir, backend, workers, timeout, lcb_timeout)


def _score_run(run_dir: Path, data_dir: Path, backend="docker", workers=4, timeout=10, lcb_timeout=6):
    import importlib.metadata
    # Refuse silent fallbacks to string matching for mathematical equivalence.
    math_version = importlib.metadata.version("math-verify")
    if backend == "docker":
        subprocess.run(["docker", "image", "inspect", "gbv-code-eval:py311"],
                       check=True, stdout=subprocess.DEVNULL)
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    data_manifest, data = load_prepared(data_dir, manifest["dataset_names"])
    if data_manifest != manifest["data_manifest"]:
        raise ValueError("Scoring data differ from generation data")
    lookup = {(r["dataset"], r["source_id"]): r for r in data}
    records = read_jsonl(run_dir / "results.jsonl")
    if len({key(r) for r in records}) != len(records):
        raise ValueError("Duplicate generation records")
    image_id = subprocess.check_output(["docker", "image", "inspect", "gbv-code-eval:py311", "--format", "{{.Id}}"], text=True).strip() if backend == "docker" else None
    scoring_id = digest({"run_id": manifest["run_id"], "backend": backend, "image_id": image_id,
                         "math_verify": math_version, "timeout": timeout,
                         "lcb_timeout_per_test": lcb_timeout, "scorer_sources": source_hashes()})
    path = run_dir / "scores.jsonl"
    previous = list(resume_records(path, manifest["run_id"]).values())
    if any(r["scoring_id"] != scoring_id for r in previous):
        raise ValueError("Scoring configuration changed; move the old scores.jsonl first")
    completed = {key(r) for r in previous}
    generation_lookup = {key(r): r for r in records}
    if any(key(r) not in generation_lookup or r["prediction_sha256"] != digest(generation_lookup[key(r)]["text"]) for r in previous):
        raise ValueError("Stale scores or changed predictions")

    def one(record):
        ref = lookup[(record["dataset"], record["source_id"])]
        if record["run_id"] != manifest["run_id"] or record["prompt_sha256"] != ref["prompt_sha256"]:
            raise ValueError("Generation/reference mismatch")
        evaluation = resolve_evaluation(ref, data_dir)
        if evaluation["kind"] == "math":
            result = math_score(record["text"], evaluation["answer"])
            metric = "accuracy"
        elif evaluation["kind"] == "livecodebench":
            from .lcb import evaluate_lcb
            result = evaluate_lcb(record["text"], evaluation, backend, lcb_timeout)
            metric = "pass@1"
        elif evaluation["kind"] == "mt-bench":
            result = {"passed": None, "reason": "external_judge_not_run"}
            metric = "not_scored"
        else:
            result = evaluate_code(record["text"], evaluation, backend, timeout)
            metric = "pass@1"
        return {"run_id": manifest["run_id"], "scoring_id": scoring_id, **{k: record[k] for k in ("variant", "dataset", "source_id", "seed")},
                "metric": metric,
                "prediction_sha256": digest(record["text"]), **result}

    pending = [r for r in records if key(r) not in completed]
    # Math-Verify uses signal timeouts: evaluate math in the main thread.
    with path.open("a") as stream:
        for record in pending:
            if lookup[(record["dataset"], record["source_id"])]["evaluation"]["kind"] == "math":
                stream.write(canonical(one(record)) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
        code = [r for r in pending if lookup[(r["dataset"], r["source_id"])]["evaluation"]["kind"] != "math"]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for result in pool.map(one, code):
                stream.write(canonical(result) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
    write_json(run_dir / "scoring_manifest.json", {"scoring_id": scoring_id,
               "backend": backend, "image_id": image_id, "math_verify": math_version, "timeout_seconds": timeout,
               "lcb_timeout_seconds_per_test": lcb_timeout,
               "mt_bench_quality": "not_scored; export-mtbench can prepare official judge inputs"})


def validate_gold(data_dir: Path, names: list[str], output: Path, backend="docker", timeout=10):
    """Check all available references; report reference-free tasks explicitly."""
    manifest, rows = load_prepared(data_dir, names)
    results = []
    for row in rows:
        evaluation = row["evaluation"]
        if evaluation["kind"] == "math":
            result = math_score("\\boxed{" + evaluation["answer"] + "}", evaluation["answer"])
        elif evaluation["kind"] in {"livecodebench", "mt-bench"}:
            result = {"passed": None, "reason": "no_canonical_solution_in_source",
                      "validation": "data_structure_and_hash_checks_only"}
        else:
            result = evaluate_code(evaluation["reference_code"], evaluation, backend, timeout)
        results.append({"dataset": row["dataset"], "source_id": row["source_id"], **result})
    failures = [r for r in results if r["passed"] is False]
    report = {"data_manifest_sha256": digest(manifest), "backend": backend, "count": len(rows),
              "passed": not failures, "canonical_answers_checked": sum(r["passed"] is not None for r in results),
              "without_canonical_answer": sum(r["passed"] is None for r in results),
              "failures": failures, "results": results}
    write_json(output, report)
    if failures:
        raise ValueError(f"{len(failures)} official references failed scoring; inspect {output}. The evaluation split was not filtered.")
    return report
