from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path

import pytest

from gbv_experiments.common import ROOT, digest, file_hash, write_json
from gbv_experiments.config import build_variants, load_config
from gbv_experiments.data import DATASETS, format_record, formatter_id, load_prepared
from gbv_experiments.report import clustered_ci, report, validate_results
from gbv_experiments.runner import make_plan, resume_records
from gbv_experiments.scoring import make_program, math_score, evaluate_code


def test_full_plan_and_ablation_controls():
    cfg = load_config(ROOT / "configs/gbv_paper_full.json")
    plan = make_plan(cfg)
    assert sum(plan["datasets"].values()) == 3648
    assert plan["datasets"]["aime25"] == 30
    assert plan["datasets"]["livecodebench"] == 1055
    assert plan["datasets"]["mt-bench"] == 80
    assert sum(plan["user_turns"].values()) == 3728
    assert plan["expected_generations"] == 3728 * 3 * plan["variant_count"]
    assert plan["expected_records"] == 3648 * 3 * plan["variant_count"]
    entries = build_variants(cfg, ["block_verification"])
    methods = {e["variant"]["method"] for e in entries}
    assert methods == {"target", "token", "bv"}
    assert all(e["variant"]["paths"] == 1 for e in entries)
    for group in ("prefix_sharing", "draft_cache", "bidirectional_attention", "target_features", "probability_precision"):
        candidates = [e["variant"] for e in build_variants(cfg, [group]) if e["variant"]["method"] != "target"]
        a, b = candidates
        assert sum(a[k] != b[k] for k in a if k != "name") == 1


def test_prompts_keep_answers_and_hidden_tests_out():
    row = format_record("gsm8k", {"question": "What is 3 times 4?", "answer": "secret reasoning #### 12"}, 0)
    assert "secret reasoning" not in row["prompt"]
    assert row["evaluation"]["answer"] == "12"
    mbpp = format_record("mbpp", {"task_id": 1, "text": "Add one.", "test_list": ["assert f(1)==2", "assert f(99)==100"], "code": "secret_solution"}, 0)
    assert "f(1)==2" in mbpp["prompt"]
    assert "f(99)" not in mbpp["prompt"] and "secret_solution" not in mbpp["prompt"]
    assert mbpp["evaluation"]["tests"][1] == "assert f(99)==100"


def test_full_split_count_and_hash_are_enforced(tmp_path):
    part = [format_record("gsm8k", {"question": f"question {i}", "answer": "#### 42"}, i) for i in range(1319)]
    path = tmp_path / "gsm8k.jsonl"
    path.write_text("".join(json.dumps(x) + "\n" for x in part))
    manifest = {"coverage": "full_evaluation_split", "formatter_id": formatter_id(), "datasets": {"gsm8k": {
        **asdict(DATASETS["gsm8k"]), "file": path.name, "count": 1319, "sha256": file_hash(path)}}}
    write_json(tmp_path / "manifest.json", manifest)
    assert len(load_prepared(tmp_path, ["gsm8k"])[1]) == 1319
    path.write_text("".join(json.dumps(x) + "\n" for x in part[:-1]))
    with pytest.raises(ValueError, match="hash"):
        load_prepared(tmp_path, ["gsm8k"])
    manifest["datasets"]["gsm8k"]["sha256"] = file_hash(path)
    write_json(tmp_path / "manifest.json", manifest)
    with pytest.raises(ValueError, match="Incomplete"):
        load_prepared(tmp_path, ["gsm8k"])


def test_resume_only_recovers_partial_last_line(tmp_path):
    path = tmp_path / "results.jsonl"
    row = {"variant": "x", "dataset": "gsm8k", "source_id": "1", "seed": 17, "run_id": "a"}
    line = json.dumps(row) + "\n"
    path.write_text(line + '{"broken":')
    assert len(resume_records(path, "a")) == 1
    assert path.read_text() == line
    path.write_text(line + line)
    with pytest.raises(ValueError, match="duplicate"):
        resume_records(path, "a")
    path.write_text(line)
    with pytest.raises(ValueError, match="mismatch"):
        resume_records(path, "b")


def results_fixture():
    variants = [{"variant": {"name": name, "method": method, "temperature": 1., "paths": 1, "length": 3}, "groups": ["main"]}
                for name, method in [("ar", "target"), ("gbv", "gbv")]]
    manifest = {"run_id": "id", "coverage": "full_evaluation_split", "variants": variants,
                "prompt_ids": [["gsm8k", str(i), "h"] for i in range(2)], "seeds": [17, 29],
                "max_new_tokens": 3, "dataset_names": ["gsm8k"]}
    records, scores = [], []
    for v in variants:
        for _, ident, h in manifest["prompt_ids"]:
            for seed in manifest["seeds"]:
                decode = 20 if v["variant"]["name"] == "ar" else 10
                r = {"variant": v["variant"]["name"], "dataset": "gsm8k", "source_id": ident,
                     "prompt_sha256": h, "run_id": "id", "seed": seed, "generated_tokens": 3,
                     "generated_token_ids": [1, 2, 3], "decode_tokens": 2, "prefill_ms": 5,
                     "decode_ms": decode, "e2e_ms": decode + 5, "rounds": [],
                     "target_forward_calls": 3, "draft_forward_calls": 0,
                     "peak_allocated_bytes": None, "finish_reason": "length", "text": "answer"}
                records.append(r)
                scores.append({k: r[k] for k in ("variant", "dataset", "source_id", "seed", "run_id")} | {"passed": True, "prediction_sha256": digest("answer")})
    return manifest, records, scores


def test_report_refuses_partial_scores_and_duplicates(tmp_path):
    manifest, rows, scores = results_fixture()
    validate_results(manifest, rows, scores)
    with pytest.raises(ValueError, match="every"):
        validate_results(manifest, rows[:-1], scores)
    with pytest.raises(ValueError, match="scores"):
        validate_results(manifest, rows, scores[:-1])
    with pytest.raises(ValueError, match="Duplicate"):
        validate_results(manifest, rows + rows[:1], scores)
    stale = deepcopy(scores)
    stale[0]["prediction_sha256"] = "changed"
    with pytest.raises(ValueError, match="Stale"):
        validate_results(manifest, rows, stale)
    write_json(tmp_path / "run_manifest.json", manifest)
    for name, items in (("results", rows), ("scores", scores)):
        (tmp_path / f"{name}.jsonl").write_text("".join(json.dumps(x)+"\n" for x in items))
    report(tmp_path, tmp_path / "report", bootstrap=30, plots=True)
    summaries = json.loads((tmp_path / "report/summary.json").read_text())["rows"]
    gbv = next(r for r in summaries if r["variant"] == "gbv")
    assert gbv["speedup_vs_ar"] == 2
    assert gbv["speedup_ci_low"] == gbv["speedup_ci_high"] == 2
    assert (tmp_path / "report/table.tex").exists()
    assert (tmp_path / "report/gsm8k_main.pdf").stat().st_size > 1000


@pytest.mark.parametrize("prediction,answer,passed", [
    (r"The answer is \boxed{\frac{1}{2}}.", "0.5", True),
    (r"\boxed{\sqrt{8}}", r"2\sqrt{2}", True),
    (r"\boxed{5}", "4", False),
    (r"</think>Final answer: \boxed{42}", "42", True),
])
def test_math_equivalence(prediction, answer, passed):
    assert math_score(prediction, answer)["passed"] is passed


@pytest.mark.parametrize("source", ["```python\ndef inc(x):\n    return x + 1\n```", "    return x + 1\n"])
def test_humaneval_full_function_or_completion(source):
    evaluation = {"kind": "humaneval", "entry_point": "inc", "prompt": "def inc(x):\n",
                  "test": "def check(candidate):\n    assert candidate(2) == 3"}
    code, test = make_program(source, evaluation)
    namespace = {}
    exec(code + "\n" + test, namespace)


def test_code_worker_correctness_and_timeout():
    evaluation = {"kind": "mbpp", "tests": ["assert inc(4) == 5"]}
    # These are test-owned fixtures; actual model code defaults to Docker.
    assert evaluate_code("def inc(x): return x + 1", evaluation, backend="process")["passed"]
    assert not evaluate_code("def inc(x): return x", evaluation, backend="process")["passed"]
    assert not evaluate_code("while True: pass", evaluation, backend="process", timeout=.1)["passed"]


def test_score_cli_defaults(monkeypatch, tmp_path):
    import sys
    from gbv_experiments.__main__ import main
    import gbv_experiments.scoring as scoring
    calls = []
    monkeypatch.setattr(scoring, "score_run", lambda *args: calls.append(args))
    monkeypatch.setattr(sys, "argv", ["gbv_paper.py", "score", "--run-dir", str(tmp_path)])
    main()
    assert calls[0][2:] == ("docker", 4, 10, 6)
