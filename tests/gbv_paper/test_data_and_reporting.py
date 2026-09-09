from copy import deepcopy
from dataclasses import asdict
import json
from types import SimpleNamespace

import pytest

from gbv_experiments.common import ROOT, digest, file_hash, write_json
from gbv_experiments.config import build_variants, load_config
from gbv_experiments.data import DATASETS, format_record, formatter_id, load_prepared
from gbv_experiments.report import report, validate_results
from gbv_experiments.runner import make_plan, resume_records, stop_token_ids
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
    assert plan["variant_count"] == 7
    entries = build_variants(cfg, ["paths"])
    candidates = [e["variant"] for e in entries if e["variant"]["method"] != "target"]
    assert {v["paths"] for v in candidates} == {1, 2, 3, 4}
    assert all(v["length"] == 15 and v["temperature"] == 1. for v in candidates)
    assert next(v for v in candidates if v["paths"] == 1)["method"] == "bv"
    assert {g for e in build_variants(cfg) for g in e["groups"]} == {"main", "paths"}


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


@pytest.mark.parametrize("model_eos,tokenizer_eos,expected", [
    (7, 9, [7]),
    ([7, 8], 9, [7, 8]),
    (None, 9, [9]),
    (None, None, []),
])
def test_stop_token_ids_match_formal_generation(model_eos, tokenizer_eos, expected):
    target = SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=model_eos))
    engine = SimpleNamespace(target=target)
    tokenizer = SimpleNamespace(eos_token_id=tokenizer_eos)
    assert stop_token_ids(engine, tokenizer) == expected


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


def test_result_validation_rejects_global_order_balance_with_dataset_bias():
    names = ["a", "b", "c"]
    prompt_ids = [[dataset, str(index), "h"]
                  for dataset in ("d0", "d1") for index in range(3)]
    manifest = {
        "run_id":"run", "coverage":"full_evaluation_split",
        "variants":[{"variant":{"name":name, "method":"target",
                                  "temperature":1.0, "paths":1, "length":1},
                     "groups":["main"]} for name in names],
        "prompt_ids":prompt_ids, "seeds":[17], "max_new_tokens":1,
        "dataset_names":["d0", "d1"],
        "method_order":{"policy":"balanced_rotation", "seed":20260909},
    }
    rows = []
    offsets = {"d0":[0, 0, 1], "d1":[1, 2, 2]}
    for dataset, dataset_offsets in offsets.items():
        for prompt, offset in enumerate(dataset_offsets):
            for position, name in enumerate(names[offset:] + names[:offset]):
                rows.append({
                    "variant":name, "dataset":dataset, "source_id":str(prompt),
                    "prompt_sha256":"h", "run_id":"run", "seed":17,
                    "method_order_ordinal":offset,
                    "method_execution_position":position,
                    "generated_tokens":1, "generated_token_ids":[1],
                    "decode_tokens":0, "prefill_ms":1.0, "decode_ms":0.0,
                    "e2e_ms":1.0, "rounds":[], "target_forward_calls":1,
                    "draft_forward_calls":0, "peak_allocated_bytes":None,
                    "finish_reason":"eos", "text":"x",
                })
    # The two datasets together use each rotation offset twice, so the old
    # global-only check accepted this deliberately dataset-biased schedule.
    for name in names:
        positions = [row["method_execution_position"] for row in rows
                     if row["variant"] == name]
        assert [positions.count(i) for i in range(3)] == [2, 2, 2]
    with pytest.raises(ValueError, match="within dataset"):
        validate_results(manifest, rows)


def test_gbv_first_report_leaves_missing_ar_comparison_empty(tmp_path):
    manifest, rows, scores = results_fixture()
    rows = [r for r in rows if r["variant"] == "gbv"]
    scores = [r for r in scores if r["variant"] == "gbv"]
    write_json(tmp_path / "run_manifest.json", manifest)
    for name, items in (("results", rows), ("scores", scores)):
        (tmp_path / f"{name}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in items))
    with pytest.raises(ValueError, match="every"):
        report(tmp_path, tmp_path / "formal", plots=False)
    coverage = report(tmp_path, tmp_path / "partial", bootstrap=20, allow_partial=True, plots=True)
    assert not coverage["complete"]
    summary = json.loads((tmp_path / "partial/summary.json").read_text())["rows"][0]
    assert summary["speedup_vs_ar"] is None and summary["quality_delta_vs_ar"] is None
    assert summary["baseline_status"] == "missing_or_partial"
    assert summary["decode_tps"] == 200 and summary["quality"] == 1
    assert (tmp_path / "partial/gsm8k_main.pdf").exists()


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


def test_process_worker_never_executes_candidate_as_root():
    evaluation = {"kind": "mbpp", "tests": ["assert not_root()"]}
    candidate = "import os\ndef not_root(): return not hasattr(os, 'geteuid') or os.geteuid() != 0"
    assert evaluate_code(candidate, evaluation, backend="process")["passed"]


def test_mbpp_setup_runs_after_candidate_definitions_and_before_assertions():
    candidate = """class Node:
    def __init__(self, value):
        self.value = value

def node_value(node):
    return sqrt(node.value ** 2)
"""
    evaluation = {
        "kind": "mbpp",
        "setup": "from math import sqrt\nroot = Node(4)",
        "tests": ["assert node_value(root) == 4"],
    }
    assert evaluate_code(candidate, evaluation, backend="process")["passed"]


def test_score_cli_defaults(monkeypatch, tmp_path):
    import sys
    from gbv_experiments.__main__ import main
    import gbv_experiments.scoring as scoring
    calls = []
    monkeypatch.setattr(scoring, "score_run", lambda *args: calls.append(args))
    monkeypatch.setattr(sys, "argv", ["gbv_paper.py", "score", "--run-dir", str(tmp_path)])
    main()
    assert calls[0][2:] == ("docker", 4, 10, 6)
