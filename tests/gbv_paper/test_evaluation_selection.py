from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
from datasets import Dataset

from gbv_experiments import data
from gbv_experiments.__main__ import main
from gbv_experiments.audit import audit
from gbv_experiments.common import ROOT, canonical, digest, file_hash, read_jsonl, write_json
from gbv_experiments.config import load_config
from gbv_experiments.report import report, validate_results
from gbv_experiments.runner import make_plan
from gbv_experiments.scoring import score_run, validate_gold


SELECTION = {"protocol": "ddtree_counts", "sample_seed": 0}


def mock_source(monkeypatch, rows):
    monkeypatch.setattr("huggingface_hub.HfApi.dataset_info", lambda *a, **kw: SimpleNamespace(sha="a" * 40))
    monkeypatch.setattr(data, "load_source", lambda *a, **kw: iter(rows))


@pytest.fixture
def selected_gsm(monkeypatch, tmp_path):
    source = [{"id": f"original-{i}", "question": f"Compute {i} + 1.", "answer": f"#### {i + 1}"}
              for i in range(1319)]
    mock_source(monkeypatch, source)
    directory = tmp_path / "data"
    manifest = data.prepare(["gsm8k"], directory, SELECTION)
    return directory, manifest, data.load_prepared(directory, ["gsm8k"], SELECTION)[1]


def test_default_plan_and_unchanged_full_alternative(monkeypatch, tmp_path):
    output = tmp_path / "plan.json"
    monkeypatch.setattr("sys.argv", ["gbv_paper", "plan", "--output", str(output)])
    main()
    plan = json.loads(output.read_text())
    assert plan["datasets"] == {"gsm8k": 128, "math500": 128, "aime25": 30, "humaneval": 164,
                                "mbpp": 128, "livecodebench": 128, "mt-bench": 80}
    assert sum(plan["datasets"].values()) == 786
    assert sum(plan["user_turns"].values()) == 866
    assert plan["variant_count"] == 13
    assert (plan["expected_records"], plan["expected_generations"]) == (30654, 33774)
    cfg = load_config(ROOT / "configs/gbv_paper_ddtree_counts.json")
    main_plan = make_plan(cfg, ["main"])
    assert (main_plan["expected_records"], main_plan["expected_generations"]) == (11790, 12990)
    full_cfg = load_config(ROOT / "configs/gbv_paper_full.json")
    assert {k: v for k, v in cfg.items() if k != "evaluation"} == full_cfg
    full = make_plan(full_cfg)
    assert full["datasets"] == plan["source_counts"]
    assert (full["expected_records"], full["expected_generations"]) == (142272, 145392)
    cfg["seeds"] = [777]
    assert make_plan(cfg)["evaluation"] == plan["evaluation"]


@pytest.mark.parametrize("source_count,count", [(1319, 128), (500, 128), (1055, 128), (30, 30), (164, 164), (80, 80)])
def test_selection_matches_official_huggingface_rule(source_count, count):
    source = Dataset.from_dict({"index": list(range(source_count))})
    expected = source.shuffle(seed=0).select(range(count)) if source_count > count else source
    assert data.selection_indices(source_count, count, 0) == list(expected["index"])


@pytest.mark.parametrize("settings", [
    {"protocol": "ddtree_counts", "counts": {"gsm8k": 127}},
    {"protocol": "ddtree_counts", "sample_seed": 17},
    {"protocol": "ddtree_counts", "sample_seed": False},
    {"protocol": "full", "sample_seed": 0},
    {"protocol": "unknown"},
    {"protocol": "ddtree_counts", "max_samples": 128},
])
def test_reject_invalid_selection_policy(settings):
    with pytest.raises(ValueError):
        data.evaluation_policy(["gsm8k"], settings)


def test_prepared_subset_has_original_ids_and_repeatable_selection(selected_gsm, monkeypatch):
    directory, manifest, rows = selected_gsm
    expected = Dataset.from_dict({"index": list(range(1319))}).shuffle(seed=0).select(range(128))["index"]
    assert len(rows) == 128
    assert manifest["coverage"] == "fixed_evaluation_subset"
    entry = manifest["datasets"]["gsm8k"]
    assert entry["expected"] == entry["source_count"] == 1319
    assert entry["count"] == 128
    assert entry["selected_source_indices"] == list(expected)
    assert [r["source_id"] for r in rows] == [f"original-{i}" for i in expected]
    assert entry["selected_source_ids"] == [r["source_id"] for r in rows]
    monkeypatch.setattr(data, "load_source", lambda *a, **kw: pytest.fail("Resume must reuse the frozen manifest"))
    assert data.prepare(["gsm8k"], directory, SELECTION) == manifest
    with pytest.raises(ValueError, match="selection changed"):
        data.prepare(["gsm8k"], directory)


@pytest.mark.parametrize("invalid_source", ["short", "duplicate"])
def test_full_source_must_be_verified_before_accepting_subset(monkeypatch, tmp_path, invalid_source):
    source = [{"id": i, "question": f"Question {i}", "answer": "#### 42"} for i in range(1319)]
    if invalid_source == "short":
        source.pop()
    else:
        source[-1]["id"] = source[0]["id"]
    mock_source(monkeypatch, source)
    with pytest.raises(ValueError, match="full source split|Duplicate source"):
        data.prepare(["gsm8k"], tmp_path, SELECTION)
    assert not (tmp_path / "manifest.json").exists()


@pytest.mark.parametrize("change", ["index", "id", "missing_row"])
def test_load_rejects_changed_subset_even_with_updated_file_hash(selected_gsm, change):
    directory, manifest, rows = selected_gsm
    if change == "index":
        rows[0]["source_index"] = -1
    elif change == "id":
        rows[0]["source_id"] = "another-question"
    else:
        rows.pop()
    path = directory / "gsm8k.jsonl"
    path.write_text("".join(canonical(row) + "\n" for row in rows))
    manifest["datasets"]["gsm8k"]["sha256"] = file_hash(path)
    write_json(directory / "manifest.json", manifest)
    with pytest.raises(ValueError, match="selection mismatch|Incomplete"):
        data.load_prepared(directory, ["gsm8k"], SELECTION)


def test_full_data_directory_cannot_be_reused_for_subset(monkeypatch, tmp_path):
    source = [{"question": f"Question {i}", "answer": "#### 42"} for i in range(1319)]
    mock_source(monkeypatch, source)
    data.prepare(["gsm8k"], tmp_path)
    with pytest.raises(ValueError, match="selection changed"):
        data.prepare(["gsm8k"], tmp_path, SELECTION)


def test_lcb_selects_questions_but_retains_all_their_tests(monkeypatch, tmp_path):
    indices = set(data.selection_indices(1055, 128, 0))
    cases = [{"input": "1\n", "output": "1\n", "testtype": "stdin"},
             {"input": "2\n", "output": "2\n", "testtype": "stdin"}]
    source = [{"platform": "atcoder", "question_id": str(i), "question_content": f"Question {i}",
               "public_test_cases": cases[:1], "private_test_cases": cases[1:] if i in indices else "not decoded",
               "metadata": {}, "starter_code": "", "contest_date": "2025-01-01", "difficulty": "easy"}
              for i in range(1055)]
    mock_source(monkeypatch, source)
    data.prepare(["livecodebench"], tmp_path, SELECTION)
    _, rows = data.load_prepared(tmp_path, ["livecodebench"], SELECTION)
    assert len(rows) == len(list((tmp_path / "evaluations/livecodebench").glob("*.json"))) == 128
    assert all(data.resolve_evaluation(row, tmp_path)["tests"] == cases for row in rows)


def test_subset_audit_scoring_and_complete_report(selected_gsm, tmp_path):
    directory, prepared, prompts = selected_gsm
    cfg = {"datasets": ["gsm8k"], "evaluation": SELECTION}
    audit_result = audit(cfg, directory, [], tmp_path / "audit.json")
    assert audit_result["checks_passed"] and audit_result["evaluation_counts"] == {"gsm8k": 128}
    validate_gold(directory, ["gsm8k"], tmp_path / "gold.json", "process", 10, SELECTION)
    manifest = {"run_id": "fixture", "coverage": "fixed_evaluation_subset", "evaluation": prepared["evaluation"],
                "variants": [{"variant": {"name": name, "method": method, "temperature": 1., "paths": 1, "length": 3},
                              "groups": ["main"]} for name, method in [("ar", "target"), ("gbv", "gbv")]],
                "data_manifest": prepared, "dataset_names": ["gsm8k"], "seeds": [17], "max_new_tokens": 3,
                "prompt_ids": [[r["dataset"], r["source_id"], r["prompt_sha256"]] for r in prompts]}
    records = []
    for variant in manifest["variants"]:
        for row in prompts:
            record = {"variant": variant["variant"]["name"], "dataset": "gsm8k", "source_id": row["source_id"],
                      "prompt_sha256": row["prompt_sha256"], "run_id": "fixture", "seed": 17,
                      "generated_tokens": 3, "generated_token_ids": [1, 2, 3], "decode_tokens": 2,
                      "prefill_ms": 5, "decode_ms": 10, "e2e_ms": 15, "rounds": [],
                      "target_forward_calls": 3, "draft_forward_calls": 0, "peak_allocated_bytes": None,
                      "finish_reason": "length", "text": r"\boxed{" + row["evaluation"]["answer"] + "}"}
            records.append(record)
    run_dir = tmp_path / "run"
    write_json(run_dir / "run_manifest.json", manifest)
    (run_dir / "results.jsonl").write_text("".join(canonical(r) + "\n" for r in records))
    score_run(run_dir, directory, "process", workers=1)
    scores = read_jsonl(run_dir / "scores.jsonl")
    assert len(scores) == 256 and all(s["passed"] for s in scores)
    coverage = report(run_dir, run_dir / "report", bootstrap=10, plots=False)
    assert coverage["complete"] and coverage["coverage"] == "fixed_evaluation_subset"
    assert coverage["expected"] == coverage["actual"] == 256
    with pytest.raises(ValueError, match="every evaluation"):
        validate_results(manifest, records[:-1], scores)
    with pytest.raises(ValueError, match="scores"):
        validate_results(manifest, records, scores[:-1])
    missing = deepcopy(manifest)
    missing["prompt_ids"].pop()
    with pytest.raises(ValueError, match="prompt counts"):
        validate_results(missing, records, scores)
    duplicate = deepcopy(manifest)
    duplicate["prompt_ids"][0] = duplicate["prompt_ids"][1]
    with pytest.raises(ValueError, match="Duplicate prompt"):
        validate_results(duplicate, records, scores)
    changed = deepcopy(manifest)
    changed["prompt_ids"][0][1] = "unselected-question"
    with pytest.raises(ValueError, match="prepared sample IDs"):
        validate_results(changed, records, scores)
