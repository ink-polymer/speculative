from copy import deepcopy
from dataclasses import asdict
import base64
import json
import pickle
from types import SimpleNamespace
import zlib

import pytest
import torch

from gbv_experiments.common import canonical, digest, file_hash, read_jsonl, write_json
from gbv_experiments.config import Variant
from gbv_experiments.conversation import generate_conversation
from gbv_experiments.data import (DATASETS, decode_lcb_tests, format_record, load_source,
    formatter_id, persist_evaluation, prompt_hash, resolve_evaluation)
from gbv_experiments.lcb import evaluate_lcb
from gbv_experiments.report import summarize, validate_results
from gbv_experiments.runner import key, resume_records


def lcb_row(functional=False, compressed=False):
    mode = "functional" if functional else "stdin"
    private = [{"input": "7\n9" if functional else "7 9\n", "output": "16", "testtype": mode}]
    payload = json.dumps(private)
    if compressed:
        payload = base64.b64encode(zlib.compress(pickle.dumps(payload))).decode()
    return {"question_id": "42", "platform": "leetcode" if functional else "atcoder",
            "question_content": "Add two integers.", "question_title": "Sum",
            "contest_date": "2025-03-01T00:00:00", "difficulty": "easy",
            "starter_code": "class Solution:\n    def add(self, a: int, b: int) -> int:\n        pass" if functional else "",
            "public_test_cases": json.dumps([{"input": "1\n2" if functional else "1 2\n", "output": "3", "testtype": mode}]),
            "private_test_cases": payload, "metadata": json.dumps({"func_name": "add"} if functional else {})}


def mt_row(index=81):
    return format_record("mt-bench", {"question_id": index, "category": "writing",
        "turns": ["Write a story.", "Shorten the previous story."], "reference": ["hidden reference", ""]}, index - 81)


@pytest.mark.parametrize("functional,compressed", [(False, False), (False, True), (True, True)])
def test_lcb_private_cases_and_starter_preserved_without_prompt_leak(functional, compressed, tmp_path):
    row = format_record("livecodebench", lcb_row(functional, compressed), 0)
    assert row["source_id"].endswith(":42")
    assert "7 9" not in row["prompt"] and '"private_test_cases"' not in row["prompt"]
    evaluation = deepcopy(row["evaluation"])
    assert evaluation["test_count"] == 2 and evaluation["private_test_count"] == 1
    persisted = persist_evaluation(row, tmp_path)
    assert "tests" not in persisted["evaluation"]
    assert resolve_evaluation(persisted, tmp_path) == evaluation
    (tmp_path / persisted["evaluation"]["file"]).write_text("{}")
    with pytest.raises(ValueError, match="hash"):
        resolve_evaluation(persisted, tmp_path)


def test_lcb_restricted_pickle_rejects_executable_objects():
    class Payload:
        def __reduce__(self):
            return eval, ('"unsafe"',)
    encoded = base64.b64encode(zlib.compress(pickle.dumps(Payload()))).decode()
    with pytest.raises(ValueError, match="Executable"):
        decode_lcb_tests(encoded)


def test_lcb_loads_exactly_all_six_release_files(tmp_path, monkeypatch):
    import huggingface_hub
    files = ["test.jsonl"] + [f"test{i}.jsonl" for i in range(2, 7)]
    counts = [400, 111, 101, 101, 167, 175]
    for name, count in zip(files, counts):
        (tmp_path / name).write_text("".join(json.dumps({"source": name, "id": i}) + "\n" for i in range(count)))
    seen = []
    def download(repo, filename, **kwargs):
        seen.append((filename, kwargs["revision"]))
        return str(tmp_path / filename)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    rows = list(load_source(DATASETS["livecodebench"], "pinned", tmp_path))
    assert len(rows) == 1055 and seen == [(name, "pinned") for name in files]


def test_aime_uses_original_problem_id():
    row = format_record("aime25", {"problem_idx": 24, "problem": "Compute 2+3.", "answer": 5}, 0)
    assert row["source_id"] == "24" and row["evaluation"]["answer"] == "5"


def test_mt_bench_both_turns_in_identity_and_reference_hidden():
    row = mt_row()
    assert row["source_id"] == "81"
    assert "hidden reference" not in row["prompt"]
    assert len(row["user_turns"]) == 2
    modified = deepcopy(row)
    modified["user_turns"][1] = "A different instruction."
    assert prompt_hash(modified) != row["prompt_sha256"]
    with pytest.raises(ValueError, match="both"):
        format_record("mt-bench", {"question_id": 81, "category": "writing", "turns": ["Only one turn"]}, 0)


@pytest.mark.parametrize("functional", [False, True])
@pytest.mark.parametrize("correct", [False, True])
def test_official_lcb_worker_checks_public_and_private(functional, correct):
    row = format_record("livecodebench", lcb_row(functional, compressed=True), 0)
    if functional:
        code = "class Solution:\n    def add(self,a,b): return " + ("a+b" if correct else "3")
    else:
        code = "import sys\na,b=map(int,sys.stdin.buffer.read().split())\nprint(" + ("a+b" if correct else "3") + ")"
    score = evaluate_lcb(code, row["evaluation"], backend="process", timeout=1)
    assert score["passed"] is correct
    assert score["executed_tests"] == 2


def test_official_lcb_worker_rejects_timeout_and_syntax_error():
    row = format_record("livecodebench", lcb_row(), 0)
    assert not evaluate_lcb("while True: pass", row["evaluation"], backend="process", timeout=1)["passed"]
    assert not evaluate_lcb("this is invalid Python!", row["evaluation"], backend="process", timeout=1)["passed"]


def test_lcb_score_run_loads_hidden_sidecar_and_records_pass_at_one(tmp_path, monkeypatch):
    from gbv_experiments import scoring
    row = persist_evaluation(format_record("livecodebench", lcb_row(), 0), tmp_path)
    data_manifest = {"fixture": "two tests, with one private case"}
    monkeypatch.setattr(scoring, "load_prepared", lambda *args: (data_manifest, [row]))
    manifest = {"run_id": "lcb_fixture", "dataset_names": ["livecodebench"], "data_manifest": data_manifest}
    write_json(tmp_path / "run_manifest.json", manifest)
    records = [{"run_id": manifest["run_id"], "variant": variant, "dataset": row["dataset"],
                "source_id": row["source_id"], "seed": 17, "prompt_sha256": row["prompt_sha256"], "text": code}
               for variant, code in (("correct", "a,b=map(int,input().split()); print(a+b)"), ("public_only", "print(3)"))]
    (tmp_path / "results.jsonl").write_text("".join(canonical(r) + "\n" for r in records))
    scoring.score_run(tmp_path, tmp_path, backend="process", workers=2, lcb_timeout=1)
    scores = {r["variant"]: r for r in read_jsonl(tmp_path / "scores.jsonl")}
    assert scores["correct"]["passed"] is True and scores["public_only"]["passed"] is False
    assert all(r["metric"] == "pass@1" and r["executed_tests"] == 2 and r["official_scorer_commit"] for r in scores.values())


class Tokenizer:
    def __init__(self):
        self.messages = []
    def apply_chat_template(self, messages, **kwargs):
        self.messages.append(deepcopy(messages))
        return json.dumps(messages)
    def __call__(self, text, **kwargs):
        return SimpleNamespace(input_ids=torch.tensor([[1, 2, len(text)]]))
    def decode(self, tokens, **kwargs):
        return ":".join(map(str, tokens))


class Engine:
    device = "cpu"
    def __init__(self):
        self.calls = []
    def generate(self, ids, variant, maximum, stops, seed, profile):
        self.calls.append((variant.name, seed))
        # EOS in first turn must not cancel the second user request.
        tokens = [9] if len(self.calls) % 2 else [4, 5, 9]
        return {"generated_token_ids": tokens, "generated_tokens": len(tokens), "decode_tokens": len(tokens)-1,
            "prefill_ms": 2, "decode_ms": 3, "e2e_ms": 5, "target_forward_calls": 1,
            "draft_forward_calls": 1, "target_tokens_processed": 3, "rounds": [],
            "peak_allocated_bytes": None, "peak_reserved_bytes": None,
            "finish_reason": "eos", "stages": {"host_ms": {}, "cuda_event_ms": {}}}


def conversation_record(variant="gbv", seed=17, source_id="81"):
    row = mt_row(int(source_id))
    tokenizer, engine = Tokenizer(), Engine()
    result = generate_conversation(engine, tokenizer, row, Variant(name=variant), 3, [9], seed, {})
    return {"run_id": "r", "variant": variant, "dataset": "mt-bench", "source_id": source_id,
            "prompt_sha256": row["prompt_sha256"], "seed": seed, **result}, tokenizer, engine


def test_conversation_uses_own_first_answer_and_counts_two_prefills():
    result, tokenizer, engine = conversation_record()
    assert len(engine.calls) == 2 and engine.calls[0][1] != engine.calls[1][1]
    assert [m["role"] for m in tokenizer.messages[1]] == ["user", "assistant", "user"]
    assert tokenizer.messages[1][1]["content"] == "9"
    assert result["turn_count"] == 2 and result["generated_tokens"] == 4
    assert result["decode_tokens"] == 2 and result["prefill_ms"] == 4
    assert result["e2e_ms"] == 10 and json.loads(result["text"]) == ["9", "4:5:9"]
    assert result["turn_results"][0]["messages_sha256"] != result["turn_results"][1]["messages_sha256"]


def conversation_fixture():
    records = [conversation_record(v, seed, str(i))[0] for v in ("ar", "gbv") for seed in (17, 29) for i in (81, 82)]
    manifest = {"run_id": "r", "coverage": "full_evaluation_split", "max_new_tokens": 3,
        "dataset_names": ["mt-bench"], "dataset_turn_counts": {"mt-bench": 2}, "seeds": [17, 29],
        "prompt_ids": [["mt-bench", str(i), mt_row(i)["prompt_sha256"]] for i in (81, 82)],
        "variants": [{"variant": {"name": v, "method": m, "temperature": 1., "paths": 1, "length": 3}, "groups": ["main"]}
                     for v, m in (("ar", "target"), ("gbv", "gbv"))]}
    scores = [{**{k: r[k] for k in ("run_id", "variant", "dataset", "source_id", "seed")},
               "passed": None, "metric": "not_scored", "prediction_sha256": digest(r["text"])} for r in records]
    return manifest, records, scores


def test_mt_report_requires_second_turn_and_does_not_invent_quality(tmp_path):
    manifest, records, scores = conversation_fixture()
    validate_results(manifest, records, scores)
    table = summarize(manifest, records, scores, bootstrap=10)
    assert all(row["quality"] is None and row["scored_samples"] == 0 for row in table)
    assert all(row["generation_turns"] == 8 and row["ttft_mean_ms"] == 2 for row in table)
    assert table[0]["speedup_vs_ar"] == 1
    modified = deepcopy(records)
    modified[0]["turn_results"].pop()
    with pytest.raises(ValueError, match="turn"):
        validate_results(manifest, modified, scores)
    modified = deepcopy(records)
    modified[0]["turn_results"][0]["prefill_ms"] = -1
    modified[0]["turn_results"][1]["prefill_ms"] = 5
    with pytest.raises(ValueError, match="timer"):
        validate_results(manifest, modified, scores)
    path = tmp_path / "results.jsonl"
    path.write_text(canonical(records[0]) + "\n" + canonical(records[1])[:100])
    recovered = resume_records(path, "r")
    assert len(recovered) == 1 and recovered[key(records[0])]["turn_count"] == 2


def test_full_mt_scoring_export_and_judgment_import(tmp_path):
    from gbv_experiments.mtbench import export_answers, import_judgments
    from gbv_experiments.scoring import score_run, validate_gold
    data_dir, run_dir = tmp_path / "data", tmp_path / "run"
    data_dir.mkdir(); run_dir.mkdir()
    rows = [mt_row(i) for i in range(81, 161)]
    path = data_dir / "mt-bench.jsonl"
    path.write_text("".join(canonical(r) + "\n" for r in rows))
    data_manifest = {"coverage": "full_evaluation_split", "formatter_id": formatter_id(),
        "datasets": {"mt-bench": {**asdict(DATASETS["mt-bench"]), "file": path.name,
                                   "count": 80, "sha256": file_hash(path)}}}
    write_json(data_dir / "manifest.json", data_manifest)
    manifest, _, _ = conversation_fixture()
    manifest["seeds"] = [17]
    manifest["prompt_ids"] = [[r["dataset"], r["source_id"], r["prompt_sha256"]] for r in rows]
    manifest["data_manifest"] = data_manifest
    records = [conversation_record(v, 17, r["source_id"])[0] for r in rows for v in ("ar", "gbv")]
    write_json(run_dir / "run_manifest.json", manifest)
    (run_dir / "results.jsonl").write_text("".join(canonical(r) + "\n" for r in records))
    score_run(run_dir, data_dir, backend="process", workers=2)
    scores = read_jsonl(run_dir / "scores.jsonl")
    assert len(scores) == 160 and all(r["passed"] is None for r in scores)
    score_run(run_dir, data_dir, backend="process", workers=2)
    assert len(read_jsonl(run_dir / "scores.jsonl")) == 160
    validate_results(manifest, records, scores)
    gold = validate_gold(data_dir, ["mt-bench"], tmp_path / "gold.json", backend="process")
    assert gold["canonical_answers_checked"] == 0 and gold["without_canonical_answer"] == 80
    export_dir = tmp_path / "export"
    exported = export_answers(run_dir, data_dir, export_dir)
    assert exported["expected_judgments"] == 320 and exported["api_calls_made"] == 0
    assert len(read_jsonl(export_dir / "model_answer/gbv__seed17.jsonl")) == 80
    write_json(export_dir / "export_manifest.json", {**exported, "run_id": "another_run"})
    with pytest.raises(ValueError, match="another run"):
        export_answers(run_dir, data_dir, export_dir)
    write_json(export_dir / "export_manifest.json", exported)
    judgments = [{"model": model, "question_id": i, "turn": turn, "score": 8,
                  "judge": ["fixture_judge", "single-v1"]}
                 for model in exported["models"] for i in range(81, 161) for turn in (1, 2)]
    judgment_path = tmp_path / "judgments.jsonl"
    judgment_path.write_text("".join(canonical(r) + "\n" for r in judgments))
    imported = import_judgments(run_dir, export_dir, judgment_path, tmp_path / "judge_report")
    assert imported["count"] == 320 and imported["complete"]
    judgment_path.write_text("".join(canonical(r) + "\n" for r in judgments[:-1]))
    with pytest.raises(ValueError, match="Incomplete"):
        import_judgments(run_dir, export_dir, judgment_path, tmp_path / "bad")
    judgments[0]["score"] = -1
    judgment_path.write_text("".join(canonical(r) + "\n" for r in judgments))
    with pytest.raises(ValueError, match="Invalid"):
        import_judgments(run_dir, export_dir, judgment_path, tmp_path / "bad")
