from __future__ import annotations

import copy
import json
import sys
from types import SimpleNamespace

import pytest
import torch

from dflash_specblock.paper import data
from dflash_specblock.paper.__main__ import load_data, main
from dflash_specblock.paper.common import ROOT, atomic_json, load_config, load_json
from dflash_specblock.paper.evaluation import measure_case, same_dialogue
from dflash_specblock.paper.runtime import PaperRuntime


def mt_row():
    return data.make_row("mt-bench", {"question_id": 101, "category": "writing",
        "turns": ["Write a story.", "Revise your previous story."],
        "reference": ["SECRET_REFERENCE_1", "SECRET_REFERENCE_2"]}, 0, "test", "lm-sys/FastChat", None)


def test_aime_source_train_split_is_evaluation_only():
    row = data.make_row("aime25", {"problem_idx": 30, "problem": "Find x.", "answer": 999},
                        29, "train", "MathArena/aime_2025", None)
    assert row["source_id"] == "30"
    assert row["split"] == "train" and row["usage"] == "evaluation_only"
    assert data.is_evaluation_row(row)
    assert "999" not in row["prompt"]
    data.validate_rows([], [], [row], .1)
    with pytest.raises(ValueError, match="official training"):
        data.validate_rows([row], [], [], .1)
    with pytest.raises(ValueError, match="Evaluation-only"):
        data.partition_train([row], [], .1)


@pytest.mark.parametrize("starter", ["", "class Solution:\n    def solve(self, nums):\n        pass"])
def test_lcb_starter_and_private_test_isolation(starter):
    raw = {"platform": "leetcode" if starter else "atcoder", "question_id": "123",
           "question_content": "Compute the sum.", "starter_code": starter,
           "private_test_cases": "DO_NOT_DECODE_OR_EXPOSE_THIS_PICKLE", "public_test_cases": "hidden-field"}
    row = data.make_row("livecodebench", raw, 0, "test", "livecodebench/code_generation_lite", "release_v6")
    assert row["source_id"] == f"{raw['platform']}:123"
    assert "DO_NOT_DECODE" not in json.dumps(row)
    assert "private_test_cases" not in row["reference"]
    assert ("Use the provided code structure." if starter else "Read from standard input") in row["prompt"]
    if starter:
        assert starter in row["prompt"]
    data.validate_rows([], [], [row], .1)


def test_lcb_reads_all_six_release_files_without_remote_code(tmp_path, monkeypatch):
    seen = []
    for i, filename in enumerate(data.LCB_FILES):
        atomic_json(tmp_path / filename, {"id": i})
    def download(repo, filename, **kwargs):
        seen.append((repo, filename, kwargs))
        return str(tmp_path / filename)
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=download))
    rows = list(data.load_evaluation_source("livecodebench", "a" * 40))
    assert [r["id"] for r in rows] == list(range(6))
    assert [item[1] for item in seen] == list(data.LCB_FILES)
    assert all(item[2] == {"repo_type": "dataset", "revision": "a" * 40} for item in seen)


def test_mtbench_revision_is_fixed_before_network_access():
    with pytest.raises(ValueError, match="Unexpected MT-Bench revision"):
        list(data.load_evaluation_source("mt-bench", "b" * 40))


def test_mtbench_keeps_both_turns_and_checks_raw_turn_leakage():
    row = mt_row()
    assert data.user_turns(row) == row["reference"]["turns"]
    altered = copy.deepcopy(row["reference"])
    altered["turns"][1] += " different"
    other = data.make_row("mt-bench", altered, 0, "test", "lm-sys/FastChat", None)
    assert other["prompt_hash"] != row["prompt_hash"]
    training = [data.make_row("gsm8k", {"question": f"train question {i}"}, i,
                             "train", "openai/gsm8k", "main") for i in range(100)]
    leak = data.make_row("gsm8k", {"question": row["user_turns"][1]}, 100,
                         "train", "openai/gsm8k", "main")
    training.append(leak)
    train_rows, dev, removed = data.partition_train(training, [row], .2)
    assert {r["id"] for r in removed} == {leak["id"]}
    data.validate_rows(train_rows, dev, [row], .2)
    with pytest.raises(ValueError, match="both nonempty"):
        data.make_row("mt-bench", {"turns": ["only first"]}, 0, "test", "lm-sys/FastChat", None)


def test_multiturn_uses_own_generated_answer_and_sums_turn_times():
    seen = []
    def encode(messages):
        seen.append(copy.deepcopy(messages))
        return torch.tensor([[len(messages), sum(len(m["content"]) for m in messages)]])
    def generate(ids, method, policy):
        return {"tokens": [5 if method == "ar" else 7], "wall_ms": 10. if ids[0, 0] == 1 else 30., "rounds": []}
    runtime = SimpleNamespace(encode_messages=encode, generate=generate,
                              tokenizer=SimpleNamespace(decode=lambda tokens, **_: f"answer-{tokens[0]}"))
    ar = measure_case(runtime, mt_row(), "ar")
    ddtree = measure_case(runtime, mt_row(), "ddtree")
    assert seen[1][1] == {"role": "assistant", "content": "answer-5"}
    assert seen[3][1] == {"role": "assistant", "content": "answer-7"}
    assert seen[0] == seen[2] == [{"role": "user", "content": "Write a story."}]
    assert all("SECRET_REFERENCE" not in json.dumps(messages) for messages in seen)
    assert ar["wall_ms"] == ddtree["wall_ms"] == 40.
    assert ar["tokens"] == [5, 5] and len(ar["turns"]) == 2
    assert not same_dialogue(ar, ddtree)
    shifted = copy.deepcopy(ar)
    shifted["turns"][0]["tokens"] = [5, 5]
    shifted["turns"][1]["tokens"] = []
    assert not same_dialogue(ar, shifted)  # identical flattened tokens is insufficient


def test_runtime_chat_template_preserves_roles_and_checks_context():
    runtime = object.__new__(PaperRuntime)
    seen = []
    def template(messages, **kwargs):
        seen.append((copy.deepcopy(messages), kwargs))
        return torch.tensor([[1, 2, 3]])
    runtime.tokenizer = SimpleNamespace(apply_chat_template=template)
    runtime.device = torch.device("cpu")
    runtime.target = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=100))
    runtime.cfg = {"max_new_tokens": 6}
    runtime.model_cfg = SimpleNamespace(enable_thinking=False)
    messages = [{"role": "user", "content": "first"}, {"role": "assistant", "content": "own answer"},
                {"role": "user", "content": "second"}]
    assert runtime.encode_messages(messages).tolist() == [[1, 2, 3]]
    assert seen[0][0] == messages and seen[0][1]["enable_thinking"] is False
    with pytest.raises(ValueError, match="alternating"):
        runtime.encode_messages([messages[0], messages[2]])
    runtime.target.config.max_position_embeddings = 10
    with pytest.raises(ValueError, match="exceeds model context"):
        runtime.encode_messages(messages)


def test_plan_counts_two_mtbench_turns(capsys):
    main(["plan"])
    plan = json.loads(capsys.readouterr().out)
    assert sum(plan["test_counts"].values()) == 3905
    assert plan["turns_per_case"]["mt-bench"] == 2
    assert plan["test_generation_calls"] == 609705


def test_preparation_and_full_manifest_with_synthetic_sources(tmp_path, monkeypatch):
    # Exercise real prepare/load validation at the declared full cardinalities,
    # but generate clearly synthetic tasks locally; never download or run a model.
    def load_train(repo, subset, *, split, revision):
        assert split == "train" and revision == "a" * 40
        if repo == "openai/gsm8k":
            return [{"question": f"synthetic gsm train {i}"} for i in range(7473)]
        if repo == "google-research-datasets/mbpp":
            return [{"text": f"synthetic code train {i}", "task_id": 601 + i} for i in range(374)]
        assert repo == "EleutherAI/hendrycks_math"
        size = 1500 if subset == data.MATH_SUBJECTS[-1] else 1000
        return [{"problem": f"synthetic math train {subset} {i}"} for i in range(size)]
    def load_eval(name, revision):
        assert revision == (data.MTBENCH_REVISION if name == "mt-bench" else "a" * 40)
        for i in range(data.SOURCES[name][3]):
            text = f"synthetic heldout {name} {i}"
            if name == "gsm8k": yield {"question": text}
            elif name in {"math500", "aime25"}: yield {"problem": text, "problem_idx": i + 1}
            elif name in {"mbpp", "mbpp_sanitized"}: yield {"text": text, "task_id": 11 + i}
            elif name == "humaneval": yield {"prompt": text, "task_id": f"HumanEval/{i}"}
            elif name == "livecodebench": yield {"question_content": text, "question_id": str(i), "platform": "atcoder"}
            else: yield {"question_id": 101 + i, "turns": [text, text + " followup"], "category": "writing"}
    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=load_train))
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(
        HfApi=lambda: SimpleNamespace(dataset_info=lambda repo: SimpleNamespace(sha="a" * 40))))
    monkeypatch.setattr(data, "load_evaluation_source", load_eval)
    manifest = data.prepare(tmp_path, .1)
    checked, train_rows, dev, evaluation = load_data(tmp_path, 0)
    assert manifest == checked and manifest["version"] == 2
    assert len(train_rows) + len(dev) == 15347
    assert len(evaluation) == 3905
    assert sum(len(data.user_turns(row)) for row in evaluation) == 3985
    assert set(r["dataset"] for r in evaluation) == set(data.SOURCES)
    assert not any(r["dataset"] in {"aime25", "livecodebench", "mt-bench"} for r in train_rows + dev)
    assert data.prepare(tmp_path, .1) == manifest  # read-only validated resume
    path = tmp_path / "mt-bench.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    records[0]["user_turns"].pop()
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    with pytest.raises(ValueError, match="Dataset manifest mismatch"):
        load_data(tmp_path, 0)
