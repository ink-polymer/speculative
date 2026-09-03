from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import inspect
import base64
import io
import pickle
import zlib
from pathlib import Path

from .common import canonical, digest, file_hash, read_jsonl, write_json


@dataclass(frozen=True)
class DatasetSpec:
    repo: str
    config: str | None
    split: str
    expected: int
    loader: str = "hf"
    release: str | None = None
    revision: str | None = None
    turns: int = 1


DATASETS = {
    "gsm8k": DatasetSpec("openai/gsm8k", "main", "test", 1319),
    "math500": DatasetSpec("HuggingFaceH4/MATH-500", None, "test", 500),
    "humaneval": DatasetSpec("openai/openai_humaneval", None, "test", 164),
    "mbpp": DatasetSpec("google-research-datasets/mbpp", "full", "test", 500),
    "mbpp_sanitized": DatasetSpec("google-research-datasets/mbpp", "sanitized", "test", 257),
    "aime24": DatasetSpec("HuggingFaceH4/aime_2024", None, "train", 30),
    "aime25": DatasetSpec("MathArena/aime_2025", None, "train", 30),
    "livecodebench": DatasetSpec("livecodebench/code_generation_lite", None, "test", 1055,
                                  loader="lcb_jsonl", release="release_v6"),
    "mt-bench": DatasetSpec("lm-sys/FastChat", None, "fastchat/llm_judge/data/mt_bench/question.jsonl", 80,
                             loader="github_jsonl", revision="587d5cfa1609a43d192cedb8441cac3c17db105d", turns=2),
}


def formatter_id():
    return digest([inspect.getsource(format_record), inspect.getsource(decode_lcb_tests)])


class DataOnlyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise ValueError("Executable pickle objects are not allowed in benchmark data")


def decode_lcb_tests(value):
    if isinstance(value, list):
        tests = value
    else:
        try:
            tests = json.loads(value)
        except json.JSONDecodeError:
            raw = zlib.decompress(base64.b64decode(value, validate=True))
            serialized = DataOnlyUnpickler(io.BytesIO(raw)).load()
            if not isinstance(serialized, (str, bytes)):
                raise ValueError("Expected a pickled JSON string")
            tests = json.loads(serialized)
    if not isinstance(tests, list):
        raise ValueError("Invalid LiveCodeBench tests")
    for test in tests:
        if set(test) != {"input", "output", "testtype"} or test["testtype"] not in {"stdin", "functional"}:
            raise ValueError("Invalid LiveCodeBench test schema")
        if not isinstance(test["input"], str) or not isinstance(test["output"], str):
            raise ValueError("LiveCodeBench input/output must be strings")
    return tests


def user_turns(row):
    return row.get("user_turns", [row["prompt"]])


def prompt_hash(row):
    turns = user_turns(row)
    return digest(turns if len(turns) > 1 else turns[0])


def format_record(name: str, row: dict, index: int) -> dict:
    source_id = str(row.get("task_id", row.get("question_id", row.get("problem_idx", row.get("id", index)))))
    evaluation = {}
    if name in {"gsm8k", "math500", "aime24", "aime25"}:
        question = row["question"] if name == "gsm8k" else row["problem"]
        prompt = question + "\nPlease reason step by step, and put your final answer within \\boxed{}."
        answer = str(row["answer"])
        if name == "gsm8k":
            if "####" not in answer:
                raise ValueError(f"Missing GSM8K answer delimiter: {source_id}")
            answer = answer.rsplit("####", 1)[1].strip()
        evaluation = {"kind": "math", "answer": answer}
    elif name == "humaneval":
        prompt = "Write a solution to the following problem and make sure that it passes the tests:\n```python\n" + row["prompt"] + "\n```"
        evaluation = {"kind": "humaneval", "prompt": row["prompt"],
                      "test": row["test"], "entry_point": row["entry_point"],
                      "reference_code": row.get("canonical_solution", "")}
    elif name in {"mbpp", "mbpp_sanitized"}:
        description = row["text"] if name == "mbpp" else row["prompt"]
        tests = row["test_list"]
        if not tests:
            raise ValueError(f"Missing MBPP tests: {source_id}")
        # One public example specifies the function signature; remaining tests stay hidden.
        prompt = description + "\nReturn only Python code.\nExample:\n```python\n" + tests[0] + "\n```"
        setup = row.get("test_setup_code", "") or "\n".join(row.get("test_imports", []))
        evaluation = {"kind": "mbpp", "setup": setup, "tests": tests, "public_test_count": 1,
                      "challenge_tests": row.get("challenge_test_list", []),
                      "reference_code": row.get("code", "")}
    elif name == "livecodebench":
        source_id = f"{row['platform']}:{row['question_id']}"
        public = decode_lcb_tests(row["public_test_cases"])
        private = decode_lcb_tests(row["private_test_cases"])
        if not public and not private:
            raise ValueError(f"LiveCodeBench has no test cases: {source_id}")
        metadata = json.loads(row["metadata"]) if isinstance(row["metadata"], str) else row["metadata"]
        fn_name = metadata.get("func_name")
        expected_type = "functional" if fn_name else "stdin"
        if any(test["testtype"] != expected_type for test in public + private):
            raise ValueError(f"LiveCodeBench test type/function mismatch: {source_id}")
        starter = row.get("starter_code") or ""
        instruction = "Use the provided code structure." if starter else "Read from standard input and write the answer to standard output."
        prompt = ("You are an expert Python programmer. Generate a correct Python program. Return only the program in a Python code block.\n\n"
                  + "### Question:\n" + row["question_content"] + "\n\n### Format:\n" + instruction
                  + "\n```python\n" + (starter or "# YOUR CODE HERE") + "\n```\n\n### Answer:\n")
        evaluation = {"kind": "livecodebench", "fn_name": fn_name, "tests": public + private,
                      "public_test_count": len(public), "private_test_count": len(private),
                      "test_count": len(public) + len(private), "release": "release_v6",
                      "platform": row["platform"], "question_id": str(row["question_id"]),
                      "contest_date": str(row["contest_date"]), "difficulty": row["difficulty"]}
    elif name == "mt-bench":
        turns = row["turns"]
        if not isinstance(turns, list) or len(turns) != 2 or any(not isinstance(t, str) or not t.strip() for t in turns):
            raise ValueError("MT-Bench requires both nonempty user turns")
        prompt = turns[0]
        evaluation = {"kind": "mt-bench", "question_id": int(row["question_id"]),
                      "category": row["category"], "reference": row.get("reference", []),
                      "quality_protocol": "external_judge_optional"}
    else:
        raise ValueError(name)
    if not prompt.strip():
        raise ValueError(f"Empty prompt: {name}/{source_id}")
    source_question = (canonical(row["turns"]) if name == "mt-bench" else row.get("question_content",
                       row.get("question", row.get("problem", row.get("text", row.get("prompt", ""))))))
    result = {"dataset": name, "source_id": source_id, "source_index": index,
            "source_question": source_question,
            "prompt": prompt, "prompt_sha256": digest(prompt), "evaluation": evaluation}
    if name == "mt-bench":
        result["user_turns"] = row["turns"]
        result["prompt_sha256"] = prompt_hash(result)
    return result


def load_source(spec, revision, output):
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download
    if spec.loader == "lcb_jsonl":
        # Read precisely the cumulative release_v6 files; no date filtering.
        if spec.release != "release_v6":
            raise ValueError("Unsupported LiveCodeBench release")
        for filename in ["test.jsonl"] + [f"test{i}.jsonl" for i in range(2, 7)]:
            path = hf_hub_download(spec.repo, filename, repo_type="dataset", revision=revision)
            with Path(path).open() as stream:
                for line in stream:
                    if line.strip():
                        yield json.loads(line)
    elif spec.loader == "github_jsonl":
        import requests
        url = f"https://raw.githubusercontent.com/{spec.repo}/{revision}/{spec.split}"
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        for line in response.text.splitlines():
            if line.strip():
                yield json.loads(line)
    else:
        yield from load_dataset(spec.repo, spec.config, split=spec.split, revision=revision)


def persist_evaluation(record, directory):
    if record["evaluation"]["kind"] != "livecodebench":
        return record
    evaluation = record["evaluation"]
    relative = f"evaluations/livecodebench/{digest(record['source_id'])}.json"
    path = directory / relative
    write_json(path, evaluation)
    record["evaluation"] = {k: v for k, v in evaluation.items() if k != "tests"}
    record["evaluation"].update({"file": relative, "sha256": file_hash(path)})
    return record


def evaluation_path(directory, evaluation):
    path = (directory / evaluation["file"]).resolve()
    if not path.is_relative_to(directory.resolve()):
        raise ValueError("Evaluation path escapes dataset directory")
    return path


def resolve_evaluation(record, directory):
    evaluation = record["evaluation"]
    if "file" in evaluation:
        path = evaluation_path(directory, evaluation)
        if file_hash(path) != evaluation["sha256"]:
            raise ValueError("Evaluation sidecar hash mismatch")
        return json.loads(path.read_text())
    return evaluation


def prepare(names: list[str], output: Path) -> dict:
    from huggingface_hub import HfApi

    unknown = set(names) - set(DATASETS)
    if unknown:
        raise ValueError(f"Unknown datasets: {sorted(unknown)}")
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if set(manifest["datasets"]) != set(names):
            raise ValueError("Existing data manifest has different datasets; use a new output directory")
        load_prepared(output, names)
        return manifest
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"schema": 1, "coverage": "full_evaluation_split", "formatter_id": formatter_id(), "datasets": {}}
    api = HfApi()
    for name in names:
        spec = DATASETS[name]
        revision = spec.revision or api.dataset_info(spec.repo).sha
        rows = []
        for i, source in enumerate(load_source(spec, revision, output)):
            # Keep large private tests in per-question sidecars, not in RAM for all questions.
            rows.append(persist_evaluation(format_record(name, dict(source), i), output))
        if len(rows) != spec.expected:
            raise ValueError(f"{name}: expected full split of {spec.expected}, got {len(rows)}")
        if len({r["source_id"] for r in rows}) != len(rows):
            raise ValueError(f"Duplicate source IDs in {name}")
        path = output / f"{name}.jsonl"
        tmp = path.with_suffix(".tmp")
        tmp.write_text("".join(canonical(row) + "\n" for row in rows))
        tmp.replace(path)
        manifest["datasets"][name] = {**asdict(spec), "revision": revision,
                                       "count": len(rows), "file": path.name,
                                       "sha256": file_hash(path)}
        print(f"Prepared {name}: {len(rows)}/{spec.expected}", flush=True)
    write_json(manifest_path, manifest)
    return manifest


def load_prepared(directory: Path, names: list[str]) -> tuple[dict, list[dict]]:
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("coverage") != "full_evaluation_split":
        raise ValueError("Data manifest is not a full evaluation split")
    if manifest.get("formatter_id") != formatter_id():
        raise ValueError("Prompt formatting code changed; prepare a new dataset directory")
    rows = []
    for name in names:
        entry = manifest["datasets"][name]
        if any(entry.get(k) != value for k, value in asdict(DATASETS[name]).items() if k != "revision"):
            raise ValueError(f"Dataset source/config/split mismatch: {name}")
        if DATASETS[name].revision and entry["revision"] != DATASETS[name].revision:
            raise ValueError(f"Pinned dataset revision mismatch: {name}")
        path = directory / entry["file"]
        if file_hash(path) != entry["sha256"]:
            raise ValueError(f"Dataset hash mismatch: {path}")
        part = read_jsonl(path)
        if len(part) != DATASETS[name].expected or len(part) != entry["count"]:
            raise ValueError(f"Incomplete evaluation set: {name}")
        if len({r["source_id"] for r in part}) != len(part):
            raise ValueError(f"Duplicate IDs: {name}")
        if any(r["dataset"] != name or prompt_hash(r) != r["prompt_sha256"] or len(user_turns(r)) != DATASETS[name].turns for r in part):
            raise ValueError(f"Invalid prompt metadata: {name}")
        for record in part:
            evaluation = record["evaluation"]
            if "file" in evaluation and file_hash(evaluation_path(directory, evaluation)) != evaluation["sha256"]:
                raise ValueError(f"Evaluation sidecar hash mismatch: {name}/{record['source_id']}")
        rows.extend(part)
    return manifest, rows
