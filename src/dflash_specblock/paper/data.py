"""Historical non-default full-split loader; current official data uses official_data.py."""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

from .common import atomic_json, digest, file_hash, load_json, read_rows, run_lock

SOURCES = {
    "gsm8k": ("openai/gsm8k", "main", "test", 1319),
    "math500": ("HuggingFaceH4/MATH-500", None, "test", 500),
    "humaneval": ("openai/openai_humaneval", None, "test", 164),
    "mbpp": ("google-research-datasets/mbpp", "full", "test", 500),
    # The upstream AIME split is called train, but is evaluation-only here.
    "aime25": ("MathArena/aime_2025", None, "train", 30),
    "livecodebench": ("livecodebench/code_generation_lite", "release_v6", "test", 1055),
    "mt-bench": ("lm-sys/FastChat", None, "test", 80),
    # Separate dataset/row in every table, never pooled with MBPP-full.
    "mbpp_sanitized": ("google-research-datasets/mbpp", "sanitized", "test", 257),
}
MAIN_DATASETS = tuple(name for name in SOURCES if name != "mbpp_sanitized")
MTBENCH_REVISION = "587d5cfa1609a43d192cedb8441cac3c17db105d"
MTBENCH_FILE = "fastchat/llm_judge/data/mt_bench/question.jsonl"
LCB_FILES = ("test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl", "test6.jsonl")


def normalize(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def expected_turns(name):
    return 2 if name == "mt-bench" else 1


def user_turns(row):
    turns = row.get("user_turns", [row["prompt"]])
    if (not isinstance(turns, list) or len(turns) != expected_turns(row["dataset"])
            or any(not isinstance(t, str) or not t.strip() for t in turns)):
        raise ValueError("Invalid or incomplete user turns")
    return turns


def is_evaluation_row(row):
    spec = SOURCES.get(row["dataset"])
    return spec is not None and (row["source"], row["subset"], row["split"]) == spec[:3]


def question_hashes(row):
    """Also check each raw question/turn, regardless of prompt wrapper changes."""
    reference = row["reference"]
    raw = reference.get("turns") if row["dataset"] == "mt-bench" else [reference.get("question_content",
        reference.get("question", reference.get("problem", reference.get("text", reference.get("prompt")))))]
    return {row["prompt_hash"]} | {digest(normalize(text)) for text in raw if isinstance(text, str)}


def canonical_prompt(name, row):
    if name == "gsm8k":
        question = row["question"]
    elif name.startswith("math") or name == "aime25":
        question = row["problem"]
    elif name == "humaneval":
        return "Write a solution to the following problem and make sure that it passes the tests:\n```python\n" + row["prompt"] + "\n```"
    elif name == "livecodebench":
        question, starter = row["question_content"], row.get("starter_code") or ""
        if not isinstance(question, str) or not question.strip() or not isinstance(starter, str):
            raise ValueError("Invalid LiveCodeBench question/starter code")
        instruction = ("Use the provided code structure." if starter else
                       "Read from standard input and write the answer to standard output.")
        return ("You are an expert Python programmer. Generate a correct Python program. "
                "Return only the program in a Python code block.\n\n### Question:\n" + question
                + "\n\n### Format:\n" + instruction + "\n```python\n"
                + (starter or "# YOUR CODE HERE") + "\n```\n\n### Answer:\n")
    elif name == "mt-bench":
        turns = row["turns"]
        if (not isinstance(turns, list) or len(turns) != 2
                or any(not isinstance(t, str) or not t.strip() for t in turns)):
            raise ValueError("MT-Bench requires both nonempty user turns")
        return turns[0]
    else:
        return row.get("prompt", row.get("text"))
    return question + "\nPlease reason step by step, and put your final answer within \\boxed{}."


def make_row(name, row, index, split, source, subset):
    prompt = canonical_prompt(name, row)
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"Missing prompt in {name}:{split}:{index}")
    source_id = str(row.get("task_id", row.get("unique_id", row.get("question_id",
                    row.get("problem_idx", row.get("id", index))))))
    if name == "livecodebench":
        source_id = f"{row['platform']}:{row['question_id']}"
        # This is a generation-speed protocol, not an automatic code judge. Keep
        # only public task metadata in RAM; never unpickle remote private tests.
        keep = {"question_title", "question_content", "platform", "question_id", "contest_id",
                "contest_date", "starter_code", "difficulty", "metadata"}
        row = {key: value for key, value in row.items() if key in keep}
    result = {"id": f"{name}/{subset or 'default'}/{split}/{source_id}",
            "dataset": name, "source_id": source_id, "split": split,
            "source": source, "subset": subset, "source_index": index,
            "prompt": prompt, "prompt_hash": digest(normalize(prompt)),
            "reference": dict(row)}
    if name == "mt-bench":
        result["user_turns"] = list(row["turns"])
        result["prompt_hash"] = digest([normalize(turn) for turn in row["turns"]])
    result["usage"] = "evaluation_only" if is_evaluation_row(result) else "training_source"
    return result


def source_layout():
    return {"livecodebench": {"release": "release_v6", "files": list(LCB_FILES),
                               "scenario": "code_generation", "date_filter": None},
            "mt-bench": {"provider": "github", "file": MTBENCH_FILE,
                         "revision": MTBENCH_REVISION, "turns": 2},
            "aime25": {"source_split": "train", "usage": "evaluation_only"}}


def load_evaluation_source(name, revision):
    """Stream raw JSONL for LCB/MT; never execute a downloaded dataset script."""
    repo, subset, split, _ = SOURCES[name]
    if name == "livecodebench":
        from huggingface_hub import hf_hub_download
        for filename in LCB_FILES:
            path = hf_hub_download(repo, filename, repo_type="dataset", revision=revision)
            with Path(path).open(encoding="utf-8") as stream:
                for line in stream:
                    if line.strip():
                        yield json.loads(line)
    elif name == "mt-bench":
        from urllib.request import urlopen
        if revision != MTBENCH_REVISION:
            raise ValueError("Unexpected MT-Bench revision")
        url = f"https://raw.githubusercontent.com/{repo}/{revision}/{MTBENCH_FILE}"
        with urlopen(url, timeout=30) as stream:
            for line in stream:
                if line.strip():
                    yield json.loads(line)
    else:
        from datasets import load_dataset
        args = [repo] + ([subset] if subset else [])
        yield from load_dataset(*args, split=split, revision=revision)


def validate_evaluation(rows):
    seen = set()
    for row in rows:
        if row["id"] in seen or not is_evaluation_row(row) or row["usage"] != "evaluation_only":
            raise ValueError("Duplicate or non-evaluation source row")
        seen.add(row["id"])
        rebuilt = make_row(row["dataset"], row["reference"], row["source_index"],
                           row["split"], row["source"], row["subset"])
        if rebuilt != row:
            raise ValueError("Prompt/reference identity or hash changed")


def check_manifest(directory: Path):
    manifest = load_json(directory / "manifest.json")
    expected = {f"{name}.jsonl" for name in SOURCES}
    if (manifest.get("version") != 3 or manifest.get("training") is not False
            or manifest.get("full_evaluation_splits") is not True or set(manifest["files"]) != expected):
        raise ValueError("Expected evaluation-only v3 manifest; use a new data directory")
    revisions = manifest["source_revisions"]
    if (set(revisions) != {spec[0] for spec in SOURCES.values()}
            or any(not isinstance(v, str) or not re.fullmatch(r"[0-9a-f]{40}", v) for v in revisions.values())):
        raise ValueError("Dataset revisions must be immutable commit SHAs")
    if load_json(directory / "source_revisions.json") != revisions:
        raise ValueError("Dataset source revision lock changed")
    if revisions["lm-sys/FastChat"] != MTBENCH_REVISION or manifest["source_layout"] != source_layout():
        raise ValueError("Frozen release layout changed")
    all_rows = []
    for name, spec in SOURCES.items():
        filename = f"{name}.jsonl"
        path, info = directory / filename, manifest["files"][filename]
        rows = read_rows(path)
        if (file_hash(path) != info["sha256"] or len(rows) != info["rows"]
                or len(rows) != spec[3] or any(r["dataset"] != name for r in rows)):
            raise ValueError(f"Full dataset manifest mismatch: {filename}")
        all_rows.extend(rows)
    validate_evaluation(all_rows)
    return manifest


def prepare(directory: Path):
    from huggingface_hub import HfApi
    with run_lock(directory):
        if (directory / "manifest.json").exists():
            return check_manifest(directory)
        # Do not silently inherit the former RL train/dev suite.
        if any((directory / f).exists() for f in ("train.jsonl", "dev.jsonl")):
            raise ValueError("Old training data directory; use a new evaluation-only directory")
        lock_path = directory / "source_revisions.json"
        repositories = {spec[0] for spec in SOURCES.values()}
        if lock_path.exists():
            revisions = load_json(lock_path)
        else:
            api = HfApi()
            revisions = {repo: MTBENCH_REVISION if repo == "lm-sys/FastChat" else api.dataset_info(repo).sha
                         for repo in sorted(repositories)}
            atomic_json(lock_path, revisions)
        if (set(revisions) != repositories or revisions.get("lm-sys/FastChat") != MTBENCH_REVISION
                or any(not isinstance(v, str) or not re.fullmatch(r"[0-9a-f]{40}", v) for v in revisions.values())):
            raise ValueError("Invalid frozen source lock")
        files = {}
        for name, (repo, subset, split, expected) in SOURCES.items():
            rows = [make_row(name, row, i, split, repo, subset)
                    for i, row in enumerate(load_evaluation_source(name, revisions[repo]))]
            if len(rows) != expected:
                raise ValueError(f"Unexpected full evaluation count for {name}: {len(rows)} != {expected}")
            validate_evaluation(rows)
            path = directory / f"{name}.jsonl"
            temp = path.with_suffix(".jsonl.tmp")
            with temp.open("w", encoding="utf-8") as stream:
                for row in rows:
                    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            temp.replace(path)
            files[path.name] = {"rows": len(rows), "sha256": file_hash(path)}
        manifest = {"version": 3, "training": False, "full_evaluation_splits": True,
                    "source_revisions": revisions, "files": files, "source_layout": source_layout(),
                    "main_datasets": list(MAIN_DATASETS), "auxiliary_datasets": ["mbpp_sanitized"]}
        atomic_json(directory / "manifest.json", manifest)
        return check_manifest(directory)
