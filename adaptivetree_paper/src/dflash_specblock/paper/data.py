"""Full official splits, immutable dataset revisions, and auditable decontamination."""
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
MATH_SUBJECTS = ("algebra", "counting_and_probability", "geometry", "intermediate_algebra",
                 "number_theory", "prealgebra", "precalculus")


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


def partition_train(training, evaluation, fraction):
    if not 0 < fraction < 1:
        raise ValueError("validation_fraction must be in (0,1)")
    if any(row.get("usage") == "evaluation_only" for row in training):
        raise ValueError("Evaluation-only datasets cannot enter the training source pool")
    heldout = set().union(*(question_hashes(row) for row in evaluation))
    heldout_code = {row["source_id"] for row in evaluation if row["dataset"].startswith("mbpp")}
    seen, train, dev, removed = set(), [], [], []
    for row in training:
        key = row["prompt_hash"]
        reason = None
        if question_hashes(row) & heldout or (row["dataset"].startswith("mbpp") and row["source_id"] in heldout_code):
            reason = "evaluation_overlap"
        elif key in seen:
            reason = "training_duplicate"
        if reason:
            removed.append({"id": row["id"], "reason": reason})
            continue
        seen.add(key)
        # Prompt-group partition, not a per-round random split.
        destination = dev if int(key[:16], 16) / 2**64 < fraction else train
        destination.append(row)
    if not train or not dev:
        raise ValueError("Train/development partition is empty")
    return train, dev, removed


def check_manifest(directory: Path):
    manifest = load_json(directory / "manifest.json")
    if manifest.get("version") != 2:
        raise ValueError("Dataset suite/schema changed; prepare the expanded full suite in a new data directory")
    expected = {"train.jsonl", "dev.jsonl"} | {f"{name}.jsonl" for name in SOURCES}
    if set(manifest["files"]) != expected or manifest.get("full_evaluation_splits") is not True:
        raise ValueError("Incomplete dataset manifest")
    revisions = manifest["source_revisions"]
    repositories = {spec[0] for spec in SOURCES.values()} | {"EleutherAI/hendrycks_math"}
    if (set(revisions) != repositories or
            any(not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{40}", sha) for sha in revisions.values())):
        raise ValueError("Dataset revisions must be immutable commit SHAs")
    if load_json(directory / "source_revisions.json") != revisions:
        raise ValueError("Dataset source revision lock changed")
    if revisions["lm-sys/FastChat"] != MTBENCH_REVISION or manifest.get("source_layout") != source_layout():
        raise ValueError("MT-Bench revision or LiveCodeBench release layout changed")
    for filename, info in manifest["files"].items():
        path = directory / filename
        if file_hash(path) != info["sha256"] or len(read_rows(path)) != info["rows"]:
            raise ValueError(f"Dataset manifest mismatch: {path}")
    return manifest


def validate_rows(training, development, evaluation, fraction):
    """Recompute prompt identity and role checks instead of trusting saved labels."""
    seen_ids = set()
    for role, rows in (("train", training), ("dev", development), ("test", evaluation)):
        for row in rows:
            if row["id"] in seen_ids:
                raise ValueError("A source row appears in multiple partitions")
            seen_ids.add(row["id"])
            rebuilt = make_row(row["dataset"], row["reference"], row["source_index"],
                               row["split"], row["source"], row["subset"])
            if rebuilt != row:
                raise ValueError("Prompt/reference identity or hash changed")
            if role == "test":
                if not is_evaluation_row(row) or row["usage"] != "evaluation_only":
                    raise ValueError("Invalid evaluation source/split")
            else:
                valid_source = ((row["dataset"], row["source"], row["subset"]) in {
                    ("gsm8k", "openai/gsm8k", "main"),
                    ("mbpp", "google-research-datasets/mbpp", "full")}
                    or (row["dataset"] == "math_train" and row["source"] == "EleutherAI/hendrycks_math"
                        and row["subset"] in MATH_SUBJECTS))
                if row["split"] != "train" or not valid_source:
                    raise ValueError("Training/development must originate from official training splits")
                assigned = "dev" if int(row["prompt_hash"][:16], 16) / 2**64 < fraction else "train"
                if assigned != role:
                    raise ValueError("Train/development hash partition changed")
    train_hashes = {r["prompt_hash"] for r in training}
    dev_hashes = {r["prompt_hash"] for r in development}
    test_hashes = {r["prompt_hash"] for r in evaluation}
    if (len(train_hashes) != len(training) or len(dev_hashes) != len(development)
            or train_hashes & (dev_hashes | test_hashes) or dev_hashes & test_hashes):
        raise ValueError("Prompt duplication/leakage detected")
    heldout_questions = set().union(*(question_hashes(row) for row in evaluation))
    if any(question_hashes(row) & heldout_questions for row in training + development):
        raise ValueError("Raw question/turn leakage detected")
    heldout_code = {r["source_id"] for r in evaluation if r["dataset"].startswith("mbpp")}
    if any(r["source_id"] in heldout_code for r in training + development if r["dataset"] == "mbpp"):
        raise ValueError("MBPP task-ID leakage detected")


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


def prepare(directory: Path, fraction=0.1):
    from datasets import load_dataset
    from huggingface_hub import HfApi
    with run_lock(directory):
        if (directory / "manifest.json").exists():
            existing = check_manifest(directory)
            if existing["validation_fraction"] != fraction:
                raise ValueError("Data partition configuration changed")
            return existing
        lock_path = directory / "source_revisions.json"
        repositories = {spec[0] for spec in SOURCES.values()} | {"EleutherAI/hendrycks_math"}
        if lock_path.exists():
            revisions = load_json(lock_path)
            if set(revisions) != repositories or revisions["lm-sys/FastChat"] != MTBENCH_REVISION:
                raise ValueError("Source list changed; use a new data directory")
        else:
            api = HfApi()
            revisions = {repo: MTBENCH_REVISION if repo == "lm-sys/FastChat" else api.dataset_info(repo).sha
                         for repo in sorted(repositories)}
            atomic_json(lock_path, revisions)
        evaluations, training, raw_counts = {}, [], {}

        def fetch(name, repo, subset, split, expected=None):
            args = [repo] + ([subset] if subset else [])
            data = load_dataset(*args, split=split, revision=revisions[repo])
            if expected is not None and len(data) != expected:
                raise ValueError(f"Unexpected full split count {repo}/{subset}/{split}: {len(data)} != {expected}")
            raw_counts[f"{repo}/{subset}/{split}"] = len(data)
            return [make_row(name, row, i, split, repo, subset) for i, row in enumerate(data)]

        for name, (repo, subset, split, expected) in SOURCES.items():
            rows = [make_row(name, row, i, split, repo, subset)
                    for i, row in enumerate(load_evaluation_source(name, revisions[repo]))]
            if len(rows) != expected:
                raise ValueError(f"Unexpected full evaluation count for {name}: {len(rows)} != {expected}")
            raw_counts[f"{repo}/{subset}/{split}"] = len(rows)
            evaluations[name] = rows
        training += fetch("gsm8k", "openai/gsm8k", "main", "train", 7473)
        training += fetch("mbpp", "google-research-datasets/mbpp", "full", "train", 374)
        math_training = []
        for subset in MATH_SUBJECTS:
            math_training += fetch("math_train", "EleutherAI/hendrycks_math", subset, "train")
        if len(math_training) != 7500:
            raise ValueError(f"Expected the complete 7500-row MATH training split, got {len(math_training)}")
        training += math_training
        train, dev, removed = partition_train(training, sum(evaluations.values(), []), fraction)
        outputs = {"train.jsonl": train, "dev.jsonl": dev}
        outputs.update({f"{name}.jsonl": rows for name, rows in evaluations.items()})
        files = {}
        for filename, rows in outputs.items():
            path = directory / filename
            if len({r["id"] for r in rows}) != len(rows):
                raise ValueError(f"Duplicate source IDs in {filename}")
            temp = path.with_suffix(".jsonl.tmp")
            with temp.open("w", encoding="utf-8") as stream:
                for row in rows:
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            temp.replace(path)
            files[filename] = {"rows": len(rows), "sha256": file_hash(path)}
        manifest = {"version": 2, "full_evaluation_splits": True,
                    "validation_fraction": fraction, "source_revisions": revisions,
                    "raw_counts": raw_counts, "excluded_training_rows": removed, "files": files,
                    "source_layout": source_layout(), "main_datasets": list(MAIN_DATASETS),
                    "auxiliary_datasets": ["mbpp_sanitized"]}
        atomic_json(directory / "manifest.json", manifest)
        return manifest
