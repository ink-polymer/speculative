"""Offline FastChat-compatible exports and audited imports; no API requests."""
from __future__ import annotations

from collections import defaultdict
import csv
import json
from pathlib import Path

from .common import canonical, digest, file_hash, read_jsonl, write_json
from .data import load_prepared, user_turns
from .report import validate_results
from .runner import key


def export_answers(run_dir: Path, data_dir: Path, output: Path):
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    records = read_jsonl(run_dir / "results.jsonl")
    validate_results(manifest, records)
    data_manifest, rows = load_prepared(data_dir, ["mt-bench"])
    if data_manifest != manifest["data_manifest"]:
        raise ValueError("Export data do not match generation data")
    lookup = {r["source_id"]: r for r in rows}
    if len(lookup) != 80:
        raise ValueError("MT-Bench export requires all 80 conversations")
    existing = output / "export_manifest.json"
    if existing.exists() and json.loads(existing.read_text())["run_id"] != manifest["run_id"]:
        raise ValueError("Existing MT-Bench export belongs to another run; use a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    questions = [{"question_id": r["evaluation"]["question_id"], "category": r["evaluation"]["category"],
                  "turns": user_turns(r), **({"reference": r["evaluation"]["reference"]} if r["evaluation"]["reference"] else {})} for r in rows]
    (output / "question.jsonl").write_text("".join(canonical(q) + "\n" for q in questions))
    answers, identities = defaultdict(list), {}
    for record in records:
        if record["dataset"] != "mt-bench":
            continue
        model_id = f"{record['variant']}__seed{record['seed']}"
        turns = record.get("turn_results", [])
        if len(turns) != 2:
            raise ValueError("Incomplete MT-Bench conversation")
        question_id = lookup[record["source_id"]]["evaluation"]["question_id"]
        answers[model_id].append({"question_id": question_id, "answer_id": digest([manifest["run_id"], key(record)])[:20],
                                 "model_id": model_id, "choices": [{"index": 0, "turns": [t["text"] for t in turns]}], "tstamp": 0})
        identities[f"{model_id}:{question_id}"] = {"variant": record["variant"], "seed": record["seed"],
            "source_id": record["source_id"], "prediction_sha256": digest(record["text"])}
    if not answers:
        raise ValueError("No MT-Bench results in this run")
    answer_dir = output / "model_answer"
    answer_dir.mkdir(exist_ok=True)
    for name, items in answers.items():
        if len(items) != 80:
            raise ValueError(f"Incomplete answers for {name}")
        (answer_dir / f"{name}.jsonl").write_text("".join(canonical(item) + "\n" for item in sorted(items, key=lambda x: x["question_id"])))
    files = [output / "question.jsonl"] + sorted(answer_dir.glob("*.jsonl"))
    result = {"run_id": manifest["run_id"], "models": sorted(answers), "records": identities,
              "expected_judgments": 2 * len(identities),
              "files": {str(p.relative_to(output)): file_hash(p) for p in files},
              "api_calls_made": 0}
    write_json(output / "export_manifest.json", result)
    return result


def import_judgments(run_dir: Path, export_dir: Path, judgments: Path, output: Path):
    exported = json.loads((export_dir / "export_manifest.json").read_text())
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    if exported["run_id"] != manifest["run_id"]:
        raise ValueError("Judgment export belongs to another run")
    for name, expected in exported["files"].items():
        if file_hash(export_dir / name) != expected:
            raise ValueError("Exported answers/questions changed")
    records = read_jsonl(run_dir / "results.jsonl")
    validate_results(manifest, records)
    lookup = {key(r): r for r in records}
    expected = {(name, turn) for name in exported["records"] for turn in (1, 2)}
    seen, scores, judge_models = set(), [], set()
    for row in read_jsonl(judgments):
        identity = f"{row['model']}:{row['question_id']}"
        item = (identity, row["turn"])
        if item not in expected or item in seen:
            raise ValueError("Duplicate or unexpected MT-Bench judgment")
        value = row["score"]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 1 <= value <= 10:
            raise ValueError("Invalid/unparsed MT-Bench judge score")
        ref = exported["records"][identity]
        prediction = lookup[(ref["variant"], "mt-bench", ref["source_id"], ref["seed"])]
        if digest(prediction["text"]) != ref["prediction_sha256"]:
            raise ValueError("Judgments refer to changed model answers")
        judge = row["judge"]
        if not isinstance(judge, list) or len(judge) != 2:
            raise ValueError("Expected FastChat [judge_model, prompt_template] metadata")
        judge_models.add(judge[0])
        scores.append({"run_id": manifest["run_id"], **ref, "turn": row["turn"],
                       "metric": "mtbench_judge_1_to_10", "score": float(value), "judge": judge})
        seen.add(item)
    if seen != expected:
        raise ValueError(f"Incomplete MT-Bench judgments: {len(seen)}/{len(expected)}")
    if len(judge_models) != 1:
        raise ValueError("MT-Bench judgments mix different judge models")
    output.mkdir(parents=True, exist_ok=True)
    (output / "mt_bench_judge_scores.jsonl").write_text("".join(canonical(r) + "\n" for r in scores))
    grouped = defaultdict(list)
    for score in scores:
        grouped[score["variant"], score["turn"]].append(score["score"])
        grouped[score["variant"], "all"].append(score["score"])
    table = [{"variant": name, "turn": turn, "count": len(values), "mean_score_1_to_10": sum(values) / len(values)}
             for (name, turn), values in grouped.items()]
    with (output / "mt_bench_judge_summary.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    result = {"run_id": manifest["run_id"], "judge_model": next(iter(judge_models)),
              "judgments_sha256": file_hash(judgments), "count": len(scores), "complete": True,
              "note": "External 1–10 judge scores, separate from objective accuracy/pass@1."}
    write_json(output / "mt_bench_judge_manifest.json", result)
    return result
