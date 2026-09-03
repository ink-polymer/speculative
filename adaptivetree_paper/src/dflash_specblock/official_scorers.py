"""官方 benchmark scorer。"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


_ARTICLES = {"a", "an", "the"}


def normalize_qa_text(text: str) -> str:
    lowered = text.lower()
    cleaned = re.sub(r"[^a-z0-9\s]", " ", lowered)
    tokens = [token for token in cleaned.split() if token and token not in _ARTICLES]
    return " ".join(tokens)


def qa_f1_score(prediction: str, reference: str) -> float:
    pred_tokens = normalize_qa_text(prediction).split()
    ref_tokens = normalize_qa_text(reference).split()
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    common: dict[str, int] = {}
    for token in pred_tokens:
        common[token] = common.get(token, 0) + 1
    overlap = 0
    for token in ref_tokens:
        count = common.get(token, 0)
        if count > 0:
            overlap += 1
            common[token] = count - 1
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def score_nq_open(prediction: str, answers: list[str]) -> dict[str, Any]:
    normalized_prediction = normalize_qa_text(prediction)
    normalized_answers = [normalize_qa_text(answer) for answer in answers if isinstance(answer, str)]
    exact_match = any(normalized_prediction == answer for answer in normalized_answers)
    f1 = max((qa_f1_score(prediction, answer) for answer in answers if isinstance(answer, str)), default=0.0)
    return {"exact_match": float(exact_match), "f1": f1}


def normalize_math_answer(text: str) -> str:
    value = text.strip()
    value = value.replace("\\left", "").replace("\\right", "")
    value = value.replace("$", "").replace(" ", "")
    value = re.sub(r"\\boxed\{(.+)\}", r"\1", value)
    value = re.sub(r"\\text\{([^}]*)\}", r"\1", value)
    value = value.strip(".;,")
    return value.lower()


def extract_math_answer(text: str) -> str:
    boxed = re.findall(r"\\boxed\{([^}]*)\}", text)
    if boxed:
        return boxed[-1]
    final_patterns = [
        r"(?i)final answer(?: is|:)\s*([^\n]+)",
        r"(?i)answer(?: is|:)\s*([^\n]+)",
    ]
    for pattern in final_patterns:
        matched = re.findall(pattern, text)
        if matched:
            return matched[-1].strip()
    numbers = re.findall(r"-?\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?", text)
    if numbers:
        return numbers[-1]
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def score_math_500(prediction: str, reference: str) -> dict[str, Any]:
    predicted_answer = normalize_math_answer(extract_math_answer(prediction))
    target_answer = normalize_math_answer(reference)
    return {
        "predicted_answer": predicted_answer,
        "reference_answer": target_answer,
        "correct": float(bool(predicted_answer and predicted_answer == target_answer)),
    }


def extract_code_block(text: str) -> str:
    fenced = re.findall(r"```(?:python)?\s*([\s\S]*?)```", text)
    if fenced:
        return fenced[0].strip("\n")
    return text.rstrip()


def _clean_humaneval_completion(text: str) -> str:
    value = extract_code_block(text)
    stop_markers = [
        "\nif __name__ ==",
        "\n```",
        "\nclass ",
        "\nprint(",
    ]
    for marker in stop_markers:
        if marker in value:
            value = value.split(marker, 1)[0]
    return value.rstrip()


def build_humaneval_program(prompt: str, completion: str, test: str, entry_point: str) -> str:
    body = _clean_humaneval_completion(completion)
    if re.search(rf"^\s*def\s+{re.escape(entry_point)}\s*\(", body, flags=re.MULTILINE):
        candidate = body
    else:
        candidate = prompt + body
    return (
        f"{candidate}\n\n"
        f"{test}\n\n"
        f"check({entry_point})\n"
        f"print('PASS')\n"
    )


def score_humaneval(
    prompt: str,
    completion: str,
    test: str,
    entry_point: str,
    timeout_seconds: float = 5.0,
) -> dict[str, Any]:
    program = build_humaneval_program(prompt, completion, test, entry_point)
    with tempfile.TemporaryDirectory(prefix="humaneval_") as temp_dir:
        path = Path(temp_dir) / "candidate.py"
        path.write_text(program, encoding="utf-8")
        try:
            result = subprocess.run(
                [sys.executable, str(path)],
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return {"passed": 0.0, "status": "timeout"}
    passed = result.returncode == 0 and "PASS" in result.stdout
    status = "passed" if passed else "failed"
    error = result.stderr.strip() or result.stdout.strip()
    return {"passed": float(passed), "status": status, "error": error}


def _simple_sentence_bleu(prediction: str, reference: str) -> float:
    pred_tokens = prediction.strip().split()
    ref_tokens = reference.strip().split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    overlap = 0
    counts: dict[str, int] = {}
    for token in ref_tokens:
        counts[token] = counts.get(token, 0) + 1
    for token in pred_tokens:
        count = counts.get(token, 0)
        if count > 0:
            overlap += 1
            counts[token] = count - 1
    precision = overlap / len(pred_tokens)
    brevity_penalty = min(1.0, len(pred_tokens) / len(ref_tokens))
    return 100.0 * precision * brevity_penalty


def summarize_translation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    predictions = [str(row["prediction"]) for row in rows]
    references = [[str(row["reference"])] for row in rows]
    try:
        import sacrebleu  # type: ignore

        bleu = sacrebleu.corpus_bleu(predictions, references).score
        return {"bleu": bleu, "metric": "sacrebleu"}
    except Exception:
        scores = [
            _simple_sentence_bleu(str(row["prediction"]), str(row["reference"]))
            for row in rows
        ]
        return {
            "bleu": sum(scores) / len(scores) if scores else 0.0,
            "metric": "simple_bleu",
        }


def summarize_task_metrics(dataset: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    if dataset == "humaneval":
        values = [float(row["score"].get("passed", 0.0)) for row in rows]
        return {"pass_at_1": sum(values) / len(values) if values else 0.0}
    if dataset == "math_500":
        values = [float(row["score"].get("correct", 0.0)) for row in rows]
        return {"accuracy": sum(values) / len(values) if values else 0.0}
    if dataset == "nq_open":
        exact = [float(row["score"].get("exact_match", 0.0)) for row in rows]
        f1 = [float(row["score"].get("f1", 0.0)) for row in rows]
        count = len(rows)
        return {
            "exact_match": sum(exact) / count if count else 0.0,
            "f1": sum(f1) / count if count else 0.0,
        }
    if dataset == "translation":
        translation_rows = [
            {
                "prediction": row["prediction"],
                "reference": row["score"]["reference"],
            }
            for row in rows
        ]
        return summarize_translation(translation_rows)
    if dataset == "mt_bench":
        values = [row["score"].get("judge_score") for row in rows if row["score"].get("judge_score") is not None]
        return {"judge_score": sum(values) / len(values) if values else None}
    if dataset == "alpaca":
        values = [row["score"].get("win_rate") for row in rows if row["score"].get("win_rate") is not None]
        return {"win_rate": sum(values) / len(values) if values else None}
    return {}


def score_dataset_row(dataset: str, row: dict[str, Any], prediction: str) -> dict[str, Any]:
    metadata = row.get("metadata") or {}
    if dataset == "humaneval":
        return score_humaneval(
            prompt=str(row["prompt"]),
            completion=prediction,
            test=str(metadata["test"]),
            entry_point=str(metadata["entry_point"]),
        )
    if dataset == "math_500":
        return score_math_500(prediction, str(metadata["answer"]))
    if dataset == "nq_open":
        answers = metadata.get("answers") or []
        return score_nq_open(prediction, [str(answer) for answer in answers])
    if dataset == "translation":
        return {"reference": str(metadata["reference"])}
    return {}


def dumps_json(data: dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True)
