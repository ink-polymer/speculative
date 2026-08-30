#!/usr/bin/env python3
"""Remove rank-training rows whose dataset/source ID appears in an evaluation set."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def canonical_dataset(row: dict[str, Any], dataset_field: str) -> str:
    value = row.get(dataset_field)
    if value is None and dataset_field == "source_dataset":
        value = row.get("dataset")
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def key(row: dict[str, Any], dataset_field: str) -> tuple[str, str]:
    return canonical_dataset(row, dataset_field), str(row.get("source_id"))


def normalized_prompt(row: dict[str, Any]) -> str:
    return re.sub(r"\s+", " ", str(row.get("prompt", ""))).strip().lower()


def prompts_overlap(left: str, right: str) -> bool:
    if not left or not right:
        return False
    if left == right:
        return True
    # Some suites wrap the original problem in an instruction template.  Long
    # containment catches the same underlying task without matching generic fragments.
    return min(len(left), len(right)) >= 64 and (left in right or right in left)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--eval", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    eval_rows = [
        json.loads(line)
        for line in args.eval.open("r", encoding="utf-8")
        if line.strip()
    ]
    eval_keys = {key(row, "dataset") for row in eval_rows}
    eval_prompts_by_dataset: dict[str, list[str]] = {}
    for row in eval_rows:
        dataset = canonical_dataset(row, "dataset")
        eval_prompts_by_dataset.setdefault(dataset, []).append(normalized_prompt(row))
    kept: list[dict[str, Any]] = []
    removed: list[dict[str, str]] = []
    for line in args.train.open("r", encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        row_key = key(row, "source_dataset")
        if row_key in eval_keys:
            removed.append({"reason": "dataset_source_id", "dataset": row_key[0], "source_id": row_key[1]})
            continue
        train_prompt = normalized_prompt(row)
        if any(
            prompts_overlap(train_prompt, eval_prompt)
            for eval_prompt in eval_prompts_by_dataset.get(row_key[0], [])
        ):
            removed.append({"reason": "prompt_content", "dataset": row_key[0], "source_id": row_key[1]})
            continue
        kept.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for row in kept:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({
        "input_rows": len(kept) + len(removed),
        "kept_rows": len(kept),
        "removed_rows": len(removed),
        "removed": removed,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
