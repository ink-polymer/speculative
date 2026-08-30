#!/usr/bin/env python3
"""Replace train/eval overlaps in an existing benchmark with deterministic unseen rows."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("dataset")), str(row.get("source_id"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--base-eval", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    train_keys = {key(row) for row in load_jsonl(args.train)}
    base_rows = load_jsonl(args.base_eval)
    kept = [row for row in base_rows if key(row) not in train_keys]
    removed = len(base_rows) - len(kept)
    kept_keys = {key(row) for row in kept}
    removed_by_dataset = Counter(
        str(row.get("dataset")) for row in base_rows if key(row) in train_keys
    )
    candidates_by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in load_jsonl(args.pool):
        if key(row) not in train_keys and key(row) not in kept_keys:
            candidates_by_dataset[str(row.get("dataset"))].append(row)
    rng = random.Random(args.seed)
    replacements = [
        row
        for dataset in sorted(removed_by_dataset)
        for row in rng.sample(candidates_by_dataset[dataset], removed_by_dataset[dataset])
    ]
    rows = kept + replacements
    if len({key(row) for row in rows}) != len(rows):
        raise RuntimeError("output contains duplicate dataset/source_id keys")
    if any(key(row) in train_keys for row in rows):
        raise RuntimeError("output still overlaps training rows")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    counts = Counter(str(row.get("dataset")) for row in rows)
    print(json.dumps({
        "rows": len(rows),
        "removed_overlaps": removed,
        "replacement_rows": len(replacements),
        "dataset_counts": dict(sorted(counts.items())),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
