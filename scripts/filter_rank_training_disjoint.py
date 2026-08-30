#!/usr/bin/env python3
"""Remove rank-training rows whose dataset/source ID appears in an evaluation set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def key(row: dict[str, Any], dataset_field: str) -> tuple[str, str]:
    return str(row.get(dataset_field)), str(row.get("source_id"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--eval", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    eval_keys = {
        key(json.loads(line), "dataset")
        for line in args.eval.open("r", encoding="utf-8")
        if line.strip()
    }
    kept: list[dict[str, Any]] = []
    removed: list[tuple[str, str]] = []
    for line in args.train.open("r", encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        row_key = key(row, "source_dataset")
        if row_key in eval_keys:
            removed.append(row_key)
        else:
            kept.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for row in kept:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({
        "input_rows": len(kept) + len(removed),
        "kept_rows": len(kept),
        "removed_rows": len(removed),
        "removed_keys": removed,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
