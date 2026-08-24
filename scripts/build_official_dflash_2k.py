#!/usr/bin/env python3
"""Build a deterministic 2K suite from DFlash's official single-turn benchmarks."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party" / "dflash_official"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=2000)
    args = parser.parse_args()

    from datasets import load_dataset
    from dflash.benchmark import DATASETS

    # MT-Bench is deliberately excluded from the fixed JSONL because its second
    # turn depends on each method's first-turn answer. These four are the exact
    # official DFlash single-turn loaders and prompt formatters.
    names = ("gsm8k", "math500", "humaneval", "mbpp")
    rows: list[dict] = []
    for name in names:
        cfg = DATASETS[name]
        dataset = load_dataset(*cfg["load_args"], **cfg["load_kwargs"])
        order = list(range(len(dataset)))
        random.Random(42).shuffle(order)
        for source_index in order:
            item = dataset[source_index]
            rows.append({
                "dataset": name,
                "source_id": str(item.get("id", item.get("task_id", source_index))),
                "prompt": cfg["format"](item),
            })

    random.Random(42).shuffle(rows)
    rows = rows[: args.count]
    if len(rows) != args.count:
        raise RuntimeError(f"requested {args.count} prompts but only found {len(rows)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for index, row in enumerate(rows):
            stream.write(json.dumps({"index": index, **row}, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(args.output), "rows": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
