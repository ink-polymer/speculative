#!/usr/bin/env python3
"""Build the held-out part of the four-domain official DFlash prompt pool.

The official 2K suite is sampled from 2,240 test prompts.  This script removes
every prompt/source id present in that suite and emits the remaining 240 rows.
It is therefore distribution-matched while remaining prompt-disjoint from the
2K evaluation set.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Callable


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-suite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-output", type=Path, default=None)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument(
        "--arrow-root",
        type=Path,
        default=None,
        help="Optional Hugging Face datasets cache root containing the four Arrow files.",
    )
    parser.add_argument("--seed", type=int, default=1729)
    return parser


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.expanduser().resolve().read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _find_arrow(root: Path, filename: str) -> Path:
    matches = sorted(root.expanduser().resolve().glob(f"**/{filename}"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected exactly one {filename} below {root}, found {len(matches)}"
        )
    return matches[0]


def _load_sources(arrow_root: Path | None) -> dict[str, Any]:
    from datasets import Dataset, load_dataset

    if arrow_root is not None:
        return {
            "gsm8k": Dataset.from_file(str(_find_arrow(arrow_root, "gsm8k-test.arrow"))),
            "math500": Dataset.from_file(str(_find_arrow(arrow_root, "math-500-test.arrow"))),
            "humaneval": Dataset.from_file(
                str(_find_arrow(arrow_root, "openai_humaneval-test.arrow"))
            ),
            "mbpp": Dataset.from_file(str(_find_arrow(arrow_root, "mbpp-test.arrow"))),
        }
    return {
        "gsm8k": load_dataset("openai/gsm8k", "main", split="test"),
        "math500": load_dataset("HuggingFaceH4/MATH-500", split="test"),
        "humaneval": load_dataset("openai/openai_humaneval", split="test"),
        "mbpp": load_dataset(
            "google-research-datasets/mbpp", "sanitized", split="test"
        ),
    }


def main() -> None:
    args = _parser().parse_args()
    evaluation = _read_jsonl(args.official_suite)
    evaluation_keys = {
        (str(row["dataset"]), str(row["source_id"])) for row in evaluation
    }
    evaluation_prompts = {str(row["prompt"]) for row in evaluation}
    if len(evaluation) != len(evaluation_keys):
        raise ValueError("official suite contains duplicate source ids")

    formatters: dict[str, Callable[[dict[str, Any]], str]] = {
        "gsm8k": lambda x: (
            f"{x['question']}\nPlease reason step by step, and put your final answer "
            "within \\boxed{}."
        ),
        "math500": lambda x: (
            f"{x['problem']}\nPlease reason step by step, and put your final answer "
            "within \\boxed{}."
        ),
        "humaneval": lambda x: (
            "Write a solution to the following problem and make sure that it passes "
            f"the tests:\n```python\n{x['prompt']}\n```"
        ),
        "mbpp": lambda x: str(x["prompt"]),
    }
    sources = _load_sources(args.arrow_root)
    rows: list[dict[str, Any]] = []
    full_counts: dict[str, int] = {}
    heldout_counts: dict[str, int] = {}
    for dataset_name in ("gsm8k", "math500", "humaneval", "mbpp"):
        dataset = sources[dataset_name]
        full_counts[dataset_name] = len(dataset)
        kept = 0
        for source_index, item in enumerate(dataset):
            source_id = str(item.get("id", item.get("task_id", source_index)))
            prompt = formatters[dataset_name](item)
            if (dataset_name, source_id) in evaluation_keys or prompt in evaluation_prompts:
                continue
            rows.append(
                {
                    "dataset": dataset_name,
                    "source_id": source_id,
                    "prompt": prompt,
                }
            )
            kept += 1
        heldout_counts[dataset_name] = kept

    if len(rows) != 240:
        raise RuntimeError(
            f"expected 240 held-out prompts from the 2,240-row pool, found {len(rows)}; "
            f"per-domain={heldout_counts}"
        )
    if any((row["dataset"], row["source_id"]) in evaluation_keys for row in rows):
        raise AssertionError("source-id leakage into calibration set")
    if any(row["prompt"] in evaluation_prompts for row in rows):
        raise AssertionError("prompt leakage into calibration set")

    validation_rows: list[dict[str, Any]] = []
    if args.validation_output is not None:
        if not 0.0 < args.validation_fraction < 1.0:
            raise ValueError("validation_fraction 必须位于 (0, 1)")
        training_rows: list[dict[str, Any]] = []
        for domain_index, dataset_name in enumerate(
            ("gsm8k", "math500", "humaneval", "mbpp")
        ):
            domain_rows = [row for row in rows if row["dataset"] == dataset_name]
            random.Random(args.seed + domain_index + 1).shuffle(domain_rows)
            validation_count = int(round(len(domain_rows) * args.validation_fraction))
            validation_rows.extend(domain_rows[:validation_count])
            training_rows.extend(domain_rows[validation_count:])
        rows = training_rows
        random.Random(args.seed).shuffle(rows)
        random.Random(args.seed + 10_000).shuffle(validation_rows)
    else:
        random.Random(args.seed).shuffle(rows)

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for index, row in enumerate(rows):
            stream.write(json.dumps({"index": index, **row}, ensure_ascii=False) + "\n")
    validation_output: Path | None = None
    if args.validation_output is not None:
        validation_output = args.validation_output.expanduser().resolve()
        validation_output.parent.mkdir(parents=True, exist_ok=True)
        with validation_output.open("w", encoding="utf-8") as stream:
            for index, row in enumerate(validation_rows):
                stream.write(
                    json.dumps({"index": index, **row}, ensure_ascii=False) + "\n"
                )
    print(
        json.dumps(
            {
                "output": str(output),
                "rows": len(rows),
                "validation_output": (
                    str(validation_output) if validation_output is not None else None
                ),
                "validation_rows": len(validation_rows),
                "official_rows": len(evaluation),
                "full_counts": full_counts,
                "heldout_counts": heldout_counts,
                "source_overlap": 0,
                "prompt_overlap": 0,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
