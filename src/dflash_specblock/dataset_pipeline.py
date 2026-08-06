"""SpecBlock benchmark 数据下载与 prompt 统一。"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from urllib.request import urlopen

from tqdm import tqdm


MT_BENCH_URL = (
    "https://raw.githubusercontent.com/lm-sys/FastChat/main/"
    "fastchat/llm_judge/data/mt_bench/question.jsonl"
)


@dataclass(slots=True, frozen=True)
class PromptRecord:
    prompt: str
    dataset: str
    source_id: str
    metadata: dict[str, Any]

    def to_row(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "dataset": self.dataset,
            "source_id": self.source_id,
            "metadata": self.metadata,
        }


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_output_dir(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else (_project_root() / value).resolve()


def _ensure_non_empty_prompt(prompt: str, dataset: str, source_id: str) -> str:
    text = prompt.strip()
    if not text:
        raise ValueError(f"{dataset}:{source_id} 生成了空 prompt")
    return text


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def _take_limit(records: list[PromptRecord], max_samples: int) -> list[PromptRecord]:
    return records if max_samples <= 0 else records[:max_samples]


def _format_mt_bench_prompt(row: dict[str, Any]) -> str:
    turns = row.get("turns")
    if not isinstance(turns, list) or not turns:
        raise ValueError("MT-Bench 样本缺少 turns")
    formatted = []
    for index, turn in enumerate(turns, start=1):
        if not isinstance(turn, str) or not turn.strip():
            raise ValueError("MT-Bench turn 必须是非空字符串")
        formatted.append(f"[User Turn {index}]\n{turn.strip()}")
    return "\n\n".join(formatted)


def _format_alpaca_prompt(row: dict[str, Any]) -> str:
    instruction = row.get("instruction")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("Alpaca 样本缺少 instruction")
    extra_input = row.get("input")
    if isinstance(extra_input, str) and extra_input.strip():
        return f"{instruction.strip()}\n\nInput:\n{extra_input.strip()}"
    return instruction.strip()


def _format_math_prompt(row: dict[str, Any]) -> str:
    problem = row.get("problem", row.get("question"))
    if not isinstance(problem, str) or not problem.strip():
        raise ValueError("MATH-500 样本缺少 problem/question")
    return problem.strip()


def _format_nq_prompt(row: dict[str, Any]) -> str:
    question = row.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Natural Questions 样本缺少 question")
    return question.strip()


def _extract_translation_text(row: dict[str, Any], source_lang: str, target_lang: str) -> tuple[str, str]:
    translation = row.get("translation")
    if isinstance(translation, dict):
        source = translation.get(source_lang)
        target = translation.get(target_lang)
        if isinstance(source, str) and isinstance(target, str):
            return source.strip(), target.strip()
    source = row.get(source_lang)
    target = row.get(target_lang)
    if isinstance(source, str) and isinstance(target, str):
        return source.strip(), target.strip()
    raise ValueError(
        f"翻译样本缺少字段: source={source_lang}, target={target_lang}"
    )


def _format_translation_prompt(source_text: str, source_lang: str, target_lang: str) -> str:
    return (
        f"Translate the following text from {source_lang} to {target_lang}.\n\n"
        f"{source_text}"
    )


def _load_mt_bench(max_samples: int) -> list[PromptRecord]:
    records: list[PromptRecord] = []
    with urlopen(MT_BENCH_URL) as response:
        raw_lines = [line for line in response.read().decode("utf-8").splitlines() if line.strip()]
        if max_samples > 0:
            raw_lines = raw_lines[:max_samples]
        for raw_line in tqdm(raw_lines, desc="download mt_bench", unit="sample"):
            if not raw_line.strip():
                continue
            row = json.loads(raw_line)
            question_id = str(row.get("question_id", len(records)))
            prompt = _ensure_non_empty_prompt(
                _format_mt_bench_prompt(row),
                dataset="mt_bench",
                source_id=question_id,
            )
            records.append(
                PromptRecord(
                    prompt=prompt,
                    dataset="mt_bench",
                    source_id=question_id,
                    metadata={
                        "category": row.get("category"),
                        "turns": len(row.get("turns", [])),
                    },
                )
            )
    return _take_limit(records, max_samples)


def _load_hf_records(
    dataset_name: str,
    path: str,
    config_name: str | None,
    split: str,
    max_samples: int,
) -> list[dict[str, Any]]:
    from datasets import load_dataset

    dataset = load_dataset(path, config_name, split=split)
    if max_samples > 0:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    records: list[dict[str, Any]] = []
    for row in tqdm(dataset, desc=f"download {dataset_name}", unit="sample"):
        records.append(dict(row))
    return records


def _load_humaneval(max_samples: int) -> list[PromptRecord]:
    rows = _load_hf_records(
        dataset_name="humaneval",
        path="openai_humaneval",
        config_name=None,
        split="test",
        max_samples=max_samples,
    )
    records: list[PromptRecord] = []
    for row in rows:
        task_id = str(row.get("task_id", len(records)))
        prompt = row.get("prompt")
        if not isinstance(prompt, str):
            raise ValueError("HumanEval 样本缺少 prompt")
        records.append(
            PromptRecord(
                prompt=_ensure_non_empty_prompt(prompt, "humaneval", task_id),
                dataset="humaneval",
                source_id=task_id,
                metadata={
                    "entry_point": row.get("entry_point"),
                    "test": row.get("test"),
                    "canonical_solution": row.get("canonical_solution"),
                },
            )
        )
    return records


def _load_math_500(max_samples: int) -> list[PromptRecord]:
    rows = _load_hf_records(
        dataset_name="math_500",
        path="HuggingFaceH4/MATH-500",
        config_name=None,
        split="test",
        max_samples=max_samples,
    )
    records: list[PromptRecord] = []
    for index, row in enumerate(rows):
        source_id = str(row.get("unique_id", row.get("id", index)))
        prompt = _ensure_non_empty_prompt(
            _format_math_prompt(row),
            dataset="math_500",
            source_id=source_id,
        )
        records.append(
            PromptRecord(
                prompt=prompt,
                dataset="math_500",
                source_id=source_id,
                metadata={
                    "subject": row.get("subject"),
                    "level": row.get("level"),
                    "answer": row.get("answer"),
                    "solution": row.get("solution"),
                },
            )
        )
    return records


def _load_alpaca(max_samples: int) -> list[PromptRecord]:
    rows = _load_hf_records(
        dataset_name="alpaca",
        path="tatsu-lab/alpaca",
        config_name=None,
        split="train",
        max_samples=max_samples,
    )
    records: list[PromptRecord] = []
    for index, row in enumerate(rows):
        source_id = str(row.get("id", index))
        prompt = _ensure_non_empty_prompt(
            _format_alpaca_prompt(row),
            dataset="alpaca",
            source_id=source_id,
        )
        records.append(
            PromptRecord(
                prompt=prompt,
                dataset="alpaca",
                source_id=source_id,
                metadata={
                    "has_input": bool(row.get("input")),
                    "reference": row.get("output"),
                    "instruction": row.get("instruction"),
                    "input": row.get("input"),
                },
            )
        )
    return records


def _load_nq_open(max_samples: int) -> list[PromptRecord]:
    rows = _load_hf_records(
        dataset_name="nq_open",
        path="nq_open",
        config_name=None,
        split="validation",
        max_samples=max_samples,
    )
    records: list[PromptRecord] = []
    for index, row in enumerate(rows):
        source_id = str(row.get("id", index))
        prompt = _ensure_non_empty_prompt(
            _format_nq_prompt(row),
            dataset="nq_open",
            source_id=source_id,
        )
        answer = row.get("answer")
        answer_count = len(answer) if isinstance(answer, list) else None
        records.append(
            PromptRecord(
                prompt=prompt,
                dataset="nq_open",
                source_id=source_id,
                metadata={
                    "answer_count": answer_count,
                    "answers": answer if isinstance(answer, list) else [],
                },
            )
        )
    return records


def _load_translation(
    max_samples: int,
    dataset_path: str,
    config_name: str,
    split: str,
    source_lang: str,
    target_lang: str,
) -> list[PromptRecord]:
    rows = _load_hf_records(
        dataset_name="translation",
        path=dataset_path,
        config_name=config_name,
        split=split,
        max_samples=max_samples,
    )
    records: list[PromptRecord] = []
    for index, row in enumerate(rows):
        source_id = str(row.get("id", index))
        source_text, target_text = _extract_translation_text(row, source_lang, target_lang)
        prompt = _ensure_non_empty_prompt(
            _format_translation_prompt(source_text, source_lang, target_lang),
            dataset="translation",
            source_id=source_id,
        )
        records.append(
            PromptRecord(
                prompt=prompt,
                dataset="translation",
                source_id=source_id,
                metadata={
                    "source_lang": source_lang,
                    "target_lang": target_lang,
                    "reference": target_text,
                },
            )
        )
    return records


def build_specblock_official_suite(
    max_samples_per_dataset: int,
    translation_dataset: str,
    translation_config: str,
    translation_split: str,
    source_lang: str,
    target_lang: str,
) -> dict[str, list[PromptRecord]]:
    builders = [
        ("mt_bench", lambda: _load_mt_bench(max_samples_per_dataset)),
        ("humaneval", lambda: _load_humaneval(max_samples_per_dataset)),
        ("math_500", lambda: _load_math_500(max_samples_per_dataset)),
        ("alpaca", lambda: _load_alpaca(max_samples_per_dataset)),
        ("nq_open", lambda: _load_nq_open(max_samples_per_dataset)),
        (
            "translation",
            lambda: _load_translation(
                max_samples=max_samples_per_dataset,
                dataset_path=translation_dataset,
                config_name=translation_config,
                split=translation_split,
                source_lang=source_lang,
                target_lang=target_lang,
            ),
        ),
    ]
    suite: dict[str, list[PromptRecord]] = {}
    for name, builder in tqdm(builders, desc="datasets", unit="dataset"):
        suite[name] = builder()
    return suite


def write_suite(output_dir: Path, suite: dict[str, list[PromptRecord]]) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    all_rows: list[dict[str, Any]] = []
    for name, records in tqdm(suite.items(), desc="write jsonl", unit="dataset"):
        path = output_dir / f"{name}.jsonl"
        rows = [record.to_row() for record in records]
        _write_jsonl(path, rows)
        written[name] = path
        all_rows.extend(rows)
    combined = output_dir / "prompts_all.jsonl"
    _write_jsonl(combined, all_rows)
    written["all"] = combined
    manifest = output_dir / "manifest.json"
    manifest_rows = {
        "suite": "specblock_official",
        "files": {name: str(path) for name, path in written.items()},
        "counts": {name: len(records) for name, records in suite.items()},
    }
    with manifest.open("w", encoding="utf-8") as stream:
        json.dump(manifest_rows, stream, ensure_ascii=False, indent=2)
    written["manifest"] = manifest
    return written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download and normalize SpecBlock benchmark datasets")
    parser.add_argument("--output-dir", default="datasets/processed/specblock_official")
    parser.add_argument("--max-samples-per-dataset", type=int, default=0)
    parser.add_argument("--translation-dataset", default="wmt14")
    parser.add_argument("--translation-config", default="de-en")
    parser.add_argument("--translation-split", default="test")
    parser.add_argument("--source-lang", default="de")
    parser.add_argument("--target-lang", default="en")
    return parser


def download_main() -> None:
    args = build_parser().parse_args()
    if args.max_samples_per_dataset < 0:
        raise ValueError("max-samples-per-dataset 不能为负数")
    output_dir = _resolve_output_dir(args.output_dir)
    suite = build_specblock_official_suite(
        max_samples_per_dataset=args.max_samples_per_dataset,
        translation_dataset=args.translation_dataset,
        translation_config=args.translation_config,
        translation_split=args.translation_split,
        source_lang=args.source_lang,
        target_lang=args.target_lang,
    )
    written = write_suite(output_dir, suite)
    summary = {
        "output_dir": str(output_dir),
        "files": {name: str(path) for name, path in written.items()},
        "counts": {name: len(records) for name, records in suite.items()},
        "translation_dataset": {
            "path": args.translation_dataset,
            "config": args.translation_config,
            "split": args.translation_split,
            "source_lang": args.source_lang,
            "target_lang": args.target_lang,
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    download_main()
