"""数据集 prompt 规范化测试。"""

from __future__ import annotations

import json

from dflash_specblock.dataset_pipeline import (
    PromptRecord,
    _extract_translation_text,
    _format_alpaca_prompt,
    _format_mt_bench_prompt,
    write_suite,
)


def test_mt_bench_turns_are_joined_in_order() -> None:
    prompt = _format_mt_bench_prompt({"turns": ["hello", "world"]})
    assert "[User Turn 1]" in prompt
    assert "[User Turn 2]" in prompt
    assert "hello" in prompt
    assert "world" in prompt


def test_alpaca_prompt_includes_optional_input() -> None:
    prompt = _format_alpaca_prompt({"instruction": "solve", "input": "1 + 1"})
    assert prompt == "solve\n\nInput:\n1 + 1"


def test_translation_text_prefers_translation_dict() -> None:
    source, target = _extract_translation_text(
        {"translation": {"de": "hallo", "en": "hello"}},
        source_lang="de",
        target_lang="en",
    )
    assert source == "hallo"
    assert target == "hello"


def test_write_suite_creates_split_and_combined_files(tmp_path) -> None:
    suite = {
        "alpaca": [
            PromptRecord(
                prompt="demo prompt",
                dataset="alpaca",
                source_id="0",
                metadata={"has_input": False},
            )
        ]
    }
    written = write_suite(tmp_path, suite)
    assert written["alpaca"].is_file()
    assert written["all"].is_file()
    assert written["manifest"].is_file()

    rows = written["all"].read_text(encoding="utf-8").strip().splitlines()
    assert len(rows) == 1
    row = json.loads(rows[0])
    assert row["prompt"] == "demo prompt"
    assert row["dataset"] == "alpaca"
