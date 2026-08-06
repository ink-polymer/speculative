"""官方 scorer 测试。"""

from __future__ import annotations

from dflash_specblock.official_scorers import (
    extract_math_answer,
    score_humaneval,
    score_math_500,
    score_nq_open,
)


def test_nq_open_accepts_normalized_exact_match() -> None:
    score = score_nq_open("The Eiffel Tower", ["eiffel tower", "Paris"])
    assert score["exact_match"] == 1.0
    assert score["f1"] == 1.0


def test_math_500_prefers_boxed_answer() -> None:
    assert extract_math_answer("We compute and get \\boxed{42}.") == "42"
    score = score_math_500("Final answer: \\boxed{42}", "42")
    assert score["correct"] == 1.0


def test_humaneval_pass_case() -> None:
    prompt = "def add(a, b):\n"
    completion = "    return a + b\n"
    test = (
        "def check(candidate):\n"
        "    assert candidate(1, 2) == 3\n"
        "    assert candidate(5, 7) == 12\n"
    )
    score = score_humaneval(
        prompt=prompt,
        completion=completion,
        test=test,
        entry_point="add",
        timeout_seconds=2.0,
    )
    assert score["passed"] == 1.0


def test_humaneval_fail_case() -> None:
    prompt = "def add(a, b):\n"
    completion = "    return a - b\n"
    test = (
        "def check(candidate):\n"
        "    assert candidate(1, 2) == 3\n"
    )
    score = score_humaneval(
        prompt=prompt,
        completion=completion,
        test=test,
        entry_point="add",
        timeout_seconds=2.0,
    )
    assert score["passed"] == 0.0
