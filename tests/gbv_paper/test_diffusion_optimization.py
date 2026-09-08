import pytest

from gbv_experiments.diffusion_optimization import (
    BASELINE_NAMES, SUPPORT_SIZES, TEMPERATURES, optimization_variants,
    profile_names, summarize, support_optimization_variants,
)


@pytest.mark.parametrize("temperature", TEMPERATURES)
def test_optimization_grid_is_valid_unique_and_keeps_controls(temperature):
    variants = optimization_variants(temperature)
    assert len(variants) == 22
    assert [variant.name for variant in variants[:2]] == list(BASELINE_NAMES)
    assert len({variant.name for variant in variants}) == len(variants)
    assert {variant.temperature for variant in variants} == {temperature}
    assert all(variant.paths * variant.length <= variant.tree_budget for variant in variants)
    candidates = variants[2:]
    assert {variant.method for variant in candidates} == {
        "diffusion_scaffold_bv", "diffusion_scaffold_ancestral",
    }
    assert all((variant.paths + 1) * variant.length <= variant.tree_budget
               for variant in candidates)


@pytest.mark.parametrize("temperature", TEMPERATURES)
def test_support_grid_is_valid_unique_and_covers_widths(temperature):
    variants = support_optimization_variants(temperature)
    assert len(variants) == 42
    assert [variant.name for variant in variants[:2]] == list(BASELINE_NAMES)
    assert len({variant.name for variant in variants}) == len(variants)
    candidates = variants[2:]
    assert {variant.diffusion_support_size for variant in candidates} == set(SUPPORT_SIZES)
    assert all((variant.paths + 1) * variant.length <= variant.tree_budget
               for variant in candidates)


def row(temperature, variant, decode_ms, sha, prompt=0, repeat=0):
    method = ("dflash" if variant == BASELINE_NAMES[0]
              else "ddtree" if variant == BASELINE_NAMES[1]
              else "diffusion_scaffold_bv")
    return {
        "temperature": temperature,
        "variant": variant,
        "method": method,
        "paths": 1,
        "length": 15,
        "tree_budget": 45,
        "diffusion_support_size": 8,
        "prompt": prompt,
        "repeat": repeat,
        "generated_sha256": sha,
        "generated_tokens": 11,
        "decode_tokens": 10,
        "decode_ms": decode_ms,
        "e2e_ms": decode_ms + 1,
        "rounds": [{"accepted_draft_tokens": 2, "committed_tokens": 3, "tree_nodes": 45}],
    }


def test_summary_uses_weighted_tpot_speedups_and_repeat_equality():
    rows = []
    for temperature in TEMPERATURES:
        rows.extend((
            row(temperature, BASELINE_NAMES[0], 100, "d"),
            row(temperature, BASELINE_NAMES[1], 80, "t"),
            row(temperature, "candidate_fast", 50, "a"),
            row(temperature, "candidate_slow", 120, "b"),
        ))
    # A second, identical repeat must not make repeat_equal false.
    rows.append(row(0.3, "candidate_fast", 50, "a", repeat=1))
    summary = summarize(rows)
    fast = next(item for item in summary["per_temperature"]
                if item["temperature"] == 0.3 and item["variant"] == "candidate_fast")
    assert fast["decode_ms_per_token"] == pytest.approx(5)
    assert fast["speedup_vs_dflash"] == pytest.approx(2)
    assert fast["speedup_vs_ddtree"] == pytest.approx(1.6)
    assert fast["repeat_equal"]
    assert summary["overall_ranking"][0]["variant"] == "candidate_fast"
    assert summary["overall_ranking"][0]["geomean_speedup_vs_dflash"] == pytest.approx(2)
    assert profile_names(summary, 0.3, count=1) == [*BASELINE_NAMES, "candidate_fast"]


def test_summary_detects_nondeterministic_repeat():
    rows = [
        row(0.3, BASELINE_NAMES[0], 100, "d"),
        row(0.3, BASELINE_NAMES[1], 80, "t"),
        row(0.3, "candidate", 50, "a"),
        row(0.3, "candidate", 50, "changed", repeat=1),
    ]
    summary = summarize(rows)
    candidate = next(item for item in summary["per_temperature"] if item["variant"] == "candidate")
    assert not candidate["repeat_equal"]
