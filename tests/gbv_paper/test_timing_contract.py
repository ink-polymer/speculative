from gbv_experiments.common import ROOT
from gbv_experiments.engine import summarize_stage_profile


def test_vendored_ddtree_excludes_first_draft_but_counts_anchor_token():
    source = (ROOT / "third_party/ddtree_pinned/ddtree.py").read_text()
    first_draft_end = source.index(
        "draft_stage_elapsed = cuda_time() - draft_stage_start"
    )
    timer_reset = source.index("decode_start = cuda_time()", first_draft_end)
    first_tree_build = source.index("tree_build_start = cuda_time()")
    output_count = source.index(
        "num_output_tokens = output_ids.shape[1] - num_input_tokens"
    )
    tpot = source.index(
        "time_per_output_token = total_decode_time / max(num_output_tokens, 1)"
    )
    assert first_draft_end < timer_reset < first_tree_build
    assert output_count < tpot


def test_stage_summary_ranks_gpu_work_and_removes_official_exclusions():
    stages = {
        "host_ms": {
            "prefill": 12.0,
            "draft_prefill": 8.0,
            "first_draft_boundary_sync": 2.0,
            "draft": 10.0,
            "tree_build": 4.0,
            "tree_compile": 3.0,
            "verify": 40.0,
            "select_and_correct": 6.0,
            "stop_check": 1.0,
            "commit": 5.0,
        },
        "cuda_event_ms": {
            "prefill": 11.0,
            "draft_prefill": 7.0,
            "first_draft_boundary_sync": 0.0,
            "draft": 9.0,
            "tree_build": 2.0,
            "tree_compile": 1.0,
            "verify": 38.0,
            "select_and_correct": 5.0,
            "stop_check": 0.0,
            "commit": 4.0,
        },
    }
    summary = summarize_stage_profile(
        stages, e2e_ms=100.0, decode_ms=88.0,
        official_scope_decode_ms=78.0,
    )
    official_host = summary["host_official_scope"]
    official_cuda = summary["cuda_official_scope"]
    assert official_host["longest_stage"] == "verify"
    assert official_cuda["longest_stage"] == "verify"
    assert "draft_prefill" not in official_host["measured_stage_ms"]
    assert "first_draft_boundary_sync" not in official_cuda["measured_stage_ms"]
    assert official_host["unattributed_ms"] == 9.0
    assert summary["diagnostic_only"] is True
    assert summary["eligible_for_primary_timing"] is False
