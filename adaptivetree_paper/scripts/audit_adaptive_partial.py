"""Read-only diagnosis of completed AdaptiveTree result pairs.

The report contains aggregate timings, acceptance lengths, and controller
decisions only.  It deliberately omits prompts, generated text, and token ids.
Every ``.pt`` artifact is hash/identity checked before deserialization.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import statistics

from dflash_specblock.paper.common import load_json
from dflash_specblock.paper.official_reporting import load_completed, official_rows


def _mean(values):
    values = list(values)
    return statistics.fmean(values) if values else None


def _result_metrics(run, method):
    results = [response[method] for response in run["responses"]]
    stage_names = sorted({name for result in results for name in result.stage_times})
    return {
        "responses": len(results),
        "mean_decode_tpot_ms": 1000 * _mean(
            result.time_per_output_token for result in results
        ),
        "mean_acceptance_length": _mean(
            accepted
            for result in results
            for accepted in result.acceptance_lengths
        ),
        "mean_decode_rounds": _mean(result.decode_rounds for result in results),
        "mean_stage_ms_per_output_token": {
            name: 1000 * _mean(
                result.stage_times.get(name, 0.0) / result.num_output_tokens
                for result in results
            )
            for name in stage_names
        },
    }


def _controller_metrics(run, method, exploration_interval):
    actions = [
        action
        for response in run["responses"]
        for action in response[method].adaptive_decisions
    ]
    budget_counts = Counter(
        int(action["decision"]["budget"])
        for action in actions
        if action.get("decision") is not None
    )
    by_budget = defaultdict(lambda: {"rounds": 0, "latency_ms": 0.0,
                                     "committed_tokens": 0})
    scheduled, ordinary = [], []
    for index, action in enumerate(actions):
        decision = action.get("decision")
        if decision is None:
            continue
        budget = int(decision["budget"])
        latency = float(action["draft_ms"]) + float(action["verify_ms"])
        committed = 1 + int(action["accepted_draft_tokens"])
        item = by_budget[budget]
        item["rounds"] += 1
        item["latency_ms"] += latency
        item["committed_tokens"] += committed
        # _select_node_count checks the old zero-based decision count and then
        # increments it.  Once the six one-shot warmups are complete, counts
        # 64, 128, ... are the forced exploration decisions.
        is_scheduled = bool(
            exploration_interval > 0
            and index >= 6
            and index % exploration_interval == 0
        )
        (scheduled if is_scheduled else ordinary).append((latency, committed, budget))

    def group(rows):
        return {
            "rounds": len(rows),
            "mean_round_ms": _mean(row[0] for row in rows),
            "mean_committed_tokens": _mean(row[1] for row in rows),
            "tokens_per_ms": (
                sum(row[1] for row in rows) / sum(row[0] for row in rows)
                if rows and sum(row[0] for row in rows) > 0 else None
            ),
            "budget_counts": dict(sorted(Counter(row[2] for row in rows).items())),
        }

    return {
        "total_rounds": len(actions),
        "budget_counts": dict(sorted(budget_counts.items())),
        "by_budget": {
            str(budget): {
                **value,
                "tokens_per_ms": value["committed_tokens"] / value["latency_ms"],
            }
            for budget, value in sorted(by_budget.items())
            if value["latency_ms"] > 0
        },
        "scheduled_exploration": group(scheduled),
        "ordinary": group(ordinary),
    }


def audit(run_dir: Path):
    contract = load_json(run_dir / "contract.json")
    identity = contract["identity"]
    config = contract["metadata"]["config"]
    interval = int(config["adaptive"]["exploration_interval"])
    datasets = []
    for sdpa_path in sorted(run_dir.glob("*__sdpa.pt")):
        flash_path = sdpa_path.with_name(
            sdpa_path.name.replace("__sdpa.pt", "__flash_attn.pt")
        )
        if not (sdpa_path.with_suffix(".complete.json").exists()
                and flash_path.exists()
                and flash_path.with_suffix(".complete.json").exists()):
            continue
        sdpa = load_completed(sdpa_path, identity)
        flash = load_completed(flash_path, identity)
        rows = official_rows(sdpa, flash, config["variants"])
        by_label = {row["method"]: row for row in rows}
        dflash_row = by_label["DFlash"]
        ddtree_row = by_label["DDTree-best"]
        dflash_run = (sdpa if dflash_row["method_backend"] == "sdpa" else flash)
        selected = {
            "dflash": _result_metrics(dflash_run, "dflash"),
            "ddtree": _result_metrics(sdpa, ddtree_row["selected_key"]),
            "adaptive": _result_metrics(sdpa, "adaptive"),
            "no_exploration": _result_metrics(sdpa, "no_exploration"),
        }
        datasets.append({
            "dataset": sdpa["args"]["dataset"],
            "model": sdpa["args"]["model_name_or_path"],
            "dflash_backend": dflash_row["method_backend"],
            "best_ddtree_key": ddtree_row["selected_key"],
            "speedup": {
                "ddtree_vs_dflash": (
                    dflash_row["mean_decode_tpot_seconds"]
                    / ddtree_row["mean_decode_tpot_seconds"]
                ),
                "adaptive_vs_dflash": by_label["adaptive"]["speedup_vs_target"]
                    / dflash_row["speedup_vs_target"],
                "adaptive_vs_ddtree": by_label["adaptive"]["speedup_vs_best_ddtree"],
                "no_exploration_vs_dflash": by_label["no_exploration"]["speedup_vs_target"]
                    / dflash_row["speedup_vs_target"],
                "no_exploration_vs_ddtree": by_label["no_exploration"]["speedup_vs_best_ddtree"],
            },
            "method_metrics": selected,
            "controllers": {
                "adaptive": _controller_metrics(sdpa, "adaptive", interval),
                "no_exploration": _controller_metrics(sdpa, "no_exploration", 0),
            },
        })
    if not datasets:
        raise ValueError("No completed SDPA/FlashAttention result pairs found")

    def geomean(field):
        values = [row["speedup"][field] for row in datasets]
        return math.exp(statistics.fmean(math.log(value) for value in values))

    return {
        "kind": "read_only_adaptivetree_partial_audit",
        "run_dir": run_dir.name,
        "completed_dataset_pairs": len(datasets),
        "limitations": [
            "Partial completed datasets only",
            "record-bf16-mismatches runs are not strict lossless claims",
            "Stage attribution follows fields emitted by the pinned official runner",
        ],
        "geomean_speedup": {
            field: geomean(field)
            for field in (
                "ddtree_vs_dflash", "adaptive_vs_dflash", "adaptive_vs_ddtree",
                "no_exploration_vs_dflash", "no_exploration_vs_ddtree",
            )
        },
        "datasets": datasets,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.run_dir.resolve()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
