"""Optional, secret-safe Weights & Biases monitoring for official workers."""
from __future__ import annotations

import math
import os
from pathlib import Path
from statistics import fmean


def wandb_contract(args):
    """Return public W&B settings only; credentials must never enter contracts."""
    project = getattr(args, "wandb_project", None)
    if not project:
        return None
    return {"project":project,
            "entity":getattr(args, "wandb_entity", None),
            "group":getattr(args, "wandb_group", None)}


def initialize_wandb(args, *, model_name, draft_name, backend, rank, world_size,
                     diagnostic_variants, controller_configs):
    settings = wandb_contract(args)
    if settings is None:
        return None
    mode = os.environ.get("WANDB_MODE", "online").lower()
    if mode not in {"offline", "disabled"} and not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError(
            "W&B monitoring requested but WANDB_API_KEY is not set; "
            "provide it through the job environment, never source control"
        )
    try:
        import wandb
    except ImportError as error:
        raise RuntimeError(
            "W&B monitoring requested but wandb is not installed; "
            "install requirements-paper.txt"
        ) from error

    directory = Path(args.run_dir) / "wandb"
    directory.mkdir(parents=True, exist_ok=True)
    group = settings["group"] or str(args.identity)[:12]
    model_tag = model_name.rsplit("/", 1)[-1]
    run = wandb.init(
        project=settings["project"],
        entity=settings["entity"],
        group=group,
        name=f"adaptive-{model_tag}-{args.dataset}-{backend}-r{rank}",
        job_type=backend,
        dir=str(directory),
        settings=wandb.Settings(x_disable_stats=True, x_disable_meta=True),
        save_code=False,
        config={
            "protocol_identity":args.identity,
            "dataset":args.dataset,
            "model":model_name,
            "draft_model":draft_name,
            "backend":backend,
            "rank":rank,
            "world_size":world_size,
            "temperature":0,
            "smoke_count":args.smoke_count,
            "diagnostic_variants":list(diagnostic_variants),
            "diagnostic_controllers":controller_configs,
        },
        tags=["adaptivetree", "t0", backend,
              "diagnostic" if diagnostic_variants else "official"],
    )
    run.summary["status"] = "running"
    return run


def _positive_finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value) and value > 0


def log_response(run, response, *, step, index, turn):
    if run is None:
        return
    baseline_tpot = response["baseline"].time_per_output_token
    metrics = {"progress/responses":step + 1,
               "progress/case_index":index,
               "progress/turn":turn}
    for method, result in response.items():
        if method.startswith("_"):
            continue
        tpot = result.time_per_output_token
        if not _positive_finite(tpot) or result.num_output_tokens <= 0:
            raise ValueError(f"Invalid W&B timing payload for {method}")
        prefix = f"methods/{method}"
        metrics[f"{prefix}/decode_tpot_ms"] = 1000 * tpot
        metrics[f"{prefix}/tokens_per_second"] = 1 / tpot
        metrics[f"{prefix}/output_tokens"] = result.num_output_tokens
        if _positive_finite(baseline_tpot):
            metrics[f"{prefix}/speedup_vs_run_target"] = baseline_tpot / tpot
        acceptance = list(result.acceptance_lengths)
        if acceptance:
            metrics[f"{prefix}/mean_acceptance_length"] = fmean(acceptance)
            metrics[f"{prefix}/decode_rounds"] = len(acceptance)
        for stage, seconds in getattr(result, "stage_times", {}).items():
            if isinstance(seconds, (int, float)) and math.isfinite(seconds) and seconds >= 0:
                metrics[f"{prefix}/stage_ms_per_output_token/{stage}"] = (
                    1000 * seconds / result.num_output_tokens
                )
        decisions = getattr(result, "adaptive_decisions", None)
        if decisions:
            budgets = [item["decision"]["budget"] for item in decisions]
            metrics[f"{prefix}/mean_draft_nodes"] = fmean(budgets)
            metrics[f"{prefix}/max_draft_nodes"] = max(budgets)
            metrics[f"{prefix}/share_rounds_above_128"] = (
                sum(budget > 128 for budget in budgets) / len(budgets)
            )
            last = decisions[-1]["decision"]
            metrics[f"{prefix}/last_draft_nodes"] = last["budget"]
            for field in ("expected_draft_tokens", "predicted_round_ms",
                          "predicted_tokens_per_ms"):
                value = last.get(field)
                if isinstance(value, (int, float)) and math.isfinite(value):
                    metrics[f"{prefix}/last_{field}"] = value
    run.log(metrics, step=step)


def finish_wandb(run, responses, methods):
    if run is None:
        return
    for method in methods:
        values = [response[method].time_per_output_token for response in responses]
        if not values:
            continue
        mean_tpot = fmean(values)
        run.summary[f"aggregate/{method}/mean_decode_tpot_ms"] = 1000 * mean_tpot
        run.summary[f"aggregate/{method}/tokens_per_second"] = 1 / mean_tpot
        baseline = fmean(response["baseline"].time_per_output_token for response in responses)
        run.summary[f"aggregate/{method}/speedup_vs_run_target"] = baseline / mean_tpot
    run.summary["completed_responses"] = len(responses)
    run.summary["status"] = "complete"
    run.finish()
