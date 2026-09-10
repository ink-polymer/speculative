#!/usr/bin/env python3
"""Held-out GSM8K-train gate for AdaptiveTree architecture changes.

This is deliberately separate from the registered formal test protocol.  It
uses deterministic examples from GSM8K *train*, balances method execution
order, includes every decode/tree/controller cost in TPOT, and refuses to pass
unless all greedy outputs exactly match an equal-node-cap official DDTree.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import time

import pyarrow.parquet as pq
import torch

from dflash_specblock.paper.adaptive_official import adaptive_generate
from dflash_specblock.paper.common import (PRIMARY_ADAPTIVE_METHOD, atomic_json,
                                           code_identity, load_json)
from dflash_specblock.paper.controller import (CONTEXTUAL_V8_VARIANT,
                                                DYNAMIC_B192_V12_VARIANT,
                                                PREBUILD_V11_VARIANT,
                                                controller_config,
                                                make_paper_builder)
from dflash_specblock.paper.official_spec import (MODELS, PINNED_MODEL_REVISIONS,
                                                  upstream)


def parse_args():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--config", type=Path,
                        default=Path(__file__).resolve().parents[1]
                        / "configs/paper_t0_full.json")
    parser.add_argument("--train-parquet", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--experimental-positive-temperature", action="store_true")
    parser.add_argument("--candidate", choices=("primary", "contextual-v8",
                                                 "prebuild-v11",
                                                 "dynamic-b192-v12"),
                        default="primary")
    parser.add_argument("--reference-budget", type=int, choices=(128, 192),
                        default=128)
    parser.add_argument("--additional-ddtree-budget", type=int,
                        choices=(128, 160, 256, 512, 1024), action="append",
                        default=[])
    parser.add_argument("--proposal-temperature", type=float)
    parser.add_argument("--maximum-tree-depth", type=int)
    parser.add_argument("--rank-head-checkpoint", type=Path)
    parser.add_argument("--rank-calibration-strength", type=float, default=1.0)
    parser.add_argument("--ratio-transport-checkpoint", type=Path)
    parser.add_argument("--ratio-transport-strength", type=float, default=1.0)
    parser.add_argument("--contextual-mass-retention-ratio", type=float)
    parser.add_argument("--contextual-minimum-history", type=int)
    parser.add_argument("--contextual-minimum-support", type=float)
    parser.add_argument("--contextual-refresh-interval", type=int)
    parser.add_argument("--contextual-floor-budget", type=int)
    parser.add_argument("--prebuild-min-top1-mean", type=float)
    parser.add_argument("--include-primary-control", action="store_true")
    parser.add_argument("--include-fixed-b192-control", action="store_true")
    parser.add_argument("--save-output-token-ids", action="store_true")
    return parser.parse_args()


def locate_train_parquet(explicit):
    if explicit is not None:
        return explicit.resolve()
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
    matches = sorted((hf_home / "hub/datasets--openai--gsm8k/snapshots").glob(
        "*/main/train-00000-of-00001.parquet"))
    if len(matches) != 1:
        raise RuntimeError("Pass --train-parquet; expected exactly one cached GSM8K train file")
    return matches[0].resolve()


def mean(values):
    return statistics.fmean(values) if values else float("nan")


def aggregate(records, method):
    rows = [row for row in records if row["method"] == method]
    tokens = sum(row["output_tokens"] for row in rows)
    stages = {name: 1000 * sum(row["stage_times"].get(name, 0.) for row in rows)
              / max(tokens, 1)
              for name in ("draft", "tree_build", "tree_compile", "verify", "commit")}
    tpot = mean([row["tpot_ms"] for row in rows])
    return {
        "responses": len(rows),
        "mean_tpot_ms": tpot,
        "median_tpot_ms": statistics.median(row["tpot_ms"] for row in rows),
        "tokens_per_second": 1000 / tpot,
        "mean_acceptance_length": mean([row["acceptance_length"] for row in rows]),
        "decode_rounds": sum(row["decode_rounds"] for row in rows),
        "output_tokens": tokens,
        "exact_match_rate_vs_equal_cap_reference": mean([
            float(row["matches_reference"]) for row in rows]),
        "stage_ms_per_output_token": stages,
    }


def main():
    args = parse_args()
    if min(args.samples, args.repeats, args.max_new_tokens) < 1:
        raise ValueError("samples, repeats, and max-new-tokens must be positive")
    if not math.isfinite(args.temperature) or args.temperature < 0:
        raise ValueError("temperature must be finite and nonnegative")
    if args.temperature > 0 and not args.experimental_positive_temperature:
        raise ValueError("T>0 requires --experimental-positive-temperature")
    if (args.contextual_mass_retention_ratio is not None
            and not 0. < args.contextual_mass_retention_ratio <= 1.):
        raise ValueError("contextual mass retention must be in (0, 1]")
    if (args.contextual_minimum_support is not None
            and not 0. < args.contextual_minimum_support <= 1.):
        raise ValueError("contextual minimum support must be in (0, 1]")
    if (args.proposal_temperature is not None
            and (not math.isfinite(args.proposal_temperature)
                 or args.proposal_temperature <= 0.)):
        raise ValueError("proposal temperature must be finite and positive")
    if (args.prebuild_min_top1_mean is not None
            and (not math.isfinite(args.prebuild_min_top1_mean)
                 or not 0. <= args.prebuild_min_top1_mean <= 1.)):
        raise ValueError("prebuild minimum top-1 mean must be in [0, 1]")
    if (args.maximum_tree_depth is not None
            and not 1 <= args.maximum_tree_depth <= 15):
        raise ValueError("maximum tree depth must be in [1, 15]")
    if (not math.isfinite(args.rank_calibration_strength)
            or not 0. <= args.rank_calibration_strength <= 1.):
        raise ValueError("rank calibration strength must be in [0, 1]")
    if (args.rank_head_checkpoint is not None
            and not args.rank_head_checkpoint.is_file()):
        raise FileNotFoundError(args.rank_head_checkpoint)
    if (args.ratio_transport_checkpoint is not None
            and not args.ratio_transport_checkpoint.is_file()):
        raise FileNotFoundError(args.ratio_transport_checkpoint)
    if (args.rank_head_checkpoint is not None
            and args.ratio_transport_checkpoint is not None):
        raise ValueError("rank and ratio checkpoints are mutually exclusive")
    if (not math.isfinite(args.ratio_transport_strength)
            or not 0. <= args.ratio_transport_strength <= 2.):
        raise ValueError("ratio transport strength must be in [0, 2]")
    for value in (args.contextual_minimum_history,
                  args.contextual_refresh_interval,
                  args.contextual_floor_budget):
        if value is not None and value < 1:
            raise ValueError("contextual integer overrides must be positive")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")

    config = load_json(args.config)
    adaptive_cfg = config["adaptive"]
    parquet = locate_train_parquet(args.train_parquet)
    table = pq.read_table(parquet, columns=["question"])
    questions = table.column("question").to_pylist()
    indices = random.Random(args.seed).sample(range(len(questions)), args.samples)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    u = upstream()
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    target_name, draft_name = MODELS[0]
    target = AutoModelForCausalLM.from_pretrained(
        target_name, revision=PINNED_MODEL_REVISIONS[target_name],
        attn_implementation="sdpa", dtype=torch.bfloat16,
        local_files_only=True).to(device).eval()
    draft = u.model.DFlashDraftModel.from_pretrained(
        draft_name, revision=PINNED_MODEL_REVISIONS[draft_name],
        attn_implementation="flash_attention_2", dtype=torch.bfloat16,
        local_files_only=True).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        target_name, revision=PINNED_MODEL_REVISIONS[target_name],
        local_files_only=True)
    rank_head = None
    if args.rank_head_checkpoint is not None:
        from dflash_specblock.rank_head import load_rank_head
        rank_head = load_rank_head(
            args.rank_head_checkpoint, int(draft.config.hidden_size), device,
            expected_metadata={
                "block_size": 15,
                "target_model_id": target_name,
                "target_revision": PINNED_MODEL_REVISIONS[target_name],
                "draft_model_id": draft_name,
                "draft_revision": PINNED_MODEL_REVISIONS[draft_name],
            },
        )
    ratio_transport_head = None
    if args.ratio_transport_checkpoint is not None:
        from dflash_specblock.rank_head import load_ratio_transport_head
        ratio_transport_head = load_ratio_transport_head(
            args.ratio_transport_checkpoint, int(draft.config.hidden_size),
            device,
            expected_metadata={
                "target_revision": PINNED_MODEL_REVISIONS[target_name],
                "draft_revision": PINNED_MODEL_REVISIONS[draft_name],
            },
            dtype=draft.dtype,
        )
    if draft.block_size != 16:
        raise ValueError("Architecture gate requires official K=15 draft horizon")
    u.ddtree.maybe_enable_cpp_compact(True)
    if u.ddtree.load_cpp_compact_module() is None:
        raise RuntimeError("Official C++ cache compaction is unavailable")

    def encode(text):
        rendered = tokenizer.apply_chat_template(
            [{"role":"user", "content":text}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        return tokenizer.encode(rendered, return_tensors="pt").to(device)

    common = dict(model=draft, target=target,
                  mask_token_id=draft.mask_token_id,
                  max_new_tokens=args.max_new_tokens,
                  block_size=draft.block_size,
                  stop_token_ids=[tokenizer.eos_token_id],
                  temperature=args.temperature)
    records = []
    exact = True
    candidate_method = {
        "primary": "adaptive_guarded_raw",
        "contextual-v8": CONTEXTUAL_V8_VARIANT,
        "prebuild-v11": PREBUILD_V11_VARIANT,
        "dynamic-b192-v12": DYNAMIC_B192_V12_VARIANT,
    }[args.candidate]
    candidate_builder_method = {
        "primary": PRIMARY_ADAPTIVE_METHOD,
        "contextual-v8": CONTEXTUAL_V8_VARIANT,
        "prebuild-v11": PREBUILD_V11_VARIANT,
        "dynamic-b192-v12": DYNAMIC_B192_V12_VARIANT,
    }[args.candidate]
    reference_method = f"ddtree_b{args.reference_budget}"
    if args.include_primary_control and args.candidate == "primary":
        raise ValueError("Primary candidate cannot also be its own control")
    if ((args.candidate == "dynamic-b192-v12")
            != (args.reference_budget == 192)):
        raise ValueError("Dynamic B192 must use DDTree B192, and only it may use that cap")
    if args.include_primary_control and args.reference_budget != 128:
        raise ValueError("The B128 primary control requires DDTree B128")
    if args.include_fixed_b192_control and args.candidate != "dynamic-b192-v12":
        raise ValueError("The fixed B192 control is only valid for dynamic B192")
    controls = []
    if args.include_primary_control:
        controls.append("adaptive_guarded_raw")
    if args.include_fixed_b192_control:
        controls.append("adaptive_fixed_b192_control")
    additional_references = tuple(
        f"ddtree_b{budget}" for budget in args.additional_ddtree_budget)
    if len(additional_references) != len(set(additional_references)):
        raise ValueError("Additional DDTree budgets must be unique")
    if reference_method in additional_references:
        raise ValueError("Additional DDTree budget duplicates the reference")
    method_names = (reference_method, *additional_references, *controls,
                    candidate_method)
    candidate_overrides = {
        "proposal_temperature": args.proposal_temperature,
        "maximum_tree_depth": args.maximum_tree_depth,
        "contextual_mass_retention_ratio": args.contextual_mass_retention_ratio,
        "contextual_minimum_history": args.contextual_minimum_history,
        "contextual_minimum_support": args.contextual_minimum_support,
        "contextual_refresh_interval": args.contextual_refresh_interval,
        "contextual_floor_budget": args.contextual_floor_budget,
        "prebuild_min_top1_mean": args.prebuild_min_top1_mean,
    }
    candidate_overrides = {
        key:value for key,value in candidate_overrides.items()
        if value is not None
    }

    def candidate_builder():
        builder = make_paper_builder(adaptive_cfg, candidate_builder_method)
        contextual_keys = {
            key for key in candidate_overrides
            if key.startswith("contextual_")
        }
        if contextual_keys and args.candidate not in {
                "contextual-v8", "prebuild-v11", "dynamic-b192-v12"}:
            raise ValueError("Contextual overrides require a contextual candidate")
        for key, value in candidate_overrides.items():
            setattr(builder, key, value)
        builder.rank_head = rank_head
        builder.rank_calibration_strength = args.rank_calibration_strength
        builder.ratio_transport_head = ratio_transport_head
        builder.ratio_transport_token_embeddings = (
            target.get_output_embeddings().weight
            if ratio_transport_head is not None else None)
        builder.ratio_transport_strength = args.ratio_transport_strength
        if builder.tree_budget != args.reference_budget:
            raise ValueError("Candidate and DDTree must have the same maximum node cap")
        return builder

    def fixed_b192_control_builder():
        builder = make_paper_builder(adaptive_cfg, DYNAMIC_B192_V12_VARIANT)
        # The impossible threshold forces B192 while retaining the identical
        # top-k, transfer, controller and raw-tree implementation as v12.
        builder.prebuild_min_top1_mean = 2.
        return builder
    started = time.time()
    for repeat in range(args.repeats):
        guarded_by_method = {candidate_method: candidate_builder()}
        if args.include_primary_control:
            guarded_by_method["adaptive_guarded_raw"] = make_paper_builder(
                adaptive_cfg, PRIMARY_ADAPTIVE_METHOD)
        if args.include_fixed_b192_control:
            guarded_by_method["adaptive_fixed_b192_control"] = (
                fixed_b192_control_builder())
        def generate(method, ids, maximum):
            kwargs = {**common, "input_ids":ids, "max_new_tokens":maximum}
            if method.startswith("ddtree_b"):
                return u.ddtree.ddtree_generate(
                    **kwargs, tree_budget=int(method.removeprefix("ddtree_b")))
            return adaptive_generate(
                **kwargs, builder=guarded_by_method[method],
                experimental_allow_positive_temperature=(args.temperature > 0),
            )

        warmup = encode("Warmup")
        for method in method_names:
            generate(method, warmup, min(args.max_new_tokens, 32))
        # Hardware warmup must not pretrain either online controller.
        guarded_by_method = {candidate_method: candidate_builder()}
        if args.include_primary_control:
            guarded_by_method["adaptive_guarded_raw"] = make_paper_builder(
                adaptive_cfg, PRIMARY_ADAPTIVE_METHOD)
        if args.include_fixed_b192_control:
            guarded_by_method["adaptive_fixed_b192_control"] = (
                fixed_b192_control_builder())
        for ordinal, index in enumerate(indices):
            prompt = (questions[index]
                      + "\nPlease reason step by step, and put your final answer within \\boxed{}.")
            ids = encode(prompt)
            order = method_names[(ordinal + repeat) % len(method_names):]
            order += method_names[:(ordinal + repeat) % len(method_names)]
            outputs = {}
            record_start = len(records)
            for method in order:
                if args.temperature > 0:
                    pair_seed = args.seed + repeat * len(questions) + index
                    torch.manual_seed(pair_seed)
                    torch.cuda.manual_seed_all(pair_seed)
                result = generate(method, ids, args.max_new_tokens)
                tokens = result.output_ids[0, result.num_input_tokens:].tolist()
                outputs[method] = tokens
                decisions = getattr(result, "adaptive_decisions", [])
                records.append({
                    "repeat": repeat,
                    "source_index": index,
                    "method": method,
                    "execution_position": order.index(method),
                    "tpot_ms": 1000 * result.time_per_output_token,
                    "output_tokens": result.num_output_tokens,
                    **({"output_token_ids": tokens}
                       if args.save_output_token_ids else {}),
                    "decode_rounds": result.decode_rounds,
                    "acceptance_length": mean(result.acceptance_lengths),
                    "stage_times": result.stage_times,
                    "selected_budget_counts": {
                        str(budget): sum(
                            (row.get("decision") or {}).get("budget") == budget
                            for row in decisions)
                        for budget in sorted({
                            (row.get("decision") or {}).get("budget")
                            for row in decisions
                            if (row.get("decision") or {}).get("budget") is not None
                        })
                    } if decisions else {},
                    "selected_budget_sequence": [
                        (row.get("decision") or {}).get("budget")
                        for row in decisions
                    ] if decisions else [],
                    "guard_reason_counts": {
                        reason: sum((row.get("guard") or {}).get("reason") == reason
                                    for row in decisions)
                        for reason in sorted({
                            (row.get("guard") or {}).get("reason")
                            for row in decisions
                            if (row.get("guard") or {}).get("reason") is not None
                        })
                    } if decisions else {},
                    "adaptive_decisions": decisions,
                    "cache_compaction": getattr(
                        result, "cache_compaction", None),
                    "topk_width_counts": {
                        str(width): sum(row.get("topk_width") == width
                                        for row in decisions)
                        for width in sorted({row.get("topk_width")
                                           for row in decisions
                                           if row.get("topk_width") is not None})
                    } if decisions else {},
                })
            reference = outputs[reference_method]
            for row in records[record_start:]:
                row["matches_ddtree"] = outputs[row["method"]] == reference
                same_cap = (row["method"] if row["method"].startswith("ddtree_b")
                            else reference_method)
                row["equal_cap_reference"] = same_cap
                row["matches_reference"] = (
                    outputs[row["method"]] == outputs[same_cap])
            exact &= all(row["matches_reference"]
                         for row in records[record_start:])
            print(f"repeat={repeat+1}/{args.repeats} sample={ordinal+1}/{args.samples}",
                  flush=True)

    summary = {method:aggregate(records, method) for method in method_names}
    reference_tpot = summary[reference_method]["mean_tpot_ms"]
    for method in method_names:
        if method != reference_method:
            summary[method]["speedup_vs_ddtree"] = (
                reference_tpot / summary[method]["mean_tpot_ms"])
    speedup = summary[candidate_method]["speedup_vs_ddtree"]
    greedy_gate = args.temperature == 0
    artifact = {
        "kind":"adaptive_architecture_development_gate_v1",
        "formal_result":False,
        "dataset":"openai/gsm8k/main/train",
        "dataset_path":str(parquet),
        "selection_seed":args.seed,
        "source_indices":indices,
        "samples":args.samples,
        "repeats":args.repeats,
        "max_new_tokens":args.max_new_tokens,
        "model":target_name,
        "draft_model":draft_name,
        "node_cap":args.reference_budget,
        "temperature":args.temperature,
        "experimental_positive_temperature":bool(args.temperature > 0),
        "candidate":candidate_method,
        "included_primary_control":args.include_primary_control,
        "included_fixed_b192_control":args.include_fixed_b192_control,
        "candidate_overrides":candidate_overrides,
        "candidate_controller_config":controller_config(candidate_builder()),
        "rank_calibration":({
            "checkpoint":str(args.rank_head_checkpoint.resolve()),
            "sha256":hashlib.sha256(
                args.rank_head_checkpoint.read_bytes()).hexdigest(),
            "strength":args.rank_calibration_strength,
            "training_data_separation_requires_formal_audit":True,
        } if args.rank_head_checkpoint is not None else None),
        "ratio_transport":({
            "checkpoint":str(args.ratio_transport_checkpoint.resolve()),
            "sha256":hashlib.sha256(
                args.ratio_transport_checkpoint.read_bytes()).hexdigest(),
            "strength":args.ratio_transport_strength,
            "training_data_separation_requires_formal_audit":True,
        } if args.ratio_transport_checkpoint is not None else None),
        "fixed_b192_control_config":(
            controller_config(fixed_b192_control_builder())
            if args.include_fixed_b192_control else None),
        "method_order":"balanced cyclic rotation",
        "includes_tree_build_and_controller_in_tpot":True,
        "exact_output_match":exact,
        "pass_rule":((
            "exact outputs and the selected candidate mean TPOT at least 1% "
            f"below DDTree B{args.reference_budget}"
        ) if greedy_gate else
            "exploratory T>0 smoke only; never passes the T=0 development gate"),
        "performance_passed":bool(greedy_gate and speedup >= 1.01),
        "strict_output_passed":bool(greedy_gate and exact),
        "passed":bool(greedy_gate and exact and speedup >= 1.01),
        "summary":summary,
        "records":records,
        "code_identity":code_identity(),
        "elapsed_seconds":time.time() - started,
    }
    atomic_json(args.output, artifact)
    print(json.dumps({key:artifact[key] for key in (
        "exact_output_match", "passed", "summary", "elapsed_seconds")},
        indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
