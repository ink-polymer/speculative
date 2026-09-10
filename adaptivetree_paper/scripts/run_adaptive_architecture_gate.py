#!/usr/bin/env python3
"""Held-out GSM8K-train gate for AdaptiveTree architecture changes.

This is deliberately separate from the registered formal test protocol.  It
uses deterministic examples from GSM8K *train*, balances method execution
order, includes every decode/tree/controller cost in TPOT, and refuses to pass
unless all greedy outputs exactly match official DDTree B128.
"""
from __future__ import annotations

import argparse
import json
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
from dflash_specblock.paper.controller import PaperAdaptiveBuilder, make_paper_builder
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
                  stop_token_ids=[tokenizer.eos_token_id], temperature=0.)
    records = []
    exact = True
    temperature_methods = {}
    extended_cfg = {
        **adaptive_cfg,
        "budget_candidates":[30,45,60,80,100,128,160,192],
        "initial_budget":128,
    }
    method_names = ("ddtree_b128", "ddtree_b160", "ddtree_b192",
                    "adaptive_raw_b128", "adaptive_guarded_raw",
                    "adaptive_raw_b192", "adaptive_guarded_b192")
    started = time.time()
    for repeat in range(args.repeats):
        previous = PaperAdaptiveBuilder(
            adaptive_cfg, "no_exploration", timing_partition="budget_aware")
        guarded = make_paper_builder(adaptive_cfg, PRIMARY_ADAPTIVE_METHOD)
        raw_fixed = make_paper_builder(adaptive_cfg, PRIMARY_ADAPTIVE_METHOD)
        raw_fixed.initial_latency_samples = 0
        raw_fixed.minimum_latency_samples = 0
        raw_fixed.minimum_latency_saving_ratio = 1.
        raw_fixed.minimum_utility_gain_ratio = 1.
        tempered = {}
        for name, temperature in temperature_methods.items():
            builder = make_paper_builder(adaptive_cfg, PRIMARY_ADAPTIVE_METHOD)
            builder.initial_latency_samples = 0
            builder.minimum_latency_samples = 0
            builder.minimum_latency_saving_ratio = 1.
            builder.minimum_utility_gain_ratio = 1.
            builder.proposal_temperature = temperature
            tempered[name] = builder
        guarded192 = PaperAdaptiveBuilder(
            extended_cfg, "guarded_raw_prefix", timing_partition="budget_aware")
        raw192 = PaperAdaptiveBuilder(
            extended_cfg, "guarded_raw_prefix", timing_partition="budget_aware")
        raw192.initial_latency_samples = 0
        raw192.minimum_latency_samples = 0
        raw192.minimum_latency_saving_ratio = 1.
        raw192.minimum_utility_gain_ratio = 1.

        def generate(method, ids, maximum):
            kwargs = {**common, "input_ids":ids, "max_new_tokens":maximum}
            if method.startswith("ddtree_b"):
                return u.ddtree.ddtree_generate(
                    **kwargs, tree_budget=int(method.removeprefix("ddtree_b")))
            if method == "adaptive_previous":
                builder = previous
            elif method == "adaptive_raw_b128":
                builder = raw_fixed
            elif method in tempered:
                builder = tempered[method]
            elif method == "adaptive_raw_b192":
                builder = raw192
            elif method == "adaptive_guarded_b192":
                builder = guarded192
            else:
                builder = guarded
            return adaptive_generate(**kwargs, builder=builder)

        warmup = encode("Warmup")
        for method in method_names:
            generate(method, warmup, min(args.max_new_tokens, 32))
        # Hardware warmup must not pretrain either online controller.
        previous = PaperAdaptiveBuilder(
            adaptive_cfg, "no_exploration", timing_partition="budget_aware")
        guarded = make_paper_builder(adaptive_cfg, PRIMARY_ADAPTIVE_METHOD)
        raw_fixed = make_paper_builder(adaptive_cfg, PRIMARY_ADAPTIVE_METHOD)
        raw_fixed.initial_latency_samples = 0
        raw_fixed.minimum_latency_samples = 0
        raw_fixed.minimum_latency_saving_ratio = 1.
        raw_fixed.minimum_utility_gain_ratio = 1.
        tempered = {}
        for name, temperature in temperature_methods.items():
            builder = make_paper_builder(adaptive_cfg, PRIMARY_ADAPTIVE_METHOD)
            builder.initial_latency_samples = 0
            builder.minimum_latency_samples = 0
            builder.minimum_latency_saving_ratio = 1.
            builder.minimum_utility_gain_ratio = 1.
            builder.proposal_temperature = temperature
            tempered[name] = builder
        guarded192 = PaperAdaptiveBuilder(
            extended_cfg, "guarded_raw_prefix", timing_partition="budget_aware")
        raw192 = PaperAdaptiveBuilder(
            extended_cfg, "guarded_raw_prefix", timing_partition="budget_aware")
        raw192.initial_latency_samples = 0
        raw192.minimum_latency_samples = 0
        raw192.minimum_latency_saving_ratio = 1.
        raw192.minimum_utility_gain_ratio = 1.

        for ordinal, index in enumerate(indices):
            prompt = (questions[index]
                      + "\nPlease reason step by step, and put your final answer within \\boxed{}.")
            ids = encode(prompt)
            order = method_names[(ordinal + repeat) % len(method_names):]
            order += method_names[:(ordinal + repeat) % len(method_names)]
            outputs = {}
            record_start = len(records)
            for method in order:
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
                    "decode_rounds": result.decode_rounds,
                    "acceptance_length": mean(result.acceptance_lengths),
                    "stage_times": result.stage_times,
                    "selected_budget_counts": {
                        str(budget): sum(
                            (row.get("decision") or {}).get("budget") == budget
                            for row in decisions)
                        for budget in extended_cfg["budget_candidates"]
                    } if decisions else {},
                })
            reference = outputs["ddtree_b128"]
            for row in records[record_start:]:
                row["matches_ddtree"] = outputs[row["method"]] == reference
                same_cap = ("ddtree_b192" if row["method"] in {
                    "adaptive_raw_b192", "adaptive_guarded_b192"
                } else (row["method"] if row["method"].startswith("ddtree_b")
                        else "ddtree_b128"))
                row["equal_cap_reference"] = same_cap
                row["matches_reference"] = (
                    outputs[row["method"]] == outputs[same_cap])
            exact &= all(row["matches_reference"]
                         for row in records[record_start:])
            print(f"repeat={repeat+1}/{args.repeats} sample={ordinal+1}/{args.samples}",
                  flush=True)

    summary = {method:aggregate(records, method) for method in method_names}
    reference_tpot = summary["ddtree_b128"]["mean_tpot_ms"]
    for method in method_names:
        if method != "ddtree_b128":
            summary[method]["speedup_vs_ddtree"] = (
                reference_tpot / summary[method]["mean_tpot_ms"])
    speedup = summary["adaptive_guarded_raw"]["speedup_vs_ddtree"]
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
        "node_cap":128,
        "method_order":"balanced cyclic rotation",
        "includes_tree_build_and_controller_in_tpot":True,
        "exact_output_match":exact,
        "pass_rule":"exact outputs and guarded-spine mean TPOT at least 1% below DDTree B128",
        "performance_passed":bool(speedup >= 1.01),
        "strict_output_passed":exact,
        "passed":bool(exact and speedup >= 1.01),
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
