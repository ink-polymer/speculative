"""Historical non-default controlled evaluation; current tables use official_reporting.py."""
from __future__ import annotations

import csv
import math
import random
from pathlib import Path

import numpy as np
import torch

from .common import BASELINES, VARIANTS, atomic_json, contract, digest, file_hash, load_json, run_lock, verify_contract
from .data import SOURCES, expected_turns, is_evaluation_row, user_turns
from .controller import PaperAdaptiveBuilder


def result_metrics(result):
    rounds = result["rounds"]
    if not math.isfinite(result["wall_ms"]) or result["wall_ms"] <= 0:
        raise ValueError("Invalid measured latency")
    return {"tokens": result["tokens"], "wall_ms": result["wall_ms"],
            "rounds": len(rounds), "accepted": sum(r["accepted"] for r in rounds),
            "committed": sum(r["committed"] for r in rounds),
            "nodes": sum(r["nodes"] for r in rounds),
            "policy_ms": sum(r.get("policy_ms", 0.) for r in rounds),
            "build_ms": sum(r.get("build_ms", 0.) for r in rounds),
            "actions": result.get("decisions", [])}


def measure_case(runtime, row, method, replica=0):
    """Each method builds its own complete dialogue; no reference answer injection.

    A turn is cold-prefilled from its chat history. Template/tokenization/decoding
    remain outside the generation timers for every method, as in single-turn tests.
    """
    prompts = user_turns(row)
    messages, measured = [], []
    for prompt in prompts:
        messages.append({"role": "user", "content": prompt})
        ids = runtime.encode(prompt) if len(prompts) == 1 else runtime.encode_messages(messages)
        input_sha256 = digest(ids.detach().cpu().tolist())
        result = runtime.generate(ids, method, replica)
        measured.append({**result_metrics(result), "input_sha256": input_sha256})
        answer = runtime.tokenizer.decode(result["tokens"], skip_special_tokens=True)
        messages.append({"role": "assistant", "content": answer})
    aggregate = {key: sum(turn[key] for turn in measured) for key in
                 ("wall_ms", "rounds", "accepted", "committed", "nodes", "policy_ms", "build_ms")}
    aggregate.update({"tokens": [token for turn in measured for token in turn["tokens"]],
                      "actions": [action for turn in measured for action in turn["actions"]],
                      "turns": measured})
    return aggregate


def same_dialogue(left, right):
    return (left["tokens"] == right["tokens"] and
            [(t["tokens"], t["input_sha256"]) for t in left["turns"]] ==
            [(t["tokens"], t["input_sha256"]) for t in right["turns"]])


def initial_states(cfg):
    return {f"{method}:{repeat}": PaperAdaptiveBuilder(cfg, method).state_dict()
            for method in cfg["variants"] for repeat in range(cfg["eval_repeats"])}


def validate_states(states, cfg):
    expected = initial_states(cfg)
    if set(states) != set(expected):
        raise ValueError("Missing/extra controller state")
    for key, state in states.items():
        method, _ = key.rsplit(":", 1)
        PaperAdaptiveBuilder(cfg, method).load_state_dict(state)


def evaluate(runtime, rows, directory, metadata, seed):
    methods = list(BASELINES) + runtime.cfg["variants"]
    with run_lock(directory):
        contract(directory, {**metadata, "stage": "evaluate", "seed": seed})
        runtime.warmup()
        runtime.restore_controllers(initial_states(runtime.cfg))
        order = list(range(len(rows)))
        random.Random(seed).shuffle(order)
        for completed, index in enumerate(order):
            row = rows[index]
            path = directory / "prompts" / f"{index:06d}.json"
            before = digest(runtime.controller_states())
            if path.exists():
                record = load_json(path)
                if (record["id"] != row["id"] or record["prompt_hash"] != row["prompt_hash"]
                        or not record["exact_match"] or record["state_before_sha256"] != before
                        or record["order_index"] != completed):
                    raise ValueError(f"Resume state chain/identity failure: {path}")
                validate_states(record["controller_states_after"], runtime.cfg)
                if digest(record["controller_states_after"]) != record["state_after_sha256"]:
                    raise ValueError("Corrupted controller checkpoint")
                runtime.restore_controllers(record["controller_states_after"])
                continue
            observations = {method: [] for method in methods}
            peak = {}
            for repeat in range(runtime.cfg["eval_repeats"]):
                shuffled = methods.copy()
                random.Random(f"{seed}:{row['id']}:{repeat}").shuffle(shuffled)
                for method in shuffled:
                    if runtime.device.type == "cuda":
                        torch.cuda.reset_peak_memory_stats(runtime.device)
                    result = measure_case(runtime, row, method, repeat)
                    observations[method].append(result)
                    if runtime.device.type == "cuda":
                        peak[method] = max(peak.get(method, 0), torch.cuda.max_memory_allocated(runtime.device))
            reference = observations["ar"][0]
            exact = all(same_dialogue(item, reference) for values in observations.values() for item in values)
            texts = [runtime.tokenizer.decode(turn["tokens"], skip_special_tokens=True) for turn in reference["turns"]]
            states = runtime.controller_states()
            record = {"order_index": completed, "state_before_sha256": before,
                      "controller_states_after": states, "state_after_sha256": digest(states),
                      "id": row["id"], "dataset": row["dataset"], "prompt_hash": row["prompt_hash"],
                      "seed": seed, "exact_match": exact, "observations": observations,
                      "peak_allocated_bytes": peak,
                      "generated_text": texts[0] if len(texts) == 1 else None,
                      "generated_turn_texts": texts}
            atomic_json(path, record)
            if not exact:
                raise RuntimeError(f"Greedy mismatch at {row['id']}; saved diagnostic, no official result table")
            print(f"evaluate seed={seed} {completed + 1}/{len(rows)} {row['dataset']}", flush=True)
        atomic_json(directory / "complete.json", {"prompts": len(rows), "methods": methods,
            "repeats": runtime.cfg["eval_repeats"], "seed": seed,
            "files": {f"prompts/{i:06d}.json": file_hash(directory / "prompts" / f"{i:06d}.json")
                      for i in range(len(rows))}})


def paired_bootstrap(baseline, method, draws=2000, seed=2026):
    a, b = np.asarray(baseline, dtype=float), np.asarray(method, dtype=float)
    if (a.shape != b.shape or a.ndim != 1 or not len(a) or not np.isfinite(a).all()
            or not np.isfinite(b).all() or np.any(a <= 0) or np.any(b <= 0) or draws < 1):
        raise ValueError("Invalid paired timing data")
    rng = np.random.default_rng(seed)
    ratios = []
    for _ in range(draws):
        indices = rng.integers(0, len(a), len(a))
        ratios.append(float(a[indices].sum() / b[indices].sum()))
    return [float(x) for x in np.quantile(ratios, [0.025, 0.975])]


def summarize(run_root: Path, rows, cfg, *, smoke=False):
    metadata = None
    if not smoke:
        if (set(cfg["variants"]) != set(VARIANTS) or len(cfg["seeds"]) < 3
                or cfg["eval_repeats"] < 3):
            raise ValueError("Formal matrix requires all ablations, >=3 seeds and >=3 timing repetitions")
        counts = {name: sum(r["dataset"] == name for r in rows) for name in SOURCES}
        if counts != {name: spec[3] for name, spec in SOURCES.items()} or sum(counts.values()) != len(rows):
            raise ValueError("Publication requires every full official test split")
        if any(not is_evaluation_row(r) for r in rows) or len({r["id"] for r in rows}) != len(rows):
            raise ValueError("Invalid formal evaluation split or duplicate source IDs")
        root_contract = load_json(run_root / "contract.json")
        protocol = root_contract["metadata"]
        verify_contract(run_root, protocol)
        if (protocol["smoke"] or protocol["config"] != cfg
                or protocol["test_ids"] != digest([r["id"] for r in rows])):
            raise ValueError("Formal run configuration/data contract mismatch")
        env = load_json(run_root / "gpu_preflight.json")
        if not env.get("gpu") or not env.get("transformers_import_ok"):
            raise ValueError("No valid GPU preflight for formal evaluation")
        metadata = {**protocol, "environment": env}
    tables, complete = [], []
    datasets = sorted({row["dataset"] for row in rows})
    expected_methods = list(BASELINES) + cfg["variants"]
    for seed in cfg["seeds"]:
        directory = run_root / "evaluation" / str(seed)
        if metadata is not None:
            verify_contract(directory, {**metadata, "stage": "evaluate", "seed": seed})
        completion = load_json(directory / "complete.json")
        if (completion["prompts"] != len(rows) or completion["methods"] != expected_methods
                or completion["seed"] != seed or completion["repeats"] != cfg["eval_repeats"]
                or len(completion["files"]) != len(rows)):
            raise ValueError("Incomplete formal evaluation or missing ablations")
        records = []
        for i, row in enumerate(rows):
            relative = f"prompts/{i:06d}.json"
            path = directory / relative
            if file_hash(path) != completion["files"].get(relative):
                raise ValueError(f"Changed or missing evaluation record: {path}")
            record = load_json(path)
            if (record["id"] != row["id"] or record["prompt_hash"] != row["prompt_hash"]
                    or record["dataset"] != row["dataset"] or record["seed"] != seed or not record["exact_match"]):
                raise ValueError("Evaluation identity/exactness failure")
            if set(record["observations"]) != set(expected_methods):
                raise ValueError("Missing method in a prompt record")
            reference = record["observations"]["ar"][0]
            for observations in record["observations"].values():
                if len(observations) != cfg["eval_repeats"]:
                    raise ValueError("Missing timing repetitions")
                for observation in observations:
                    if not same_dialogue(observation, reference) or not reference["tokens"]:
                        raise ValueError("Recorded token sequence differs from AR")
                    if not math.isfinite(observation["wall_ms"]) or observation["wall_ms"] <= 0:
                        raise ValueError("Invalid recorded latency")
                    turns = observation["turns"]
                    if len(turns) != expected_turns(row["dataset"]):
                        raise ValueError("Missing conversation turn")
                    if observation["tokens"] != [t for turn in turns for t in turn["tokens"]]:
                        raise ValueError("Aggregate tokens do not match turn boundaries")
                    if any(not turn["tokens"] or not math.isfinite(turn["wall_ms"]) or turn["wall_ms"] <= 0 for turn in turns):
                        raise ValueError("Invalid conversation turn measurement")
                    for key in ("wall_ms", "rounds", "accepted", "committed", "nodes", "policy_ms", "build_ms"):
                        if not math.isclose(observation[key], sum(turn[key] for turn in turns), rel_tol=1e-12, abs_tol=1e-9):
                            raise ValueError("Aggregate metric does not match conversation turns")
            records.append(record)
        order = list(range(len(rows)))
        random.Random(seed).shuffle(order)
        state_hash = digest(initial_states(cfg))
        for position, index in enumerate(order):
            record = records[index]
            states = record["controller_states_after"]
            validate_states(states, cfg)
            if (record["order_index"] != position or record["state_before_sha256"] != state_hash
                    or record["state_after_sha256"] != digest(states)):
                raise ValueError("Controller state chain broken")
            state_hash = record["state_after_sha256"]
        complete.append({"seed": seed, "prompts": len(records)})
        for dataset in datasets:
            selected = [r for r in records if r["dataset"] == dataset]
            def times(method):
                return [sum(v["wall_ms"] for v in r["observations"][method]) for r in selected]
            baseline, ddtree = times("ar"), times("ddtree")
            for method in expected_methods:
                latency = times(method)
                measured = [v for r in selected for v in r["observations"][method]]
                tokens = sum(len(v["tokens"]) for v in measured)
                round_count = sum(v["rounds"] for v in measured)
                tables.append({"dataset": dataset, "seed": seed, "method": method,
                    "prompts": len(selected), "repeats": cfg["eval_repeats"], "tokens": tokens,
                    "turns_per_prompt": expected_turns(dataset),
                    "measured_turns": len(selected) * expected_turns(dataset) * cfg["eval_repeats"],
                    "wall_ms": sum(latency), "tokens_per_second": 1000 * tokens / sum(latency),
                    "speedup_vs_ar": sum(baseline) / sum(latency),
                    "speedup_vs_ddtree": sum(ddtree) / sum(latency),
                    "descriptive_paired_ar_ci95": paired_bootstrap(baseline, latency, cfg["bootstrap_samples"]),
                    "descriptive_paired_ddtree_ci95": paired_bootstrap(ddtree, latency, cfg["bootstrap_samples"]),
                    "mean_nodes_per_verify": sum(v["nodes"] for v in measured) / round_count if round_count else 0.,
                    "mean_committed_per_verify": sum(v["committed"] for v in measured) / round_count if round_count else 0.,
                    "policy_ms": sum(v["policy_ms"] for v in measured),
                    "build_ms": sum(v["build_ms"] for v in measured), "exact_match": True})
    seed_summary = []
    for dataset in datasets:
        for method in expected_methods:
            selected = [r for r in tables if r["dataset"] == dataset and r["method"] == method]
            summary = {"dataset": dataset, "method": method, "seed_count": len(selected)}
            for metric in ("speedup_vs_ar", "speedup_vs_ddtree", "tokens_per_second"):
                values = [r[metric] for r in selected]
                summary[metric + "_mean"] = float(np.mean(values))
                summary[metric + "_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
            seed_summary.append(summary)
    report = {"publication_eligible": not smoke, "full_split": not smoke,
              "eligibility_scope": "automated completeness/exactness gates, not a guarantee of scientific validity",
              "measurement": "paired aggregate tokens / end-to-end seconds; all repeats included",
              "uncertainty": "prompt bootstrap is descriptive only: online states couple consecutive cases; order-seed SD reported separately",
              "training": False, "controller_state": "persistent per method/repetition, reset per order seed",
              "multi_turn": "own generated assistant history; cold prefill each turn; bootstrap by whole dialogue",
              "scope": "controlled in-repository implementations; not unmodified official runtimes",
              "seed_runs": complete, "rows": tables, "seed_summary": seed_summary}
    atomic_json(run_root / "tables.json", report)
    with (run_root / "tables.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(tables[0]))
        writer.writeheader()
        writer.writerows(tables)
    lines = ["# T=0 完整实验与消融", "", "此表来自共享模型、共享树验证器的受控实现，不代表未经修改的官方程序。", "",
             "计时含 prefill、策略、构树、验证、缓存和同步；重复测量全部计入。按 prompt 配对 bootstrap 仅描述已观测样本的离散性；在线状态引入序列依赖，不保证其名义覆盖率。另报顺序种子间标准差。", "",
             "| 数据集 | 顺序种子 | 方法 | 题/对话数 | 轮/题 | tok/s | 相对 AR | 相对 DDTree |", "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |"]
    if smoke:
        lines[0] = "# SMOKE ONLY：不可用于论文"
    for row in tables:
        lines.append(f"| {row['dataset']} | {row['seed']} | {row['method']} | {row['prompts']} | {row['turns_per_prompt']} | {row['tokens_per_second']:.2f} | {row['speedup_vs_ar']:.4f} | {row['speedup_vs_ddtree']:.4f} |")
    (run_root / "tables.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    comparison = ["# " + ("SMOKE ONLY：不可用于论文" if smoke else "跨顺序种子汇总"), "",
                  "相对 AR 加速比：独立种子的均值 ± 样本标准差；不是置信区间，也不混合 full/sanitized。", "",
                  "| 方法 | " + " | ".join(datasets) + " |",
                  "| --- | " + " | ".join("---:" for _ in datasets) + " |"]
    for method in expected_methods:
        cells = []
        for dataset in datasets:
            row = next(r for r in seed_summary if r["dataset"] == dataset and r["method"] == method)
            mean, sd = row["speedup_vs_ar_mean"], row["speedup_vs_ar_sd"]
            cells.append(f"{mean:.4f} ± {sd:.4f}" if sd is not None else f"{mean:.4f} (1 seed)")
        comparison.append("| " + method + " | " + " | ".join(cells) + " |")
    (run_root / "comparison.md").write_text("\n".join(comparison) + "\n", encoding="utf-8")
    return report
