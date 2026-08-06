"""统一官方 benchmark 评测入口。"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from .benchmark import baseline_greedy
from .cli import create_engine
from .config import ExperimentConfig
from .device import resolve_device
from .judge import OpenAIJudgeClient
from .models import render_prompt
from .official_scorers import score_dataset_row, summarize_task_metrics


def _resolve_output_dir(config: ExperimentConfig, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (config.project_root / path).resolve()


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            prompt = row.get("prompt")
            dataset = row.get("dataset")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"{path}:{line_number} 缺少非空 prompt")
            if not isinstance(dataset, str) or not dataset.strip():
                raise ValueError(f"{path}:{line_number} 缺少 dataset")
            rows.append(row)
    if not rows:
        raise ValueError(f"评测文件为空: {path}")
    return rows


def _filter_rows(rows: list[dict[str, Any]], datasets: set[str] | None, max_prompts: int) -> list[dict[str, Any]]:
    filtered = [row for row in rows if datasets is None or row["dataset"] in datasets]
    return filtered if max_prompts <= 0 else filtered[:max_prompts]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run official benchmark evaluation for DFlash-SpecBlock")
    parser.add_argument("--config", default="configs/qwen3_4b_a2.json")
    parser.add_argument("--prompts", default="datasets/processed/specblock_official/prompts_all.jsonl")
    parser.add_argument("--output-dir", default="outputs/official_eval")
    parser.add_argument("--datasets", default="")
    parser.add_argument("--mode", choices=("hybrid", "target"), default="hybrid")
    parser.add_argument("--max-prompts", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--measure-baseline", action="store_true")
    return parser


def _generate_prediction(
    mode: str,
    engine: Any,
    tokenizer: Any,
    config: ExperimentConfig,
    device: torch.device,
    prompt: str,
    max_new_tokens: int,
    stop_ids: set[int],
) -> dict[str, Any]:
    input_ids = render_prompt(tokenizer, prompt, config.enable_thinking).to(device)
    if mode == "target":
        generated_ids, elapsed_ms = baseline_greedy(
            engine.target,
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            stop_ids=stop_ids,
            device=device,
        )
        prediction = tokenizer.decode(generated_ids, skip_special_tokens=True)
        return {
            "prediction": prediction,
            "generated_tokens": int(generated_ids.numel()),
            "decode_ms": elapsed_ms,
            "prefill_ms": None,
            "verify_iterations": None,
            "average_committed_per_verify": None,
        }
    result = engine.generate(input_ids, max_new_tokens=max_new_tokens, stop_token_ids=stop_ids)
    prediction = tokenizer.decode(result.generated_ids[0], skip_special_tokens=True)
    return {
        "prediction": prediction,
        "generated_tokens": int(result.generated_ids.shape[1]),
        "decode_ms": result.total_decode_ms,
        "prefill_ms": result.prefill_ms,
        "verify_iterations": len(result.iterations),
        "average_committed_per_verify": result.average_accepted_length,
    }


def main() -> None:
    args = build_parser().parse_args()
    config = ExperimentConfig.from_json(args.config)
    if args.device:
        config.device = args.device
    if args.max_prompts < 0:
        raise ValueError("max-prompts 不能为负数")
    device = resolve_device(config.device)
    engine, tokenizer = create_engine(config, device)
    max_new_tokens = config.max_new_tokens if args.max_new_tokens is None else args.max_new_tokens
    if max_new_tokens < 1:
        raise ValueError("max-new-tokens 必须为正整数")

    judge_client = OpenAIJudgeClient.from_env(args.judge_model)
    datasets = {item.strip() for item in args.datasets.split(",") if item.strip()} or None
    rows = _filter_rows(
        _load_rows(Path(args.prompts).expanduser().resolve()),
        datasets=datasets,
        max_prompts=args.max_prompts,
    )
    stop_ids = {int(tokenizer.eos_token_id)} if tokenizer.eos_token_id is not None else set()
    output_dir = _resolve_output_dir(config, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    progress = tqdm(rows, desc="official eval", unit="sample")
    for index, row in enumerate(progress):
        dataset = str(row["dataset"])
        prompt = str(row["prompt"])
        prediction_info = _generate_prediction(
            mode=args.mode,
            engine=engine,
            tokenizer=tokenizer,
            config=config,
            device=device,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            stop_ids=stop_ids,
        )
        score = score_dataset_row(dataset, row, prediction_info["prediction"])
        if dataset == "translation":
            score["reference"] = str((row.get("metadata") or {}).get("reference", ""))
        if dataset == "mt_bench" and judge_client is not None:
            score.update(
                judge_client.judge_mt_bench(
                    prompt=prompt,
                    response=prediction_info["prediction"],
                    category=(row.get("metadata") or {}).get("category"),
                )
            )
        if dataset == "alpaca" and judge_client is not None:
            score.update(
                judge_client.judge_alpaca(
                    prompt=prompt,
                    response=prediction_info["prediction"],
                    reference=str((row.get("metadata") or {}).get("reference", "")),
                )
            )
        item = {
            "index": index,
            "dataset": dataset,
            "source_id": row.get("source_id"),
            "prediction": prediction_info["prediction"],
            "generated_tokens": prediction_info["generated_tokens"],
            "prefill_ms": prediction_info["prefill_ms"],
            "decode_ms": prediction_info["decode_ms"],
            "verify_iterations": prediction_info["verify_iterations"],
            "average_committed_per_verify": prediction_info["average_committed_per_verify"],
            "score": score,
        }
        if args.measure_baseline and args.mode == "hybrid":
            input_ids = render_prompt(tokenizer, prompt, config.enable_thinking).to(device)
            baseline_ids, baseline_ms = baseline_greedy(
                engine.target,
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                stop_ids=stop_ids,
                device=device,
            )
            baseline_prediction = tokenizer.decode(baseline_ids, skip_special_tokens=True)
            hybrid_ms = (
                (prediction_info["prefill_ms"] or 0.0) + float(prediction_info["decode_ms"])
            )
            item["baseline_ms"] = baseline_ms
            item["baseline_prediction"] = baseline_prediction
            item["greedy_exact_match"] = baseline_prediction == prediction_info["prediction"]
            item["wall_clock_speedup"] = baseline_ms / hybrid_ms if hybrid_ms > 0 else None
        results.append(item)
        grouped[dataset].append(item)
        progress.set_postfix(dataset=dataset, tokens=item["generated_tokens"])

    results_path = output_dir / "results.jsonl"
    with results_path.open("w", encoding="utf-8") as stream:
        for row in results:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "mode": args.mode,
        "prompts": len(results),
        "judge_enabled": judge_client is not None,
        "datasets": {
            dataset: {
                "count": len(items),
                "metrics": summarize_task_metrics(dataset, items),
            }
            for dataset, items in grouped.items()
        },
        "results": str(results_path),
    }
    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
