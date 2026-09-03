"""正版 DFlash（线性 speculative decoding）benchmark，输出格式与 benchmark.py 完全一致。

用于与 DFlash-SpecBlock 树模式在相同数据上对比加速比与平均接受长度。
不依赖 rank head，不需要 learned checkpoint。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from tqdm import tqdm

from .benchmark import baseline_greedy, _load_prompts
from .config import ExperimentConfig
from .device import configure_cuda_runtime, dtype_from_name, resolve_device, synchronize
from .dflash_adapter import DFlashBlockAdapter
from .models import load_models, render_prompt
from .rank_head import HeuristicRanker
from .vanilla_engine import VanillaDFlashEngine


def create_vanilla_engine(config: ExperimentConfig, device: torch.device):
    """构造正版 DFlash 引擎：DFlash draft adapter + 线性因果验证，无需 rank head。"""
    bundle = load_models(config, device)
    # vanilla DFlash 不做分支决策，ranker 仅满足 adapter 构造签名，不会被调用。
    ranker = HeuristicRanker().to(device).eval()
    adapter = DFlashBlockAdapter(
        target=bundle.target,
        draft=bundle.draft,
        ranker=ranker,
        block_size=config.block_size,
    )
    engine = VanillaDFlashEngine(
        target=bundle.target,
        adapter=adapter,
        device=device,
        dtype=dtype_from_name(config.dtype),
    )
    return engine, bundle.tokenizer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark vanilla DFlash (linear speculative decoding) on NVIDIA CUDA"
    )
    parser.add_argument("--config", default="configs/qwen3_4b_cuda_tree15.json")
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--output", default="outputs/benchmark_vanilla_dflash.jsonl")
    parser.add_argument("--max-prompts", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--device", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = ExperimentConfig.from_json(args.config)
    if args.device:
        config.device = args.device
    device = resolve_device(config.device)
    configure_cuda_runtime(device, config.allow_tf32)
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    engine, tokenizer = create_vanilla_engine(config, device)
    prompts = _load_prompts(Path(args.prompts))
    if args.max_prompts > 0:
        prompts = prompts[: args.max_prompts]
    max_new_tokens = (
        config.max_new_tokens if args.max_new_tokens is None else args.max_new_tokens
    )
    if max_new_tokens < 1:
        raise ValueError("max-new-tokens 必须为正整数")
    stop_ids = (
        {int(tokenizer.eos_token_id)} if tokenizer.eos_token_id is not None else set()
    )

    output_path = Path(args.output).expanduser()
    if not output_path.is_absolute():
        output_path = config.project_root / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    with output_path.open("w", encoding="utf-8") as stream:
        progress = tqdm(prompts, desc="vanilla-dflash", unit="prompt")
        for index, prompt in enumerate(progress):
            input_ids = render_prompt(tokenizer, prompt, config.enable_thinking).to(
                device,
                non_blocking=device.type == "cuda",
            )
            baseline_ids, baseline_ms = baseline_greedy(
                engine.target, input_ids, max_new_tokens, stop_ids, device
            )
            synchronize(device)
            hybrid_started_at = time.perf_counter()
            hybrid = engine.generate(input_ids, max_new_tokens, stop_ids)
            synchronize(device)
            hybrid_ms = (time.perf_counter() - hybrid_started_at) * 1000.0
            hybrid_stage_ms = hybrid.prefill_ms + hybrid.total_decode_ms
            hybrid_ids = hybrid.generated_ids[0].detach().cpu()
            baseline_cpu = baseline_ids.detach().cpu()
            exact_match = torch.equal(baseline_cpu, hybrid_ids)
            mismatch_index = None
            if not exact_match:
                common = min(int(baseline_cpu.numel()), int(hybrid_ids.numel()))
                differences = (baseline_cpu[:common] != hybrid_ids[:common]).nonzero()
                mismatch_index = (
                    int(differences[0].item()) if differences.numel() else common
                )
            row = {
                "index": index,
                "prompt": prompt,
                "baseline_tokens": int(baseline_ids.numel()),
                "baseline_ms": baseline_ms,
                "hybrid_tokens": int(hybrid.generated_ids.numel()),
                "hybrid_ms": hybrid_ms,
                "hybrid_stage_ms": hybrid_stage_ms,
                "wall_clock_speedup": (
                    baseline_ms / hybrid_ms if hybrid_ms > 0 else None
                ),
                "average_committed_per_verify": hybrid.average_accepted_length,
                "verify_iterations": len(hybrid.iterations),
                "greedy_exact_match": exact_match,
                "first_mismatch_index": mismatch_index,
            }
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            rows.append(row)
            progress.set_postfix(
                speedup=(
                    f"{row['wall_clock_speedup']:.2f}x"
                    if row["wall_clock_speedup"] is not None
                    else "n/a"
                ),
                match=row["greedy_exact_match"],
            )
            print(json.dumps(row, ensure_ascii=False))
            if not exact_match:
                import warnings

                warnings.warn(
                    f"提示词 {index} 的 hybrid 输出与 target greedy baseline 在 token "
                    f"{mismatch_index} 处不一致（{config.dtype} 或 attention backend 数值差异）；"
                    f"记录 mismatch 并继续，不中断 benchmark。",
                    stacklevel=2,
                )

    baseline_total = sum(row["baseline_ms"] for row in rows)
    hybrid_total = sum(row["hybrid_ms"] for row in rows)
    matched_rows = [row for row in rows if row["greedy_exact_match"]]
    mismatched_count = len(rows) - len(matched_rows)
    matched_baseline_total = sum(row["baseline_ms"] for row in matched_rows)
    matched_hybrid_total = sum(row["hybrid_ms"] for row in matched_rows)
    print(
        json.dumps(
            {
                "prompts": len(rows),
                "exact_matches": len(matched_rows),
                "mismatches": mismatched_count,
                "mismatch_rate": (
                    mismatched_count / len(rows) if rows else 0.0
                ),
                "total_speedup": (
                    baseline_total / hybrid_total if hybrid_total > 0 else None
                ),
                "matched_speedup": (
                    matched_baseline_total / matched_hybrid_total
                    if matched_hybrid_total > 0 and matched_rows
                    else None
                ),
                "mean_acceptance": (
                    sum(row["average_committed_per_verify"] for row in rows) / len(rows)
                    if rows
                    else 0.0
                ),
                "output": str(output_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
