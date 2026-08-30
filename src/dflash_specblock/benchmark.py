"""JSONL 提示集上的 baseline 与 DFlash-SpecBlock 对照实验。"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import torch
from tqdm import tqdm

from .cli import create_engine, finalize_tree_policy, persist_tree_policy
from .config import ExperimentConfig
from .device import configure_cuda_runtime, resolve_device, synchronize
from .models import render_prompt


@torch.inference_mode()
def baseline_greedy(
    target: torch.nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    stop_ids: set[int],
    device: torch.device,
) -> tuple[torch.Tensor, float]:
    """使用同一目标模型与 DynamicCache 的标准逐 token greedy 基线。"""
    from transformers import DynamicCache

    if max_new_tokens < 1:
        return torch.empty(0, dtype=torch.long), 0.0
    cache = DynamicCache()
    generated: list[torch.Tensor] = []
    token_id: int | None = None
    synchronize(device)
    started_at = time.perf_counter()
    output = target(
        input_ids=input_ids,
        past_key_values=cache,
        use_cache=True,
        # Qwen3 otherwise projects every prompt position over the full
        # vocabulary although greedy prefill consumes only the last row.
        logits_to_keep=1,
        return_dict=True,
    )
    token_tensor = output.logits[0, -1].argmax().reshape(1, 1)
    generated.append(token_tensor.reshape(-1))
    if stop_ids:
        token_id = int(token_tensor.item())
    while len(generated) < max_new_tokens and (
        not stop_ids or token_id not in stop_ids
    ):
        output = target(
            input_ids=token_tensor,
            past_key_values=output.past_key_values,
            use_cache=True,
            return_dict=True,
        )
        token_tensor = output.logits[0, -1].argmax().reshape(1, 1)
        generated.append(token_tensor.reshape(-1))
        if stop_ids:
            token_id = int(token_tensor.item())
    synchronize(device)
    elapsed_ms = (time.perf_counter() - started_at) * 1000.0
    return torch.cat(generated).detach().cpu(), elapsed_ms


def _load_prompts(path: Path) -> list[str]:
    prompts: list[str] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            prompt = item.get("prompt")
            if not isinstance(prompt, str):
                raise ValueError(f"{path}:{line_number} 缺少 prompt 字段")
            prompts.append(prompt)
    if not prompts:
        raise ValueError(f"提示集为空: {path}")
    return prompts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark DFlash-SpecBlock on NVIDIA CUDA")
    parser.add_argument("--config", default="configs/qwen3_4b_cuda.json")
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--output", default="outputs/benchmark.jsonl")
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
    engine, tokenizer = create_engine(config, device)
    prompts = _load_prompts(Path(args.prompts))
    if args.max_prompts > 0:
        prompts = prompts[: args.max_prompts]
    max_new_tokens = (
        config.max_new_tokens if args.max_new_tokens is None else args.max_new_tokens
    )
    if max_new_tokens < 1:
        raise ValueError("max-new-tokens 必须为正整数")
    stop_ids = {int(tokenizer.eos_token_id)} if tokenizer.eos_token_id is not None else set()

    output_path = Path(args.output).expanduser()
    if not output_path.is_absolute():
        output_path = config.project_root / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    with output_path.open("w", encoding="utf-8") as stream:
        progress = tqdm(prompts, desc="benchmark", unit="prompt")
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
                mismatch_index = int(differences[0].item()) if differences.numel() else common
            budget_histogram = Counter(item.tree_nodes for item in hybrid.iterations)
            row = {
                "index": index,
                "prompt": prompt,
                "baseline_tokens": int(baseline_ids.numel()),
                "baseline_ms": baseline_ms,
                "hybrid_tokens": int(hybrid.generated_ids.numel()),
                "hybrid_ms": hybrid_ms,
                "hybrid_stage_ms": hybrid_stage_ms,
                "wall_clock_speedup": baseline_ms / hybrid_ms if hybrid_ms > 0 else None,
                "average_committed_per_verify": hybrid.average_accepted_length,
                "verify_iterations": len(hybrid.iterations),
                "mean_tree_nodes": (
                    sum(item.tree_nodes for item in hybrid.iterations) / len(hybrid.iterations)
                    if hybrid.iterations
                    else 0.0
                ),
                "tree_budget_histogram": {
                    str(budget): budget_histogram[budget]
                    for budget in sorted(budget_histogram)
                },
                "draft_decode_ms": sum(item.draft_ms for item in hybrid.iterations),
                "verify_decode_ms": sum(item.verify_ms for item in hybrid.iterations),
                "tree_build_ms": sum(item.tree_build_ms for item in hybrid.iterations),
                "greedy_exact_match": exact_match,
                "first_mismatch_index": mismatch_index,
            }
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            rows.append(row)
            # Training runs checkpoint after each prompt, outside the timed decode region, so an
            # interrupted multi-hour GPU job can resume without losing the learned policy.
            persist_tree_policy(engine, config)
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

    finalize_tree_policy(engine, config)
    persist_tree_policy(engine, config)
    policy_diagnostics = getattr(engine.tree_builder, "policy_diagnostics", None)
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
                "total_speedup": baseline_total / hybrid_total if hybrid_total > 0 else None,
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
                "tree_policy": (
                    policy_diagnostics() if policy_diagnostics is not None else None
                ),
                "output": str(output_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
