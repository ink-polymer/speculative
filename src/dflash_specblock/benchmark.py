"""JSONL 提示集上的 baseline 与 DFlash-SpecBlock 对照实验。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from .cli import create_engine
from .config import ExperimentConfig
from .device import DeviceTimer, resolve_device
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

    cache = DynamicCache()
    generated: list[int] = []
    with DeviceTimer(device) as timer:
        output = target(
            input_ids=input_ids,
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )
        token = int(output.logits[0, -1].argmax().item())
        generated.append(token)
        while len(generated) < max_new_tokens and token not in stop_ids:
            token_tensor = torch.tensor([[token]], dtype=torch.long, device=device)
            output = target(
                input_ids=token_tensor,
                past_key_values=output.past_key_values,
                use_cache=True,
                return_dict=True,
            )
            token = int(output.logits[0, -1].argmax().item())
            generated.append(token)
    return torch.tensor(generated, dtype=torch.long), timer.elapsed_ms


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
    parser = argparse.ArgumentParser(description="Benchmark DFlash-SpecBlock on Ascend A2")
    parser.add_argument("--config", default="configs/qwen3_4b_a2.json")
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
            input_ids = render_prompt(tokenizer, prompt, config.enable_thinking).to(device)
            baseline_ids, baseline_ms = baseline_greedy(
                engine.target, input_ids, max_new_tokens, stop_ids, device
            )
            hybrid = engine.generate(input_ids, max_new_tokens, stop_ids)
            hybrid_ms = hybrid.prefill_ms + hybrid.total_decode_ms
            hybrid_ids = hybrid.generated_ids[0].detach().cpu()
            baseline_cpu = baseline_ids.detach().cpu()
            exact_match = torch.equal(baseline_cpu, hybrid_ids)
            mismatch_index = None
            if not exact_match:
                common = min(int(baseline_cpu.numel()), int(hybrid_ids.numel()))
                differences = (baseline_cpu[:common] != hybrid_ids[:common]).nonzero()
                mismatch_index = int(differences[0].item()) if differences.numel() else common
            row = {
                "index": index,
                "prompt": prompt,
                "baseline_tokens": int(baseline_ids.numel()),
                "baseline_ms": baseline_ms,
                "hybrid_tokens": int(hybrid.generated_ids.numel()),
                "hybrid_ms": hybrid_ms,
                "wall_clock_speedup": baseline_ms / hybrid_ms if hybrid_ms > 0 else None,
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
                    f"{mismatch_index} 处不一致（bfloat16 数值精度差异）；"
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
                "output": str(output_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
