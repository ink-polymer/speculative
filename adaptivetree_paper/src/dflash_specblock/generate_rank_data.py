"""用目标模型离线生成 rank-head 训练文本。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from .benchmark import baseline_greedy
from .config import ExperimentConfig
from .device import configure_cuda_runtime, resolve_device
from .models import load_target_model, render_prompt


def _resolve_output_path(config: ExperimentConfig, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (config.project_root / path).resolve()


def _load_prompt_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            prompt = item.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"{path}:{line_number} 缺少非空 prompt 字段")
            rows.append(item)
    if not rows:
        raise ValueError(f"prompt 文件为空: {path}")
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate target-model training texts for rank head"
    )
    parser.add_argument("--config", default="configs/qwen3_4b_cuda.json")
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--output", default="datasets/generated/rank_train.jsonl")
    parser.add_argument("--max-prompts", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--device", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = ExperimentConfig.from_json(args.config)
    if args.device:
        config.device = args.device
    if args.max_prompts < 0:
        raise ValueError("max-prompts 不能为负数")
    device = resolve_device(config.device)
    configure_cuda_runtime(device, config.allow_tf32)
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)

    rows = _load_prompt_rows(Path(args.prompts).expanduser().resolve())
    if args.max_prompts > 0:
        rows = rows[: args.max_prompts]
    max_new_tokens = (
        config.max_new_tokens if args.max_new_tokens is None else args.max_new_tokens
    )
    if max_new_tokens < 1:
        raise ValueError("max-new-tokens 必须为正整数")

    # Rank-data generation uses only the target model. Avoid loading the unused 4B
    # DFlash draft, which saves one full model's GPU memory and startup I/O.
    bundle = load_target_model(config, device)
    stop_ids = (
        {int(bundle.tokenizer.eos_token_id)}
        if bundle.tokenizer.eos_token_id is not None
        else set()
    )
    output_path = _resolve_output_path(config, args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    generated = 0
    with output_path.open("w", encoding="utf-8") as stream:
        progress = tqdm(rows, desc="generate rank train data")
        for item in progress:
            prompt = item["prompt"]
            input_ids = render_prompt(bundle.tokenizer, prompt, config.enable_thinking).to(
                device,
                non_blocking=device.type == "cuda",
            )
            generated_ids, elapsed_ms = baseline_greedy(
                bundle.target,
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                stop_ids=stop_ids,
                device=device,
            )
            full_ids = torch.cat(
                [
                    input_ids.detach().cpu(),
                    generated_ids.unsqueeze(0),
                ],
                dim=1,
            )
            text = bundle.tokenizer.decode(full_ids[0], skip_special_tokens=True)
            row = {
                "text": text,
                "prompt": prompt,
                "source_dataset": item.get("dataset"),
                "source_id": item.get("source_id"),
                "metadata": item.get("metadata"),
                "generated_tokens": int(generated_ids.numel()),
                "generation_ms": elapsed_ms,
            }
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            generated += 1
            progress.set_postfix(rows=generated, tokens=int(generated_ids.numel()))

    print(
        json.dumps(
            {
                "output": str(output_path),
                "rows": generated,
                "max_new_tokens": max_new_tokens,
                "device": str(device),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
