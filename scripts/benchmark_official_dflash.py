#!/usr/bin/env python3
"""Run the vendored official DFlash implementation on this paper's JSONL suite."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_ROOT = PROJECT_ROOT / "third_party" / "dflash_official"
sys.path.insert(0, str(OFFICIAL_ROOT))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Official z-lab/DFlash on the exact DFlash-SpecBlock prompt JSONL"
    )
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--draft", default="z-lab/Qwen3-4B-DFlash-b16")
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--draft-revision", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    return parser


def _load_rows(path: Path, limit: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.expanduser().resolve().open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            prompt = row.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"{path}:{line_number} is missing a non-empty prompt")
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    if not rows:
        raise ValueError(f"No prompts found in {path}")
    return rows


def _metrics(result: Any) -> dict[str, Any]:
    generated = result.output_ids[0, result.num_input_tokens :].detach().cpu().tolist()
    decode_s = float(result.time_per_output_token) * int(result.num_output_tokens)
    return {
        "generated_token_ids": generated,
        "generated_tokens": int(result.num_output_tokens),
        "prefill_ms": float(result.time_to_first_token) * 1000.0,
        "decode_ms": decode_s * 1000.0,
        "tokens_per_second": int(result.num_output_tokens) / max(decode_s, 1e-12),
        "acceptance_lengths": [int(value) for value in result.acceptance_lengths],
    }


def main() -> None:
    args = _parser().parse_args()
    if not OFFICIAL_ROOT.is_dir():
        raise FileNotFoundError(f"Official DFlash checkout not found: {OFFICIAL_ROOT}")

    import torch
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from dflash.model import DFlashDraftModel, dflash_generate

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Official DFlash Transformers benchmark requires an available CUDA GPU")
    torch.cuda.set_device(device)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    target_kwargs: dict[str, Any] = {
        "attn_implementation": "sdpa",
        "dtype": torch.bfloat16,
    }
    draft_kwargs = dict(target_kwargs)
    tokenizer_kwargs: dict[str, Any] = {}
    if args.model_revision:
        target_kwargs["revision"] = args.model_revision
        tokenizer_kwargs["revision"] = args.model_revision
    if args.draft_revision:
        draft_kwargs["revision"] = args.draft_revision

    target = AutoModelForCausalLM.from_pretrained(args.model, **target_kwargs).to(device).eval()
    draft = DFlashDraftModel.from_pretrained(args.draft, **draft_kwargs).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, **tokenizer_kwargs)
    block_size = int(draft.block_size if args.block_size is None else args.block_size)
    stop_ids = target.generation_config.eos_token_id or tokenizer.eos_token_id
    stop_ids = [int(stop_ids)] if isinstance(stop_ids, int) else [int(x) for x in stop_ids]
    rows = _load_rows(args.prompts, args.max_samples)

    def encode(prompt: str):
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        return tokenizer.encode(text, return_tensors="pt", add_special_tokens=False).to(device)

    warmup_ids = encode("Warmup")
    for size in (1, block_size):
        dflash_generate(
            draft,
            target,
            warmup_ids,
            min(16, args.max_new_tokens),
            stop_ids,
            args.temperature,
            block_size=size,
        )

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for index, row in enumerate(tqdm(rows, desc="official-dflash", unit="prompt")):
            input_ids = encode(row["prompt"])
            baseline_result = dflash_generate(
                draft,
                target,
                input_ids,
                args.max_new_tokens,
                stop_ids,
                args.temperature,
                block_size=1,
                return_stats=True,
            )
            dflash_result = dflash_generate(
                draft,
                target,
                input_ids,
                args.max_new_tokens,
                stop_ids,
                args.temperature,
                block_size=block_size,
                return_stats=True,
            )
            baseline = _metrics(baseline_result)
            dflash = _metrics(dflash_result)
            record = {
                "index": index,
                "dataset": row.get("dataset"),
                "source_id": row.get("source_id"),
                "prompt": row["prompt"],
                "implementation": "z-lab/dflash",
                "dtype": "bfloat16",
                "block_size_including_anchor": block_size,
                "baseline": baseline,
                "dflash": dflash,
                "greedy_exact_match": (
                    baseline["generated_token_ids"] == dflash["generated_token_ids"]
                    if args.temperature == 0.0
                    else None
                ),
            }
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()


if __name__ == "__main__":
    main()
