#!/usr/bin/env python3
"""Run the current official z-lab/dflash Transformers path on fixed JSONL prompts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = ROOT / "third_party" / "dflash_official"
sys.path.insert(0, str(OFFICIAL))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--draft", required=True)
    parser.add_argument("--max-samples", type=int, default=2000)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    import torch
    from tqdm import tqdm
    from dflash.benchmark import apply_chat_template, load_transformers_models, stop_token_ids
    from dflash.model import dflash_generate

    rows = [json.loads(line) for line in args.prompts.open(encoding="utf-8") if line.strip()]
    rows = rows[: args.max_samples]
    device = torch.device("cuda:0")
    torch.manual_seed(0)
    target, draft, tokenizer = load_transformers_models(args.model, args.draft, device)
    block_size = int(draft.block_size)
    stops = stop_token_ids(target, tokenizer)

    def encode(prompt: str):
        text = apply_chat_template(tokenizer, [{"role": "user", "content": prompt}], "off")
        return tokenizer.encode(text, return_tensors="pt", add_special_tokens=False).to(device)

    warmup = encode(rows[0]["prompt"])
    for size in (1, block_size):
        dflash_generate(draft, target, warmup, min(64, args.max_new_tokens), None, 0.0,
                        block_size=size)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = 0
    mode = "w"
    if args.resume and args.output.exists():
        existing = [json.loads(line) for line in args.output.open(encoding="utf-8") if line.strip()]
        completed = len(existing)
        for index, old in enumerate(existing):
            if old["prompt"] != rows[index]["prompt"]:
                raise ValueError(f"resume mismatch at row {index}")
        mode = "a"

    with args.output.open(mode, encoding="utf-8") as stream:
        for index in tqdm(range(completed, len(rows)), desc="official-dflash", unit="prompt"):
            row = rows[index]
            ids = encode(row["prompt"])
            results = {}
            for size, name in ((1, "baseline"), (block_size, "dflash")):
                result = dflash_generate(
                    draft, target, ids, args.max_new_tokens, stops, 0.0,
                    block_size=size, return_stats=True,
                )
                generated = result.output_ids[0, result.num_input_tokens:].detach().cpu().tolist()
                decode_ms = float(result.time_per_output_token) * result.num_output_tokens * 1000
                results[name] = {
                    "generated_token_ids": generated,
                    "generated_tokens": int(result.num_output_tokens),
                    "prefill_ms": float(result.time_to_first_token) * 1000,
                    "decode_ms": decode_ms,
                    "tokens_per_second": result.num_output_tokens / max(decode_ms / 1000, 1e-12),
                    "acceptance_lengths": [int(x) for x in result.acceptance_lengths],
                }
            record = {
                "index": index, "dataset": row["dataset"], "source_id": row["source_id"],
                "prompt": row["prompt"], "implementation": "z-lab/dflash",
                "dtype": "bfloat16", "block_size_including_anchor": block_size,
                **results,
                "dflash_greedy_exact_match": results["baseline"]["generated_token_ids"] == results["dflash"]["generated_token_ids"],
            }
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()


if __name__ == "__main__":
    main()
