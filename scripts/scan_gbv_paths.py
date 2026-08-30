#!/usr/bin/env python3
"""Compare temperature-1 DDTree with several exact-GBV path counts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_ROOT = PROJECT_ROOT / "third_party" / "ddtree_official"
sys.path.insert(0, str(OFFICIAL_ROOT))


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="models/Qwen3-4B")
    parser.add_argument("--draft", default="models/Qwen3-4B-DFlash-b16")
    parser.add_argument("--paths", default="2,3,4,5")
    parser.add_argument("--tree-budget", type=int, default=60)
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.0)
    return parser.parse_args()


def metrics(result: Any) -> dict[str, Any]:
    decode_ms = float(result.time_per_output_token) * int(result.num_output_tokens) * 1000.0
    return {
        "tokens": int(result.num_output_tokens),
        "decode_ms": decode_ms,
        "tokens_per_second": int(result.num_output_tokens) * 1000.0 / max(decode_ms, 1e-9),
        "mean_committed": sum(result.acceptance_lengths) / max(len(result.acceptance_lengths), 1),
        "rounds": int(result.decode_rounds),
        "stage_times_s": {k: float(v) for k, v in result.stage_times.items()},
    }


def main() -> None:
    cfg = args()
    import torch
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from ddtree import ddtree_generate
    from dflash import dflash_generate
    from gbv import gbv_generate
    from model import DFlashDraftModel

    path_counts = tuple(sorted({int(x) for x in cfg.paths.split(",") if x.strip()}))
    rows = [json.loads(x) for x in cfg.prompts.read_text(encoding="utf-8").splitlines() if x.strip()]
    rows = rows[: cfg.max_samples]
    target = AutoModelForCausalLM.from_pretrained(
        cfg.model, attn_implementation="sdpa", dtype=torch.bfloat16
    ).cuda().eval()
    draft = DFlashDraftModel.from_pretrained(
        cfg.draft, attn_implementation="flash_attention_2", dtype=torch.bfloat16
    ).cuda().eval()
    tokenizer = AutoTokenizer.from_pretrained(cfg.model)
    block_size = int(draft.block_size)
    stop_ids = target.generation_config.eos_token_id or tokenizer.eos_token_id
    stop_ids = [int(stop_ids)] if isinstance(stop_ids, int) else [int(x) for x in stop_ids]

    def encode(prompt: str) -> torch.Tensor:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False,
        )
        return tokenizer.encode(text, return_tensors="pt", add_special_tokens=False).cuda()

    def target_baseline(ids: torch.Tensor, length: int):
        return dflash_generate(
            model=draft, target=target, input_ids=ids, mask_token_id=draft.mask_token_id,
            max_new_tokens=length, block_size=1, stop_token_ids=stop_ids,
            temperature=cfg.temperature, verification_mode="target_match",
        )

    def tree(ids: torch.Tensor, length: int):
        return ddtree_generate(
            model=draft, target=target, input_ids=ids, mask_token_id=draft.mask_token_id,
            max_new_tokens=length, block_size=block_size, tree_budget=cfg.tree_budget,
            stop_token_ids=stop_ids, temperature=cfg.temperature,
        )

    def gbv(ids: torch.Tensor, paths: int, length: int):
        return gbv_generate(
            model=draft, target=target, input_ids=ids, mask_token_id=draft.mask_token_id,
            max_new_tokens=length, block_size=block_size, stop_token_ids=stop_ids,
            temperature=cfg.temperature, path_count=paths,
        )

    warmup = encode("Warmup")
    target_baseline(warmup, 16)
    tree(warmup, 16)
    for count in path_counts:
        gbv(warmup, count, 16)

    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    with cfg.output.open("w", encoding="utf-8") as stream:
        for index, row in enumerate(tqdm(rows, desc="gbv-scan")):
            ids = encode(row["prompt"])
            seed = 42 + index
            torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            baseline = metrics(target_baseline(ids, cfg.max_new_tokens))
            torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            fixed_tree = metrics(tree(ids, cfg.max_new_tokens))
            candidates = {}
            for count in path_counts:
                torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
                candidates[str(count)] = metrics(gbv(ids, count, cfg.max_new_tokens))
            stream.write(json.dumps({
                "index": index, "dataset": row.get("dataset"),
                "source_id": row.get("source_id"), "prompt": row["prompt"],
                "baseline": baseline, "ddtree": fixed_tree, "gbv": candidates,
                "temperature": cfg.temperature,
            }, ensure_ascii=False) + "\n")
            stream.flush()


if __name__ == "__main__":
    main()
