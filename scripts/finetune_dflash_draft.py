#!/usr/bin/env python3
"""Domain-adapt the DFlash drafter on strictly disjoint target continuations."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dflash_specblock.config import ExperimentConfig  # noqa: E402
from dflash_specblock.device import configure_cuda_runtime, resolve_device  # noqa: E402
from dflash_specblock.models import load_models  # noqa: E402


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/qwen3_4b_cuda.json")
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--anchors-per-row", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    if args.epochs < 1 or args.anchors_per_row < 1:
        raise ValueError("epochs and anchors-per-row must be positive")
    if args.gradient_accumulation < 1:
        raise ValueError("gradient-accumulation must be positive")

    config = ExperimentConfig.from_json(args.config)
    device = resolve_device(config.device)
    configure_cuda_runtime(device, config.allow_tf32)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    rows = [
        json.loads(line)
        for line in args.train_data.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError("training JSONL is empty")

    bundle = load_models(config, device)
    target = bundle.target.eval()
    draft = bundle.draft.train().float()
    for parameter in target.parameters():
        parameter.requires_grad_(False)
    for parameter in draft.parameters():
        parameter.requires_grad_(True)

    target_layer_ids = [int(value) for value in draft.target_layer_ids]
    checkpoint_block = int(getattr(draft, "block_size", config.block_size + 1))
    horizon = checkpoint_block - 1
    mask_token_id = int(draft.mask_token_id)
    tokenized: list[tuple[torch.Tensor, int]] = []
    for row in rows:
        ids = bundle.tokenizer(
            row["text"], return_tensors="pt", add_special_tokens=True
        ).input_ids[0].contiguous()
        tokenized.append((ids, int(row.get("generated_tokens", 0))))

    examples: list[tuple[int, int]] = []
    sampler = random.Random(args.seed)
    for row_index, (ids, generated_tokens) in enumerate(tokenized):
        max_anchor = int(ids.numel()) - checkpoint_block
        if max_anchor < 1:
            continue
        min_anchor = max(1, int(ids.numel()) - generated_tokens - 1)
        min_anchor = min(min_anchor, max_anchor)
        population = list(range(min_anchor, max_anchor + 1))
        if len(population) <= args.anchors_per_row:
            selected = population
        else:
            selected = sampler.sample(population, args.anchors_per_row)
        examples.extend((row_index, anchor) for anchor in selected)
    if not examples:
        raise RuntimeError("no valid assistant-side training anchors")

    optimizer = torch.optim.AdamW(
        draft.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    updates = 0
    losses: list[float] = []
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(args.epochs):
        sampler.shuffle(examples)
        progress = tqdm(examples, desc=f"draft-ft {epoch + 1}")
        for example_index, (row_index, anchor_index) in enumerate(progress, start=1):
            ids = tokenized[row_index][0].to(device)
            prefix = ids[:anchor_index].unsqueeze(0)
            anchor = ids[anchor_index : anchor_index + 1]
            labels = ids[
                anchor_index + 1 : anchor_index + 1 + horizon
            ].unsqueeze(0)

            with torch.inference_mode():
                target_output = target(
                    input_ids=prefix,
                    output_hidden_states=True,
                    use_cache=False,
                    logits_to_keep=1,
                    return_dict=True,
                )
                target_context_inference = torch.cat(
                    [target_output.hidden_states[layer_id + 1] for layer_id in target_layer_ids],
                    dim=-1,
                )
                mask_ids = torch.full(
                    (1, horizon), mask_token_id, dtype=torch.long, device=device
                )
                noise_ids = torch.cat([anchor.unsqueeze(0), mask_ids], dim=1)
                noise_embedding_inference = target.get_input_embeddings()(noise_ids)
                position_ids_inference = torch.arange(
                    target_context_inference.shape[1] + checkpoint_block,
                    dtype=torch.long,
                    device=device,
                ).unsqueeze(0)

            # Tensors created under inference_mode cannot be retained for a
            # trainable draft backward.  Clone only the compact boundary
            # tensors after leaving inference mode; target activations remain
            # frozen and are released immediately.
            target_context = target_context_inference.clone()
            noise_embedding = noise_embedding_inference.clone()
            position_ids = position_ids_inference.clone()

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                hidden_all = draft(
                    target_hidden=target_context,
                    noise_embedding=noise_embedding,
                    position_ids=position_ids,
                    past_key_values=None,
                    use_cache=False,
                    is_causal=False,
                )
                if not isinstance(hidden_all, torch.Tensor):
                    hidden_all = hidden_all.last_hidden_state
                hidden = hidden_all[:, 1 : 1 + horizon]
                logits = target.get_output_embeddings()(hidden)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]), labels.reshape(-1)
                )
                scaled_loss = loss / args.gradient_accumulation
            scaled_loss.backward()
            losses.append(float(loss.detach()))

            should_step = (
                example_index % args.gradient_accumulation == 0
                or example_index == len(examples)
            )
            if should_step:
                torch.nn.utils.clip_grad_norm_(draft.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                updates += 1
            if example_index % 20 == 0:
                progress.set_postfix(loss=f"{sum(losses[-20:]) / min(20, len(losses)):.3f}")

    args.output.mkdir(parents=True, exist_ok=True)
    draft.eval().to(torch.bfloat16).save_pretrained(
        args.output, safe_serialization=True, max_shard_size="4GB"
    )
    metadata = {
        "architecture": "dflash_disjoint_domain_adaptation_v1",
        "training_rows": len(rows),
        "training_examples_per_epoch": len(examples),
        "epochs": args.epochs,
        "optimizer_updates": updates,
        "anchors_per_row": args.anchors_per_row,
        "learning_rate": args.learning_rate,
        "mean_training_loss": sum(losses) / len(losses),
        "target_model_id": config.target_model_id,
        "draft_model_id": config.draft_model_id,
    }
    (args.output / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
