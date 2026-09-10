"""Train a one-shot top-rank tree calibrator by Target distillation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random

import torch
import torch.nn.functional as F

from dflash_specblock.dflash_adapter import DFlashBlockAdapter
from dflash_specblock.rank_head import HeuristicRanker
from gbv_experiments.config import load_config
from gbv_experiments.engine import load_models
from gbv_experiments.prefix_conditional import RankCalibratedHead
from gbv_experiments.terminal_formal import allocation_gate, model_gate


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def texts(path: Path) -> list[str]:
    result = []
    with path.open() as stream:
        for line in stream:
            if line.strip():
                result.append(json.loads(line)["text"])
    if not result:
        raise ValueError("Rank-calibrator corpus is empty")
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", type=Path, required=True)
    result.add_argument("--train-data", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--max-samples", type=int, default=384)
    result.add_argument("--holdout-samples", type=int, default=64)
    result.add_argument("--epochs", type=int, default=2)
    result.add_argument("--learning-rate", type=float, default=1e-3)
    result.add_argument("--rank", type=int, default=64)
    result.add_argument("--support-size", type=int, default=32)
    result.add_argument("--length", type=int, default=15)
    result.add_argument("--max-context", type=int, default=512)
    result.add_argument("--seed", type=int, default=20260910)
    result.add_argument("--device", default="cuda:0")
    return result


def remaining_kl(log_scores: torch.Tensor,
                 teacher: torch.Tensor) -> torch.Tensor:
    per_depth = F.kl_div(log_scores, teacher, reduction="none").sum(-1)
    weights = torch.arange(
        per_depth.shape[-1], 0, -1,
        device=per_depth.device, dtype=per_depth.dtype,
    )
    return (per_depth * (weights / weights.mean())).mean()


def main() -> None:
    args = parser().parse_args()
    if (args.output.exists() or args.max_samples < 1
            or args.holdout_samples < 1 or args.epochs < 1
            or args.learning_rate <= 0 or args.rank < 1
            or not 2 <= args.support_size <= 256
            or not 2 <= args.length <= 15 or args.max_context < 1):
        raise ValueError("Invalid or existing rank-calibrator output/controls")
    cfg = load_config(args.config.resolve())
    precision = {
        key: cfg["model"].get(key) for key in (
            "dtype", "target_attention", "draft_attention", "allow_tf32",
        )
    }
    if precision != {
            "dtype": "bfloat16", "target_attention": "sdpa",
            "draft_attention": "sdpa", "allow_tf32": False}:
        raise ValueError(f"Official precision controls changed: {precision}")
    allocation_gate(args.device)
    engine, tokenizer = load_models(cfg["model"], args.device)
    model_gate(engine)
    adapter = DFlashBlockAdapter(
        target=engine.target, draft=engine.draft,
        ranker=HeuristicRanker(), block_size=args.length,
    )
    corpus = texts(args.train_data.resolve())
    order = list(range(len(corpus)))
    random.Random(args.seed).shuffle(order)
    holdout_count = min(args.holdout_samples, max(1, len(order) // 5))
    holdout = order[:holdout_count]
    train = order[holdout_count:holdout_count + args.max_samples]
    head = RankCalibratedHead(
        int(engine.draft.config.hidden_size), args.rank,
        args.support_size, args.length,
    ).to(engine.device).train()
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.learning_rate)

    def example(index: int, seed: int):
        ids = tokenizer(
            corpus[index], return_tensors="pt", add_special_tokens=True,
        ).input_ids[0]
        if ids.numel() < args.length + 3:
            return None
        local = random.Random(seed)
        anchor_index = local.randint(1, ids.numel() - args.length - 1)
        start = max(0, anchor_index - args.max_context)
        prefix = ids[start:anchor_index][None].to(engine.device)
        anchor = ids[anchor_index:anchor_index + 1].to(engine.device)
        labels = ids[
            anchor_index + 1:anchor_index + 1 + args.length
        ].to(engine.device)
        with torch.inference_mode():
            context_output = engine.target(
                input_ids=prefix, output_hidden_states=True, use_cache=False,
                logits_to_keep=1, return_dict=True,
            )
            context = adapter.extract_target_context(
                context_output.hidden_states
            )
            draft_logits, hidden = adapter.draft_first_raw(context, anchor)
            teacher_input = torch.cat(
                (prefix, anchor[None], labels[:-1][None]), dim=1,
            )
            teacher_logits = engine.target(
                input_ids=teacher_input, use_cache=False,
                logits_to_keep=args.length, return_dict=True,
            ).logits[:, -args.length:]
        return hidden[:, :args.length].clone(), draft_logits[
            :, :args.length
        ].clone(), teacher_logits.clone()

    train_loss = baseline_loss = 0.0
    updates = 0
    for epoch in range(args.epochs):
        epoch_order = train.copy()
        random.Random(args.seed + epoch).shuffle(epoch_order)
        for step, index in enumerate(epoch_order, 1):
            batch = example(index, args.seed + epoch * len(corpus) + index)
            if batch is None:
                continue
            hidden, draft_logits, teacher_logits = batch
            optimizer.zero_grad(set_to_none=True)
            scores, candidates = head.scores(hidden, draft_logits)
            with torch.no_grad():
                teacher = torch.softmax(
                    teacher_logits.gather(2, candidates).float(), dim=-1,
                )
                baseline = remaining_kl(torch.log_softmax(
                    draft_logits.gather(2, candidates).float(), dim=-1,
                ), teacher)
            loss = remaining_kl(torch.log_softmax(
                scores.float(), dim=-1,
            ), teacher)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            updates += 1
            train_loss += float(loss.detach())
            baseline_loss += float(baseline)
            if step % 40 == 0:
                print(
                    f"epoch={epoch + 1}/{args.epochs} step={step}/{len(train)} "
                    f"kl={train_loss / updates:.6f} "
                    f"baseline={baseline_loss / updates:.6f}", flush=True,
                )

    head.eval()
    holdout_loss = holdout_baseline = 0.0
    holdout_updates = 0
    with torch.inference_mode():
        for index in holdout:
            batch = example(index, args.seed + 100_000 + index)
            if batch is None:
                continue
            hidden, draft_logits, teacher_logits = batch
            scores, candidates = head.scores(hidden, draft_logits)
            teacher = torch.softmax(
                teacher_logits.gather(2, candidates).float(), dim=-1,
            )
            holdout_loss += float(remaining_kl(
                torch.log_softmax(scores.float(), dim=-1), teacher,
            ))
            holdout_baseline += float(remaining_kl(
                torch.log_softmax(
                    draft_logits.gather(2, candidates).float(), dim=-1,
                ), teacher,
            ))
            holdout_updates += 1
    if updates < 1 or holdout_updates < 1:
        raise RuntimeError("Rank-calibrator received no usable examples")
    metadata = {
        "architecture": "parallel_rank_calibrator_v1",
        "hidden_size": head.hidden_size, "rank": head.rank,
        "support_size": head.support_size, "max_length": head.max_length,
        "updates": updates, "epochs": args.epochs,
        "learning_rate": args.learning_rate, "seed": args.seed,
        "training_examples": len(train), "holdout_examples": holdout_updates,
        "training_sha256": digest(args.train_data.resolve()),
        "target": cfg["model"]["target"], "draft": cfg["model"]["draft"],
        "train_kl": train_loss / updates,
        "baseline_train_kl": baseline_loss / updates,
        "holdout_kl": holdout_loss / holdout_updates,
        "baseline_holdout_kl": holdout_baseline / holdout_updates,
        "holdout_improved": holdout_loss < holdout_baseline,
        "adds_target_or_draft_forward_at_runtime": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": {
            name: value.detach().cpu()
            for name, value in head.state_dict().items()
        },
        "metadata": metadata,
    }, args.output)
    print(json.dumps({"saved": str(args.output), **metadata}, indent=2))


if __name__ == "__main__":
    main()
