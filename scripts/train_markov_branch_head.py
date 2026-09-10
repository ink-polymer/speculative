"""Distill a parent-token-conditioned, one-Draft DDTree proposal head."""
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
from gbv_experiments.slot_mixer import MarkovBranchHead
from gbv_experiments.terminal_formal import allocation_gate, model_gate


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", type=Path, required=True)
    result.add_argument("--train-data", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--max-samples", type=int, default=256)
    result.add_argument("--holdout-samples", type=int, default=64)
    result.add_argument("--epochs", type=int, default=2)
    result.add_argument("--learning-rate", type=float, default=5e-4)
    result.add_argument("--rank", type=int, default=32)
    result.add_argument("--support-size", type=int, default=16)
    result.add_argument("--start-depth", type=int, default=1)
    result.add_argument("--length", type=int, default=15)
    result.add_argument("--max-context", type=int, default=512)
    result.add_argument("--seed", type=int, default=20260910)
    result.add_argument("--device", default="cuda:0")
    return result


def file_hash(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def read_texts(path: Path) -> list[str]:
    result = []
    with path.open() as stream:
        for line in stream:
            if line.strip():
                text = json.loads(line).get("text")
                if not isinstance(text, str) or not text:
                    raise ValueError("Training row has no text")
                result.append(text)
    return result


def selected_kl(scores: torch.Tensor, teacher_logits: torch.Tensor,
                candidates: torch.Tensor) -> torch.Tensor:
    teacher = torch.softmax(
        teacher_logits.gather(2, candidates).float(), dim=-1,
    )
    per_depth = F.kl_div(
        torch.log_softmax(scores.float(), dim=-1), teacher,
        reduction="none",
    ).sum(-1)
    weight = torch.arange(
        per_depth.shape[-1], 0, -1,
        dtype=per_depth.dtype, device=per_depth.device,
    )
    return (per_depth * (weight / weight.mean())).mean()


def main() -> None:
    args = parser().parse_args()
    if (args.output.exists() or args.max_samples < 1
            or args.holdout_samples < 1 or args.epochs < 1
            or args.learning_rate <= 0 or args.rank < 1
            or not 2 <= args.support_size <= 64
            or not 0 <= args.start_depth < args.length
            or not 2 <= args.length <= 15):
        raise ValueError("Invalid or existing Markov-head output/controls")
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
    head = MarkovBranchHead(
        int(engine.draft.config.hidden_size), args.rank,
        args.support_size, args.length, args.start_depth,
    ).to(engine.device, dtype=torch.float32).train()
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.learning_rate)
    corpus = read_texts(args.train_data.resolve())
    order = list(range(len(corpus)))
    random.Random(args.seed).shuffle(order)
    holdout_count = min(args.holdout_samples, max(1, len(order) // 5))
    holdout = order[:holdout_count]
    train = order[holdout_count:holdout_count + args.max_samples]
    token_embeddings = engine.target.get_output_embeddings().weight

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
        ][None].to(engine.device)
        with torch.inference_mode():
            context_output = engine.target(
                input_ids=prefix, output_hidden_states=True, use_cache=False,
                logits_to_keep=1, return_dict=True,
            )
            context = adapter.extract_target_context(
                context_output.hidden_states
            )
            draft_logits, draft_hidden = adapter.draft_first_raw(
                context, anchor
            )
            teacher_input = torch.cat(
                (prefix, anchor[None], labels[:, :-1]), dim=1,
            )
            teacher_logits = engine.target(
                input_ids=teacher_input, use_cache=False,
                logits_to_keep=args.length, return_dict=True,
            ).logits[:, -args.length:]
        return (
            draft_hidden[:, :args.length].clone(),
            draft_logits[:, :args.length].clone(),
            teacher_logits.clone(), labels,
        )

    total = baseline_total = 0.0
    updates = 0
    for epoch in range(args.epochs):
        shuffled = train.copy()
        random.Random(args.seed + epoch).shuffle(shuffled)
        for step, index in enumerate(shuffled, 1):
            batch = example(index, args.seed + epoch * len(corpus) + index)
            if batch is None:
                continue
            hidden, draft_logits, teacher_logits, labels = batch
            optimizer.zero_grad(set_to_none=True)
            scores, candidates = head.teacher_forced_scores(
                hidden, draft_logits, token_embeddings, labels,
            )
            loss = selected_kl(scores, teacher_logits, candidates)
            with torch.no_grad():
                baseline = selected_kl(
                    draft_logits.gather(2, candidates),
                    teacher_logits, candidates,
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            updates += 1
            total += float(loss.detach())
            baseline_total += float(baseline)
            if step % 40 == 0:
                print(
                    f"epoch={epoch + 1}/{args.epochs} step={step}/{len(train)} "
                    f"kl={total / updates:.6f} "
                    f"baseline={baseline_total / updates:.6f}", flush=True,
                )

    head.eval()
    held = held_baseline = 0.0
    held_count = 0
    with torch.inference_mode():
        for index in holdout:
            batch = example(index, args.seed + 100_000 + index)
            if batch is None:
                continue
            hidden, draft_logits, teacher_logits, labels = batch
            scores, candidates = head.teacher_forced_scores(
                hidden, draft_logits, token_embeddings, labels,
            )
            held += float(selected_kl(scores, teacher_logits, candidates))
            held_baseline += float(selected_kl(
                draft_logits.gather(2, candidates), teacher_logits, candidates,
            ))
            held_count += 1
    if updates < 1 or held_count < 1:
        raise RuntimeError("Markov branch head received no usable examples")
    metadata = {
        "architecture": "markov_branch_topr_v1",
        "hidden_size": head.hidden_size, "rank": head.rank,
        "support_size": head.support_size, "max_length": head.max_length,
        "start_depth": head.start_depth,
        "updates": updates, "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "max_context": args.max_context, "seed": args.seed,
        "training_examples": len(train), "holdout_examples": held_count,
        "training_sha256": file_hash(args.train_data.resolve()),
        "target": cfg["model"]["target"],
        "target_revision": cfg["model"]["target_revision"],
        "draft": cfg["model"]["draft"],
        "draft_revision": cfg["model"]["draft_revision"],
        "train_kl": total / updates,
        "baseline_train_kl": baseline_total / updates,
        "holdout_kl": held / held_count,
        "baseline_holdout_kl": held_baseline / held_count,
        "holdout_improved": held < held_baseline,
        "adds_target_or_draft_forward_at_runtime": False,
        "branch_conditioned": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": {
            name: value.detach().cpu()
            for name, value in head.state_dict().items()
        }, "metadata": metadata,
    }, args.output)
    print(json.dumps({"saved": str(args.output), **metadata}, indent=2))


if __name__ == "__main__":
    main()
