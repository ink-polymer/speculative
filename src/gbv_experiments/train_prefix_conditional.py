"""Train a lightweight prefix-conditional proposal head by Target distillation."""
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

from .config import load_config
from .engine import load_models
from .prefix_conditional import PrefixConditionalHead


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_texts(path: Path) -> list[str]:
    texts = []
    with path.open() as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            text = value.get("text")
            if not isinstance(text, str) or not text:
                raise ValueError(f"{path}:{line_number} has no text")
            texts.append(text)
    if not texts:
        raise ValueError("Prefix-head training set is empty")
    return texts


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", required=True)
    result.add_argument("--train-data", required=True)
    result.add_argument("--output", required=True)
    result.add_argument("--max-samples", type=int, default=384)
    result.add_argument("--holdout-samples", type=int, default=64)
    result.add_argument("--epochs", type=int, default=1)
    result.add_argument("--learning-rate", type=float, default=1e-3)
    result.add_argument("--max-context", type=int, default=512)
    result.add_argument("--rank", type=int, default=64)
    result.add_argument("--support-size", type=int, default=32)
    result.add_argument("--length", type=int, default=15)
    result.add_argument(
        "--rollout-probability", type=float, default=0.5,
        help="Fraction of updates distilled on prefixes sampled by the head.",
    )
    result.add_argument(
        "--objective", choices=("mean_kl", "remaining_kl"),
        default="remaining_kl",
        help="remaining_kl weights an error by the suffix it can invalidate.",
    )
    result.add_argument("--seed", type=int, default=20260910)
    return result


def distillation_loss(log_scores: torch.Tensor, teacher: torch.Tensor,
                      objective: str) -> torch.Tensor:
    """KL objective aligned with expected accepted-prefix length."""
    per_depth = F.kl_div(
        log_scores, teacher, reduction="none",
    ).sum(-1)
    if objective == "mean_kl":
        return per_depth.mean()
    if objective != "remaining_kl":
        raise ValueError(f"Unknown prefix objective: {objective}")
    length = per_depth.shape[-1]
    weights = torch.arange(
        length, 0, -1, dtype=per_depth.dtype, device=per_depth.device,
    )
    weights = weights / weights.mean()
    return (per_depth * weights).mean()


def main() -> None:
    args = parser().parse_args()
    if (args.max_samples < 1 or args.holdout_samples < 1
            or args.epochs < 1 or args.learning_rate <= 0
            or args.max_context < 1 or args.rank < 1
            or not 1 <= args.support_size <= 256
            or not 2 <= args.length <= 15
            or not 0 <= args.rollout_probability <= 1):
        raise ValueError("Invalid prefix-head training controls")
    config_path = Path(args.config)
    train_path = Path(args.train_data)
    cfg = load_config(config_path)
    model_cfg = cfg["model"]
    device = torch.device("cuda:0")
    engine, tokenizer = load_models(model_cfg, str(device))
    adapter = DFlashBlockAdapter(
        target=engine.target, draft=engine.draft,
        ranker=HeuristicRanker(), block_size=args.length,
    )
    head = PrefixConditionalHead(
        hidden_size=int(engine.draft.config.hidden_size),
        rank=args.rank, support_size=args.support_size,
        max_length=args.length,
    ).to(device=device, dtype=torch.float32).train()
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.learning_rate)
    texts = load_texts(train_path)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    order = list(range(len(texts)))
    random.shuffle(order)
    holdout_count = min(args.holdout_samples, max(1, len(order) // 5))
    holdout = order[:holdout_count]
    train = order[holdout_count:holdout_count + args.max_samples]
    if not train:
        raise RuntimeError("No disjoint prefix-head training examples")
    candidate_embedding = engine.target.get_output_embeddings().weight
    prefix_embedding = engine.target.get_input_embeddings().weight

    def example(sample_index: int, anchor_seed: int, *, rollout: bool = False):
        token_ids = tokenizer(
            texts[sample_index], return_tensors="pt", add_special_tokens=True,
        ).input_ids[0]
        if token_ids.numel() < args.length + 3:
            return None
        local = random.Random(anchor_seed)
        anchor_index = local.randint(1, token_ids.numel() - args.length - 1)
        start = max(0, anchor_index - args.max_context)
        prefix = token_ids[start:anchor_index].unsqueeze(0).to(device)
        anchor = token_ids[anchor_index:anchor_index + 1].to(device)
        corpus_labels = token_ids[
            anchor_index + 1:anchor_index + 1 + args.length
        ][None].to(device)
        with torch.inference_mode():
            context_output = engine.target(
                input_ids=prefix, output_hidden_states=True, use_cache=False,
                logits_to_keep=1, return_dict=True,
            )
            context = adapter.extract_target_context(context_output.hidden_states)
            draft_logits, draft_hidden = adapter.draft_first_raw(context, anchor)
            labels = corpus_labels
            if rollout:
                noise_ids = torch.full(
                    (args.length + 1,), int(engine.draft.mask_token_id),
                    dtype=torch.long, device=device,
                )
                noise_ids[0] = anchor[0]
                rollout_generator = torch.Generator(device=device).manual_seed(
                    anchor_seed + 1_000_000,
                )
                labels = head.propose(
                    draft_hidden[:, :args.length],
                    draft_logits[0, :args.length], candidate_embedding,
                    noise_ids, args.length, 1., rollout_generator,
                    prefix_embeddings=prefix_embedding,
                ).paths()
            teacher_input = torch.cat((prefix[0], anchor, labels[0, :-1]))[None]
            target_logits = engine.target(
                input_ids=teacher_input, output_hidden_states=False,
                use_cache=False, return_dict=True,
            ).logits[:, -args.length:]
        return (
            draft_hidden[:, :args.length].clone(),
            draft_logits[:, :args.length].clone(),
            target_logits.clone(), labels,
        )

    updates = 0
    loss_total = 0.0
    baseline_kl_total = 0.0
    for epoch in range(args.epochs):
        for step, sample_index in enumerate(train, 1):
            anchor_seed = args.seed + epoch * len(texts) + sample_index
            use_rollout = random.Random(anchor_seed + 2_000_000).random() < (
                args.rollout_probability
            )
            batch = example(
                sample_index, anchor_seed, rollout=use_rollout,
            )
            if batch is None:
                continue
            hidden, draft_logits, target_logits, labels = batch
            optimizer.zero_grad(set_to_none=True)
            scores, candidates = head.teacher_forced_scores(
                hidden, draft_logits, candidate_embedding, labels,
                prefix_embeddings=prefix_embedding,
            )
            with torch.no_grad():
                teacher_selected = target_logits.gather(2, candidates).float()
                teacher = torch.softmax(teacher_selected, dim=-1)
                baseline = torch.log_softmax(
                    draft_logits.gather(2, candidates).float(), dim=-1,
                )
                baseline_kl = distillation_loss(
                    baseline, teacher, args.objective,
                )
            loss = distillation_loss(
                torch.log_softmax(scores.float(), dim=-1), teacher,
                args.objective,
            )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Nonfinite prefix-head loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            updates += 1
            loss_total += float(loss.detach())
            baseline_kl_total += float(baseline_kl)
            if step % 20 == 0:
                print(
                    f"epoch={epoch + 1}/{args.epochs} step={step}/{len(train)} "
                    f"updates={updates} mean_kl={loss_total / updates:.6f} "
                    f"baseline_kl={baseline_kl_total / updates:.6f}",
                    flush=True,
                )
    if updates < 1:
        raise RuntimeError("No prefix-head training updates")

    head.eval()
    holdout_loss = 0.0
    holdout_baseline = 0.0
    holdout_updates = 0
    with torch.inference_mode():
        for sample_index in holdout:
            batch = example(sample_index, args.seed + 10_000 + sample_index)
            if batch is None:
                continue
            hidden, draft_logits, target_logits, labels = batch
            scores, candidates = head.teacher_forced_scores(
                hidden, draft_logits, candidate_embedding, labels,
                prefix_embeddings=prefix_embedding,
            )
            teacher = torch.softmax(
                target_logits.gather(2, candidates).float(), dim=-1,
            )
            holdout_loss += float(distillation_loss(
                torch.log_softmax(scores.float(), dim=-1), teacher,
                args.objective,
            ))
            holdout_baseline += float(distillation_loss(
                torch.log_softmax(
                    draft_logits.gather(2, candidates).float(), dim=-1,
                ), teacher, args.objective,
            ))
            holdout_updates += 1
    if holdout_updates < 1:
        raise RuntimeError("No prefix-head holdout examples")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "architecture": "prefix_conditional_topr_v2_split_embeddings",
        "hidden_size": head.hidden_size,
        "rank": head.rank,
        "support_size": head.support_size,
        "max_length": head.max_length,
        "updates": updates,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "objective": args.objective,
        "rollout_probability": args.rollout_probability,
        "max_context": args.max_context,
        "seed": args.seed,
        "training_examples": len(train),
        "holdout_examples": holdout_updates,
        "training_sha256": file_sha256(train_path),
        "target": model_cfg["target"],
        "target_revision": model_cfg["target_revision"],
        "draft": model_cfg["draft"],
        "draft_revision": model_cfg["draft_revision"],
        "final_train_kl": loss_total / updates,
        "baseline_train_kl": baseline_kl_total / updates,
        "holdout_kl": holdout_loss / holdout_updates,
        "baseline_holdout_kl": holdout_baseline / holdout_updates,
    }
    torch.save({
        "state_dict": {
            name: value.detach().cpu()
            for name, value in head.state_dict().items()
        },
        "metadata": metadata,
    }, output)
    print(json.dumps({"saved": str(output), **metadata}, indent=2), flush=True)


if __name__ == "__main__":
    main()
