"""Train the frozen-model parallel causal DFlash slot mixer."""
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
from .slot_mixer import CausalSlotMixer


def parser():
    result = argparse.ArgumentParser()
    result.add_argument("--config", required=True)
    result.add_argument("--train-data", required=True)
    result.add_argument("--output", required=True)
    result.add_argument("--epochs", type=int, default=1)
    result.add_argument("--learning-rate", type=float, default=2e-4)
    result.add_argument("--max-context", type=int, default=512)
    result.add_argument("--max-samples", type=int, default=0)
    result.add_argument("--seed", type=int, default=2026)
    return result


def load_texts(path: Path):
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
        raise ValueError("Training set is empty")
    return texts


def file_sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    args = parser().parse_args()
    if args.epochs != 1:
        raise ValueError("Architecture experiment freezes training at one epoch")
    if args.learning_rate != 2e-4:
        raise ValueError("Architecture experiment freezes learning rate at 2e-4")
    if args.max_context != 512 or args.seed != 2026:
        raise ValueError("Architecture experiment training controls are frozen")
    cfg = load_config(Path(args.config))
    model_cfg = cfg["model"]
    device = torch.device("cuda:0")
    engine, tokenizer = load_models(model_cfg, str(device))
    adapter = DFlashBlockAdapter(
        target=engine.target,
        draft=engine.draft,
        ranker=HeuristicRanker(),
        block_size=15,
    )
    mixer = CausalSlotMixer(
        hidden_size=int(engine.draft.config.hidden_size),
        rank=64,
        kernel_size=3,
    ).to(device).train()
    optimizer = torch.optim.AdamW(mixer.parameters(), lr=args.learning_rate)
    train_path = Path(args.train_data)
    texts = load_texts(train_path)
    if args.max_samples:
        texts = texts[:args.max_samples]

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    order = list(range(len(texts)))
    random.shuffle(order)
    updates = 0
    running_loss = 0.0
    for epoch in range(args.epochs):
        for step, sample_index in enumerate(order, 1):
            token_ids = tokenizer(
                texts[sample_index], return_tensors="pt", add_special_tokens=True
            ).input_ids[0]
            if token_ids.numel() < 18:
                continue
            maximum_anchor = token_ids.numel() - 16
            anchor_index = random.randint(1, maximum_anchor)
            prefix_start = max(0, anchor_index - args.max_context)
            prefix = token_ids[prefix_start:anchor_index].unsqueeze(0).to(device)
            anchor = token_ids[anchor_index:anchor_index + 1].to(device)
            labels = token_ids[anchor_index + 1:anchor_index + 16].to(device)

            with torch.inference_mode():
                target_output = engine.target(
                    input_ids=prefix,
                    output_hidden_states=True,
                    use_cache=False,
                    logits_to_keep=1,
                    return_dict=True,
                )
                target_context = adapter.extract_target_context(
                    target_output.hidden_states
                )
                _, draft_hidden = adapter.draft_first_raw(target_context, anchor)
            draft_hidden = draft_hidden.clone()
            del target_output, target_context

            optimizer.zero_grad(set_to_none=True)
            corrected = mixer(draft_hidden)
            logits = engine.target.get_output_embeddings()(corrected)[0].float()
            loss = F.cross_entropy(logits, labels)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Nonfinite slot mixer loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mixer.parameters(), 1.0)
            optimizer.step()
            updates += 1
            running_loss += float(loss.detach())
            if step % 20 == 0:
                print(
                    f"step={step}/{len(order)} updates={updates} "
                    f"mean_loss={running_loss / updates:.6f}",
                    flush=True,
                )

    if updates < 1:
        raise RuntimeError("No slot mixer updates were completed")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": {
            name: value.detach().cpu() for name, value in mixer.state_dict().items()
        },
        "metadata": {
            "architecture": "parallel_causal_slot_mixer_v1",
            "hidden_size": mixer.hidden_size,
            "rank": mixer.rank,
            "kernel_size": mixer.kernel_size,
            "updates": updates,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "max_context": args.max_context,
            "seed": args.seed,
            "training_examples": len(texts),
            "training_sha256": file_sha256(train_path),
            "target": model_cfg["target"],
            "target_revision": model_cfg["target_revision"],
            "draft": model_cfg["draft"],
            "draft_revision": model_cfg["draft_revision"],
            "final_mean_loss": running_loss / updates,
        },
    }, output)
    print(f"saved={output} updates={updates} mean_loss={running_loss / updates:.6f}")


if __name__ == "__main__":
    main()
