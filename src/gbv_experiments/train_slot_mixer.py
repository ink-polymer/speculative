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
from .slot_mixer import CausalSlotMixer, ResidualSlotMLP


def parser():
    result = argparse.ArgumentParser()
    result.add_argument("--config", required=True)
    result.add_argument("--train-data", required=True)
    result.add_argument("--output", required=True)
    result.add_argument("--epochs", type=int, default=3)
    result.add_argument("--learning-rate", type=float, default=2e-5)
    result.add_argument("--max-context", type=int, default=512)
    result.add_argument("--max-samples", type=int, default=0)
    result.add_argument("--seed", type=int, default=2026)
    result.add_argument("--validation-fraction", type=float, default=.1)
    result.add_argument("--patience", type=int, default=2)
    result.add_argument("--adapter-kind", choices=("causal-mixer", "slot-mlp"),
                        default="causal-mixer")
    result.add_argument("--bottleneck", type=int, default=512)
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
    if args.epochs < 1 or args.learning_rate <= 0 or args.max_context < 1:
        raise ValueError("Invalid slot-mixer training controls")
    if not 0 < args.validation_fraction < .5 or args.patience < 1:
        raise ValueError("Invalid validation fraction or patience")
    cfg = load_config(Path(args.config))
    model_cfg = cfg["model"]
    device = torch.device("cuda:0")
    engine, tokenizer = load_models(model_cfg, str(device))
    engine.target.requires_grad_(False)
    engine.draft.requires_grad_(False)
    adapter = DFlashBlockAdapter(
        target=engine.target,
        draft=engine.draft,
        ranker=HeuristicRanker(),
        block_size=15,
    )
    hidden_size = int(engine.draft.config.hidden_size)
    if args.adapter_kind == "slot-mlp":
        mixer = ResidualSlotMLP(
            hidden_size=hidden_size, bottleneck=args.bottleneck,
        ).to(device).train()
    else:
        mixer = CausalSlotMixer(
            hidden_size=hidden_size, rank=64, kernel_size=3,
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
    validation_count = max(1, round(len(order) * args.validation_fraction))
    validation_order = order[:validation_count]
    training_order = order[validation_count:]
    if not training_order:
        raise ValueError("Validation split leaves no training rows")

    encoded = [
        tokenizer(text, return_tensors="pt", add_special_tokens=True).input_ids[0]
        for text in texts
    ]
    validation_anchors = {
        index: random.Random(args.seed + index * 1_000_003).randint(
            1, encoded[index].numel() - 16)
        for index in validation_order if encoded[index].numel() >= 18
    }
    if not validation_anchors:
        raise ValueError("Validation split has no sufficiently long examples")

    def example_loss(sample_index: int, anchor_index: int):
        token_ids = encoded[sample_index]
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
                target_output.hidden_states)
            _, draft_hidden = adapter.draft_first_raw(target_context, anchor)
        corrected = mixer(draft_hidden.clone())
        logits = engine.target.get_output_embeddings()(corrected)[0].float()
        return F.cross_entropy(logits, labels)

    def measure_validation_loss():
        mixer.eval()
        total = 0.
        with torch.inference_mode():
            for sample_index, anchor_index in validation_anchors.items():
                total += float(example_loss(sample_index, anchor_index))
        mixer.train()
        return total / len(validation_anchors)

    updates = 0
    initial_validation_loss = measure_validation_loss()
    print(f"initial_validation={initial_validation_loss:.6f}", flush=True)
    best_validation_loss = float("inf")
    best_epoch = 0
    best_state = None
    stale_epochs = 0
    epoch_records = []
    for epoch in range(args.epochs):
        epoch_order = training_order.copy()
        random.Random(args.seed + epoch).shuffle(epoch_order)
        running_loss = 0.0
        epoch_updates = 0
        for step, sample_index in enumerate(epoch_order, 1):
            token_ids = encoded[sample_index]
            if token_ids.numel() < 18:
                continue
            anchor_index = random.randint(1, token_ids.numel() - 16)
            optimizer.zero_grad(set_to_none=True)
            loss = example_loss(sample_index, anchor_index)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Nonfinite slot mixer loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mixer.parameters(), 1.0)
            optimizer.step()
            updates += 1
            epoch_updates += 1
            running_loss += float(loss.detach())
            if step % 50 == 0:
                print(
                    f"epoch={epoch + 1}/{args.epochs} "
                    f"step={step}/{len(epoch_order)} updates={updates} "
                    f"mean_loss={running_loss / epoch_updates:.6f}",
                    flush=True,
                )

        training_loss = running_loss / max(epoch_updates, 1)
        validation_loss = measure_validation_loss()
        epoch_records.append({
            "epoch": epoch + 1,
            "training_loss": training_loss,
            "validation_loss": validation_loss,
            "updates": epoch_updates,
        })
        print(
            f"epoch={epoch + 1} train={training_loss:.6f} "
            f"validation={validation_loss:.6f}", flush=True)
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch + 1
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in mixer.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"early_stop_epoch={epoch + 1}", flush=True)
                break

    if updates < 1 or best_state is None:
        raise RuntimeError("No slot mixer updates were completed")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": best_state,
        "metadata": {
            "architecture": (
                "residual_slot_mlp_v1" if args.adapter_kind == "slot-mlp"
                else "parallel_causal_slot_mixer_v1"),
            "hidden_size": mixer.hidden_size,
            **({"bottleneck": mixer.bottleneck}
               if args.adapter_kind == "slot-mlp" else {
                   "rank": mixer.rank,
                   "kernel_size": mixer.kernel_size,
               }),
            "updates": updates,
            "epochs_requested": args.epochs,
            "epochs_completed": len(epoch_records),
            "best_epoch": best_epoch,
            "learning_rate": args.learning_rate,
            "max_context": args.max_context,
            "seed": args.seed,
            "training_examples": len(texts),
            "training_split_examples": len(training_order),
            "validation_examples": len(validation_anchors),
            "validation_fraction": args.validation_fraction,
            "patience": args.patience,
            "initial_validation_loss": initial_validation_loss,
            "training_sha256": file_sha256(train_path),
            "target": model_cfg["target"],
            "target_revision": model_cfg["target_revision"],
            "draft": model_cfg["draft"],
            "draft_revision": model_cfg["draft_revision"],
            "best_validation_loss": best_validation_loss,
            "epoch_records": epoch_records,
        },
    }, output)
    print(
        f"saved={output} updates={updates} best_epoch={best_epoch} "
        f"best_validation_loss={best_validation_loss:.6f}")


if __name__ == "__main__":
    main()
