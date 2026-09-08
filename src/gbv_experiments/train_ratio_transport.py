"""Train the candidate-restricted target/draft log-density ratio head."""
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
from .slot_mixer import CandidateRatioTransportHead


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cfg = load_config(Path(args.config))
    model_cfg = cfg["model"]
    device = torch.device("cuda:0")
    engine, tokenizer = load_models(model_cfg, str(device))
    adapter = DFlashBlockAdapter(
        target=engine.target, draft=engine.draft,
        ranker=HeuristicRanker(), block_size=15,
    )
    head = CandidateRatioTransportHead(
        int(engine.draft.config.hidden_size), rank=32, candidates=45,
    ).to(device=device, dtype=next(engine.draft.parameters()).dtype).train()
    optimizer = torch.optim.AdamW(head.parameters(), lr=2e-4)
    train_path = Path(args.train_data)
    texts = []
    with train_path.open() as stream:
        for line in stream:
            if line.strip():
                texts.append(json.loads(line)["text"])
    random.seed(2026)
    torch.manual_seed(2026)
    torch.cuda.manual_seed_all(2026)
    order = list(range(len(texts)))
    random.shuffle(order)
    updates, loss_total = 0, 0.0
    for step, sample_index in enumerate(order, 1):
        ids = tokenizer(
            texts[sample_index], return_tensors="pt", add_special_tokens=True
        ).input_ids[0]
        if ids.numel() < 18:
            continue
        anchor_index = random.randint(1, ids.numel() - 16)
        start = max(0, anchor_index - 512)
        prefix = ids[start:anchor_index].unsqueeze(0).to(device)
        anchor = ids[anchor_index:anchor_index + 1].to(device)
        future = ids[anchor_index + 1:anchor_index + 16].to(device)
        with torch.inference_mode():
            context_output = engine.target(
                input_ids=prefix, output_hidden_states=True, use_cache=False,
                logits_to_keep=1, return_dict=True,
            )
            context = adapter.extract_target_context(context_output.hidden_states)
            draft_logits, draft_hidden = adapter.draft_first_raw(context, anchor)
            teacher_input = torch.cat((prefix[0], anchor, future[:-1]))[None]
            teacher_logits = engine.target(
                input_ids=teacher_input, output_hidden_states=False,
                use_cache=False, return_dict=True,
            ).logits[:, -15:]
            candidate_ids = draft_logits.topk(45, dim=-1, sorted=True).indices
            draft_selected = draft_logits.gather(2, candidate_ids).float()
            target_selected = teacher_logits.gather(2, candidate_ids).float()
            desired = target_selected - draft_selected
            desired = desired - desired.mean(-1, keepdim=True)
        # Tensors created inside inference_mode cannot be saved by autograd.
        # The frozen-model values are copied without changing their numerics.
        draft_hidden = draft_hidden.clone()
        candidate_ids = candidate_ids.clone()
        desired = desired.clone()
        optimizer.zero_grad(set_to_none=True)
        predicted = head.candidate_corrections(
            draft_hidden, engine.target.get_output_embeddings().weight,
            candidate_ids,
        ).float()
        predicted = predicted - predicted.mean(-1, keepdim=True)
        loss = F.smooth_l1_loss(predicted, desired)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Nonfinite ratio-transport loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        optimizer.step()
        updates += 1
        loss_total += float(loss.detach())
        if step % 20 == 0:
            print(f"step={step}/{len(order)} mean_loss={loss_total / updates:.6f}", flush=True)
    if not updates:
        raise RuntimeError("No ratio-head training updates")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": {k: v.detach().cpu() for k, v in head.state_dict().items()},
        "metadata": {
            "architecture": "candidate_ratio_transport_v1",
            "hidden_size": head.hidden_size,
            "rank": head.rank,
            "candidates": head.candidates,
            "updates": updates,
            "epochs": 1,
            "learning_rate": 2e-4,
            "max_context": 512,
            "seed": 2026,
            "training_examples": len(texts),
            "training_sha256": file_sha256(train_path),
            "target_revision": model_cfg["target_revision"],
            "draft_revision": model_cfg["draft_revision"],
            "final_mean_loss": loss_total / updates,
        },
    }, output)
    print(f"saved={output} updates={updates} mean_loss={loss_total / updates:.6f}")


if __name__ == "__main__":
    main()
