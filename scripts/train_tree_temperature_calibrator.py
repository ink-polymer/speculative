"""Fit a zero-extra-forward depth calibrator for DDTree proposal allocation.

The learned parameters affect only the proposal-tree topology.  Target
verification remains exact at the configured sampling temperature.  Training
uses either next-token labels or Target-distribution distillation on a disjoint
text corpus and never reads benchmark answers or timing results.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random

import torch
import torch.nn.functional as F

from dflash_specblock.dflash_adapter import DFlashBlockAdapter
from dflash_specblock.rank_head import HeuristicRanker

from gbv_experiments.common import write_json
from gbv_experiments.config import load_config
from gbv_experiments.engine import load_models
from gbv_experiments.terminal_formal import allocation_gate, model_gate, telemetry


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def load_texts(path: Path) -> list[str]:
    rows = []
    with path.open() as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            text = value.get("text")
            if not isinstance(text, str) or not text:
                raise ValueError(f"{path}:{line_number} has no text")
            rows.append(text)
    if not rows:
        raise ValueError("Tree-calibration corpus is empty")
    return rows


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", type=Path, required=True)
    result.add_argument("--train-data", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--max-samples", type=int, default=128)
    result.add_argument("--holdout-samples", type=int, default=32)
    result.add_argument("--epochs", type=int, default=2)
    result.add_argument("--learning-rate", type=float, default=0.03)
    result.add_argument("--max-context", type=int, default=512)
    result.add_argument("--length", type=int, default=15)
    result.add_argument("--seed", type=int, default=20260910)
    result.add_argument("--device", default="cuda:0")
    result.add_argument(
        "--objective", choices=("token_nll", "target_kl"),
        default="target_kl",
    )
    result.add_argument("--vocab-bias", action="store_true")
    result.add_argument("--bias-l2", type=float, default=1e-4)
    return result


def draft_example(engine, adapter, tokenizer, text: str, length: int,
                  max_context: int, anchor_seed: int,
                  objective: str):
    token_ids = tokenizer(
        text, return_tensors="pt", add_special_tokens=True,
    ).input_ids[0]
    if token_ids.numel() < length + 3:
        return None
    local = random.Random(anchor_seed)
    anchor_index = local.randint(1, token_ids.numel() - length - 1)
    start = max(0, anchor_index - max_context)
    prefix = token_ids[start:anchor_index].unsqueeze(0).to(engine.device)
    anchor = token_ids[anchor_index:anchor_index + 1].to(engine.device)
    labels = token_ids[
        anchor_index + 1:anchor_index + 1 + length
    ].to(engine.device)
    with torch.inference_mode():
        context_output = engine.target(
            input_ids=prefix, output_hidden_states=True, use_cache=False,
            logits_to_keep=1, return_dict=True,
        )
        context = adapter.extract_target_context(context_output.hidden_states)
        draft_logits, _ = adapter.draft_first_raw(context, anchor)
        inference_logits = draft_logits[0, :length].float()
        teacher_probabilities = None
        if objective == "target_kl":
            teacher_input = torch.cat(
                (prefix, anchor[None], labels[:-1][None]), dim=1,
            )
            teacher_output = engine.target(
                input_ids=teacher_input, use_cache=False,
                logits_to_keep=length, return_dict=True,
            )
            teacher_probabilities = torch.softmax(
                teacher_output.logits[0, -length:].float(), dim=-1,
            )
    # A clone made after leaving inference mode is a normal leaf tensor.  That
    # lets autograd differentiate the tiny temperature vector without ever
    # retaining a model graph.
    return (
        inference_logits.clone(), labels,
        None if teacher_probabilities is None
        else teacher_probabilities.clone(),
    )


def nll(logits: torch.Tensor, labels: torch.Tensor,
        temperatures: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(
        logits / temperatures[:, None], labels, reduction="none",
    )


def objective_loss(logits: torch.Tensor, labels: torch.Tensor,
                   teacher_probabilities: torch.Tensor | None,
                   temperatures: torch.Tensor,
                   objective: str) -> torch.Tensor:
    if objective == "token_nll":
        return nll(logits, labels, temperatures)
    if teacher_probabilities is None:
        raise RuntimeError("Target-KL objective requires teacher probabilities")
    return -(
        teacher_probabilities
        * F.log_softmax(logits / temperatures[:, None], dim=-1)
    ).sum(-1)


def main() -> None:
    args = parser().parse_args()
    if (args.max_samples < 1 or args.holdout_samples < 1
            or args.epochs < 1 or args.learning_rate <= 0
            or args.max_context < 1 or not 2 <= args.length <= 15
            or args.bias_l2 < 0):
        raise ValueError("Invalid tree-calibrator controls")
    if args.output.exists():
        raise ValueError("Output exists; choose a fresh calibration file")
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
    texts = load_texts(args.train_data.resolve())
    order = list(range(len(texts)))
    random.Random(args.seed).shuffle(order)
    holdout_count = min(args.holdout_samples, max(1, len(order) // 5))
    holdout = order[:holdout_count]
    train = order[holdout_count:holdout_count + args.max_samples]
    if not train:
        raise RuntimeError("No disjoint tree-calibrator training examples")

    log_temperatures = torch.nn.Parameter(torch.zeros(
        args.length, device=engine.device, dtype=torch.float32,
    ))
    vocab_bias = (
        torch.nn.Parameter(torch.zeros(
            engine.target.config.vocab_size, device=engine.device,
            dtype=torch.float32,
        )) if args.vocab_bias else None
    )
    parameters = [log_temperatures]
    if vocab_bias is not None:
        parameters.append(vocab_bias)
    optimizer = torch.optim.Adam(parameters, lr=args.learning_rate)
    minimum, maximum = math.log(0.2), math.log(3.0)
    updates = 0
    train_loss = 0.0
    baseline_loss = 0.0
    for epoch in range(args.epochs):
        epoch_order = train.copy()
        random.Random(args.seed + epoch).shuffle(epoch_order)
        for step, sample_index in enumerate(epoch_order, 1):
            example = draft_example(
                engine, adapter, tokenizer, texts[sample_index], args.length,
                args.max_context,
                args.seed + epoch * len(texts) + sample_index,
                args.objective,
            )
            if example is None:
                continue
            logits, labels, teacher_probabilities = example
            optimizer.zero_grad(set_to_none=True)
            temperatures = log_temperatures.exp()
            calibrated_logits = (
                logits if vocab_bias is None else logits + vocab_bias[None]
            )
            losses = objective_loss(
                calibrated_logits, labels, teacher_probabilities, temperatures,
                args.objective,
            )
            loss = losses.mean()
            if vocab_bias is not None:
                loss = loss + args.bias_l2 * vocab_bias.square().mean()
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Nonfinite calibration loss")
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                log_temperatures.clamp_(minimum, maximum)
                if vocab_bias is not None:
                    vocab_bias.clamp_(-4.0, 4.0)
            updates += 1
            train_loss += float(loss.detach())
            baseline_loss += float(objective_loss(
                logits, labels, teacher_probabilities,
                torch.ones_like(temperatures), args.objective,
            ).mean())
            if step % 20 == 0:
                print(
                    f"epoch={epoch + 1}/{args.epochs} "
                    f"step={step}/{len(epoch_order)} updates={updates} "
                    f"objective={train_loss / updates:.6f} "
                    f"baseline={baseline_loss / updates:.6f}",
                    flush=True,
                )
    if updates < 1:
        raise RuntimeError("No tree-calibrator updates")

    calibrated_holdout = torch.zeros(args.length, device=engine.device)
    baseline_holdout = torch.zeros(args.length, device=engine.device)
    holdout_updates = 0
    temperatures = log_temperatures.detach().exp()
    for sample_index in holdout:
        example = draft_example(
            engine, adapter, tokenizer, texts[sample_index], args.length,
            args.max_context, args.seed + 10_000 + sample_index,
            args.objective,
        )
        if example is None:
            continue
        logits, labels, teacher_probabilities = example
        with torch.no_grad():
            calibrated_logits = (
                logits if vocab_bias is None else logits + vocab_bias[None]
            )
            calibrated_holdout += objective_loss(
                calibrated_logits, labels, teacher_probabilities, temperatures,
                args.objective,
            )
            baseline_holdout += objective_loss(
                logits, labels, teacher_probabilities,
                torch.ones_like(temperatures), args.objective,
            )
        holdout_updates += 1
    if holdout_updates < 1:
        raise RuntimeError("No tree-calibrator holdout examples")
    calibrated_holdout /= holdout_updates
    baseline_holdout /= holdout_updates
    learned = [float(value) for value in temperatures.cpu()]
    report = {
        "kind": "depthwise_tree_proposal_temperature_calibrator",
        "formal_complete": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "official_precision": precision,
        "target_sampling_temperature": 1.0,
        "changes_target_distribution": False,
        "adds_model_forward_at_runtime": False,
        "objective": args.objective,
        "length": args.length,
        "temperatures": learned,
        "vocab_bias": (
            None if vocab_bias is None
            else [float(value) for value in vocab_bias.detach().cpu()]
        ),
        "training": {
            "corpus": str(args.train_data.resolve()),
            "corpus_sha256": sha256(args.train_data.resolve()),
            "examples": len(train), "epochs": args.epochs,
            "updates": updates, "learning_rate": args.learning_rate,
            "seed": args.seed,
            "vocab_bias_enabled": args.vocab_bias,
            "bias_l2": args.bias_l2,
            "mean_online_objective": train_loss / updates,
            "mean_online_baseline_objective": baseline_loss / updates,
        },
        "holdout": {
            "examples": holdout_updates,
            "per_depth_objective": [
                float(value) for value in calibrated_holdout.cpu()
            ],
            "per_depth_baseline_objective": [
                float(value) for value in baseline_holdout.cpu()
            ],
            "mean_objective": float(calibrated_holdout.mean()),
            "mean_baseline_objective": float(baseline_holdout.mean()),
            "improved": bool(
                calibrated_holdout.mean() < baseline_holdout.mean()
            ),
        },
        "telemetry_end": telemetry(args.device),
    }
    write_json(args.output.resolve(), report)
    printable = {key: value for key, value in report.items()
                 if key != "vocab_bias"}
    printable["vocab_bias_parameters"] = (
        0 if vocab_bias is None else vocab_bias.numel()
    )
    print(json.dumps(printable, indent=2), flush=True)


if __name__ == "__main__":
    main()
