"""Fine-tune a prefix head on the actual finite-tree node-selection task."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import torch
import torch.nn.functional as F

from dflash_specblock.dflash_adapter import DFlashBlockAdapter
from dflash_specblock.rank_head import HeuristicRanker
from gbv_experiments.config import load_config
from gbv_experiments.engine import load_models
from gbv_experiments.prefix_conditional import (
    PrefixConditionalHead, load_prefix_conditional_head,
)
from gbv_experiments.sampling import probabilities
from gbv_experiments.train_prefix_conditional import file_sha256, load_texts


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", type=Path, required=True)
    result.add_argument("--train-data", type=Path, required=True)
    result.add_argument("--initial-checkpoint", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--max-samples", type=int, default=64)
    result.add_argument("--holdout-samples", type=int, default=16)
    result.add_argument("--epochs", type=int, default=2)
    result.add_argument("--learning-rate", type=float, default=2e-4)
    result.add_argument("--budget", type=int, default=45)
    result.add_argument("--pool-budget", type=int, default=90)
    result.add_argument("--support-size", type=int, default=32)
    result.add_argument("--max-context", type=int, default=384)
    result.add_argument("--seed", type=int, default=20260910)
    return result


def node_paths(tree, candidate_ids: torch.Tensor) -> torch.Tensor:
    """One teacher-forced candidate path for every non-root pool node."""
    length = candidate_ids.shape[0]
    paths = candidate_ids[:, 0][None].expand(len(tree.tokens), -1).clone()
    prefixes: list[list[int]] = [[]]
    for node, (parent, token) in enumerate(
            zip(tree.parents[1:], tree.tokens), 1):
        prefix = prefixes[parent] + [token]
        prefixes.append(prefix)
        paths[node - 1, :len(prefix)] = paths.new_tensor(prefix)
    return paths


def predicted_node_log_mass(head: PrefixConditionalHead,
                            hidden: torch.Tensor,
                            draft_logits: torch.Tensor,
                            output_embeddings: torch.Tensor,
                            input_embeddings: torch.Tensor,
                            q: torch.Tensor, pool_budget: int):
    width = head.support_size
    q_values, candidate_ids = q.topk(width, dim=-1, sorted=True)
    tree = head._sparse_probability_tree(
        candidate_ids, q_values, pool_budget,
    )
    paths = node_paths(tree, candidate_ids)
    batch = paths.shape[0]
    # Candidate ids are identical across the 512 prefix rows.  Calling the
    # public teacher-forced helper on an expanded full-vocabulary tensor would
    # redundantly execute the same 151K-way top-k once per row.  Reuse the one
    # support computed above and batch only the recurrent branch states.
    dtype = head.hidden_down.weight.dtype
    expanded_hidden = hidden.expand(batch, -1, -1).to(dtype)
    base = head.hidden_down(head.hidden_norm(expanded_hidden))
    previous = torch.zeros_like(expanded_hidden)
    previous[:, 1:] = F.embedding(
        paths[:, :-1], input_embeddings,
    ).to(dtype)
    recurrent_input = base + head.prefix_down(head.token_norm(previous))
    states, _ = head.recurrent(recurrent_input)
    candidate_embeddings = F.embedding(
        candidate_ids, output_embeddings,
    ).to(dtype)
    keys = head.key(head.token_norm(candidate_embeddings))
    queries = head.query(states)[:, :, None]
    correction = (queries * keys[None]).sum(-1) / head.rank ** .5
    correction = correction + head.depth_bias[
        :hidden.shape[1], :width
    ][None]
    draft_values = draft_logits[0].gather(1, candidate_ids).to(dtype)
    scores = (
        draft_values[None]
        + head.correction_scale * correction
    )
    base_log = torch.log_softmax(draft_logits.float(), dim=-1)
    selected_base = base_log.gather(2, candidate_ids[None])
    correction = scores.float() - draft_values.float()[None]
    adjusted = selected_base.expand(batch, -1, -1) + correction
    tail = (1. - selected_base.exp().sum(-1)).clamp_min(1e-12).log()
    normalizer = torch.logaddexp(
        torch.logsumexp(adjusted, -1), tail.expand(batch, -1),
    )
    log_probability = adjusted - normalizer[..., None]

    rank_lookup = [
        {token: rank for rank, token in enumerate(row)}
        for row in candidate_ids.tolist()
    ]
    edge = []
    for node, token in enumerate(tree.tokens, 1):
        depth = tree.depths[node] - 1
        edge.append(log_probability[
            node - 1, depth, rank_lookup[depth][token]
        ])
    cumulative = [hidden.new_zeros((), dtype=torch.float32)]
    for node, value in enumerate(edge, 1):
        cumulative.append(cumulative[tree.parents[node]] + value)
    return tree, torch.stack(cumulative), log_probability, candidate_ids


def target_node_log_mass(tree, p: torch.Tensor) -> torch.Tensor:
    cumulative = [p.new_zeros((), dtype=torch.float32)]
    for node, (parent, token) in enumerate(
            zip(tree.parents[1:], tree.tokens), 1):
        cumulative.append(
            cumulative[parent] + p[parent, token].float().clamp_min(1e-12).log()
        )
    return torch.stack(cumulative)


def selection_loss(predicted: torch.Tensor, target: torch.Tensor,
                   budget: int) -> torch.Tensor:
    """Mass-weighted fit plus model-hard top-B membership ranking."""
    predicted = predicted[1:]
    target = target[1:].detach()
    teacher = torch.softmax(target, 0)
    listwise = -(teacher * torch.log_softmax(predicted, 0)).sum()
    order = target.argsort(descending=True)
    positive = order[:budget]
    negative = order[budget:]
    if negative.numel() == 0:
        return listwise
    # Optimize the mistakes the current selector would actually make.
    positive_scores = predicted.index_select(0, positive)
    negative_scores = predicted.index_select(0, negative)
    positive = positive.index_select(
        0, positive_scores.argsort()[:min(32, positive.numel())],
    )
    negative = negative.index_select(
        0, negative_scores.argsort(descending=True)[:min(32, negative.numel())],
    )
    boundary = F.softplus(-(
        predicted.index_select(0, positive)[:, None]
        - predicted.index_select(0, negative)[None]
    )).mean()
    return listwise + 4. * boundary


def branch_coverage_loss(tree, student_log_probability: torch.Tensor,
                         candidate_ids: torch.Tensor, p: torch.Tensor,
                         target_log_mass: torch.Tensor) -> torch.Tensor:
    """Exact Target KL on every distinct branch prefix in the large pool."""
    first_child: dict[int, int] = {}
    for child, parent in enumerate(tree.parents[1:], 1):
        first_child.setdefault(parent, child)
    losses, weights = [], []
    for parent, child in first_child.items():
        depth = tree.depths[parent]
        if depth >= candidate_ids.shape[0]:
            continue
        selected = p[parent].gather(0, candidate_ids[depth])
        tail = (1. - selected.sum()).clamp_min(0.)
        teacher = torch.cat((selected, tail[None])).float()
        student = torch.cat((
            student_log_probability[child - 1, depth],
            torch.log1p(-student_log_probability[
                child - 1, depth
            ].exp().sum().clamp(max=1. - 1e-7))[None],
        ))
        losses.append(F.kl_div(student, teacher, reduction="sum"))
        weights.append(target_log_mass[parent].detach().exp())
    if not losses:
        raise RuntimeError("Large candidate pool exposed no branch prefixes")
    loss_tensor = torch.stack(losses)
    weight_tensor = torch.stack(weights).to(loss_tensor.dtype)
    weight_tensor = weight_tensor / weight_tensor.mean().clamp_min(1e-12)
    return (loss_tensor * weight_tensor).mean()


def main() -> None:
    args = parser().parse_args()
    if (args.max_samples < 1 or args.holdout_samples < 1 or args.epochs < 1
            or args.learning_rate <= 0 or args.max_context < 1
            or not 1 <= args.budget < args.pool_budget
            or not 1 <= args.support_size <= 256):
        raise ValueError("Invalid tree-selector training controls")
    cfg = load_config(args.config)["model"]
    device = torch.device("cuda:0")
    engine, tokenizer = load_models(cfg, str(device))
    adapter = DFlashBlockAdapter(
        target=engine.target, draft=engine.draft,
        ranker=HeuristicRanker(), block_size=15,
    )
    expected = {
        "target": cfg["target"], "target_revision": cfg["target_revision"],
        "draft": cfg["draft"], "draft_revision": cfg["draft_revision"],
    }
    initial_head = load_prefix_conditional_head(
        args.initial_checkpoint, int(engine.draft.config.hidden_size),
        device, expected,
    )
    if args.support_size < initial_head.support_size:
        raise ValueError("Selector support cannot narrow the initial head")
    if args.support_size == initial_head.support_size:
        head = initial_head
    else:
        head = PrefixConditionalHead(
            hidden_size=initial_head.hidden_size,
            rank=initial_head.rank,
            support_size=args.support_size,
            max_length=initial_head.max_length,
        ).to(device=device, dtype=next(initial_head.parameters()).dtype)
        state = initial_head.state_dict()
        widened = head.state_dict()
        for key, value in state.items():
            if key == "depth_bias":
                widened[key][:, :value.shape[1]].copy_(value)
            else:
                widened[key].copy_(value)
        head.load_state_dict(widened, strict=True)
    head.train()
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.learning_rate)
    output_embeddings = engine.target.get_output_embeddings().weight
    input_embeddings = engine.target.get_input_embeddings().weight
    texts = load_texts(args.train_data)
    order = list(range(len(texts)))
    random.Random(args.seed).shuffle(order)
    holdout = order[:args.holdout_samples]
    train = order[
        args.holdout_samples:args.holdout_samples + args.max_samples
    ]

    def example(sample_index: int, anchor_seed: int):
        ids = tokenizer(
            texts[sample_index], return_tensors="pt", add_special_tokens=True,
        ).input_ids[0]
        if ids.numel() < 18:
            return None
        local = random.Random(anchor_seed)
        anchor_index = local.randint(1, ids.numel() - 16)
        start = max(0, anchor_index - args.max_context)
        prefix = ids[start:anchor_index][None].to(device)
        anchor = ids[anchor_index:anchor_index + 1].to(device)
        with torch.inference_mode():
            cache = engine.cache_factory()
            context_output = engine.target_forward(
                prefix, cache, hidden=True, last_only=False,
            )
            context = adapter.extract_target_context(
                context_output.hidden_states
            )
            draft_logits, draft_hidden = adapter.draft_first_raw(
                context, anchor,
            )
            q = probabilities(draft_logits[0], 1., torch.float64)
        tree, predicted, student_rows, candidate_ids = predicted_node_log_mass(
            head, draft_hidden, draft_logits, output_embeddings,
            input_embeddings, q, args.pool_budget,
        )
        with torch.inference_mode():
            tree_ids = torch.tensor(
                [[int(anchor)] + tree.tokens], device=device,
            )
            positions = (
                torch.tensor(tree.depths, device=device) + prefix.shape[1]
            )[None]
            output = engine.target_forward(
                tree_ids, cache, hidden=False, positions=positions,
                mask=tree.mask(
                    prefix.shape[1], next(engine.target.parameters()).dtype,
                    device,
                ),
            )
            p = probabilities(output.logits[0], 1., torch.float64)
            target = target_node_log_mass(tree, p)
        return (
            selection_loss(predicted, target, args.budget)
            + branch_coverage_loss(
                tree, student_rows, candidate_ids, p, target,
            )
        )

    updates = 0
    total = 0.
    for epoch in range(args.epochs):
        for step, sample_index in enumerate(train, 1):
            loss = example(
                sample_index, args.seed + epoch * len(texts) + sample_index,
            )
            if loss is None:
                continue
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.)
            optimizer.step()
            updates += 1
            total += float(loss.detach())
            if step % 10 == 0:
                print(
                    f"epoch={epoch + 1}/{args.epochs} step={step}/{len(train)} "
                    f"updates={updates} mean_selection_loss={total / updates:.6f}",
                    flush=True,
                )
    if updates < 1:
        raise RuntimeError("No tree-selector updates")

    head.eval()
    holdout_total = 0.
    holdout_count = 0
    with torch.no_grad():
        for sample_index in holdout:
            loss = example(sample_index, args.seed + 1_000_000 + sample_index)
            if loss is not None:
                holdout_total += float(loss)
                holdout_count += 1
    if holdout_count < 1:
        raise RuntimeError("No tree-selector holdout examples")
    initial = torch.load(
        args.initial_checkpoint, map_location="cpu", weights_only=True,
    )
    metadata = dict(initial["metadata"])
    metadata.update({
        "support_size": head.support_size,
        "objective": "finite_tree_topb_selection_branch_kl_v2",
        "selector_updates": updates,
        "selector_epochs": args.epochs,
        "selector_learning_rate": args.learning_rate,
        "selector_budget": args.budget,
        "selector_pool_budget": args.pool_budget,
        "selector_holdout_loss": holdout_total / holdout_count,
        "selector_training_sha256": file_sha256(args.train_data),
        "selector_initial_sha256": file_sha256(args.initial_checkpoint),
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": {
            key: value.detach().cpu() for key, value in head.state_dict().items()
        },
        "metadata": metadata,
    }, args.output)
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
