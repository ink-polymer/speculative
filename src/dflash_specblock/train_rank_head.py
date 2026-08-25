"""训练 DFlash-SpecBlock 的四分类 rank head。

输入 JSONL 每行至少含一个 ``text`` 字段，并应优先使用目标模型自行生成的回复。训练只更新
rank head：DFlash 与目标模型全部冻结。每个样本随机取 anchor，预测后续 K 个位置，使用目标
token 在 draft 分布中的 rank bucket 作为标签，并应用 SpecBlock valid-prefix mask。
"""

from __future__ import annotations

import argparse
import json
import random
import warnings
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import ExperimentConfig
from .device import configure_cuda_runtime, dtype_from_name, resolve_device
from .dflash_adapter import DFlashBlockAdapter
from .models import load_models
from .rank_head import (
    DFlashRankHead,
    HeuristicRanker,
    distribution_summary,
    target_rank_buckets,
    valid_prefix_mask,
)


def _load_texts(path: Path) -> list[str]:
    texts: list[str] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            text = item.get("text")
            if not isinstance(text, str) or not text:
                raise ValueError(f"{path}:{line_number} 缺少非空 text 字段")
            texts.append(text)
    if not texts:
        raise ValueError("训练文件中没有样本")
    return texts


def _autocast_context(device: torch.device, dtype: torch.dtype):
    if device.type == "cuda" and dtype in {torch.float16, torch.bfloat16}:
        return torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


def _make_grad_scaler(enabled: bool):
    if not enabled:
        return None
    try:
        return torch.amp.GradScaler("cuda", enabled=True)
    except (AttributeError, TypeError):
        # Compatibility with PyTorch releases that still expose the CUDA-specific API.
        return torch.cuda.amp.GradScaler(enabled=True)


def _make_optimizer(parameters, learning_rate: float, device: torch.device):
    parameters = list(parameters)
    kwargs = {"lr": learning_rate}
    if device.type == "cuda":
        try:
            return torch.optim.AdamW(parameters, fused=True, **kwargs)
        except (RuntimeError, TypeError) as error:
            warnings.warn(
                f"fused AdamW is unavailable; falling back to the standard implementation: {error}",
                stacklevel=2,
            )
    return torch.optim.AdamW(parameters, **kwargs)


def _classify_rank_features(
    rank_head: DFlashRankHead,
    hidden: torch.Tensor,
    summary: torch.Tensor,
) -> torch.Tensor:
    """Run the trainable MLP from compact, precomputed frozen-model features."""
    parameter_dtype = rank_head.classifier[0].weight.dtype
    rank_input = torch.cat([hidden.float(), summary.float()], dim=-1)
    return rank_head.classifier(rank_input.to(parameter_dtype))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train DFlash-SpecBlock rank head")
    parser.add_argument("--config", default="configs/qwen3_4b_cuda.json")
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--output", default="checkpoints/rank_head.pt")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--device", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.epochs < 1:
        raise ValueError("epochs 必须为正整数")
    if args.learning_rate <= 0:
        raise ValueError("learning-rate 必须为正数")
    if args.max_samples < 0:
        raise ValueError("max-samples 不能为负数")
    if args.num_workers < 0:
        raise ValueError("num-workers 不能为负数")
    if args.log_every < 1:
        raise ValueError("log-every 必须为正整数")
    config = ExperimentConfig.from_json(args.config)
    if args.device:
        config.device = args.device
    device = resolve_device(config.device)
    configure_cuda_runtime(device, config.allow_tf32)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)

    bundle = load_models(config, device)
    rank_head = DFlashRankHead(int(bundle.draft.config.hidden_size)).to(device).train()
    adapter = DFlashBlockAdapter(
        target=bundle.target,
        draft=bundle.draft,
        ranker=HeuristicRanker(),
        block_size=config.block_size,
    )
    optimizer = _make_optimizer(rank_head.parameters(), args.learning_rate, device)
    model_dtype = dtype_from_name(config.dtype)
    scaler = _make_grad_scaler(device.type == "cuda" and model_dtype == torch.float16)
    texts = _load_texts(Path(args.train_data))
    if args.max_samples > 0:
        texts = texts[: args.max_samples]

    # Tokenize once instead of serializing tokenizer work with every GPU update.
    tokenized = [
        bundle.tokenizer(text, return_tensors="pt", add_special_tokens=True)
        .input_ids[0]
        .contiguous()
        for text in texts
    ]
    loader_generator = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(
        tokenized,
        batch_size=None,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        generator=loader_generator,
    )

    total_updates = 0
    for epoch in range(args.epochs):
        progress = tqdm(loader, total=len(tokenized), desc=f"rank epoch {epoch + 1}")
        for ids in progress:
            # 跨块 curriculum 最多还要读取 cut+K 个未来 token，因此至少预留 2K+2。
            if ids.numel() < config.block_size * 2 + 3:
                continue
            # Move one contiguous token tensor instead of three independent slices.
            ids = ids.to(device, non_blocking=device.type == "cuda")
            max_anchor = ids.numel() - config.block_size * 2 - 1
            anchor_index = random.randint(1, max_anchor)
            prefix = ids[:anchor_index].unsqueeze(0)
            anchor = ids[anchor_index : anchor_index + 1]
            labels = ids[
                anchor_index + 1 : anchor_index + 1 + config.block_size
            ].unsqueeze(0)

            with torch.inference_mode():
                target_output = bundle.target(
                    input_ids=prefix,
                    output_hidden_states=True,
                    use_cache=False,
                    # Training consumes target hidden states, not prompt logits.
                    # Retaining one row avoids a full [B, prompt, vocab] LM-head output.
                    logits_to_keep=1,
                    return_dict=True,
                )
                target_context = adapter.extract_target_context(target_output.hidden_states)
                draft_logits, draft_hidden = adapter.draft_first_raw(target_context, anchor)
                buckets = target_rank_buckets(draft_logits, labels)
                valid = valid_prefix_mask(draft_logits, labels)
                draft_summary = distribution_summary(draft_logits)

                # SpecBlock §3.3：均匀采样 cut∈{1,...,K}，用第一块 cached h(L)
                # 绕过 target projection 作为下一块条件，并将监督序列平移 cut 个位置。
                # 训练/推理一致性：max_blocks=1 时推理不会触发 continuation，训练也不做。
                do_continuation = config.max_blocks >= 2
                cut = random.randint(1, config.block_size) if do_continuation else 0
                if do_continuation:
                    continuation_anchor = labels[:, cut - 1]
                    continuation_labels = ids[
                        anchor_index + cut + 1 : anchor_index + cut + 1 + config.block_size
                    ].unsqueeze(0)
                    continuation_logits, continuation_hidden = adapter.draft_continuation_raw(
                        draft_context=draft_hidden[:, cut - 1, :],
                        anchor_ids=continuation_anchor,
                    )
                    continuation_buckets = target_rank_buckets(
                        continuation_logits, continuation_labels
                    )
                    continuation_valid = valid_prefix_mask(
                        continuation_logits, continuation_labels
                    )
                    continuation_summary = distribution_summary(continuation_logits)
                else:
                    continuation_logits = None
                    continuation_hidden = None
                    continuation_summary = None
                    continuation_buckets = None
                    continuation_valid = None
                # SpecBlock 官方明确「Block 间不做 filter」：推理触发下一块时，起点位置的
                # top-1 恰好可能是错的，用「前 cut 位全对」过滤掉的正是推理真实分布中的
                # 核心场景。这里只保留块内 valid-prefix mask，不再叠加跨块前缀正确性条件。

            # 只把 compact hidden/15维摘要以及小型标签复制成普通 tensor。不要把
            # [B,K,V]（Qwen3 为约 151K 词表）logits clone 到 autograd 路径。
            draft_hidden = draft_hidden.clone()
            draft_summary = draft_summary.clone()
            buckets = buckets.clone()
            valid = valid.clone()
            if continuation_buckets is not None:
                continuation_hidden = continuation_hidden.clone()
                continuation_summary = continuation_summary.clone()
                continuation_buckets = continuation_buckets.clone()
                continuation_valid = continuation_valid.clone()
            del draft_logits, continuation_logits, target_output, target_context

            optimizer.zero_grad(set_to_none=True)
            with _autocast_context(device, model_dtype):
                class_logits = _classify_rank_features(
                    rank_head,
                    draft_hidden,
                    draft_summary,
                )
                # valid_prefix_mask always includes the first position, so this avoids a
                # per-step GPU->CPU synchronization from ``valid.any()``.
                losses = [F.cross_entropy(class_logits[valid], buckets[valid])]
                if continuation_hidden is not None:
                    continuation_class_logits = _classify_rank_features(
                        rank_head,
                        continuation_hidden,
                        continuation_summary,
                    )
                    losses.append(
                        F.cross_entropy(
                            continuation_class_logits[continuation_valid],
                            continuation_buckets[continuation_valid],
                        )
                    )
                loss = torch.stack(losses).mean()

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(rank_head.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(rank_head.parameters(), 1.0)
                optimizer.step()
            total_updates += 1
            if total_updates % args.log_every == 0:
                valid_count = valid.sum()
                if continuation_valid is not None:
                    valid_count = valid_count + continuation_valid.sum()
                loss_value, valid_value = torch.stack(
                    [loss.detach().float(), valid_count.float()]
                ).cpu().tolist()
                progress.set_postfix(
                    loss=f"{loss_value:.4f}",
                    valid=int(valid_value),
                )

    if total_updates == 0:
        raise RuntimeError(
            "没有产生任何 rank-head 更新；请检查训练数据长度、tokenizer 和 valid-prefix 样本"
        )

    output_path = Path(args.output).expanduser()
    if not output_path.is_absolute():
        output_path = config.project_root / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": {
                name: tensor.detach().cpu() for name, tensor in rank_head.state_dict().items()
            },
            "metadata": {
                "hidden_size": rank_head.hidden_size,
                "projection_size": rank_head.projection_size,
                "architecture": "specblock_h15_mlp_v1",
                "block_size": config.block_size,
                "max_blocks": config.max_blocks,
                "updates": total_updates,
                "target_model_id": config.target_model_id,
                "target_revision": config.target_revision,
                "draft_model_id": config.draft_model_id,
                "draft_revision": config.draft_revision,
                "training_dtype": config.dtype,
                "amp_enabled": device.type == "cuda" and model_dtype in {
                    torch.float16,
                    torch.bfloat16,
                },
                "allow_tf32": config.allow_tf32,
                "attn_implementation": config.attn_implementation,
                "draft_attn_implementation": (
                    config.draft_attn_implementation or config.attn_implementation
                ),
            },
        },
        output_path,
    )
    print(f"rank head 已保存到 {output_path}，更新步数={total_updates}")


if __name__ == "__main__":
    main()
