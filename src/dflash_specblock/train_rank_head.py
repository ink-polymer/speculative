"""训练 DFlash-SpecBlock 的四分类 rank head。

输入 JSONL 每行至少含一个 ``text`` 字段，并应优先使用目标模型自行生成的回复。训练只更新
rank head：DFlash 与目标模型全部冻结。每个样本随机取 anchor，预测后续 K 个位置，使用目标
token 在 draft 分布中的 rank bucket 作为标签，并应用 SpecBlock valid-prefix mask。
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .config import ExperimentConfig
from .device import resolve_device
from .dflash_adapter import DFlashBlockAdapter
from .models import load_models
from .rank_head import DFlashRankHead, HeuristicRanker, target_rank_buckets, valid_prefix_mask


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train DFlash-SpecBlock rank head")
    parser.add_argument("--config", default="configs/qwen3_4b_a2.json")
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--output", default="checkpoints/rank_head.pt")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-samples", type=int, default=0)
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
    config = ExperimentConfig.from_json(args.config)
    if args.device:
        config.device = args.device
    device = resolve_device(config.device)
    random.seed(config.seed)
    torch.manual_seed(config.seed)

    bundle = load_models(config, device)
    rank_head = DFlashRankHead(int(bundle.draft.config.hidden_size)).to(device).train()
    adapter = DFlashBlockAdapter(
        target=bundle.target,
        draft=bundle.draft,
        ranker=HeuristicRanker(),
        block_size=config.block_size,
    )
    optimizer = torch.optim.AdamW(rank_head.parameters(), lr=args.learning_rate)
    texts = _load_texts(Path(args.train_data))
    if args.max_samples > 0:
        texts = texts[: args.max_samples]

    total_updates = 0
    for epoch in range(args.epochs):
        random.shuffle(texts)
        progress = tqdm(texts, desc=f"rank epoch {epoch + 1}")
        for text in progress:
            ids = bundle.tokenizer(text, return_tensors="pt", add_special_tokens=True).input_ids[0]
            # 跨块 curriculum 最多还要读取 cut+K 个未来 token，因此至少预留 2K+2。
            if ids.numel() < config.block_size * 2 + 3:
                continue
            max_anchor = ids.numel() - config.block_size * 2 - 1
            anchor_index = random.randint(1, max_anchor)
            prefix = ids[:anchor_index].unsqueeze(0).to(device)
            anchor = ids[anchor_index : anchor_index + 1].to(device)
            labels = ids[
                anchor_index + 1 : anchor_index + 1 + config.block_size
            ].unsqueeze(0).to(device)

            with torch.inference_mode():
                target_output = bundle.target(
                    input_ids=prefix,
                    output_hidden_states=True,
                    use_cache=False,
                    return_dict=True,
                )
                target_context = adapter.extract_target_context(target_output.hidden_states)
                draft_logits, draft_hidden = adapter.draft_first_raw(target_context, anchor)
                buckets = target_rank_buckets(draft_logits, labels)
                valid = valid_prefix_mask(draft_logits, labels)

                # SpecBlock §3.3：均匀采样 cut∈{1,...,K}，用第一块 cached h(L)
                # 绕过 target projection 作为下一块条件，并将监督序列平移 cut 个位置。
                cut = random.randint(1, config.block_size)
                continuation_anchor = labels[:, cut - 1]
                continuation_labels = ids[
                    anchor_index + cut + 1 : anchor_index + cut + 1 + config.block_size
                ].unsqueeze(0).to(device)
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
                # SpecBlock 官方明确「Block 间不做 filter」：推理触发下一块时，起点位置的
                # top-1 恰好可能是错的，用「前 cut 位全对」过滤掉的正是推理真实分布中的
                # 核心场景。这里只保留块内 valid-prefix mask，不再叠加跨块前缀正确性条件。

            # inference_mode 下创建的张量不能参与 autograd 记录的计算，用它们做索引会触发
            # "Inference tensors cannot be saved for backward"。标签与 mask 只在退出
            # inference_mode 后 clone 一次，前向仍然完全在 inference_mode 中完成。
            buckets = buckets.clone()
            valid = valid.clone()
            continuation_buckets = continuation_buckets.clone()
            continuation_valid = continuation_valid.clone()

            class_logits = rank_head(draft_hidden, draft_logits)
            continuation_class_logits = rank_head(continuation_hidden, continuation_logits)
            losses: list[torch.Tensor] = []
            if valid.any():
                losses.append(F.cross_entropy(class_logits[valid], buckets[valid]))
            if continuation_valid.any():
                losses.append(
                    F.cross_entropy(
                        continuation_class_logits[continuation_valid],
                        continuation_buckets[continuation_valid],
                    )
                )
            if not losses:
                continue
            loss = torch.stack(losses).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(rank_head.parameters(), 1.0)
            optimizer.step()
            total_updates += 1
            progress.set_postfix(
                loss=f"{loss.item():.4f}",
                valid=int(valid.sum().item() + continuation_valid.sum().item()),
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
            "state_dict": rank_head.state_dict(),
            "metadata": {
                "hidden_size": rank_head.hidden_size,
                "projection_size": rank_head.projection_size,
                "architecture": "specblock_h15_mlp_v1",
                "block_size": config.block_size,
                "updates": total_updates,
                "target_model_id": config.target_model_id,
                "target_revision": config.target_revision,
                "draft_model_id": config.draft_model_id,
                "draft_revision": config.draft_revision,
            },
        },
        output_path,
    )
    print(f"rank head 已保存到 {output_path}，更新步数={total_updates}")


if __name__ == "__main__":
    main()
