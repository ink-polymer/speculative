"""SpecBlock rank head 及其 15 维分布摘要。

论文把目标 token 在 draft 分布中的真实 rank 压缩为四类：1、2-4、5-10、>10。rank head
只读取停止梯度后的 DFlash 隐状态与分布摘要，避免分类任务反向干扰扩散 drafter。
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn


def distribution_summary(logits: torch.Tensor) -> torch.Tensor:
    """返回 SpecBlock 官方 rank head 使用的 15 维分布摘要。

    与官方 ``_llama3_specblock_base`` 完全一致地采用 top-20 快速路径：归一化常数用 top-20
    的 logsumexp 近似全词表 lse，entropy 也只在 top-20 上累加。这不是精度妥协，而是必须
    对齐的口径——rank head 的权重是在该特征分布上训练的，改用全词表 lse 会把 top1 log-prob
    和 entropy 整体平移，使已训练的 rank head 失配。

    15 维构成：top-10 log-prob (10) + top1相对 rank2/3/5 的 logit gap (3)
    + top1 probability (1) + entropy (1)。
    """
    if logits.shape[-1] < 20:
        raise ValueError("词表至少需要 20 个 token 才能计算官方 top-20 摘要")
    work = logits.float()
    top20_values = torch.topk(work, k=20, dim=-1).values
    log_z = torch.logsumexp(top20_values, dim=-1, keepdim=True)

    top10_values = top20_values[..., :10]
    top_log_probs = top10_values - log_z
    # 官方使用 gap_12 / gap_13 / gap_15，即 top1 与第2、3、5 个 logit 的差。
    gaps = torch.stack(
        [top10_values[..., 0] - top10_values[..., index] for index in (1, 2, 4)], dim=-1
    )
    top1_probability = top_log_probs[..., :1].exp()
    top20_log_probs = top20_values - log_z
    top20_probabilities = top20_log_probs.exp()
    entropy = -(top20_probabilities * top20_log_probs).nan_to_num(0.0).sum(
        dim=-1, keepdim=True
    )
    return torch.cat([top_log_probs, gaps, top1_probability, entropy], dim=-1)


def target_rank_buckets(logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    """不做全词表排序，直接统计比目标 token logit 更大的元素数量。"""
    target_logit = logits.gather(-1, target_ids.unsqueeze(-1))
    ranks = (logits > target_logit).sum(dim=-1) + 1
    buckets = torch.full_like(ranks, 3)
    buckets = torch.where(ranks <= 10, torch.full_like(buckets, 2), buckets)
    buckets = torch.where(ranks <= 4, torch.full_like(buckets, 1), buckets)
    buckets = torch.where(ranks == 1, torch.zeros_like(buckets), buckets)
    return buckets


def valid_prefix_mask(logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    """第 k 位仅在此前所有 greedy token 都正确时参与 loss。"""
    correct = logits.argmax(dim=-1).eq(target_ids)
    prefix_before = torch.ones_like(correct, dtype=torch.bool)
    if correct.shape[-1] > 1:
        prefix_before[..., 1:] = correct[..., :-1].cumprod(dim=-1).bool()
    return prefix_before


class DFlashRankHead(nn.Module):
    """严格对应 SpecBlock 的 ``(H + 15) -> 256 -> 4`` 四分类头。

    DFlash 官方 checkpoint 输出的 ``hidden`` 已经过 drafter 最终 RMSNorm，因此这里不再
    额外引入 LayerNorm 或拆分投影。这样除“把 SpecBlock hidden 换成 DFlash hidden”这一
    组合点外，rank head 的结构与 SpecBlock 官方实现保持一致。
    """

    def __init__(self, hidden_size: int, projection_size: int = 256) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.projection_size = int(projection_size)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size + 15, projection_size, bias=False),
            nn.SiLU(),
            nn.Linear(projection_size, 4, bias=False),
        )

    def forward(self, hidden: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        # rank loss 不得反向改变 DFlash trunk 或 LM head；论文中的 sg(.) 在此显式实现。
        # adapter 的 raw forward 由 inference_mode 保护。训练 rank head 时，inference tensor
        # 不能被 autograd 保存，因此只在已经退出 inference_mode 后复制成普通 tensor；正式
        # 推理仍保持零复制路径。
        detached_hidden = hidden.detach()
        detached_logits = logits.detach()
        if not torch.is_inference_mode_enabled():
            if detached_hidden.is_inference():
                detached_hidden = detached_hidden.clone()
            if detached_logits.is_inference():
                detached_logits = detached_logits.clone()
        rank_input = torch.cat(
            [detached_hidden.float(), distribution_summary(detached_logits)], dim=-1
        )
        parameter_dtype = self.classifier[0].weight.dtype
        return self.classifier(rank_input.to(parameter_dtype))


class HeuristicRanker(nn.Module):
    """仅用于工程连通性测试的无参数近似器，不等价于论文训练后的 rank head。"""

    def forward(self, hidden: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        del hidden
        top = torch.topk(logits.float(), k=2, dim=-1).values
        margin = top[..., 0] - top[..., 1]
        buckets = torch.full_like(margin, 3, dtype=torch.long)
        buckets = torch.where(margin >= 0.20, torch.full_like(buckets, 2), buckets)
        buckets = torch.where(margin >= 0.80, torch.full_like(buckets, 1), buckets)
        buckets = torch.where(margin >= 2.00, torch.zeros_like(buckets), buckets)
        return torch.nn.functional.one_hot(buckets, num_classes=4).float() * 20.0


def load_rank_head(
    checkpoint: str | Path,
    hidden_size: int,
    device: torch.device,
    expected_metadata: Mapping[str, Any] | None = None,
) -> DFlashRankHead:
    """严格加载与当前 target/draft/K 绑定的 SpecBlock rank checkpoint。"""
    payload = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise ValueError("rank checkpoint 必须包含 state_dict 与 metadata")
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("rank checkpoint metadata 必须是字典")
    for field in ("projection_size", "updates"):
        if field not in metadata:
            raise ValueError(f"rank checkpoint 缺少严格复现实验元数据: {field}")
    if int(metadata["projection_size"]) < 1:
        raise ValueError("rank checkpoint projection_size 必须为正整数")
    if int(metadata["updates"]) < 1:
        raise ValueError("rank checkpoint 没有有效训练更新，不能用于正式实验")
    required_metadata: dict[str, Any] = {
        "architecture": "specblock_h15_mlp_v1",
        "hidden_size": hidden_size,
    }
    required_metadata.update(dict(expected_metadata or {}))
    missing = sorted(key for key in required_metadata if key not in metadata)
    if missing:
        raise ValueError(f"rank checkpoint 缺少严格复现实验元数据: {missing}")
    mismatched = {
        key: (metadata[key], expected)
        for key, expected in required_metadata.items()
        if metadata[key] != expected
    }
    if mismatched:
        raise ValueError(f"rank checkpoint 与当前实验配置不一致: {mismatched}")

    saved_hidden = int(metadata["hidden_size"])
    if saved_hidden != hidden_size:
        raise ValueError(f"rank head hidden_size={saved_hidden}，draft hidden_size={hidden_size}")
    model = DFlashRankHead(
        hidden_size=hidden_size,
        projection_size=int(metadata["projection_size"]),
    )
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.to(device).eval()
