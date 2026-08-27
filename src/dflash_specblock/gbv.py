"""Temperature>0 的多路径树构建与 Greedy Block Verification。

实现对应 ``ink-polymer/speculative`` 的 ``temperature1-tree-gbv-2k`` 分支：
从 DFlash 各并行 slot 的完整分布独立采样多条路径，合并为一棵前缀树；target
一次验证整棵树后，按 GBV 的次序统计量选择路径，并用联合 Block Verification
的残差分布提交 token。不能把单路径 verifier 独立应用到普通 DDTree 的所有叶子，
因为叶事件共享前缀且相互重叠，那样不保持目标采样分布。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from .tree import BlockProposal, DraftTree


def sampling_probs(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """用 FP32 softmax 构造 temperature sampling 分布。"""
    if temperature <= 0:
        raise ValueError("概率验证要求 temperature > 0")
    return torch.softmax(logits.float() / temperature, dim=-1)


def sample_probs(probs: torch.Tensor) -> torch.Tensor:
    """从最后一维的每个离散分布各采一个 token。"""
    shape = probs.shape[:-1]
    return torch.multinomial(probs.reshape(-1, probs.shape[-1]), 1).reshape(shape)


def block_rejection_sample(
    drafts: torch.Tensor,
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
) -> tuple[int, torch.Tensor]:
    """Sun et al. (2024) Appendix A 的联合 Block Verification。"""
    gamma = int(drafts.numel())
    vocab = int(target_probs.shape[-1])
    if target_probs.shape != (gamma + 1, vocab):
        raise ValueError("target_probs 必须是 [gamma + 1, vocab]")
    if draft_probs.shape != (gamma, vocab):
        raise ValueError("draft_probs 必须是 [gamma, vocab]")

    accept_probability = torch.ones((), dtype=torch.float32, device=drafts.device)
    best_length = 0
    bonus: torch.Tensor | None = None
    zero_q = torch.zeros((vocab,), dtype=target_probs.dtype, device=drafts.device)
    for index in range(gamma + 1):
        q_row = draft_probs[index] if index < gamma else zero_q
        residual = (target_probs[index] * accept_probability - q_row).clamp_min(0)
        reject = (1.0 - accept_probability).clamp_min(0).reshape(1)
        weights = torch.cat((residual, reject))
        total = weights.sum()
        if not bool(torch.isfinite(total)) or float(total.item()) <= 0:
            chosen = int(sample_probs(target_probs[index : index + 1])[0].item())
        else:
            chosen = int(torch.multinomial(weights / total, 1).item())
        if chosen < vocab:
            best_length = index
            bonus = torch.tensor(chosen, dtype=torch.long, device=drafts.device)

        if index < gamma:
            token = drafts[index]
            p_token = target_probs[index, token]
            q_token = draft_probs[index, token]
            ratio = p_token / q_token.clamp_min(torch.finfo(q_token.dtype).tiny)
            accept_probability = torch.minimum(
                torch.ones_like(accept_probability), accept_probability * ratio
            )

    if bonus is None:
        bonus = sample_probs(target_probs[best_length : best_length + 1])[0]
    return best_length, bonus


def gbv_select_path_and_probs(
    paths: torch.Tensor,
    target_probs_by_path: torch.Tensor,
    draft_probs: torch.Tensor,
) -> tuple[int, torch.Tensor]:
    """按 GBV 次序统计量选路径，并计算被选路径的偏斜 proposal 分布。"""
    if paths.ndim != 2:
        raise ValueError("paths 必须是 [path_count, length]")
    path_count, length = paths.shape
    vocab = draft_probs.shape[-1]
    if draft_probs.shape != (length, vocab):
        raise ValueError("draft_probs 必须是 [length, vocab]")
    if target_probs_by_path.shape != (path_count, length + 1, vocab):
        raise ValueError("target_probs_by_path 必须是 [path_count, length + 1, vocab]")

    tiny = torch.finfo(draft_probs.dtype).tiny
    ratios: list[list[float]] = []
    for path_index in range(path_count):
        token_p = target_probs_by_path[path_index, :length].gather(
            -1, paths[path_index, :, None]
        )[:, 0]
        token_q = draft_probs.gather(
            -1, paths[path_index, :, None]
        )[:, 0]
        ratios.append((token_p / token_q.clamp_min(tiny)).tolist())

    # Python tuple 的字典序与论文中的逐位置次序统计量一致；token id 只负责稳定破平。
    selected = max(
        range(path_count),
        key=lambda path_index: tuple(
            (ratios[path_index][depth], int(paths[path_index, depth]))
            for depth in range(length)
        ),
    )

    selected_path = paths[selected]
    selected_target = target_probs_by_path[selected]
    q_prefix = torch.ones((), dtype=draft_probs.dtype, device=paths.device)
    q_gamma_prefix = torch.ones_like(q_prefix)
    lower_path_mass = torch.zeros_like(q_prefix)
    q_gamma_rows: list[torch.Tensor] = []
    for depth in range(length):
        q_row = draft_probs[depth]
        p_row = selected_target[depth]
        ratio_row = p_row / q_row.clamp_min(tiny)
        order = torch.argsort(ratio_row, stable=True)
        ordered_q = q_row.index_select(0, order)
        lower_ordered = torch.cat(
            (torch.zeros_like(ordered_q[:1]), ordered_q.cumsum(0)[:-1])
        )
        lower = torch.empty_like(q_row)
        lower.scatter_(0, order, lower_ordered)

        a = lower_path_mass + q_prefix * lower
        joint = (a + q_prefix * q_row).pow(path_count) - a.pow(path_count)
        conditional = joint / q_gamma_prefix.clamp_min(tiny)
        conditional = conditional.clamp_min(0)
        conditional = conditional / conditional.sum().clamp_min(tiny)
        q_gamma_rows.append(conditional)

        token = selected_path[depth]
        lower_path_mass = a[token]
        q_prefix = q_prefix * q_row[token]
        q_gamma_prefix = joint[token]

    return selected, torch.stack(q_gamma_rows)


@dataclass(slots=True)
class GBVMetadata:
    paths: torch.Tensor
    path_node_indices: list[list[int]]
    draft_probs: torch.Tensor
    temperature: float


class GBVDraftTree(DraftTree):
    """带有 GBV 路径分布元数据的、去重后的采样前缀树。"""

    def __init__(self, metadata: GBVMetadata) -> None:
        super().__init__()
        self.gbv = metadata


class GBVTreeBuilder:
    """从一次 DFlash block 的完整分布采样 K 条路径并合并成树。"""

    requires_rank = False
    manages_budget = True

    def __init__(self, block_size: int, path_count: int, temperature: float) -> None:
        self.block_size = int(block_size)
        self.path_count = int(path_count)
        self.temperature = float(temperature)
        self.tree_budget = self.block_size * self.path_count
        if self.block_size < 1:
            raise ValueError("GBV block_size 必须为正整数")
        if self.path_count < 1:
            raise ValueError("gbv_paths 必须为正整数")
        if self.temperature <= 0:
            raise ValueError("GBV 要求 temperature > 0")

    def build(
        self,
        first: BlockProposal,
        expand: Callable[[torch.Tensor, torch.Tensor], BlockProposal],
        budget: int | None = None,
    ) -> GBVDraftTree:
        del expand, budget
        first.validate()
        if first.logits.shape[:2] != (1, self.block_size):
            raise ValueError("GBV 只支持 batch=1，且 proposal block_size 必须与配置一致")

        draft_probs = sampling_probs(first.logits[0], self.temperature)
        paths = sample_probs(
            draft_probs.unsqueeze(0).expand(self.path_count, -1, -1)
        )
        metadata = GBVMetadata(
            paths=paths,
            path_node_indices=[],
            draft_probs=draft_probs,
            temperature=self.temperature,
        )
        tree = GBVDraftTree(metadata)
        children: dict[tuple[int, int], int] = {}
        path_nodes: list[list[int]] = []
        for path in paths.tolist():
            parent = -1
            nodes: list[int] = []
            for depth, token in enumerate(path):
                key = (parent, int(token))
                node_index = children.get(key)
                if node_index is None:
                    node_index = tree.add_node(
                        token_id=int(token),
                        parent=parent,
                        # GBV 的路径概率保留在完整 draft_probs 中；树节点分数不参与选择。
                        cumulative_log_probability=0.0,
                        block_index=0,
                        slot_index=depth,
                        rank_bucket=0,
                    )
                    children[key] = node_index
                nodes.append(node_index)
                parent = node_index
            path_nodes.append(nodes)
        tree.gbv.path_node_indices = path_nodes
        tree.validate(self.tree_budget)
        return tree


def select_gbv_path(
    current_logits: torch.Tensor,
    tree: GBVDraftTree,
) -> tuple[list[int], list[int], int]:
    """在 target 树 logits 上执行路径选择和联合 Block Verification。"""
    if current_logits.ndim != 2 or current_logits.shape[0] != len(tree) + 1:
        raise ValueError("current_logits 必须是 [1 + tree_nodes, vocab]")
    metadata = tree.gbv
    target_probs = sampling_probs(current_logits, metadata.temperature)
    target_by_path = torch.stack(
        [
            target_probs[
                torch.tensor(
                    [0] + [node + 1 for node in path_nodes],
                    dtype=torch.long,
                    device=current_logits.device,
                )
            ]
            for path_nodes in metadata.path_node_indices
        ]
    )
    selected, skewed_q = gbv_select_path_and_probs(
        metadata.paths, target_by_path, metadata.draft_probs
    )
    accepted, bonus = block_rejection_sample(
        metadata.paths[selected], target_by_path[selected], skewed_q
    )
    selected_nodes = metadata.path_node_indices[selected][:accepted]
    accepted_tokens = metadata.paths[selected, :accepted].tolist()
    return selected_nodes, [int(token) for token in accepted_tokens], int(bonus.item())
