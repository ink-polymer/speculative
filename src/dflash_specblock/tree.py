"""SpecBlock 风格的 block-iterative 动态草稿树。

实现遵循论文和官方代码的关键拓扑：一个 block 先形成 greedy 主链，各位置的 top-b 其余
token 作为该位置的兄弟节点；rank-guided pending 节点被批量送进下一 block。这里故意使用
纯 PyTorch/Python 数据结构，不依赖 Triton，从而能在 Ascend 910B A2 上执行。
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import torch


@dataclass(slots=True)
class BlockProposal:
    """一次 DFlash block forward 的输出。"""

    logits: torch.Tensor  # [B, K, V]
    hidden: torch.Tensor  # [B, K, H]
    rank_logits: torch.Tensor  # [B, K, 4]

    def validate(self) -> None:
        if self.logits.ndim != 3 or self.hidden.ndim != 3 or self.rank_logits.ndim != 3:
            raise ValueError("BlockProposal 三个张量都必须是 [B, K, *]")
        if self.logits.shape[:2] != self.hidden.shape[:2]:
            raise ValueError("logits 与 hidden 的 [B, K] 不一致")
        if self.logits.shape[:2] != self.rank_logits.shape[:2]:
            raise ValueError("logits 与 rank_logits 的 [B, K] 不一致")
        if self.rank_logits.shape[-1] != 4:
            raise ValueError("rank_logits 最后一维必须为四个 bucket")


@dataclass(slots=True)
class TreeNode:
    token_id: int
    parent: int
    depth: int
    cumulative_log_probability: float
    block_index: int
    slot_index: int
    rank_bucket: int


@dataclass(slots=True)
class _PendingStart:
    node_index: int
    token_id: int
    cumulative_log_probability: float
    hidden: torch.Tensor


class DraftTree:
    """根节点是当前已验证 anchor；nodes 中只保存待验证候选。"""

    def __init__(self, nodes: Iterable[TreeNode] | None = None) -> None:
        self.nodes: list[TreeNode] = list(nodes or [])

    def add_node(
        self,
        token_id: int,
        parent: int,
        cumulative_log_probability: float,
        block_index: int,
        slot_index: int,
        rank_bucket: int,
    ) -> int:
        if parent >= len(self.nodes):
            raise ValueError("父节点必须先于子节点加入")
        depth = 1 if parent < 0 else self.nodes[parent].depth + 1
        self.nodes.append(
            TreeNode(
                token_id=int(token_id),
                parent=int(parent),
                depth=depth,
                cumulative_log_probability=float(cumulative_log_probability),
                block_index=int(block_index),
                slot_index=int(slot_index),
                rank_bucket=int(rank_bucket),
            )
        )
        return len(self.nodes) - 1

    def __len__(self) -> int:
        return len(self.nodes)

    def validate(self, budget: int | None = None) -> None:
        """验证目标树前向依赖的全部拓扑不变量。"""
        if budget is not None and len(self.nodes) > budget:
            raise AssertionError(f"树节点数 {len(self.nodes)} 超过预算 {budget}")
        for index, node in enumerate(self.nodes):
            if node.parent >= index:
                raise AssertionError(f"节点 {index} 的 parent={node.parent} 不是拓扑前序")
            expected_depth = 1 if node.parent < 0 else self.nodes[node.parent].depth + 1
            if node.depth != expected_depth:
                raise AssertionError(
                    f"节点 {index} depth={node.depth}，按父链应为 {expected_depth}"
                )

    def children(self) -> dict[int, list[int]]:
        result: dict[int, list[int]] = defaultdict(list)
        for index, node in enumerate(self.nodes):
            result[node.parent].append(index)
        return dict(result)

    def ancestor_mask(self, device: torch.device | None = None) -> torch.Tensor:
        """mask[i,j]=True 表示候选 i 可关注候选祖先 j（含自身）。"""
        count = len(self.nodes)
        mask = torch.zeros((count, count), dtype=torch.bool, device=device)
        for node_index in range(count):
            current = node_index
            while current >= 0:
                mask[node_index, current] = True
                current = self.nodes[current].parent
        return mask

    def retrieve_indices(self, device: torch.device | None = None) -> torch.Tensor:
        """返回所有根到叶路径；-1 是对齐填充，便于与 SpecBlock/EAGLE 工具对照。"""
        children = self.children()
        leaves = [index for index in range(len(self.nodes)) if index not in children]
        max_depth = max((self.nodes[index].depth for index in leaves), default=0)
        result = torch.full((len(leaves), max_depth), -1, dtype=torch.long, device=device)
        for row, leaf in enumerate(leaves):
            path: list[int] = []
            current = leaf
            while current >= 0:
                path.append(current)
                current = self.nodes[current].parent
            path.reverse()
            result[row, : len(path)] = torch.tensor(path, dtype=torch.long, device=device)
        return result

    def token_tensor(self, device: torch.device) -> torch.Tensor:
        return torch.tensor([node.token_id for node in self.nodes], dtype=torch.long, device=device)

    def prune(self, budget: int) -> None:
        """按累计 log-prob 选择节点，同时保证任何保留节点的整条祖先链存在。"""
        if len(self.nodes) <= budget:
            return
        order = sorted(
            range(len(self.nodes)),
            key=lambda index: self.nodes[index].cumulative_log_probability,
            reverse=True,
        )
        kept: set[int] = set()
        for candidate in order:
            chain: list[int] = []
            current = candidate
            while current >= 0 and current not in kept:
                chain.append(current)
                current = self.nodes[current].parent
            if len(kept) + len(chain) <= budget:
                kept.update(chain)
            if len(kept) >= budget:
                break

        kept_ordered = sorted(kept)
        remap = {old: new for new, old in enumerate(kept_ordered)}
        compacted: list[TreeNode] = []
        for old in kept_ordered:
            node = self.nodes[old]
            compacted.append(
                TreeNode(
                    token_id=node.token_id,
                    parent=remap.get(node.parent, -1),
                    depth=node.depth,
                    cumulative_log_probability=node.cumulative_log_probability,
                    block_index=node.block_index,
                    slot_index=node.slot_index,
                    rank_bucket=node.rank_bucket,
                )
            )
        self.nodes = compacted


ExpandCallback = Callable[[torch.Tensor, torch.Tensor], BlockProposal]


class SpecBlockTreeBuilder:
    """将 DFlash 的一批并行 token 分布组装成 SpecBlock 动态树。"""

    def __init__(
        self,
        block_size: int,
        max_blocks: int,
        tree_budget: int,
        beam_width: int,
        branch_factors: Sequence[int] = (2, 4, 10, 0),
    ) -> None:
        self.block_size = int(block_size)
        self.max_blocks = int(max_blocks)
        self.tree_budget = int(tree_budget)
        self.beam_width = int(beam_width)
        self.branch_factors = tuple(int(x) for x in branch_factors)
        if self.block_size < 1 or self.max_blocks < 1:
            raise ValueError("block_size 与 max_blocks 必须为正整数")
        if self.tree_budget < self.block_size or self.beam_width < 1:
            raise ValueError("tree_budget 不能小于 block_size，beam_width 必须为正整数")
        if len(self.branch_factors) != 4:
            raise ValueError("branch_factors 必须包含四个 bucket")
        if any(value < 0 for value in self.branch_factors):
            raise ValueError("branch_factors 不能包含负数")

    @staticmethod
    def _deduplicate_and_limit(pending: list[_PendingStart], limit: int) -> list[_PendingStart]:
        best: dict[int, _PendingStart] = {}
        for item in pending:
            existing = best.get(item.node_index)
            if (
                existing is None
                or item.cumulative_log_probability > existing.cumulative_log_probability
            ):
                best[item.node_index] = item
        return sorted(
            best.values(), key=lambda item: item.cumulative_log_probability, reverse=True
        )[:limit]

    def _adaptive_beam(self, node_count: int) -> int:
        """官方 adaptive beam：``min(beam_width, max(1, remaining_budget // K))``。"""
        remaining_budget = self.tree_budget - node_count
        return min(self.beam_width, max(1, remaining_budget // self.block_size))

    def _expand_rows(
        self,
        tree: DraftTree,
        proposal: BlockProposal,
        starts: list[_PendingStart],
        block_index: int,
        collect_pending: bool,
        pending_limit: int | None,
    ) -> list[_PendingStart]:
        proposal.validate()
        batch, block_size, _ = proposal.logits.shape
        if batch != len(starts) or block_size != self.block_size:
            raise ValueError("proposal 的 [B,K] 与 starts/config 不匹配")

        rank_buckets = proposal.rank_logits.argmax(dim=-1)
        all_next: list[_PendingStart] = []
        requested_topk = max(max(self.branch_factors), self.beam_width if block_index == 0 else 1)
        max_branch = min(max(requested_topk, 1), proposal.logits.shape[-1])

        for row, start in enumerate(starts):
            row_logits = proposal.logits[row]
            log_z = torch.logsumexp(row_logits.float(), dim=-1)
            top_values, top_ids = torch.topk(row_logits, k=max_branch, dim=-1)
            top_log_probs = top_values.float() - log_z.unsqueeze(-1)

            greedy_nodes: list[int] = []
            slot_parents: list[int] = []
            previous = start.node_index
            running_score = start.cumulative_log_probability

            # 官方 reference path 的 block-1 slot0 固定使用 beam_width 保证根部多样性；
            # 其余位置使用 rank bucket 映射，并在 b3->0 时停止处理更深位置的兄弟分支。
            nonzero_slots = (rank_buckets[row] != 0).nonzero(as_tuple=True)[0]
            primary_give_up = bool(
                nonzero_slots.numel()
                and int(rank_buckets[row, int(nonzero_slots[0])].item()) == 3
            )
            slot_widths = [0] * self.block_size
            stopped_by_give_up = False
            for slot in range(self.block_size):
                bucket = int(rank_buckets[row, slot].item())
                if block_index == 0 and slot == 0:
                    slot_widths[slot] = min(self.beam_width, max_branch)
                    continue
                width = min(self.branch_factors[bucket], max_branch)
                if bucket == 3 and width == 0:
                    stopped_by_give_up = True
                    break
                slot_widths[slot] = width

            # 一次 block 的 top-1 位置按顺序连接为主链。
            for slot in range(self.block_size):
                bucket = int(rank_buckets[row, slot].item())
                slot_parents.append(previous)
                running_score += float(top_log_probs[slot, 0].item())
                node_index = tree.add_node(
                    token_id=int(top_ids[slot, 0].item()),
                    parent=previous,
                    cumulative_log_probability=running_score,
                    block_index=block_index,
                    slot_index=slot,
                    rank_bucket=bucket,
                )
                greedy_nodes.append(node_index)
                previous = node_index

            # 每个位置剩余 top-b token 都与该位置的 greedy token 互为兄弟。
            for slot in range(self.block_size):
                bucket = int(rank_buckets[row, slot].item())
                width = slot_widths[slot]
                if width <= 0:
                    continue
                parent = slot_parents[slot]
                parent_score = (
                    start.cumulative_log_probability
                    if parent < 0
                    else tree.nodes[parent].cumulative_log_probability
                )
                hidden = proposal.hidden[row, slot].detach()

                if collect_pending and width > 1:
                    greedy = greedy_nodes[slot]
                    all_next.append(
                        _PendingStart(
                            node_index=greedy,
                            token_id=tree.nodes[greedy].token_id,
                            cumulative_log_probability=(
                                tree.nodes[greedy].cumulative_log_probability
                            ),
                            hidden=hidden,
                        )
                    )
                for alternative in range(1, width):
                    score = parent_score + float(top_log_probs[slot, alternative].item())
                    alt_node = tree.add_node(
                        token_id=int(top_ids[slot, alternative].item()),
                        parent=parent,
                        cumulative_log_probability=score,
                        block_index=block_index,
                        slot_index=slot,
                        rank_bucket=bucket,
                    )
                    if collect_pending:
                        all_next.append(
                            _PendingStart(
                                node_index=alt_node,
                                token_id=tree.nodes[alt_node].token_id,
                                cumulative_log_probability=score,
                                hidden=hidden,
                            )
                        )

            # SpecBlock 的 hitchhike：没有 give-up 时，主链末端也可启动下一块。
            hitchhike_allowed = (
                not primary_give_up if block_index == 0 else not stopped_by_give_up
            )
            if collect_pending and hitchhike_allowed:
                last = greedy_nodes[-1]
                all_next.append(
                    _PendingStart(
                        node_index=last,
                        token_id=tree.nodes[last].token_id,
                        cumulative_log_probability=tree.nodes[last].cumulative_log_probability,
                        hidden=proposal.hidden[row, -1].detach(),
                    )
                )

        if pending_limit is None:
            return self._deduplicate_and_limit(all_next, len(all_next))
        return self._deduplicate_and_limit(all_next, pending_limit)

    def build(self, first_block: BlockProposal, expand: ExpandCallback) -> DraftTree:
        """构建最多 M 个 diffusion block 的草稿树。"""
        first_block.validate()
        if first_block.logits.shape[0] != 1:
            raise ValueError("第一块必须只有一个当前已验证 anchor")

        tree = DraftTree()
        hidden_size = first_block.hidden.shape[-1]
        root_start = _PendingStart(
            node_index=-1,
            token_id=-1,
            cumulative_log_probability=0.0,
            hidden=torch.empty(hidden_size, device=first_block.hidden.device),
        )
        # 官方在每个 block 结束后立刻用 adaptive_beam裁剪 pending，再对剩余起点做下一次
        # forward；先forward 再裁剪会白算被丢弃的行，也会放大 batch。
        pending = self._expand_rows(
            tree,
            first_block,
            [root_start],
            block_index=0,
            collect_pending=self.max_blocks > 1,
            pending_limit=self._adaptive_beam(len(tree)),
        )

        for block_index in range(1, self.max_blocks):
            if not pending:
                break
            anchor_tokens = torch.tensor(
                [item.token_id for item in pending],
                dtype=torch.long,
                device=first_block.logits.device,
            )
            contexts = torch.stack([item.hidden for item in pending], dim=0)
            proposal = expand(contexts, anchor_tokens)
            pending = self._expand_rows(
                tree,
                proposal,
                pending,
                block_index=block_index,
                collect_pending=block_index + 1 < self.max_blocks,
                pending_limit=self._adaptive_beam(len(tree)),
            )

        tree.prune(self.tree_budget)
        tree.validate(self.tree_budget)
        return tree
