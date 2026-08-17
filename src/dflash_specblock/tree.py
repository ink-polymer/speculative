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
    """根节点是当前已验证 anchor；nodes 中只保存待验证候选。

    惰性缓存：构建阶段（add_node 频繁）不触发；验证阶段（只读）一次计算后复用。
    """

    def __init__(self, nodes: Iterable[TreeNode] | None = None) -> None:
        self.nodes: list[TreeNode] = list(nodes or [])
        self._parents_cache: list[int] | None = None
        self._tokens_cache: list[int] | None = None
        self._children_cache: dict[int, list[int]] | None = None

    def _invalidate_caches(self) -> None:
        self._parents_cache = None
        self._tokens_cache = None
        self._children_cache = None

    def _parents(self) -> list[int]:
        if self._parents_cache is None:
            self._parents_cache = [node.parent for node in self.nodes]
        return self._parents_cache

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
        self._invalidate_caches()
        return len(self.nodes) - 1

    def __len__(self) -> int:
        return len(self.nodes)

    def validate(self, budget: int | None = None) -> None:
        """验证目标树前向依赖的全部拓扑不变量。"""
        if budget is not None and len(self.nodes) > budget:
            raise AssertionError(f"树节点数 {len(self.nodes)} 超过预算 {budget}")
        parents = self._parents()
        for index in range(len(self.nodes)):
            parent = parents[index]
            if parent >= index:
                raise AssertionError(f"节点 {index} 的 parent={parent} 不是拓扑前序")
            expected_depth = 1 if parent < 0 else self.nodes[parent].depth + 1
            if self.nodes[index].depth != expected_depth:
                raise AssertionError(
                    f"节点 {index} depth={self.nodes[index].depth}，按父链应为 {expected_depth}"
                )

    def children(self) -> dict[int, list[int]]:
        if self._children_cache is None:
            result: dict[int, list[int]] = defaultdict(list)
            parents = self._parents()
            for index in range(len(self.nodes)):
                result[parents[index]].append(index)
            self._children_cache = dict(result)
        return self._children_cache

    def ancestor_mask(self, device: torch.device | None = None) -> torch.Tensor:
        """mask[i,j]=True 表示候选 i 可关注候选祖先 j（含自身）。

        向量化对数并行祖先传播：每次迭代 mask[i] |= mask[parent[i]]，
        重复 ceil(log2(N)) 次保证覆盖最深路径，全部为张量并行操作，
        避免 Python while 逐节点走父链的单元素赋值。
        """
        count = len(self.nodes)
        if count == 0:
            return torch.zeros((0, 0), dtype=torch.bool, device=device)
        mask = torch.eye(count, dtype=torch.bool, device=device)
        if count == 1:
            return mask
        parents = torch.as_tensor(self._parents(), dtype=torch.long, device=device)
        valid = parents >= 0
        safe_parents = parents.clamp(min=0)
        iterations = count.bit_length()
        for _ in range(iterations):
            parent_rows = mask.index_select(0, safe_parents)
            mask = torch.where(valid.unsqueeze(-1), mask | parent_rows, mask)
        return mask

    def retrieve_indices(self, device: torch.device | None = None) -> torch.Tensor:
        """返回所有根到叶路径；-1 是对齐填充，便于与 SpecBlock/EAGLE 工具对照。"""
        children = self.children()
        parents = self._parents()
        node_count = len(self.nodes)
        leaves = [index for index in range(node_count) if index not in children]
        max_depth = max((self.nodes[index].depth for index in leaves), default=0)
        if max_depth == 0:
            return torch.full((len(leaves), 0), -1, dtype=torch.long, device=device)
        result = torch.full((len(leaves), max_depth), -1, dtype=torch.long, device=device)
        for row, leaf in enumerate(leaves):
            path: list[int] = []
            current = leaf
            while current >= 0:
                path.append(current)
                current = parents[current]
            path.reverse()
            result[row, : len(path)] = torch.as_tensor(path, dtype=torch.long, device=device)
        return result

    def token_tensor(self, device: torch.device) -> torch.Tensor:
        if self._tokens_cache is None:
            self._tokens_cache = [node.token_id for node in self.nodes]
        return torch.as_tensor(self._tokens_cache, dtype=torch.long, device=device)

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
        self._invalidate_caches()


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
        batch, block_size, vocab = proposal.logits.shape
        if batch != len(starts) or block_size != self.block_size:
            raise ValueError("proposal 的 [B,K] 与 starts/config 不匹配")

        requested_topk = max(max(self.branch_factors), self.beam_width if block_index == 0 else 1)
        max_branch = min(max(requested_topk, 1), vocab)

        # 批量化：一次 topk / logsumexp 处理整个 [B, K, V]，
        # 代替原逐 row 循环；在 NPU 上把 B 次 kernel 合并为 1 次。
        log_z = torch.logsumexp(proposal.logits.float(), dim=-1)  # [B, K]
        top_values, top_ids = torch.topk(proposal.logits, k=max_branch, dim=-1)  # [B, K, max_branch]
        top_log_probs = top_values.float() - log_z.unsqueeze(-1)  # [B, K, max_branch]
        rank_buckets = proposal.rank_logits.argmax(dim=-1)  # [B, K]

        # 一次性 host<-device 取回（NPU 上 3 次同步代替原来每个 slot 多次 .item()）。
        # 后续 Python 循环全部操作原生 int/float，无任何同步。
        rank_buckets_cpu: list[list[int]] = rank_buckets.tolist()
        top_ids_cpu: list[list[list[int]]] = top_ids.tolist()
        top_log_probs_cpu: list[list[list[float]]] = top_log_probs.tolist()

        all_next: list[_PendingStart] = []
        beam_width = self.beam_width
        branch_factors = self.branch_factors
        add_node = tree.add_node
        hidden_source = proposal.hidden

        for row, start in enumerate(starts):
            row_buckets = rank_buckets_cpu[row]
            row_top_ids = top_ids_cpu[row]
            row_top_log_probs = top_log_probs_cpu[row]
            row_hidden = hidden_source[row]

            greedy_nodes: list[int] = []
            slot_parents: list[int] = []
            previous = start.node_index
            running_score = start.cumulative_log_probability

            # 官方 reference path 的 block-1 slot0 固定使用 beam_width 保证根部多样性；
            # 其余位置使用 rank bucket 映射，并在 b3->0 时停止处理更深位置的兄弟分支。
            nonzero_slots = [s for s in range(block_size) if row_buckets[s] != 0]
            primary_give_up = bool(nonzero_slots and row_buckets[nonzero_slots[0]] == 3)
            slot_widths = [0] * block_size
            stopped_by_give_up = False
            for slot in range(block_size):
                bucket = row_buckets[slot]
                if block_index == 0 and slot == 0:
                    slot_widths[slot] = min(beam_width, max_branch)
                    continue
                width = min(branch_factors[bucket], max_branch)
                if bucket == 3 and width == 0:
                    stopped_by_give_up = True
                    break
                slot_widths[slot] = width

            # 一次 block 的 top-1 位置按顺序连接为主链。
            for slot in range(block_size):
                bucket = row_buckets[slot]
                slot_parents.append(previous)
                running_score += row_top_log_probs[slot][0]
                node_index = add_node(
                    token_id=row_top_ids[slot][0],
                    parent=previous,
                    cumulative_log_probability=running_score,
                    block_index=block_index,
                    slot_index=slot,
                    rank_bucket=bucket,
                )
                greedy_nodes.append(node_index)
                previous = node_index

            # 每个位置剩余 top-b token 都与该位置的 greedy token 互为兄弟。
            nodes = tree.nodes
            for slot in range(block_size):
                bucket = row_buckets[slot]
                width = slot_widths[slot]
                if width <= 0:
                    continue
                parent = slot_parents[slot]
                parent_score = (
                    start.cumulative_log_probability
                    if parent < 0
                    else nodes[parent].cumulative_log_probability
                )
                hidden = row_hidden[slot].detach()

                if collect_pending and width > 1:
                    greedy = greedy_nodes[slot]
                    all_next.append(
                        _PendingStart(
                            node_index=greedy,
                            token_id=nodes[greedy].token_id,
                            cumulative_log_probability=nodes[greedy].cumulative_log_probability,
                            hidden=hidden,
                        )
                    )
                for alternative in range(1, width):
                    score = parent_score + row_top_log_probs[slot][alternative]
                    alt_node = add_node(
                        token_id=row_top_ids[slot][alternative],
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
                                token_id=nodes[alt_node].token_id,
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
                        token_id=nodes[last].token_id,
                        cumulative_log_probability=nodes[last].cumulative_log_probability,
                        hidden=row_hidden[-1].detach(),
                    )
                )

        limit = pending_limit if pending_limit is not None else len(all_next)
        return self._deduplicate_and_limit(all_next, limit)

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
