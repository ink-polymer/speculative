"""SpecBlock 风格的 block-iterative 动态草稿树。

实现遵循论文和官方代码的关键拓扑：一个 block 先形成 greedy 主链，各位置的 top-b 其余
token 作为该位置的兄弟节点；rank-guided pending 节点被批量送进下一 block。树拓扑
保留为轻量 Python host 控制流，大张量计算和验证则留在 NVIDIA GPU。
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
    top20_values: torch.Tensor | None = None  # [B, K, 20] float32, 预算好的 top-20 logit 值
    top20_ids: torch.Tensor | None = None  # [B, K, 20] int64, 预算好的 top-20 token id

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
    # P0 修复：greedy 主链节点标记为 True，prune 时无条件保留。
    # 主链是接受长度的唯一来源，且不额外占 draft 成本（top-1 已在 block
    # forward 中算出），不应参与竞争性剪枝。
    is_main_chain: bool = False


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
        self._retrieve_paths_cache: list[list[int]] | None = None
        self._ancestor_mask_cache: torch.Tensor | None = None

    def _invalidate_caches(self) -> None:
        self._parents_cache = None
        self._tokens_cache = None
        self._children_cache = None
        self._retrieve_paths_cache = None
        self._ancestor_mask_cache = None

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
        is_main_chain: bool = False,
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
                is_main_chain=bool(is_main_chain),
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

        树拓扑由 Python 在 host 上构建，且生产预算通常只有几十个节点。
        因此在 host 上一次生成并缓存 mask，需要时只做一次小 H2D copy，
        比在 GPU 上启动 ceil(log2(N)) 组小 kernel 更适合低延迟解码。
        """
        count = len(self.nodes)
        if self._ancestor_mask_cache is None:
            mask = torch.zeros((count, count), dtype=torch.bool)
            parents = self._parents()
            for node_index in range(count):
                current = node_index
                while current >= 0:
                    mask[node_index, current] = True
                    current = parents[current]
            self._ancestor_mask_cache = mask
        if device is None or torch.device(device).type == "cpu":
            return self._ancestor_mask_cache
        return self._ancestor_mask_cache.to(device=device, non_blocking=True)

    def preset_ancestor_mask(self, mask: torch.Tensor) -> None:
        """注入已在 host 上算好的祖先可见性矩阵，跳过 :meth:`ancestor_mask` 的重复推导。

        供 DDTree 构建器使用：它在建树时已经用「父行继承」递推出整张 visibility，
        再让验证阶段逐节点上溯父链重算一次纯属浪费。任何后续 ``add_node``/``prune``
        都会通过 ``_invalidate_caches`` 丢弃这份缓存，因此不存在陈旧风险。
        """
        if mask.ndim != 2 or mask.shape != (len(self.nodes), len(self.nodes)):
            raise ValueError("ancestor mask 必须是 [N, N]，N 为当前节点数")
        if mask.dtype != torch.bool:
            raise ValueError("ancestor mask 必须是 torch.bool")
        if mask.device.type != "cpu":
            raise ValueError("ancestor mask 缓存保存在 host，必须是 CPU 张量")
        self._ancestor_mask_cache = mask

    def _retrieve_paths(self) -> list[list[int]]:
        """Return padded root-to-leaf paths as host control metadata."""
        if self._retrieve_paths_cache is not None:
            return self._retrieve_paths_cache

        children = self.children()
        parents = self._parents()
        node_count = len(self.nodes)
        leaves = [index for index in range(node_count) if index not in children]
        max_depth = max((self.nodes[index].depth for index in leaves), default=0)
        paths: list[list[int]] = []
        for leaf in leaves:
            path: list[int] = []
            current = leaf
            while current >= 0:
                path.append(current)
                current = parents[current]
            path.reverse()
            path.extend([-1] * (max_depth - len(path)))
            paths.append(path)
        self._retrieve_paths_cache = paths
        return paths

    def retrieve_paths(self) -> list[list[int]]:
        """返回缓存在 host 的根到叶路径，供 Python 验证控制流使用。"""
        return self._retrieve_paths()

    def retrieve_indices(self, device: torch.device | None = None) -> torch.Tensor:
        """返回所有根到叶路径；-1 是对齐填充。"""
        paths = self._retrieve_paths()
        if not paths:
            return torch.empty((0, 0), dtype=torch.long, device=device)
        return torch.tensor(paths, dtype=torch.long, device=device)

    def token_tensor(self, device: torch.device) -> torch.Tensor:
        if self._tokens_cache is None:
            self._tokens_cache = [node.token_id for node in self.nodes]
        return torch.as_tensor(self._tokens_cache, dtype=torch.long, device=device)

    def prune(self, budget: int) -> None:
        """按累计 log-prob 选择节点，同时保证任何保留节点的整条祖先链存在。

        P0 修复：greedy 主链节点（is_main_chain=True）无条件保留，不参与竞争性
        剪枝。主链是接受长度的唯一来源，且不额外占 draft 成本（top-1 已在 block
        forward 中算出）。浅层兄弟的 cum_lp 天然高于深层主链节点，若主链参与
        竞争会被挤掉，最坏损失 18.6% 的 τ，而那 60 个节点的 verify 成本一分没省。
        """
        if len(self.nodes) <= budget:
            return

        # Step 1: 无条件保留全部主链节点及其祖先链。跨块主链的父节点可能是
        # 上一块的兄弟节点（非主链），必须一并保留，否则 compaction 后 parent
        # 变成 -1 但 depth 不变，validate 会报拓扑不一致。
        main_chain = [
            i for i, node in enumerate(self.nodes) if node.is_main_chain
        ]
        kept: set[int] = set()
        for idx in main_chain:
            current = idx
            while current >= 0 and current not in kept:
                kept.add(current)
                current = self.nodes[current].parent

        # 主链 + 祖先超出预算时只保留浅层部分（节点按索引拓扑序排列）。
        if len(kept) > budget:
            kept = set(sorted(kept)[:budget])
        # Step 2: 剩余预算竞争性分配给非主链节点。
        elif len(kept) < budget:
            non_main = [
                i for i in range(len(self.nodes)) if i not in main_chain
            ]
            order = sorted(
                non_main,
                key=lambda index: self.nodes[index].cumulative_log_probability,
                reverse=True,
            )
            for candidate in order:
                if len(kept) >= budget:
                    break
                chain: list[int] = []
                current = candidate
                while current >= 0 and current not in kept:
                    chain.append(current)
                    current = self.nodes[current].parent
                if len(kept) + len(chain) <= budget:
                    kept.update(chain)

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
                    is_main_chain=node.is_main_chain,
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
        # Reused page-locked host workspaces for the small amount of dynamic
        # tree metadata that Python must inspect.  Three D2H copies are queued
        # on one CUDA stream and paid for with a single synchronization.
        self._host_metadata_signature: tuple[tuple[tuple[int, ...], torch.dtype], ...] | None = None
        self._host_metadata_buffers: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None

    def _metadata_to_host(
        self,
        rank_buckets: torch.Tensor,
        top_ids: torch.Tensor,
        top_log_probs: torch.Tensor,
    ) -> tuple[list[list[int]], list[list[list[int]]], list[list[list[float]]]]:
        tensors = (rank_buckets, top_ids, top_log_probs)
        if rank_buckets.device.type == "cuda":
            if any(tensor.device != rank_buckets.device for tensor in tensors[1:]):
                raise ValueError("tree metadata tensors 必须位于同一 CUDA device")
            signature = tuple((tuple(tensor.shape), tensor.dtype) for tensor in tensors)
            if self._host_metadata_buffers is None or signature != self._host_metadata_signature:
                self._host_metadata_buffers = tuple(
                    torch.empty(
                        tensor.shape,
                        dtype=tensor.dtype,
                        device="cpu",
                        pin_memory=True,
                    )
                    for tensor in tensors
                )
                self._host_metadata_signature = signature

            for host, source in zip(self._host_metadata_buffers, tensors):
                host.copy_(source, non_blocking=True)
            # All copies were enqueued on the current stream, so one wait is
            # sufficient.  Calling ``tolist`` on the CPU buffers below cannot
            # trigger additional device synchronizations.
            torch.cuda.current_stream(rank_buckets.device).synchronize()
            rank_host, ids_host, probabilities_host = self._host_metadata_buffers
        else:
            rank_host, ids_host, probabilities_host = (
                tensor.detach().cpu() for tensor in tensors
            )

        return rank_host.tolist(), ids_host.tolist(), probabilities_host.tolist()

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

        # The draft logits are typically BF16/FP16 and can be [B, K, ~150K].
        # Promoting that entire tensor to FP32 creates a large transient allocation.
        # Branch scores only guide which candidates are verified (they cannot change
        # lossless target acceptance), so reduce in the model dtype and promote the
        # compact [B, K] normalizer for the subsequent host-side score bookkeeping.
        log_z = torch.logsumexp(proposal.logits, dim=-1).float()  # [B, K]

        if proposal.top20_values is not None and proposal.top20_ids is not None:
            top_values = proposal.top20_values[:, :, :max_branch]
            top_ids = proposal.top20_ids[:, :, :max_branch]
        else:
            top_values, top_ids = torch.topk(proposal.logits, k=max_branch, dim=-1)
            top_values = top_values.float()

        top_log_probs = top_values - log_z.unsqueeze(-1)  # [B, K, max_branch]
        rank_buckets = proposal.rank_logits.argmax(dim=-1)  # [B, K]

        # Python 只读取一次 host metadata。CUDA 使用 page-locked 复用缓冲区，
        # 三个异步 D2H copy 共享一个同步点；后续循环不再触发设备同步。
        rank_buckets_cpu, top_ids_cpu, top_log_probs_cpu = self._metadata_to_host(
            rank_buckets, top_ids, top_log_probs
        )

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
                    is_main_chain=True,
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

    def build(
        self,
        first_block: BlockProposal,
        expand: ExpandCallback,
        budget: int | None = None,
    ) -> DraftTree:
        """构建最多 M 个 diffusion block 的草稿树。

        ``budget`` 为 None 时使用 ``self.tree_budget``；否则用指定值（自适应预算）。
        """
        first_block.validate()
        if first_block.logits.shape[0] != 1:
            raise ValueError("第一块必须只有一个当前已验证 anchor")

        effective_budget = budget if budget is not None else self.tree_budget
        if effective_budget < self.block_size:
            raise ValueError(f"budget={effective_budget} 不能小于 block_size={self.block_size}")

        tree = DraftTree()
        hidden_size = first_block.hidden.shape[-1]
        root_start = _PendingStart(
            node_index=-1,
            token_id=-1,
            cumulative_log_probability=0.0,
            hidden=torch.empty(hidden_size, device=first_block.hidden.device),
        )
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

        tree.prune(effective_budget)
        tree.validate(effective_budget)
        return tree
