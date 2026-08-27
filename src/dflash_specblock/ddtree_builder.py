"""DDTree（Diffusion Draft Tree）的 best-first 草稿树构建器。

论文与官方实现：

- DDTree paper：<https://arxiv.org/abs/2604.12989>
- DDTree code：<https://github.com/liranringel/ddtree>
  （固定 commit ``c96427a185677bf4133ed865dd1626a5041aef9b``，见
  ``third_party/OFFICIAL_SOURCES.md``）

与 ``tree.py`` 中的 SpecBlock 构建器的本质差异
------------------------------------------------

SpecBlock 是「局部宽度分配」：先连出 greedy 主链，再由 rank head 预测的 bucket 决定
*每个 slot* 展开多少兄弟，并可跨 block 继续扩展。宽度决策是逐位置、局部的。

DDTree 是「全局最优预算分配」：把 DFlash 一次 block forward 得到的 ``K`` 个 slot 分布
视作条件独立，于是任一候选序列的对数联合概率就是各 slot log-prob 之和。在这个可加
性下，用一个全局最大堆按累计 log-prob 依次弹出节点，等价于在给定节点预算内枚举出
概率最高的候选前缀集合——不需要 rank head，也不需要跨 block continuation。

官方 ``build_ddtree_tree`` 的两条推入规则保证了堆的正确性：

1. **sibling**：弹出 ``(depth, rank)`` 后推入 ``(depth, rank+1)``，父节点不变。同一 slot
   的 topk 按 log-prob 降序，因此 ``rank+1`` 的分数必然不高于 ``rank``。
2. **child**：弹出节点后推入它的 ``(depth+1, rank=0)`` 子节点。

任一未生成节点的分数都不高于其「已生成的前驱」（父节点或前一个兄弟）的分数，所以堆顶
始终是全局最优的未生成节点，最先弹出的 ``budget`` 个节点就是分数最高的 ``budget`` 个
候选前缀。这也解释了为什么树的节点预算可以在构建过程中直接截断，而不像 SpecBlock 那样
需要事后 ``prune``。

本模块产出与 ``tree.py`` 完全相同的 :class:`~dflash_specblock.tree.DraftTree`，因此
``verification.py`` 的 ancestor-only 4D mask、最长 greedy 接受路径与 KV cache 压缩可以
原样复用，无需为 DDTree 另写一套验证器。等价性见 ``tests/test_ddtree_builder.py``。
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np
import torch

from .tree import BlockProposal, DraftTree


@dataclass(frozen=True, slots=True)
class BudgetDecision:
    """一次 latency-aware DDTree 的预算决策诊断。"""

    budget: int
    expected_draft_tokens: float
    predicted_round_ms: float | None
    predicted_tokens_per_ms: float | None


def _rank_bucket(rank: int) -> int:
    """把 topk rank 映射到 SpecBlock 的四个 bucket，仅作为可观测的诊断元数据。

    DDTree 不使用 rank head，这个字段不参与任何决策；保留它是为了让 DDTree 与
    SpecBlock 产出的 ``DraftTree`` 拥有一致的字段语义，便于共用统计与调试工具。
    """
    if rank == 0:
        return 0
    if rank <= 3:
        return 1
    if rank <= 9:
        return 2
    return 3


class DDTreeBuilder:
    """按 DDTree 官方 best-first 规则，从一次 DFlash block forward 构建草稿树。

    ``requires_rank = False`` 让 :class:`~dflash_specblock.engine.DFlashSpecBlockEngine`
    跳过 rank head 前向与 top-20 摘要：DDTree 的宽度分配完全由 log-prob 决定，多算的
    rank logits 不会被读取。

    Parameters
    ----------
    block_size:
        本工程口径的 ``K``，即 DFlash 一次 block forward 预测的未来位置数
        （官方 DDTree 的 ``block_size`` 含 anchor，等于 ``K + 1``）。它同时是树的
        深度上限，因为深度 ``d`` 的节点取自第 ``d - 1`` 个 slot 的分布。
    tree_budget:
        anchor 之外的最大节点数，与官方 ``tree_budget`` 同义。
    reserve_greedy_chain:
        默认 ``False``，即严格复现官方行为。置为 ``True`` 时启用本工程的可选改动，
        详见 :meth:`build_from_logits`。
    """

    requires_rank = False
    manages_budget = False

    def __init__(
        self,
        block_size: int,
        tree_budget: int,
        reserve_greedy_chain: bool = False,
    ) -> None:
        self.block_size = int(block_size)
        self.tree_budget = int(tree_budget)
        self.reserve_greedy_chain = bool(reserve_greedy_chain)
        if self.block_size < 1:
            raise ValueError("block_size 必须为正整数")
        if self.tree_budget < 1:
            raise ValueError("tree_budget 必须为正整数")
        # 复用 page-locked host 缓冲区：每轮的 topk 形状固定为 [K, topk]，
        # 两个异步 D2H copy 共享一个同步点。官方每轮都新分配 host 张量并隐式同步。
        self._host_signature: tuple[tuple[tuple[int, ...], torch.dtype], ...] | None = None
        self._host_buffers: tuple[torch.Tensor, torch.Tensor] | None = None
        # 最近一次枚举的 slot top-k。CROF 在同一轮直接读取这份摘要做跨轮对齐，
        # 从而不为共识门控重复启动 top-k / logsumexp kernel。
        self.last_top_log_probs: np.ndarray | None = None
        self.last_top_token_ids: np.ndarray | None = None

    def _to_host(
        self,
        top_log_probs: torch.Tensor,
        top_token_ids: torch.Tensor,
    ) -> tuple[np.ndarray, np.ndarray]:
        """把 topk 元数据搬到 host，返回可直接索引的 numpy 视图。

        堆的比较、父子关系和 visibility 都是 Python/numpy 控制流，必须在 host 上进行。
        CUDA 路径复用 pinned 缓冲区，把两次 copy 排在同一个 stream 上，只付一次同步。
        """
        tensors = (top_log_probs, top_token_ids)
        if top_log_probs.device.type == "cuda":
            if top_token_ids.device != top_log_probs.device:
                raise ValueError("DDTree topk 元数据必须位于同一 CUDA device")
            signature = tuple((tuple(t.shape), t.dtype) for t in tensors)
            if self._host_buffers is None or signature != self._host_signature:
                self._host_buffers = tuple(
                    torch.empty(t.shape, dtype=t.dtype, device="cpu", pin_memory=True)
                    for t in tensors
                )
                self._host_signature = signature
            for host, source in zip(self._host_buffers, tensors):
                host.copy_(source, non_blocking=True)
            # 两个 copy 都排在当前 stream 上，一次等待即可；后续 numpy 访问不会再触发同步。
            torch.cuda.current_stream(top_log_probs.device).synchronize()
            log_probs_host, token_ids_host = self._host_buffers
        else:
            log_probs_host, token_ids_host = (t.detach().cpu() for t in tensors)
        return log_probs_host.numpy(), token_ids_host.numpy()

    def build_from_logits(
        self,
        draft_logits: torch.Tensor,
        budget: int | None = None,
    ) -> DraftTree:
        """从 ``[K, V]`` draft logits 构建 DDTree。

        ``draft_logits[d]`` 是第 ``d`` 个未来位置的分布，对应树深度 ``d + 1``。

        ``reserve_greedy_chain=True`` 时先无条件保留 all-rank-0 的 greedy 链，再把剩余
        预算交给同样的 best-first 分配。这不改变 sibling/child 的推入规则，只是把 greedy
        链的弹出顺序提前，因此仍然是一棵合法的 DDTree；动机与理论依据见
        ``docs/METHOD.md``。默认关闭以保持与官方逐节点一致。
        """
        if draft_logits.ndim != 2:
            raise ValueError("draft_logits 必须是 [K, V]")

        effective_budget = self.tree_budget if budget is None else int(budget)
        if effective_budget < 0:
            raise ValueError("budget 不能为负数")

        depth_limit = min(int(draft_logits.shape[0]), self.block_size)
        vocabulary = int(draft_logits.shape[-1])
        if effective_budget == 0 or depth_limit == 0:
            return DraftTree()

        topk = min(effective_budget, vocabulary)

        # 与官方一致地在 FP32 上做 topk 与 logsumexp。这里的张量是 [K, V]（K 通常为 15），
        # FP32 副本只有几 MB，对运行 4B 目标模型的显存无实质影响；而堆的跨深度比较会把
        # 不同 slot 的 log_z 相加，用低精度归一化常数可能扰动近似并列节点的顺序，
        # 因此这份开销换取的是与官方逐节点一致的树。
        logits = draft_logits.float()
        top_values, top_token_ids = torch.topk(logits, k=topk, dim=-1)
        log_z = torch.logsumexp(logits, dim=-1, keepdim=True)
        top_log_probs_np, top_token_ids_np = self._to_host(
            top_values - log_z, top_token_ids.to(torch.int64)
        )
        self.last_top_log_probs = top_log_probs_np
        self.last_top_token_ids = top_token_ids_np

        # 索引 0 是 anchor（已验证的当前 token），节点从 1 开始，与官方保持一致；
        # 转成 DraftTree 时再整体左移一位，把 anchor 表示为 parent = -1。
        node_token_ids = np.empty(effective_budget, dtype=np.int64)
        node_depths = np.empty(effective_budget, dtype=np.int64)
        node_ranks = np.empty(effective_budget, dtype=np.int64)
        node_scores = np.empty(effective_budget, dtype=np.float64)
        parents = np.empty(effective_budget + 1, dtype=np.int64)
        parents[0] = -1
        node_count = 0

        # 堆元素与官方完全一致：(-logw, ranks, parent_index, depth, rank, logw)。
        # ``ranks`` 元组既是路径标识，也是分数并列时的确定性 tiebreak。
        heap: list[tuple[float, tuple[int, ...], int, int, int, float]] = []

        def push_sibling(ranks: tuple[int, ...], parent: int, depth: int, rank: int, logw: float) -> None:
            if rank + 1 >= topk:
                return
            sibling_logw = (
                logw
                - float(top_log_probs_np[depth - 1, rank])
                + float(top_log_probs_np[depth - 1, rank + 1])
            )
            heapq.heappush(
                heap,
                (-sibling_logw, ranks[:-1] + (rank + 1,), parent, depth, rank + 1, sibling_logw),
            )

        def push_child(ranks: tuple[int, ...], node_index: int, depth: int, logw: float) -> None:
            if depth >= depth_limit:
                return
            child_logw = logw + float(top_log_probs_np[depth, 0])
            heapq.heappush(
                heap, (-child_logw, ranks + (0,), node_index, depth + 1, 0, child_logw)
            )

        def emit(token_id: int, parent: int, depth: int, rank: int, logw: float) -> int:
            nonlocal node_count
            index = node_count + 1
            node_token_ids[node_count] = token_id
            node_depths[node_count] = depth
            node_ranks[node_count] = rank
            node_scores[node_count] = logw
            parents[index] = parent
            node_count += 1
            return index

        if self.reserve_greedy_chain:
            # 先铺满 greedy 链（每层 rank 0），再按官方规则补充剩余预算。sibling/child
            # 的推入规则不变，等价于把 greedy 链的弹出顺序提前到堆序之前。
            chain_length = min(depth_limit, effective_budget)
            parent = 0
            logw = 0.0
            for depth in range(1, chain_length + 1):
                logw += float(top_log_probs_np[depth - 1, 0])
                ranks = (0,) * depth
                node_index = emit(
                    token_id=int(top_token_ids_np[depth - 1, 0]),
                    parent=parent,
                    depth=depth,
                    rank=0,
                    logw=logw,
                )
                push_sibling(ranks, parent, depth, 0, logw)
                if depth == chain_length:
                    push_child(ranks, node_index, depth, logw)
                parent = node_index
        else:
            first_logw = float(top_log_probs_np[0, 0])
            heap.append((-first_logw, (0,), 0, 1, 0, first_logw))

        while heap and node_count < effective_budget:
            _, ranks, parent_index, depth, rank, logw = heapq.heappop(heap)
            node_index = emit(
                token_id=int(top_token_ids_np[depth - 1, rank]),
                parent=parent_index,
                depth=depth,
                rank=rank,
                logw=logw,
            )
            push_sibling(ranks, parent_index, depth, rank, logw)
            push_child(ranks, node_index, depth, logw)

        selected_count = self._select_node_count(node_scores[:node_count])
        if not 0 <= selected_count <= node_count:
            raise AssertionError(
                f"预算策略返回非法节点数 {selected_count}，可用范围是 [0, {node_count}]"
            )
        node_count = selected_count

        return self._to_draft_tree(
            node_token_ids[:node_count],
            node_depths[:node_count],
            node_ranks[:node_count],
            node_scores[:node_count],
            parents[: node_count + 1],
        )

    def _select_node_count(self, node_scores: np.ndarray) -> int:
        """从 best-first 前缀中选择要验证的节点数；基础 DDTree 保留全部节点。"""
        return int(node_scores.shape[0])

    @staticmethod
    def _to_draft_tree(
        node_token_ids: np.ndarray,
        node_depths: np.ndarray,
        node_ranks: np.ndarray,
        node_scores: np.ndarray,
        parents: np.ndarray,
    ) -> DraftTree:
        """把官方索引口径（0 = anchor）转换为 ``DraftTree`` 口径（parent = -1 表示 anchor）。"""
        node_count = int(node_token_ids.shape[0])
        tree = DraftTree()
        if node_count == 0:
            return tree

        # greedy 链只作为诊断标记：DDTree 在构建期就截断预算，不会调用 prune，
        # 因此这个字段不影响任何行为。
        is_main_chain = np.zeros(node_count, dtype=np.bool_)
        for index in range(node_count):
            parent = int(parents[index + 1])
            if node_ranks[index] != 0:
                continue
            if parent == 0 or bool(is_main_chain[parent - 1]):
                is_main_chain[index] = True

        for index in range(node_count):
            tree.add_node(
                token_id=int(node_token_ids[index]),
                parent=int(parents[index + 1]) - 1,
                cumulative_log_probability=float(node_scores[index]),
                block_index=0,
                slot_index=int(node_depths[index]) - 1,
                rank_bucket=_rank_bucket(int(node_ranks[index])),
                is_main_chain=bool(is_main_chain[index]),
            )

        # 用官方的「父行继承」递推一次性算出祖先可见性矩阵，并直接注入 DraftTree 缓存。
        # 递推按行复用父节点已算好的结果，是 O(N^2) 的 numpy 行拷贝；DraftTree 默认的
        # 逐节点上溯父链会为每个 (node, ancestor) 对做一次 Python 级张量赋值。两者结果
        # 相同（见 tests/test_ddtree_builder.py），这里省掉的是验证阶段的重复计算。
        visibility = np.zeros((node_count + 1, node_count + 1), dtype=np.bool_)
        visibility[0, 0] = True
        for index in range(1, node_count + 1):
            parent = int(parents[index])
            visibility[index, :index] = visibility[parent, :index]
            visibility[index, index] = True
        tree.preset_ancestor_mask(torch.from_numpy(visibility[1:, 1:].copy()))

        tree.validate()
        return tree

    def build(
        self,
        first_block: BlockProposal,
        expand: object | None = None,
        budget: int | None = None,
    ) -> DraftTree:
        """与 :meth:`SpecBlockTreeBuilder.build` 同签名的适配入口。

        DDTree 是单块方法：整棵树来自同一次 DFlash block forward，因此 ``expand``
        （SpecBlock 的跨块 continuation 回调）不会被调用。
        """
        del expand
        first_block.validate()
        if first_block.logits.shape[0] != 1:
            raise ValueError("DDTree 只从单个已验证 anchor 出发构建")
        return self.build_from_logits(first_block.logits[0], budget=budget)


class LatencyAwareDDTreeBuilder(DDTreeBuilder):
    """按当前 proposal 质量和实测 GPU 延迟选择 DDTree 节点预算。

    DDTree 的 best-first 输出有一个关键性质：前 ``b`` 个节点对任意预算 ``b`` 都是合法且
    在 factorized draft surrogate 下最优的树。因此可以一次枚举到最大预算，再从若干嵌套
    前缀中选择预计吞吐最高者，无需第二次 draft forward。

    对候选预算 ``b``，当前 block 的期望 draft 接受数是前 ``b`` 个节点概率质量之和。
    本类用在线观测校准这份 surrogate，并用每个预算在当前 GPU 上的 verify EWMA 延迟估计
    ``expected committed tokens / round latency``。首次运行只需各预算少量 warmup；之后主要
    走预测最优预算，并按固定间隔重新探索，适应上下文长度与运行环境变化。
    """

    manages_budget = True

    def __init__(
        self,
        block_size: int,
        tree_budget: int,
        budget_candidates: tuple[int, ...],
        initial_budget: int,
        warmup_rounds_per_budget: int = 1,
        ewma_alpha: float = 0.2,
        exploration_interval: int = 64,
    ) -> None:
        super().__init__(
            block_size=block_size,
            tree_budget=tree_budget,
            reserve_greedy_chain=False,
        )
        candidates = tuple(sorted({int(value) for value in budget_candidates}))
        if not candidates:
            raise ValueError("budget_candidates 不能为空")
        if candidates[0] < self.block_size:
            raise ValueError("每个候选预算都必须不少于 block_size")
        if candidates[-1] > self.tree_budget:
            raise ValueError("候选预算不能超过 tree_budget")
        if int(initial_budget) not in candidates:
            raise ValueError("initial_budget 必须包含在 budget_candidates 中")
        if int(warmup_rounds_per_budget) < 1:
            raise ValueError("warmup_rounds_per_budget 必须至少为 1")
        if not 0.0 < float(ewma_alpha) <= 1.0:
            raise ValueError("ewma_alpha 必须在 (0, 1] 内")
        if int(exploration_interval) < 0:
            raise ValueError("exploration_interval 不能为负数")

        self.budget_candidates = candidates
        self.initial_budget = int(initial_budget)
        self.warmup_rounds_per_budget = int(warmup_rounds_per_budget)
        self.ewma_alpha = float(ewma_alpha)
        self.exploration_interval = int(exploration_interval)

        # 先复测已有的 60-node 基线，再按与它的距离逐步探索，避免冷启动第一轮直接使用
        # 最大树。相同距离时优先较小预算。
        self._warmup_order = tuple(
            sorted(candidates, key=lambda value: (value != self.initial_budget, abs(value - self.initial_budget), value))
        )
        self._observations = {budget: 0 for budget in candidates}
        self._verify_ms: dict[int, float] = {}
        self._fixed_ms: float | None = None
        self._acceptance_scale = 1.0
        self._decision_count = 0
        self._last_mass_by_budget: dict[int, float] = {}
        self._last_selected_budget: int | None = None
        self._last_expected_draft_tokens: float | None = None
        self.last_decision: BudgetDecision | None = None

    @staticmethod
    def _ewma(previous: float | None, value: float, alpha: float) -> float:
        return value if previous is None else (1.0 - alpha) * previous + alpha * value

    def _next_warmup_budget(self, available: tuple[int, ...]) -> int | None:
        for budget in self._warmup_order:
            if budget in available and self._observations[budget] < self.warmup_rounds_per_budget:
                return budget
        return None

    def _select_node_count(self, node_scores: np.ndarray) -> int:
        node_count = int(node_scores.shape[0])
        available = tuple(value for value in self.budget_candidates if value <= node_count)
        if not available:
            return node_count

        # 节点按 prefix probability 递减产生；累加 exp(log q(prefix)) 正是 DDTree
        # surrogate 下的期望 draft 接受数。clip 只防止极小概率下溢，不改变有效项。
        probability_mass = np.exp(np.clip(node_scores.astype(np.float64), -745.0, 0.0))
        cumulative_mass = np.cumsum(probability_mass)
        mass_by_budget = {budget: float(cumulative_mass[budget - 1]) for budget in available}
        self._last_mass_by_budget = mass_by_budget

        selected = self._next_warmup_budget(available)
        if selected is None and self.exploration_interval > 0:
            if self._decision_count > 0 and self._decision_count % self.exploration_interval == 0:
                selected = min(available, key=lambda value: (self._observations[value], value))

        utility: float | None = None
        predicted_ms: float | None = None
        if selected is None:
            measured = tuple(value for value in available if value in self._verify_ms)
            if not measured or self._fixed_ms is None:
                selected = self.initial_budget if self.initial_budget in available else available[0]
            else:
                def score(budget: int) -> tuple[float, int]:
                    expected_draft = min(
                        float(self.block_size),
                        self._acceptance_scale * mass_by_budget[budget],
                    )
                    round_ms = self._fixed_ms + self._verify_ms[budget]
                    value = (1.0 + expected_draft) / max(round_ms, 1e-6)
                    return value, -budget

                selected = max(measured, key=score)
                utility = score(selected)[0]
                predicted_ms = self._fixed_ms + self._verify_ms[selected]

        expected = min(
            float(self.block_size),
            self._acceptance_scale * mass_by_budget[selected],
        )
        if predicted_ms is None and self._fixed_ms is not None and selected in self._verify_ms:
            predicted_ms = self._fixed_ms + self._verify_ms[selected]
            utility = (1.0 + expected) / max(predicted_ms, 1e-6)

        self._last_selected_budget = selected
        self._last_expected_draft_tokens = mass_by_budget[selected]
        self._decision_count += 1
        self.last_decision = BudgetDecision(
            budget=selected,
            expected_draft_tokens=expected,
            predicted_round_ms=predicted_ms,
            predicted_tokens_per_ms=utility,
        )
        return selected

    def select_budget_from_scores(self, node_scores: np.ndarray) -> int:
        """只运行预算策略，不重新枚举 DDTree。

        CROF 已经为了共识/分歧分析枚举了最大 DDTree；无重叠或低置信时通过本入口
        复用原 latency-aware 回退策略，避免第二次处理同一份 draft logits。
        """
        return self._select_node_count(node_scores)

    def observe(
        self,
        *,
        tree_nodes: int,
        draft_ms: float,
        verify_ms: float,
        accepted_draft_tokens: int,
    ) -> None:
        """把一轮真实延迟和接受长度反馈给在线预算策略。"""
        budget = self._last_selected_budget
        expected = self._last_expected_draft_tokens
        if budget is None or expected is None:
            return
        if int(tree_nodes) != budget:
            raise ValueError(
                f"观测树节点数 {tree_nodes} 与最近预算决策 {budget} 不一致"
            )
        self._observations[budget] += 1
        self._fixed_ms = self._ewma(self._fixed_ms, max(float(draft_ms), 0.0), self.ewma_alpha)
        self._verify_ms[budget] = self._ewma(
            self._verify_ms.get(budget), max(float(verify_ms), 0.0), self.ewma_alpha
        )
        if expected > 1e-9:
            ratio = max(0.0, min(2.0, float(accepted_draft_tokens) / expected))
            self._acceptance_scale = self._ewma(
                self._acceptance_scale, ratio, self.ewma_alpha
            )
