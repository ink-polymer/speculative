"""CROF：复用上一轮残余预测的跨轮重叠共识森林。

本模块不执行额外 draft forward。每轮仍先由 DFlash 产生一次 ``[K,V]`` proposal，
随后把上一轮尚未越过的 slot top-1 与当前 slot top-1 按绝对生成位置对齐：

* 连续高共识：显式保留当前 greedy 深走廊，并用 30/45 一类小预算验证；
* 首次高置信分歧：同时保留当前走廊和“旧 token + 当前后缀”修复走廊；
* 无重叠、双视角低置信或旧视角在线命中率过低：回退 latency-aware DDTree。

森林里的每个 token 都只是候选，仍由 target 的 ancestor-only forward 验证，因此不会
改变 temperature=0 greedy 输出。跨轮历史只改变候选拓扑与节点预算。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .ddtree_builder import DDTreeBuilder, LatencyAwareDDTreeBuilder
from .tree import BlockProposal, DraftTree


@dataclass(frozen=True, slots=True)
class CROFDecision:
    """最近一轮 CROF 门控与预算诊断。depth 均为 0-based slot 深度。"""

    mode: str
    budget: int
    overlap_slots: int
    consensus_prefix: int
    divergence_depth: int | None
    new_hit_rate: float
    old_hit_rate: float
    consensus_hit_rate: float


@dataclass(slots=True)
class _Calibration:
    prior: float
    hits: int = 0
    total: int = 0

    @property
    def rate(self) -> float:
        # 对称 Beta 先验，冷启动为 0.5；prior 表示等效总样本数。
        return (self.hits + self.prior * 0.5) / (self.total + self.prior)

    def update(self, hit: bool) -> None:
        self.hits += int(bool(hit))
        self.total += 1


@dataclass(frozen=True, slots=True)
class _PredictionView:
    token_ids: tuple[int, ...]
    margins: tuple[float, ...]
    advance: int = 0


@dataclass(frozen=True, slots=True)
class _Alignment:
    old_token_ids: tuple[int, ...]
    old_margins: tuple[float, ...]
    overlap_slots: int
    consensus_prefix: int
    divergence_depth: int | None


class CrossRoundConsensusForestBuilder:
    """基于跨轮预测重叠选择深走廊、双走廊或 DDTree 回退。"""

    requires_rank = False
    manages_budget = True

    def __init__(
        self,
        *,
        block_size: int,
        tree_budget: int,
        budget_candidates: tuple[int, ...],
        initial_budget: int,
        warmup_rounds_per_budget: int = 1,
        ewma_alpha: float = 0.2,
        exploration_interval: int = 64,
        consensus_budgets: tuple[int, int] = (30, 45),
        repair_budgets: tuple[int, int, int] = (60, 80, 100),
        min_consensus_slots: int = 2,
        confidence_margin: float = 1.0,
        old_min_hit_rate: float = 0.2,
        calibration_prior: float = 2.0,
    ) -> None:
        self.block_size = int(block_size)
        self.tree_budget = int(tree_budget)
        self.consensus_budgets = tuple(int(value) for value in consensus_budgets)
        self.repair_budgets = tuple(int(value) for value in repair_budgets)
        self.min_consensus_slots = int(min_consensus_slots)
        self.confidence_margin = float(confidence_margin)
        self.old_min_hit_rate = float(old_min_hit_rate)
        self.calibration_prior = float(calibration_prior)

        if len(self.consensus_budgets) != 2:
            raise ValueError("consensus_budgets 必须恰好包含 2 个预算")
        if len(self.repair_budgets) != 3:
            raise ValueError("repair_budgets 必须恰好包含 3 个预算")
        gated = self.consensus_budgets + self.repair_budgets
        if any(value < self.block_size or value > self.tree_budget for value in gated):
            raise ValueError("CROF 门控预算必须位于 [block_size, tree_budget]")
        if tuple(sorted(self.consensus_budgets)) != self.consensus_budgets:
            raise ValueError("consensus_budgets 必须递增")
        if tuple(sorted(self.repair_budgets)) != self.repair_budgets:
            raise ValueError("repair_budgets 必须递增")
        if self.min_consensus_slots < 1:
            raise ValueError("min_consensus_slots 必须为正整数")
        if self.confidence_margin < 0.0:
            raise ValueError("confidence_margin 不能为负数")
        if not 0.0 <= self.old_min_hit_rate <= 1.0:
            raise ValueError("old_min_hit_rate 必须位于 [0,1]")
        if self.calibration_prior <= 0.0:
            raise ValueError("calibration_prior 必须为正数")

        # 最大树只枚举一次：其 top-k 摘要同时服务 DDTree 与 CROF 门控。
        self._enumerator = DDTreeBuilder(
            block_size=self.block_size,
            tree_budget=self.tree_budget,
        )
        self._fallback_policy = LatencyAwareDDTreeBuilder(
            block_size=self.block_size,
            tree_budget=self.tree_budget,
            budget_candidates=budget_candidates,
            initial_budget=initial_budget,
            warmup_rounds_per_budget=warmup_rounds_per_budget,
            ewma_alpha=ewma_alpha,
            exploration_interval=exploration_interval,
        )
        self._new_calibration = _Calibration(self.calibration_prior)
        self._old_calibration = _Calibration(self.calibration_prior)
        self._consensus_calibration = _Calibration(self.calibration_prior)
        self._mode_counts = {"fallback": 0, "consensus": 0, "repair": 0}
        self._previous: _PredictionView | None = None
        self._current: _PredictionView | None = None
        self._last_alignment: _Alignment | None = None
        self._last_used_fallback = False
        self.last_decision: CROFDecision | None = None

    def reset_generation(self) -> None:
        """清除只属于单条生成序列的历史；硬件延迟与命中率校准跨样本保留。"""
        self._previous = None
        self._current = None
        self._last_alignment = None
        self._last_used_fallback = False
        self.last_decision = None

    @staticmethod
    def _prediction_view(
        top_log_probs: np.ndarray,
        top_token_ids: np.ndarray,
    ) -> _PredictionView:
        token_ids = tuple(int(value) for value in top_token_ids[:, 0])
        if top_log_probs.shape[1] >= 2:
            margins = tuple(
                float(value)
                for value in top_log_probs[:, 0] - top_log_probs[:, 1]
            )
        else:
            margins = tuple(float("inf") for _ in token_ids)
        return _PredictionView(token_ids=token_ids, margins=margins)

    def _align(self, current: _PredictionView) -> _Alignment | None:
        previous = self._previous
        if previous is None or previous.advance >= len(previous.token_ids):
            return None
        overlap = min(
            len(current.token_ids),
            len(previous.token_ids) - previous.advance,
        )
        if overlap <= 0:
            return None
        old_tokens = previous.token_ids[previous.advance : previous.advance + overlap]
        old_margins = previous.margins[previous.advance : previous.advance + overlap]
        consensus = 0
        while consensus < overlap and old_tokens[consensus] == current.token_ids[consensus]:
            consensus += 1
        divergence = consensus if consensus < overlap else None
        return _Alignment(
            old_token_ids=old_tokens,
            old_margins=old_margins,
            overlap_slots=overlap,
            consensus_prefix=consensus,
            divergence_depth=divergence,
        )

    def _choose_mode(self, current: _PredictionView, alignment: _Alignment | None) -> str:
        if alignment is None:
            return "fallback"
        divergence = alignment.divergence_depth
        if divergence is None:
            agreed = alignment.consensus_prefix
            if agreed < self.min_consensus_slots:
                return "fallback"
            joint_margin = min(
                min(current.margins[:agreed]),
                min(alignment.old_margins[:agreed]),
            )
            return "consensus" if joint_margin >= self.confidence_margin else "fallback"

        # 首次分歧处两个视角都平坦时，不为噪声建立修复分支。
        if max(current.margins[divergence], alignment.old_margins[divergence]) < self.confidence_margin:
            return "fallback"
        # 累积到足够在线证据后，淘汰长期无效的旧视角；冷启动仍允许探索。
        if (
            self._old_calibration.total >= 4
            and self._old_calibration.rate < self.old_min_hit_rate
        ):
            return "fallback"
        return "repair"

    def _consensus_budget(self, alignment: _Alignment) -> int:
        small, large = self.consensus_budgets
        strong_prefix = max(4, self.min_consensus_slots + 1)
        if (
            alignment.consensus_prefix >= strong_prefix
            and self._consensus_calibration.rate >= 0.6
        ):
            return small
        return large

    def _repair_budget(self, alignment: _Alignment) -> int:
        small, medium, large = self.repair_budgets
        divergence = int(alignment.divergence_depth or 0)
        old_rate = self._old_calibration.rate
        if divergence >= 4 and old_rate >= 0.35:
            return small
        if divergence >= 2 or old_rate >= 0.25:
            return medium
        return large

    @staticmethod
    def _prefix_tree(full_tree: DraftTree, budget: int) -> DraftTree:
        tree = DraftTree(full_tree.nodes[: int(budget)])
        tree.validate(int(budget))
        return tree

    @staticmethod
    def _slot_log_probability(
        token_id: int,
        depth: int,
        top_log_probs: np.ndarray,
        top_token_ids: np.ndarray,
    ) -> float:
        matches = np.flatnonzero(top_token_ids[depth] == int(token_id))
        if matches.size:
            return float(top_log_probs[depth, int(matches[0])])
        # 极少数旧 token 不在当前 top-B；该值只用于诊断排序，不参与 target 验证。
        return float(top_log_probs[depth, -1]) - 1.0

    def _merge_forest(
        self,
        full_tree: DraftTree,
        current: _PredictionView,
        alignment: _Alignment,
        mode: str,
        budget: int,
        top_log_probs: np.ndarray,
        top_token_ids: np.ndarray,
    ) -> DraftTree:
        tree = DraftTree()
        child_by_key: dict[tuple[int, int], int] = {}

        def add_path(tokens: tuple[int, ...], *, main_chain: bool) -> None:
            parent = -1
            score = 0.0
            for depth, token_id in enumerate(tokens):
                score += self._slot_log_probability(
                    token_id, depth, top_log_probs, top_token_ids
                )
                key = (parent, int(token_id))
                node_index = child_by_key.get(key)
                if node_index is None:
                    if len(tree) >= budget:
                        return
                    node_index = tree.add_node(
                        token_id=token_id,
                        parent=parent,
                        cumulative_log_probability=score,
                        block_index=0,
                        slot_index=depth,
                        rank_bucket=0 if main_chain else 3,
                        is_main_chain=main_chain,
                    )
                    child_by_key[key] = node_index
                elif main_chain:
                    tree.nodes[node_index].is_main_chain = True
                parent = node_index

        # 小预算也无条件保留 K 深的当前走廊。
        add_path(current.token_ids, main_chain=True)
        if mode == "repair":
            divergence = alignment.divergence_depth
            if divergence is None:
                raise AssertionError("repair 模式必须存在首次分歧")
            repair_path = (
                current.token_ids[:divergence]
                + (alignment.old_token_ids[divergence],)
                + current.token_ids[divergence + 1 :]
            )
            add_path(repair_path, main_chain=False)

        # 按原 DDTree best-first 次序填满剩余预算；mandatory 路径与已有节点去重。
        source_to_merged: dict[int, int] = {}
        for source_index, source in enumerate(full_tree.nodes):
            parent = -1 if source.parent < 0 else source_to_merged[source.parent]
            key = (parent, source.token_id)
            merged_index = child_by_key.get(key)
            if merged_index is None:
                if len(tree) >= budget:
                    break
                merged_index = tree.add_node(
                    token_id=source.token_id,
                    parent=parent,
                    cumulative_log_probability=source.cumulative_log_probability,
                    block_index=source.block_index,
                    slot_index=source.slot_index,
                    rank_bucket=source.rank_bucket,
                    is_main_chain=source.is_main_chain,
                )
                child_by_key[key] = merged_index
            source_to_merged[source_index] = merged_index

        tree.validate(budget)
        return tree

    def build(
        self,
        first_block: BlockProposal,
        expand: object | None = None,
        budget: int | None = None,
    ) -> DraftTree:
        del expand
        first_block.validate()
        if first_block.logits.shape[0] != 1:
            raise ValueError("CROF 只支持 batch=1 的单 anchor proposal")
        max_budget = self.tree_budget if budget is None else min(int(budget), self.tree_budget)
        if max_budget < self.block_size:
            raise ValueError("CROF 枚举预算不能小于 block_size")

        full_tree = self._enumerator.build_from_logits(
            first_block.logits[0], budget=max_budget
        )
        top_log_probs = self._enumerator.last_top_log_probs
        top_token_ids = self._enumerator.last_top_token_ids
        if top_log_probs is None or top_token_ids is None:
            raise AssertionError("DDTree 枚举未产生 top-k 摘要")
        current = self._prediction_view(top_log_probs, top_token_ids)
        alignment = self._align(current)
        mode = self._choose_mode(current, alignment)
        scores = np.asarray(
            [node.cumulative_log_probability for node in full_tree.nodes],
            dtype=np.float64,
        )

        if mode == "fallback":
            selected_budget = self._fallback_policy.select_budget_from_scores(scores)
            tree = self._prefix_tree(full_tree, selected_budget)
            self._last_used_fallback = True
        else:
            if alignment is None:
                raise AssertionError("CROF 门控模式缺少跨轮对齐")
            selected_budget = (
                self._consensus_budget(alignment)
                if mode == "consensus"
                else self._repair_budget(alignment)
            )
            selected_budget = min(selected_budget, max_budget, len(full_tree))
            tree = self._merge_forest(
                full_tree,
                current,
                alignment,
                mode,
                selected_budget,
                top_log_probs,
                top_token_ids,
            )
            self._last_used_fallback = False

        self._current = current
        self._last_alignment = alignment
        self._mode_counts[mode] += 1
        self.last_decision = CROFDecision(
            mode=mode,
            budget=len(tree),
            overlap_slots=0 if alignment is None else alignment.overlap_slots,
            consensus_prefix=0 if alignment is None else alignment.consensus_prefix,
            divergence_depth=None if alignment is None else alignment.divergence_depth,
            new_hit_rate=self._new_calibration.rate,
            old_hit_rate=self._old_calibration.rate,
            consensus_hit_rate=self._consensus_calibration.rate,
        )
        return tree

    def observe(
        self,
        *,
        tree_nodes: int,
        draft_ms: float,
        verify_ms: float,
        accepted_draft_tokens: int,
    ) -> None:
        """保留现有 latency-aware DDTree 回退的实测延迟学习。"""
        if self._last_used_fallback:
            self._fallback_policy.observe(
                tree_nodes=tree_nodes,
                draft_ms=draft_ms,
                verify_ms=verify_ms,
                accepted_draft_tokens=accepted_draft_tokens,
            )

    def observe_verification(
        self,
        *,
        accepted_token_ids: tuple[int, ...],
        bonus_token_id: int,
    ) -> None:
        """用真实 target 前缀更新新/旧/共识视角命中率，并保存下一轮历史。"""
        current = self._current
        if current is None:
            return
        target_tokens = tuple(int(value) for value in accepted_token_ids) + (
            int(bonus_token_id),
        )
        alignment = self._last_alignment
        for depth, target_token in enumerate(target_tokens[: len(current.token_ids)]):
            new_token = current.token_ids[depth]
            self._new_calibration.update(new_token == target_token)
            if alignment is not None and depth < alignment.overlap_slots:
                old_token = alignment.old_token_ids[depth]
                self._old_calibration.update(old_token == target_token)
                if old_token == new_token:
                    self._consensus_calibration.update(new_token == target_token)

        # 前缀从旧 anchor 推进 accepted + 1（bonus 成为下一轮 anchor）。因此
        # q_new[1] 对齐 q_old[accepted+2]，即 0-based slot 偏移 accepted+1。
        self._previous = _PredictionView(
            token_ids=current.token_ids,
            margins=current.margins,
            advance=len(accepted_token_ids) + 1,
        )

    def diagnostics(self) -> dict[str, object]:
        """返回可直接写入 benchmark JSON 的累计 CROF 诊断。"""
        return {
            "mode_counts": dict(self._mode_counts),
            "new_view": {
                "hits": self._new_calibration.hits,
                "total": self._new_calibration.total,
                "posterior_hit_rate": self._new_calibration.rate,
            },
            "old_view": {
                "hits": self._old_calibration.hits,
                "total": self._old_calibration.total,
                "posterior_hit_rate": self._old_calibration.rate,
            },
            "consensus": {
                "hits": self._consensus_calibration.hits,
                "total": self._consensus_calibration.total,
                "posterior_hit_rate": self._consensus_calibration.rate,
            },
        }
