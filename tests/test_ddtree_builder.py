"""DDTree 构建器测试：拓扑不变量、预算、best-first 顺序，以及与官方实现的等价性。

等价性测试直接从 ``third_party/ddtree_official`` 导入官方 ``build_ddtree_tree``，用同一份
logits 对比逐节点的 token/parent/depth 与 visibility 矩阵。官方源码不做任何修改，其固定
commit 记录在 ``third_party/OFFICIAL_SOURCES.md``。
"""

from __future__ import annotations

import importlib.util
import itertools
import math
import sys
from pathlib import Path

import pytest
import torch

from dflash_specblock.ddtree_builder import DDTreeBuilder


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_ROOT = PROJECT_ROOT / "third_party" / "ddtree_official"


def _load_official_build_ddtree_tree():
    """只加载官方 ``ddtree.py`` 中的纯函数，绕开它对 flash-attn/loguru 的导入。

    官方 ``ddtree.py`` 顶部会 ``import loguru`` 并从 ``model`` 拉入 FlashAttention 相关
    代码，而 ``build_ddtree_tree`` 本身只依赖 heapq/numpy/torch。测试环境（CPU、无
    flash-attn）无法整模块导入，因此按源码单独编译该函数，保证比对的仍是官方逐字节源码。
    """
    source_path = OFFICIAL_ROOT / "ddtree.py"
    if not source_path.is_file():
        pytest.skip(f"官方 DDTree checkout 不存在: {source_path}")

    source = source_path.read_text(encoding="utf-8")
    marker = "def build_ddtree_tree("
    start = source.find(marker)
    if start < 0:
        pytest.skip("官方 ddtree.py 中未找到 build_ddtree_tree")
    end = source.find("\ndef compile_ddtree_tree(", start)
    if end < 0:
        pytest.skip("无法定位 build_ddtree_tree 的结束位置")

    namespace: dict[str, object] = {}
    prelude = (
        "import heapq\n"
        "import time\n"
        "import numpy as np\n"
        "import torch\n"
        "DDTREE_TREE_BUILD_STAGE_ORDER = "
        "('tree_build_copy', 'tree_build_heap', 'tree_build_visibility')\n"
        "def cuda_time():\n"
        "    return time.perf_counter()\n"
        "def empty_stage_times(names):\n"
        "    return {name: 0.0 for name in names}\n"
    )
    exec(compile(prelude + source[start:end], str(source_path), "exec"), namespace)
    return namespace["build_ddtree_tree"]


def _random_logits(depth: int, vocabulary: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(depth, vocabulary, generator=generator)


def _peaked_logits(depth: int, vocabulary: int) -> torch.Tensor:
    """构造尖峰分布，逼近真实 drafter 的 logits 形状。

    ``torch.randn`` 得到的近似均匀分布下，同一 slot 的 top-1 与 top-2 只差约 0.002 nat，
    而向下加一层要付约 3.7 nat。此时 best-first 把全部预算花在深度 1 的兄弟上是数学上
    正确的分配，测试深度相关行为必须用有明显峰值的分布。
    """
    logits = torch.full((depth, vocabulary), -12.0)
    for row in range(depth):
        logits[row, (row * 3 + 1) % vocabulary] = 8.0
        logits[row, (row * 3 + 2) % vocabulary] = 4.0
        logits[row, (row * 3 + 3) % vocabulary] = 2.0
    return logits


def _official_tree(logits: torch.Tensor, budget: int):
    build = _load_official_build_ddtree_tree()
    node_token_ids, node_depths, parents, child_maps, visibility, _ = build(logits, budget)
    return node_token_ids, node_depths, parents, child_maps, visibility


@pytest.mark.parametrize(
    ("depth", "vocabulary", "budget", "seed"),
    [
        (15, 512, 60, 0),
        (15, 512, 30, 1),
        (4, 64, 60, 2),
        (8, 128, 7, 3),
        (15, 512, 1, 4),
        (3, 32, 200, 5),
    ],
)
def test_matches_official_build_ddtree_tree(depth, vocabulary, budget, seed) -> None:
    """逐节点与官方实现完全一致：token、parent、depth 以及祖先可见性。"""
    _assert_matches_official(_random_logits(depth, vocabulary, seed), depth, budget)


@pytest.mark.parametrize("budget", [1, 7, 30, 60, 200])
def test_matches_official_on_peaked_logits(budget) -> None:
    """尖峰分布下同样逐节点一致——这是真实 drafter 会产生的深树情形。"""
    _assert_matches_official(_peaked_logits(15, 512), 15, budget)


def _assert_matches_official(logits: torch.Tensor, depth: int, budget: int) -> None:
    official_tokens, official_depths, official_parents, _, official_visibility = _official_tree(
        logits, budget
    )

    tree = DDTreeBuilder(block_size=depth, tree_budget=budget).build_from_logits(logits)

    assert len(tree) == int(official_tokens.numel())
    assert [node.token_id for node in tree.nodes] == [
        int(value) for value in official_tokens.tolist()
    ]
    assert [node.depth for node in tree.nodes] == [
        int(value) for value in official_depths.tolist()
    ]
    # 官方索引 0 表示 anchor；DraftTree 用 parent = -1 表示 anchor，故整体左移一位。
    assert [node.parent for node in tree.nodes] == [
        int(value) - 1 for value in official_parents[1:]
    ]

    # 官方 visibility 含 anchor 行/列；去掉后应与 DraftTree 的 ancestor mask 相同。
    expected_mask = official_visibility[1:, 1:]
    assert torch.equal(tree.ancestor_mask(), expected_mask)


def test_best_first_maximizes_expected_acceptance_against_brute_force() -> None:
    """best-first 在给定预算下最大化 E[接受 token 数]（对小规模实例穷举验证）。

    在 DDTree 的条件独立假设下，节点 ``v`` 被走到的概率就是 ``exp(cum_logprob(v))``，
    因此 ``E[接受数] = sum_v exp(cum_logprob(v))``。可行解是「前缀封闭」的节点集合
    （保留某节点必须保留其整条祖先链）。

    这个测试穷举所有前缀封闭子集，确认 best-first 的选集达到最优。它同时说明：任何
    「重新分配预算」的启发式（例如 ``reserve_greedy_chain``）在该目标下都不可能超过
    官方策略，最好也只是持平——这正是本工程把该开关默认关闭的依据。
    """
    slot_count, per_slot = 3, 3

    def small_logits(seed: int) -> torch.Tensor:
        generator = torch.Generator().manual_seed(seed)
        logits = torch.full((slot_count, slot_count * per_slot), -30.0)
        for slot in range(slot_count):
            probabilities = torch.softmax(torch.randn(per_slot, generator=generator) * 1.2, 0)
            for rank in range(per_slot):
                logits[slot, slot * per_slot + rank] = torch.log(probabilities[rank])
        return logits

    def score(log_probs: torch.Tensor, node_set) -> float:
        return sum(
            math.exp(
                sum(float(log_probs[slot, slot * per_slot + ranks[slot]]) for slot in range(len(ranks)))
            )
            for ranks in node_set
        )

    def selected_rank_paths(tree) -> set[tuple[int, ...]]:
        paths = set()
        for index in range(len(tree)):
            ranks: list[int] = []
            current = index
            while current >= 0:
                node = tree.nodes[current]
                ranks.append(node.token_id - (node.depth - 1) * per_slot)
                current = node.parent
            paths.add(tuple(reversed(ranks)))
        return paths

    candidates = [
        ranks
        for depth in range(1, slot_count + 1)
        for ranks in itertools.product(range(per_slot), repeat=depth)
    ]

    for seed in range(4):
        logits = small_logits(seed)
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        for budget in (2, 3, 5, 7):
            chosen = selected_rank_paths(
                DDTreeBuilder(block_size=slot_count, tree_budget=budget).build_from_logits(logits)
            )
            best = max(
                score(log_probs, subset)
                for subset in itertools.combinations(candidates, budget)
                if all(len(r) == 1 or r[:-1] in set(subset) for r in subset)
            )
            # 用统一的「重新求和」口径比较，避免官方增量更新的浮点累加噪声干扰。
            assert score(log_probs, chosen) == pytest.approx(best, rel=1e-12)


def test_reserve_greedy_chain_does_not_beat_official_expected_acceptance() -> None:
    """诚实记录 ``reserve_greedy_chain`` 的代价：它换来更深的 greedy 链，但不提升期望接受数。

    由上一个测试的最优性结论，该开关在自一致假设下至多持平。这里断言「不显著更优」，
    以防未来有人误把它当作加速改进来宣传。
    """

    def logits_with_decay(seed: int, top1: float, decay: float, depth: int = 15) -> torch.Tensor:
        generator = torch.Generator().manual_seed(seed)
        logits = torch.full((depth, 2000), -30.0)
        for row in range(depth):
            head = max(0.30, top1 - decay * row)
            tail = (1.0 - head) / 4
            picks = torch.randperm(2000, generator=generator)[:5]
            logits[row, picks[0]] = math.log(head)
            for offset in range(1, 5):
                logits[row, picks[offset]] = math.log(tail)
        return logits

    def expected_acceptance(tree) -> float:
        return sum(math.exp(node.cumulative_log_probability) for node in tree.nodes)

    for top1, decay in ((0.90, 0.0), (0.861, 0.05), (0.75, 0.0)):
        official_total = reserved_total = 0.0
        for seed in range(6):
            logits = logits_with_decay(seed, top1, decay)
            official_total += expected_acceptance(
                DDTreeBuilder(block_size=15, tree_budget=60).build_from_logits(logits)
            )
            reserved_total += expected_acceptance(
                DDTreeBuilder(
                    block_size=15, tree_budget=60, reserve_greedy_chain=True
                ).build_from_logits(logits)
            )
        assert reserved_total <= official_total + 1e-6


def test_preset_ancestor_mask_matches_generic_derivation() -> None:
    """注入的 visibility 必须与 DraftTree 自己上溯父链得到的结果一致。"""
    logits = _random_logits(15, 512, seed=11)
    tree = DDTreeBuilder(block_size=15, tree_budget=60).build_from_logits(logits)

    injected = tree.ancestor_mask().clone()
    # 清空缓存后走通用推导路径重算一次。
    tree._ancestor_mask_cache = None
    assert torch.equal(injected, tree.ancestor_mask())


def test_topology_invariants_and_budget() -> None:
    logits = _random_logits(15, 512, seed=7)
    tree = DDTreeBuilder(block_size=15, tree_budget=60).build_from_logits(logits)

    assert len(tree) == 60
    tree.validate(60)
    for index, node in enumerate(tree.nodes):
        assert node.parent < index
        expected = 1 if node.parent < 0 else tree.nodes[node.parent].depth + 1
        assert node.depth == expected
        # 深度 d 的节点取自第 d - 1 个 slot，因此不能超过 block_size。
        assert 1 <= node.depth <= 15
        assert node.slot_index == node.depth - 1


def test_scores_are_non_increasing_in_emission_order() -> None:
    """best-first 的核心性质：节点按累计 log-prob 降序产生。"""
    logits = _random_logits(15, 512, seed=3)
    tree = DDTreeBuilder(block_size=15, tree_budget=60).build_from_logits(logits)

    scores = [node.cumulative_log_probability for node in tree.nodes]
    for earlier, later in zip(scores, scores[1:]):
        assert earlier >= later - 1e-9


def test_cumulative_score_equals_sum_of_slot_log_probs() -> None:
    """累计分数必须等于路径上各 slot log-prob 之和（条件独立可加性）。"""
    depth, vocabulary, budget = 8, 128, 40
    logits = _random_logits(depth, vocabulary, seed=5)
    log_probs = torch.log_softmax(logits.float(), dim=-1)

    tree = DDTreeBuilder(block_size=depth, tree_budget=budget).build_from_logits(logits)
    for index, node in enumerate(tree.nodes):
        path: list[int] = []
        current = index
        while current >= 0:
            path.append(current)
            current = tree.nodes[current].parent
        path.reverse()
        expected = sum(
            float(log_probs[tree.nodes[step].depth - 1, tree.nodes[step].token_id])
            for step in path
        )
        assert node.cumulative_log_probability == pytest.approx(expected, abs=1e-4)


def test_depth_limited_by_block_size_not_logits_rows() -> None:
    """block_size 小于 logits 行数时，树深度必须被 block_size 截断。"""
    logits = _peaked_logits(15, 256)
    assert max(
        node.depth
        for node in DDTreeBuilder(block_size=15, tree_budget=60)
        .build_from_logits(logits)
        .nodes
    ) == 15
    tree = DDTreeBuilder(block_size=4, tree_budget=60).build_from_logits(logits)
    assert max(node.depth for node in tree.nodes) == 4
    assert all(node.depth <= 4 for node in tree.nodes)


def test_budget_zero_and_empty_logits_give_empty_tree() -> None:
    logits = _random_logits(15, 256, seed=13)
    assert len(DDTreeBuilder(block_size=15, tree_budget=60).build_from_logits(logits, budget=0)) == 0
    empty = torch.empty(0, 256)
    assert len(DDTreeBuilder(block_size=15, tree_budget=60).build_from_logits(empty)) == 0


def test_reserve_greedy_chain_keeps_full_chain_and_stays_valid() -> None:
    """可选改动：greedy 链被完整保留，且树仍满足全部拓扑不变量。"""
    depth, budget = 15, 60
    logits = _random_logits(depth, 512, seed=17)
    greedy_tokens = [int(value) for value in logits.argmax(dim=-1).tolist()]

    baseline = DDTreeBuilder(block_size=depth, tree_budget=budget).build_from_logits(logits)
    reserved = DDTreeBuilder(
        block_size=depth, tree_budget=budget, reserve_greedy_chain=True
    ).build_from_logits(logits)

    assert len(reserved) == len(baseline) == budget
    reserved.validate(budget)

    # 前 depth 个节点就是 greedy 链，且构成一条从 anchor 出发的路径。
    chain = reserved.nodes[:depth]
    assert [node.token_id for node in chain] == greedy_tokens
    assert [node.depth for node in chain] == list(range(1, depth + 1))
    assert chain[0].parent == -1
    for index in range(1, depth):
        assert chain[index].parent == index - 1

    # 官方 best-first 在该样本上的最深 greedy 前缀不足 K，这正是该改动想覆盖的情形。
    def longest_greedy_prefix(tree) -> int:
        length = 0
        parent = -1
        for step, token in enumerate(greedy_tokens):
            match = next(
                (
                    index
                    for index, node in enumerate(tree.nodes)
                    if node.parent == parent
                    and node.token_id == token
                    and node.depth == step + 1
                ),
                None,
            )
            if match is None:
                break
            length += 1
            parent = match
        return length

    assert longest_greedy_prefix(reserved) == depth
    assert longest_greedy_prefix(baseline) < depth


def test_build_from_block_proposal_ignores_continuation_callback() -> None:
    """DDTree 是单块方法，SpecBlock 的 continuation 回调不得被调用。"""
    from dflash_specblock.tree import BlockProposal

    logits = _random_logits(4, 64, seed=21).unsqueeze(0)
    proposal = BlockProposal(
        logits=logits,
        hidden=torch.zeros(1, 4, 8),
        rank_logits=torch.zeros(1, 4, 4),
    )

    def expand(*_args, **_kwargs):
        raise AssertionError("DDTree 不应触发跨块 continuation")

    tree = DDTreeBuilder(block_size=4, tree_budget=20).build(proposal, expand)
    assert len(tree) == 20
    tree.validate(20)


def test_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError):
        DDTreeBuilder(block_size=0, tree_budget=60)
    with pytest.raises(ValueError):
        DDTreeBuilder(block_size=15, tree_budget=0)
    with pytest.raises(ValueError):
        DDTreeBuilder(block_size=15, tree_budget=60).build_from_logits(
            _random_logits(15, 64, seed=1), budget=-1
        )
    with pytest.raises(ValueError):
        DDTreeBuilder(block_size=15, tree_budget=60).build_from_logits(torch.zeros(15))
