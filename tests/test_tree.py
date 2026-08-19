"""SpecBlock 主链、兄弟分支、跨块批扩展与祖先剪枝测试。"""

import torch

from dflash_specblock.tree import BlockProposal, DraftTree, SpecBlockTreeBuilder


def _proposal(batch: int, block: int, vocab: int = 16) -> BlockProposal:
    logits = torch.full((batch, block, vocab), -8.0)
    for row in range(batch):
        for slot in range(block):
            logits[row, slot, (row * 3 + slot + 1) % vocab] = 6.0
            logits[row, slot, (row * 3 + slot + 2) % vocab] = 5.0
            logits[row, slot, (row * 3 + slot + 3) % vocab] = 4.0
    hidden = torch.randn(batch, block, 8)
    rank_logits = torch.zeros(batch, block, 4)
    rank_logits[..., 0] = 10.0
    return BlockProposal(logits=logits, hidden=hidden, rank_logits=rank_logits)


def test_block_iterative_tree_is_topological_and_budgeted() -> None:
    builder = SpecBlockTreeBuilder(
        block_size=3,
        max_blocks=2,
        tree_budget=20,
        beam_width=3,
        branch_factors=(2, 4, 10, 0),
    )
    calls: list[int] = []

    def expand(contexts: torch.Tensor, anchors: torch.Tensor) -> BlockProposal:
        calls.append(anchors.numel())
        assert contexts.shape == (anchors.numel(), 8)
        return _proposal(anchors.numel(), 3)

    tree = builder.build(_proposal(1, 3), expand)
    # 官方在 forward 前先按 adaptive_beam=min(beam_width,(budget-nodes)//K) 裁剪 pending。
    assert calls == [3]
    assert len(tree) <= 20
    for index, node in enumerate(tree.nodes):
        assert node.parent < index
        assert node.depth == (1 if node.parent < 0 else tree.nodes[node.parent].depth + 1)


def test_pending_is_pruned_to_adaptive_beam_before_next_forward() -> None:
    """block1 的 pending 不能整批进入下一次 forward，必须先被 adaptive beam 裁剪。"""
    builder = SpecBlockTreeBuilder(
        block_size=4,
        max_blocks=2,
        tree_budget=60,
        beam_width=10,
        branch_factors=(2, 4, 10, 0),
    )
    proposal = _proposal(1, 4, vocab=64)
    proposal.rank_logits.zero_()
    proposal.rank_logits[0, 0, 0] = 10
    proposal.rank_logits[0, 1, 1] = 10
    proposal.rank_logits[0, 2, 2] = 10
    proposal.rank_logits[0, 3, 0] = 10

    batch_sizes: list[int] = []

    def expand(contexts: torch.Tensor, anchors: torch.Tensor) -> BlockProposal:
        batch_sizes.append(anchors.numel())
        return _proposal(anchors.numel(), 4, vocab=64)

    builder.build(proposal, expand)
    assert batch_sizes and batch_sizes[0] <= 10


def test_ancestor_mask_blocks_siblings() -> None:
    tree = DraftTree()
    first = tree.add_node(1, -1, -0.1, 0, 0, 0)
    sibling = tree.add_node(2, -1, -0.2, 0, 0, 1)
    child = tree.add_node(3, first, -0.3, 0, 1, 0)
    mask = tree.ancestor_mask()
    assert mask[child, first]
    assert mask[child, child]
    assert not mask[child, sibling]


def test_prune_keeps_ancestors() -> None:
    tree = DraftTree()
    root_child = tree.add_node(1, -1, -5.0, 0, 0, 0)
    tree.add_node(2, root_child, -0.1, 0, 1, 0)
    tree.add_node(3, -1, -1.0, 0, 0, 0)
    tree.prune(2)
    assert len(tree) == 2
    assert tree.nodes[1].parent == 0


def test_official_block1_slot0_beam_and_rank_bucket_widths() -> None:
    proposal = _proposal(1, 2, vocab=32)
    proposal.rank_logits.zero_()
    proposal.rank_logits[0, 0, 0] = 10  # top-2
    proposal.rank_logits[0, 1, 2] = 10  # top-10
    builder = SpecBlockTreeBuilder(
        block_size=2,
        max_blocks=1,
        tree_budget=32,
        beam_width=10,
        branch_factors=(2, 4, 10, 0),
    )
    tree = builder.build(proposal, lambda *_: (_ for _ in ()).throw(AssertionError()))

    # block1 slot0 固定 beam=10；slot1 的 b2 也是 top-10。
    assert len(tree) == 20
    root_children = [node for node in tree.nodes if node.parent == -1]
    slot1_children = [node for node in tree.nodes if node.parent == 0]
    assert len(root_children) == 10
    assert len(slot1_children) == 10


def test_give_up_keeps_greedy_chain_and_only_slot0_root_diversity_continues() -> None:
    proposal = _proposal(1, 3)
    proposal.rank_logits.zero_()
    proposal.rank_logits[..., 3] = 10
    calls = 0

    def expand(_: torch.Tensor, anchors: torch.Tensor) -> BlockProposal:
        nonlocal calls
        calls += 1
        return _proposal(anchors.numel(), 3)

    builder = SpecBlockTreeBuilder(
        block_size=3,
        max_blocks=2,
        tree_budget=20,
        beam_width=10,
        branch_factors=(2, 4, 10, 0),
    )
    tree = builder.build(proposal, expand)
    # 全部 b3 时，greedy K 个节点仍免费保留；block1 slot0 的 beam 根候选仍进入 block2。
    assert len(tree) <= 20
    assert calls == 1


def test_prune_protects_greedy_main_chain_from_shallow_siblings() -> None:
    """P0: 浅层兄弟 cum_lp 天然高于深层主链，prune 必须保护主链不被挤掉。"""
    tree = DraftTree()
    # 主链：depth 1→5，cum_lp 逐层递减
    main_indices = []
    prev = -1
    score = 0.0
    for slot in range(5):
        score -= 0.2  # 每层 -0.2
        idx = tree.add_node(slot + 1, prev, score, 0, slot, 0, is_main_chain=True)
        main_indices.append(idx)
        prev = idx

    # 浅层兄弟：depth=1, cum_lp 远高于主链深层节点
    for alt in range(10):
        tree.add_node(100 + alt, -1, -0.01, 0, 0, 1, is_main_chain=False)

    # budget=8: 主链5个必须全保留 + 3个兄弟
    tree.prune(8)
    assert len(tree) == 8

    # 验证主链全部存活
    main_survived = [n for n in tree.nodes if n.is_main_chain]
    assert len(main_survived) == 5, (
        f"主链只保留了 {len(main_survived)}/5 个节点，prune 未保护主链"
    )
    # 验证主链深度完整（5层）
    max_main_depth = max(n.depth for n in main_survived)
    assert max_main_depth == 5, f"主链最大深度 {max_main_depth}，应为 5"

    # 验证兄弟只保留了3个
    sibling_count = sum(1 for n in tree.nodes if not n.is_main_chain)
    assert sibling_count == 3, f"兄弟保留了 {sibling_count}，应为 3"


def test_prune_main_chain_exceeds_budget_keeps_shallowest() -> None:
    """主链本身超出预算时，只保留浅层部分。"""
    tree = DraftTree()
    for slot in range(10):
        parent = -1 if slot == 0 else slot - 1
        tree.add_node(slot, parent, -0.1 * slot, 0, slot, 0, is_main_chain=True)

    tree.prune(5)
    assert len(tree) == 5
    # 应保留 index 0-4（浅层）
    for i, node in enumerate(tree.nodes):
        assert node.depth == i + 1
    assert tree.nodes[-1].depth == 5, "应保留 depth 1-5"


def test_prune_preserves_non_main_chain_ancestors_of_main_chain() -> None:
    """P0: 跨块主链的父节点可能是上一块的兄弟（非主链），必须一并保留。"""
    tree = DraftTree()
    # Block 0: 主链 0→1→2
    tree.add_node(1, -1, -0.1, 0, 0, 0, is_main_chain=True)
    tree.add_node(2, 0, -0.2, 0, 1, 0, is_main_chain=True)
    tree.add_node(3, 1, -0.3, 0, 2, 0, is_main_chain=True)
    # Block 0: 兄弟（非主链，cum_lp 更高）
    sibling = tree.add_node(99, 0, -0.05, 0, 1, 1, is_main_chain=False)
    # Block 1: 主链从兄弟节点延续
    tree.add_node(4, sibling, -0.15, 1, 0, 0, is_main_chain=True)
    tree.add_node(5, 4, -0.25, 1, 1, 0, is_main_chain=True)

    # budget=6: 主链5 + 祖先(兄弟)1 = 6，正好满
    tree.prune(6)
    assert len(tree) == 6
    # 兄弟节点必须保留（是 block1 主链的祖先）
    sibling_survived = any(n.token_id == 99 for n in tree.nodes)
    assert sibling_survived, "非主链祖先（兄弟）被剪掉，导致 block1 主链断裂"
