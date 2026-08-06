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
