"""DDTree 端到端集成测试：复用 mock 模型验证无损性与树结构。

复用 ``test_integration_speedup`` 的确定性 mock（target 预测 ``(current + 1) % V``，draft
完美匹配），把树构建器换成 :class:`DDTreeBuilder`，验证 DDTree 拓扑同样能走通
ancestor-only 验证 + KV 压缩，且输出与逐 token greedy baseline 完全一致。

无损性由 target 的验证语义保证，与草稿树的形状无关；这里测的是 DDTree 产出的
``DraftTree``（parent/depth/ancestor mask）确实满足验证器的全部前置假设。
"""

from __future__ import annotations

import torch

from dflash_specblock.ddtree_builder import DDTreeBuilder
from dflash_specblock.device import resolve_device
from dflash_specblock.dflash_adapter import DFlashBlockAdapter
from dflash_specblock.engine import DFlashSpecBlockEngine
from dflash_specblock.rank_head import HeuristicRanker
from dflash_specblock.verification import TargetTreeVerifier

from test_integration_speedup import (
    MASK_TOKEN_ID,
    TARGET_LAYER_IDS,
    VOCAB,
    _baseline_greedy,
    _make_embedding,
    _make_lm_head,
    _MockDraft,
    _MockTarget,
)


DD_BLOCK_SIZE = 15
DD_TREE_BUDGET = 60


def _create_ddtree_engine(
    device: torch.device,
    reserve_greedy_chain: bool = False,
    tree_budget: int = DD_TREE_BUDGET,
) -> tuple[DFlashSpecBlockEngine, _MockTarget]:
    embedding = _make_embedding()
    lm_head = _make_lm_head()
    target = _MockTarget(embedding, lm_head).to(device).eval()
    draft = _MockDraft(embedding, lm_head).to(device).eval()
    adapter = DFlashBlockAdapter(
        target=target,
        draft=draft,
        # DDTree 不读取 rank 输出；占位 ranker 只为满足 adapter 的字段契约。
        ranker=HeuristicRanker().to(device).eval(),
        block_size=DD_BLOCK_SIZE,
        mask_token_id=MASK_TOKEN_ID,
    )
    tree_builder = DDTreeBuilder(
        block_size=DD_BLOCK_SIZE,
        tree_budget=tree_budget,
        reserve_greedy_chain=reserve_greedy_chain,
    )
    verifier = TargetTreeVerifier(
        target=target,
        target_layer_ids=TARGET_LAYER_IDS,
        device=device,
        dtype=torch.float32,
    )
    engine = DFlashSpecBlockEngine(
        target=target,
        adapter=adapter,
        tree_builder=tree_builder,
        verifier=verifier,
        device=device,
    )
    return engine, target


def test_engine_skips_rank_head_for_ddtree() -> None:
    """DDTree 模式下 engine 必须跳过 rank head 前向。"""
    device = resolve_device("auto")
    engine, _ = _create_ddtree_engine(device)
    assert engine._compute_rank is False

    calls = 0
    original = engine.adapter.ranker.forward

    def counting_forward(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    engine.adapter.ranker.forward = counting_forward
    engine.generate(torch.tensor([[5]], dtype=torch.long, device=device), max_new_tokens=20)
    assert calls == 0, "DDTree 模式下不应调用 rank head"


def test_specblock_still_computes_rank_head() -> None:
    """回归保护：SpecBlock 路径必须仍然调用 rank head。"""
    from test_integration_speedup import _create_engine

    device = resolve_device("auto")
    engine, _ = _create_engine(device)
    assert engine._compute_rank is True

    calls = 0
    original = engine.adapter.ranker.forward

    def counting_forward(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    engine.adapter.ranker.forward = counting_forward
    engine.generate(torch.tensor([[5]], dtype=torch.long, device=device), max_new_tokens=12)
    assert calls > 0, "SpecBlock 模式必须调用 rank head"


def test_ddtree_output_matches_baseline_greedy() -> None:
    """无损保证：DDTree 输出与逐 token greedy baseline 完全一致。"""
    device = resolve_device("auto")
    engine, target = _create_ddtree_engine(device)
    input_ids = torch.tensor([[5]], dtype=torch.long, device=device)
    max_new_tokens = 30

    baseline_ids, _ = _baseline_greedy(target, input_ids, max_new_tokens, device)
    hybrid_ids = engine.generate(input_ids, max_new_tokens=max_new_tokens).generated_ids[0].tolist()

    assert hybrid_ids == baseline_ids


def test_ddtree_reserve_greedy_chain_is_also_lossless() -> None:
    """可选改动同样无损：改动只影响树形状，不触碰验证语义。"""
    device = resolve_device("auto")
    engine, target = _create_ddtree_engine(device, reserve_greedy_chain=True)
    input_ids = torch.tensor([[7]], dtype=torch.long, device=device)
    max_new_tokens = 30

    baseline_ids, _ = _baseline_greedy(target, input_ids, max_new_tokens, device)
    hybrid_ids = engine.generate(input_ids, max_new_tokens=max_new_tokens).generated_ids[0].tolist()

    assert hybrid_ids == baseline_ids


def test_ddtree_produces_sequence_following_mock_rule() -> None:
    device = resolve_device("auto")
    engine, _ = _create_ddtree_engine(device)
    start_token = 10
    result = engine.generate(
        torch.tensor([[start_token]], dtype=torch.long, device=device), max_new_tokens=10
    )
    tokens = result.generated_ids[0].tolist()
    assert tokens == [(start_token + 1 + offset) % VOCAB for offset in range(len(tokens))]


def test_ddtree_respects_budget_and_is_multi_node() -> None:
    """DDTree 在构建期截断预算，节点数不得超出，且必须是多节点树。"""
    device = resolve_device("auto")
    engine, _ = _create_ddtree_engine(device)
    result = engine.generate(
        torch.tensor([[5]], dtype=torch.long, device=device), max_new_tokens=20
    )

    assert result.iterations
    for stats in result.iterations:
        assert stats.tree_nodes <= DD_TREE_BUDGET
        assert stats.tree_nodes > 1


def test_ddtree_accepts_full_chain_with_perfect_draft() -> None:
    """完美 draft 下，reserve_greedy_chain 必须能接受完整的 K 步链。

    mock draft 的 greedy 链与 target 完全一致，因此保留完整 greedy 链后每轮应接受
    ``K`` 个 draft token。这是该改动的直接可观测收益。
    """
    device = resolve_device("auto")
    engine, _ = _create_ddtree_engine(device, reserve_greedy_chain=True)
    result = engine.generate(
        torch.tensor([[5]], dtype=torch.long, device=device), max_new_tokens=60
    )

    assert result.iterations
    # 除最后一轮可能被 max_new_tokens 截断外，都应接受满 K 个 draft token。
    assert max(stats.accepted_draft_tokens for stats in result.iterations) == DD_BLOCK_SIZE


def test_ddtree_budget_smaller_than_block_size_is_allowed() -> None:
    """预算小于 K 时 DDTree 仍应正常工作（SpecBlock 则要求 budget >= K）。"""
    device = resolve_device("auto")
    engine, target = _create_ddtree_engine(device, tree_budget=4)
    input_ids = torch.tensor([[5]], dtype=torch.long, device=device)

    baseline_ids, _ = _baseline_greedy(target, input_ids, 20, device)
    hybrid_ids = engine.generate(input_ids, max_new_tokens=20).generated_ids[0].tolist()
    assert hybrid_ids == baseline_ids
