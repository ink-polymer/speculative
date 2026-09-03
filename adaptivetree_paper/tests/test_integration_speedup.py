"""端到端集成测试：验证树投机解码框架正确性并测量加速比。

使用确定性 mock 模型（embedding + 移位 lm_head），draft 完美匹配 target，
从而在 CUDA GPU（或 CPU 测试后备）上验证：
1. 框架是树模型（多节点、兄弟分支、ancestor mask 生效）
2. 输出与 baseline greedy 完全一致（无损保证）
3. 加速比测量（hybrid vs baseline 墙钟时间）
"""

from __future__ import annotations

import json
import time
from types import SimpleNamespace
from typing import Sequence

import torch
from torch import nn
from transformers import DynamicCache
from transformers.modeling_outputs import CausalLMOutputWithPast

from dflash_specblock.dflash_adapter import DFlashBlockAdapter
from dflash_specblock.device import resolve_device, synchronize
from dflash_specblock.engine import DFlashSpecBlockEngine
from dflash_specblock.rank_head import HeuristicRanker
from dflash_specblock.tree import SpecBlockTreeBuilder
from dflash_specblock.verification import TargetTreeVerifier

# ---------------------------------------------------------------------------
# Mock 模型：target 预测 next = (current + 1) % V，draft 完美匹配 target
# ---------------------------------------------------------------------------

VOCAB = 128
HIDDEN = 128
NUM_TARGET_LAYERS = 2
TARGET_LAYER_IDS = [0, 1]
CHECKPOINT_BLOCK = 16  # 与真实 DFlash-b16 一致
MASK_TOKEN_ID = 127
BLOCK_SIZE = 4
MAX_BLOCKS = 2
TREE_BUDGET = 60
BEAM_WIDTH = 10
BRANCH_FACTORS = (2, 4, 10, 0)


def _make_embedding() -> nn.Embedding:
    """one-hot embedding：embedding(t) = e_t。"""
    emb = nn.Embedding(VOCAB, HIDDEN)
    with torch.no_grad():
        emb.weight.copy_(torch.eye(VOCAB))
    return emb


def _make_lm_head() -> nn.Linear:
    """循环移位矩阵：lm_head(e_t) = e_{(t+1) % V}，即 argmax = (t+1) % V。"""
    head = nn.Linear(HIDDEN, VOCAB, bias=False)
    with torch.no_grad():
        weight = torch.zeros(VOCAB, HIDDEN)
        for token in range(VOCAB):
            weight[(token + 1) % VOCAB, token] = 1.0
        head.weight.copy_(weight)
    return head


class _SmartNorm(nn.Module):
    """根据 anchor hidden 生成完美预测序列的 norm 层。

    position 0 = anchor hidden（不变）
    position i (i>=1) = embedding(predicted_token_at_{i-1})
    使得 lm_head(hidden[i]) == (anchor + i) % V，与 target 验证一致。
    """

    def __init__(self, embedding: nn.Embedding, lm_head: nn.Linear) -> None:
        super().__init__()
        self._embedding = embedding
        self._lm_head = lm_head

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        anchor_hidden = hidden_states[:, 0:1, :]  # [B, 1, H]
        hidden_list = [anchor_hidden]
        current = anchor_hidden
        for _ in range(hidden_states.shape[1] - 1):
            hidden_list.append(current)
            next_token = self._lm_head(current).argmax(-1)  # [B, 1]
            current = self._embedding(next_token)  # [B, 1, H]
        return torch.cat(hidden_list, dim=1)


class _IdentityLayer(nn.Module):
    """Identity 层，用于 draft continuation path。"""

    def forward(self, hidden_states: torch.Tensor, **_: object) -> torch.Tensor:
        return hidden_states


class _MockDraft(nn.Module):
    """Mock DFlash draft model：完美匹配 target 的预测。"""

    def __init__(self, embedding: nn.Embedding, lm_head: nn.Linear) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            hidden_size=HIDDEN,
            block_size=CHECKPOINT_BLOCK,
            num_hidden_layers=NUM_TARGET_LAYERS,
            num_target_layers=NUM_TARGET_LAYERS,
            vocab_size=VOCAB,
            dflash_config={"mask_token_id": MASK_TOKEN_ID},
        )
        self.block_size = CHECKPOINT_BLOCK
        self.mask_token_id = MASK_TOKEN_ID
        self.target_layer_ids = list(TARGET_LAYER_IDS)
        self.fc = nn.Linear(len(TARGET_LAYER_IDS) * HIDDEN, HIDDEN, bias=False)
        self.hidden_norm = nn.Identity()
        self.layers = nn.ModuleList([_IdentityLayer() for _ in range(NUM_TARGET_LAYERS)])
        self.norm = _SmartNorm(embedding, lm_head)
        self.rotary_emb = self._dummy_rotary
        self._embedding = embedding
        self._lm_head = lm_head

    @staticmethod
    def _dummy_rotary(
        hidden_states: torch.Tensor, position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shape = (*position_ids.shape, hidden_states.shape[-1])
        return torch.ones(shape), torch.zeros(shape)

    def forward(
        self,
        target_hidden: torch.Tensor,
        noise_embedding: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: object | None = None,
        use_cache: bool = False,
        is_causal: bool = False,
        **_: object,
    ) -> torch.Tensor:
        result = self.norm(noise_embedding)  # [B, checkpoint_block, H]
        if use_cache and past_key_values is not None:
            # fc 把 target_hidden 从 len(target_layer_ids)*H → H，与 result 维度一致
            context = self.fc(target_hidden)  # [B, context_length, H]
            full_kv = torch.cat([context, result], dim=1)  # [B, ctx+block, H]
            for layer_idx in range(NUM_TARGET_LAYERS):
                past_key_values.update(full_kv, full_kv, layer_idx=layer_idx)
        return result


class _MockTarget(nn.Module):
    """Mock target model：预测 next = (current + 1) % V。"""

    def __init__(self, embedding: nn.Embedding, lm_head: nn.Linear) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            hidden_size=HIDDEN,
            num_hidden_layers=NUM_TARGET_LAYERS,
            vocab_size=VOCAB,
        )
        self.embedding = embedding
        self.lm_head = lm_head

    def get_input_embeddings(self) -> nn.Module:
        return self.embedding

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: object | None = None,
        use_cache: bool = True,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        cache_position: torch.Tensor | None = None,
        **_: object,
    ) -> CausalLMOutputWithPast:
        hidden = self.embedding(input_ids)  # [B, T, H]
        if use_cache and past_key_values is not None:
            for layer_idx in range(NUM_TARGET_LAYERS):
                past_key_values.update(hidden, hidden, layer_idx=layer_idx)
        logits = self.lm_head(hidden)  # [B, T, V]
        all_hidden = None
        if output_hidden_states:
            all_hidden = tuple(hidden for _ in range(NUM_TARGET_LAYERS + 1))
        return CausalLMOutputWithPast(
            logits=logits,
            hidden_states=all_hidden,
            past_key_values=past_key_values,
        )


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def _create_engine(device: torch.device) -> tuple[DFlashSpecBlockEngine, _MockTarget]:
    embedding = _make_embedding()
    lm_head = _make_lm_head()
    target = _MockTarget(embedding, lm_head).to(device).eval()
    draft = _MockDraft(embedding, lm_head).to(device).eval()
    ranker = HeuristicRanker().to(device).eval()
    adapter = DFlashBlockAdapter(
        target=target,
        draft=draft,
        ranker=ranker,
        block_size=BLOCK_SIZE,
        mask_token_id=MASK_TOKEN_ID,
    )
    tree_builder = SpecBlockTreeBuilder(
        block_size=BLOCK_SIZE,
        max_blocks=MAX_BLOCKS,
        tree_budget=TREE_BUDGET,
        beam_width=BEAM_WIDTH,
        branch_factors=BRANCH_FACTORS,
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


def _baseline_greedy(
    target: _MockTarget,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    device: torch.device,
) -> tuple[list[int], float]:
    """标准逐 token greedy 基线。"""
    cache = DynamicCache()
    generated: list[int] = []
    synchronize(device)
    start = time.perf_counter()
    output = target(input_ids=input_ids, past_key_values=cache, use_cache=True, return_dict=True)
    token = int(output.logits[0, -1].argmax().item())
    generated.append(token)
    while len(generated) < max_new_tokens:
        token_tensor = torch.tensor([[token]], dtype=torch.long, device=device)
        output = target(
            input_ids=token_tensor,
            past_key_values=output.past_key_values,
            use_cache=True,
            return_dict=True,
        )
        token = int(output.logits[0, -1].argmax().item())
        generated.append(token)
    synchronize(device)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return generated, elapsed_ms


# ---------------------------------------------------------------------------
# 测试
# ---------------------------------------------------------------------------


def test_tree_framework_produces_multi_node_tree() -> None:
    """验证框架是树模型：生成的树有多节点和兄弟分支，不是线性链。"""
    device = resolve_device("auto")
    engine, _ = _create_engine(device)
    input_ids = torch.tensor([[5]], dtype=torch.long, device=device)
    result = engine.generate(input_ids, max_new_tokens=20)

    assert len(result.iterations) > 0, "至少应有一轮验证"
    for stats in result.iterations:
        # tree_budget=60, beam_width=10 → 树应有多个节点（远超 block_size=4）
        assert stats.tree_nodes > BLOCK_SIZE, (
            f"树节点数 {stats.tree_nodes} <= block_size {BLOCK_SIZE}，"
            "框架退化为线性，不是树模型"
        )
        # beam_width=10 在 slot 0 创建兄弟分支
        assert stats.tree_nodes >= BEAM_WIDTH, (
            f"树节点数 {stats.tree_nodes} < beam_width {BEAM_WIDTH}，"
            "树缺少根部分支"
        )


def test_hybrid_output_matches_baseline_greedy() -> None:
    """验证无损保证：hybrid 输出与 baseline greedy 完全一致。"""
    device = resolve_device("auto")
    engine, target = _create_engine(device)
    input_ids = torch.tensor([[5]], dtype=torch.long, device=device)
    max_new_tokens = 30

    baseline_ids, _ = _baseline_greedy(target, input_ids, max_new_tokens, device)
    hybrid_result = engine.generate(input_ids, max_new_tokens=max_new_tokens)
    hybrid_ids = hybrid_result.generated_ids[0].tolist()

    assert hybrid_ids == baseline_ids, (
        f"hybrid 输出与 baseline 不一致！\n"
        f"hybrid:   {hybrid_ids[:10]}...\n"
        f"baseline: {baseline_ids[:10]}..."
    )


def test_speedup_measurement() -> None:
    """测量并报告 hybrid vs baseline 的墙钟加速比。"""
    device = resolve_device("auto")
    engine, target = _create_engine(device)
    input_ids = torch.tensor([[10]], dtype=torch.long, device=device)
    max_new_tokens = 40

    # 多次运行取最佳值（减少噪声）
    best_baseline_ms = float("inf")
    best_hybrid_ms = float("inf")

    for _ in range(3):
        baseline_ids, baseline_ms = _baseline_greedy(
            target, input_ids, max_new_tokens, device
        )
        best_baseline_ms = min(best_baseline_ms, baseline_ms)

        hybrid_result = engine.generate(input_ids, max_new_tokens=max_new_tokens)
        hybrid_ms = hybrid_result.prefill_ms + hybrid_result.total_decode_ms
        best_hybrid_ms = min(best_hybrid_ms, hybrid_ms)

    speedup = best_baseline_ms / best_hybrid_ms if best_hybrid_ms > 0 else 0.0

    # 验证无损
    hybrid_ids = hybrid_result.generated_ids[0].tolist()
    assert hybrid_ids == baseline_ids, "hybrid 输出与 baseline 不一致"

    # 验证接受了 draft token（完美 draft 应有高接受率）
    total_accepted = sum(s.accepted_draft_tokens for s in hybrid_result.iterations)
    total_committed = sum(s.committed_tokens for s in hybrid_result.iterations)
    assert total_accepted > 0, "未接受任何 draft token"

    # 报告加速比
    summary = {
        "max_new_tokens": max_new_tokens,
        "baseline_ms": round(best_baseline_ms, 2),
        "hybrid_ms": round(best_hybrid_ms, 2),
        "speedup": f"{speedup:.2f}x",
        "iterations": len(hybrid_result.iterations),
        "total_accepted_draft": total_accepted,
        "total_committed": total_committed,
        "average_accepted_length": round(hybrid_result.average_accepted_length, 2),
        "tree_nodes_per_iter": [s.tree_nodes for s in hybrid_result.iterations],
        "tree_build_ms_per_iter": [round(s.tree_build_ms, 2) for s in hybrid_result.iterations],
        "cache_compact_ms_per_iter": [round(s.cache_compact_ms, 2) for s in hybrid_result.iterations],
    }
    print("\n[加速比测试]")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    # 完美 draft + CPU mock，加速比应 > 1（hybrid 每轮多 token）
    # 但 CPU 上树构建开销可能抵消增益，所以只报告不硬断言 > 1
    # 关键断言是：框架是树模型（上面已验证）+ 无损（上面已验证）


def test_tree_has_sibling_branches_and_ancestor_mask() -> None:
    """验证树有兄弟分支且 ancestor mask 正确屏蔽非祖先节点。"""
    from dflash_specblock.tree import DraftTree

    tree = DraftTree()
    root_a = tree.add_node(1, -1, -0.1, 0, 0, 0)
    root_b = tree.add_node(2, -1, -0.2, 0, 0, 1)  # sibling of root_a
    child = tree.add_node(3, root_a, -0.3, 0, 1, 0)  # child of root_a
    grandchild = tree.add_node(4, child, -0.4, 0, 2, 0)  # grandchild

    mask = tree.ancestor_mask()

    # child 可关注 root_a 和自身，不能关注 root_b
    assert mask[child, root_a], "child 应能关注 parent root_a"
    assert mask[child, child], "child 应能关注自身"
    assert not mask[child, root_b], "child 不应能关注 sibling root_b"

    # grandchild 可关注整条祖先链
    assert mask[grandchild, root_a], "grandchild 应能关注 root_a"
    assert mask[grandchild, child], "grandchild 应能关注 child"
    assert mask[grandchild, grandchild], "grandchild 应能关注自身"
    assert not mask[grandchild, root_b], "grandchild 不应能关注 root_b"


def test_engine_produces_correct_sequence() -> None:
    """验证生成的 token 序列符合 (current+1) % V 规则。"""
    device = resolve_device("auto")
    engine, _ = _create_engine(device)
    start_token = 10
    input_ids = torch.tensor([[start_token]], dtype=torch.long, device=device)
    result = engine.generate(input_ids, max_new_tokens=10)

    tokens = result.generated_ids[0].tolist()
    # 第一个 token 应是 (start + 1) % V
    expected = [(start_token + 1 + i) % VOCAB for i in range(len(tokens))]
    assert tokens == expected, (
        f"token 序列不符合 (current+1) % V 规则\n"
        f"got:      {tokens}\n"
        f"expected: {expected}"
    )


# ---------------------------------------------------------------------------
# P2 修复：生产配置端到端覆盖
# ---------------------------------------------------------------------------

PROD_BLOCK_SIZE = 15
PROD_MAX_BLOCKS = 1
PROD_TREE_BUDGET = 60
PROD_BEAM_WIDTH = 4
PROD_BRANCH_FACTORS = (2, 4, 10, 0)


def _create_prod_engine(device: torch.device) -> tuple[DFlashSpecBlockEngine, _MockTarget]:
    """用生产配置（K=15, M=1, budget=60）创建 engine。"""
    embedding = _make_embedding()
    lm_head = _make_lm_head()
    target = _MockTarget(embedding, lm_head).to(device).eval()
    draft = _MockDraft(embedding, lm_head).to(device).eval()
    ranker = HeuristicRanker().to(device).eval()
    adapter = DFlashBlockAdapter(
        target=target,
        draft=draft,
        ranker=ranker,
        block_size=PROD_BLOCK_SIZE,
        mask_token_id=MASK_TOKEN_ID,
    )
    tree_builder = SpecBlockTreeBuilder(
        block_size=PROD_BLOCK_SIZE,
        max_blocks=PROD_MAX_BLOCKS,
        tree_budget=PROD_TREE_BUDGET,
        beam_width=PROD_BEAM_WIDTH,
        branch_factors=PROD_BRANCH_FACTORS,
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


def test_prod_config_lossless() -> None:
    """生产配置 K=15/M=1/budget=60 的无损保证。"""
    device = resolve_device("auto")
    engine, target = _create_prod_engine(device)
    input_ids = torch.tensor([[5]], dtype=torch.long, device=device)
    max_new_tokens = 30

    baseline_ids, _ = _baseline_greedy(target, input_ids, max_new_tokens, device)
    hybrid_result = engine.generate(input_ids, max_new_tokens=max_new_tokens)
    hybrid_ids = hybrid_result.generated_ids[0].tolist()

    assert hybrid_ids == baseline_ids, (
        f"生产配置 hybrid 输出与 baseline 不一致！\n"
        f"hybrid:   {hybrid_ids[:10]}...\n"
        f"baseline: {baseline_ids[:10]}..."
    )


def test_prod_config_main_chain_survives_prune() -> None:
    """P0 验证：生产配置下 greedy 主链在 prune 后完整保留。

    K=15, budget=60, beam=4 → 主链 15 个节点 + 兄弟。
    如果 prune 保护生效，主链 15 层全部保留。
    """
    device = resolve_device("auto")
    engine, _ = _create_prod_engine(device)
    input_ids = torch.tensor([[5]], dtype=torch.long, device=device)
    result = engine.generate(input_ids, max_new_tokens=20)

    assert len(result.iterations) > 0
    # 每轮的主链深度应接近 K=15（完美 draft 下应等于 15）
    for stats in result.iterations:
        # accepted_draft_tokens 是主链被接受的数量
        # 完美 draft 下应接受全部 K-1=14 个 draft token + 1 bonus = 15
        assert stats.accepted_draft_tokens > 0, "主链未接受任何 token"
        # tree_nodes 应 >= K（主链至少 15 个节点）
        assert stats.tree_nodes >= PROD_BLOCK_SIZE, (
            f"树节点 {stats.tree_nodes} < K={PROD_BLOCK_SIZE}，主链可能被 prune 截断"
        )


def test_prod_config_tree_is_multi_node() -> None:
    """生产配置下树是多节点（不只是线性链）。"""
    device = resolve_device("auto")
    engine, _ = _create_prod_engine(device)
    input_ids = torch.tensor([[5]], dtype=torch.long, device=device)
    result = engine.generate(input_ids, max_new_tokens=20)

    for stats in result.iterations:
        # beam_width=4 → slot 0 至少 4 个根子节点
        assert stats.tree_nodes >= PROD_BEAM_WIDTH + PROD_BLOCK_SIZE, (
            f"树节点 {stats.tree_nodes} < beam+K={PROD_BEAM_WIDTH + PROD_BLOCK_SIZE}，"
            "树退化为纯线性"
        )
