"""DFlash 官方首块 cache 语义与组合创新接口的 CPU 契约测试。"""

from types import SimpleNamespace

import torch
from torch import nn

from dflash_specblock.dflash_adapter import DFlashBlockAdapter
from dflash_specblock.rank_head import HeuristicRanker


class _Target(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(32, 4)
        self.head = nn.Linear(4, 32, bias=False)

    def get_input_embeddings(self) -> nn.Module:
        return self.embedding

    def get_output_embeddings(self) -> nn.Module:
        return self.head


class _Layer(nn.Module):
    def forward(self, hidden_states: torch.Tensor, **_: object) -> torch.Tensor:
        return hidden_states


class _ForbiddenProjection(nn.Module):
    """后续块必须绕过只适用于目标多层特征的 fc。"""

    def forward(self, _: torch.Tensor) -> torch.Tensor:
        raise AssertionError("SpecBlock 后续块不应再次调用 DFlash fc projection")


class _CountingNorm(nn.Module):
    """官方 use_draft_condition 分支仍会执行 condition_norm，这里记录调用次数。"""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return hidden_states


class _Draft(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            hidden_size=4,
            block_size=5,
            dflash_config={"mask_token_id": 31},
        )
        self.block_size = 5
        self.target_layer_ids = [0]
        self.fc = _ForbiddenProjection()
        self.hidden_norm = _CountingNorm()
        self.layers = nn.ModuleList([_Layer()])
        self.norm = nn.Identity()
        self.last_position_ids: torch.Tensor | None = None

    def rotary_emb(
        self, hidden_states: torch.Tensor, position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shape = (*position_ids.shape, hidden_states.shape[-1])
        return torch.ones(shape), torch.zeros(shape)

    def forward(
        self,
        target_hidden: torch.Tensor,
        noise_embedding: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: object | None,
        **_: object,
    ) -> torch.Tensor:
        self.last_position_ids = position_ids.detach().clone()
        if past_key_values is not None:
            past_key_values.length += target_hidden.shape[1] + noise_embedding.shape[1]
        return noise_embedding


class _Cache:
    def __init__(self, length: int) -> None:
        self.length = length

    def get_seq_length(self) -> int:
        return self.length

    def crop(self, length: int) -> None:
        self.length = length


def _adapter() -> tuple[DFlashBlockAdapter, _Draft]:
    draft = _Draft()
    adapter = DFlashBlockAdapter(
        target=_Target(),
        draft=draft,
        ranker=HeuristicRanker(),
        block_size=4,
    )
    return adapter, draft


def test_first_block_uses_incremental_absolute_positions_and_crops_cache() -> None:
    adapter, draft = _adapter()
    cache = _Cache(length=2)
    target_update = torch.randn(1, 3, 4)
    logits, hidden = adapter.draft_first_raw(
        target_context=target_update,
        anchor_ids=torch.tensor([7]),
        draft_cache=cache,
        cache_prefix_length=5,
    )

    assert logits.shape == (1, 4, 32)
    assert hidden.shape == (1, 4, 4)
    assert draft.last_position_ids is not None
    assert draft.last_position_ids[0].tolist() == list(range(2, 10))
    assert cache.length == 5


def test_continuation_bypasses_fc_but_keeps_hidden_norm() -> None:
    """对齐官方 use_draft_condition：跳过 condition_proj(fc)，保留 condition_norm。"""
    adapter, draft = _adapter()
    proposal = adapter.propose_continuation(
        draft_context=torch.randn(2, 4),
        anchor_ids=torch.tensor([3, 4]),
    )
    assert proposal.logits.shape == (2, 4, 32)
    assert proposal.hidden.shape == (2, 4, 4)
    assert proposal.rank_logits.shape == (2, 4, 4)
    # fc 是 _ForbiddenProjection，能走到这里就说明它没有被调用。
    assert draft.hidden_norm.calls == 1


def test_first_block_rejects_discontinuous_cache_update() -> None:
    adapter, _ = _adapter()
    cache = _Cache(length=2)
    try:
        adapter.draft_first_raw(
            target_context=torch.randn(1, 3, 4),
            anchor_ids=torch.tensor([7]),
            draft_cache=cache,
            cache_prefix_length=6,
        )
    except ValueError as error:
        assert "不连续" in str(error)
    else:
        raise AssertionError("不连续的 DFlash cache 被静默接受")
