"""目标模型与 DFlash checkpoint 的结构契约测试。"""

from types import SimpleNamespace

import torch
from torch import nn

from dflash_specblock.models import _validate_model_pair


class _Target(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=4, num_hidden_layers=3, vocab_size=8)
        self.lm_head = nn.Linear(4, 8, bias=False)

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head


class _Draft(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            hidden_size=4,
            num_hidden_layers=2,
            num_target_layers=3,
            vocab_size=8,
        )
        self.target_layer_ids = [0, 2]
        self.fc = nn.Linear(8, 4, bias=False)
        self.hidden_norm = nn.Identity()
        self.layers = nn.ModuleList([nn.Identity(), nn.Identity()])
        self.rotary_emb = nn.Identity()


def test_exact_target_dflash_contract_is_accepted() -> None:
    _validate_model_pair(_Target(), _Draft())


def test_vocab_mismatch_is_rejected_before_inference() -> None:
    target = _Target()
    draft = _Draft()
    draft.config.vocab_size = 9
    try:
        _validate_model_pair(target, draft)
    except ValueError as error:
        assert "vocab" in str(error)
    else:
        raise AssertionError("target/draft vocab mismatch was accepted")
