"""目标模型与 DFlash checkpoint 的结构契约测试。"""

import sys
from types import SimpleNamespace

import torch
from torch import nn

from dflash_specblock.config import ExperimentConfig
from dflash_specblock.models import _validate_model_pair, load_models, load_target_model


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


class _RecordingFactory:
    def __init__(self, value: object) -> None:
        self.value = value
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def from_pretrained(self, *args: object, **kwargs: object) -> object:
        self.calls.append((args, kwargs))
        return self.value


def _fake_transformers(monkeypatch):
    target_factory = _RecordingFactory(_Target())
    draft_factory = _RecordingFactory(_Draft())
    tokenizer = object()
    tokenizer_factory = _RecordingFactory(tokenizer)
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoModelForCausalLM=target_factory,
            AutoModel=draft_factory,
            AutoTokenizer=tokenizer_factory,
        ),
    )
    return target_factory, draft_factory, tokenizer_factory, tokenizer


def test_cuda_model_loading_uses_sdpa_and_direct_device_map(monkeypatch) -> None:
    target_factory, draft_factory, _, _ = _fake_transformers(monkeypatch)
    config = ExperimentConfig(
        device="cuda:1",
        dtype="float16",
        attn_implementation="sdpa",
        draft_attn_implementation="flash_attention_2",
    )

    bundle = load_models(config, torch.device("cuda:1"))

    assert bundle.target is target_factory.value
    assert bundle.draft is draft_factory.value
    target_kwargs = target_factory.calls[0][1]
    draft_kwargs = draft_factory.calls[0][1]
    assert target_kwargs["device_map"] == {"": 1}
    assert draft_kwargs["device_map"] == {"": 1}
    assert target_kwargs["attn_implementation"] == "sdpa"
    assert draft_kwargs["attn_implementation"] == "flash_attention_2"
    assert all(not parameter.requires_grad for parameter in bundle.target.parameters())
    assert all(not parameter.requires_grad for parameter in bundle.draft.parameters())


def test_target_only_loader_does_not_instantiate_draft(monkeypatch) -> None:
    target_factory, draft_factory, _, tokenizer = _fake_transformers(monkeypatch)
    config = ExperimentConfig(device="cuda:0", dtype="float16")

    bundle = load_target_model(config, torch.device("cuda:0"))

    assert bundle.target is target_factory.value
    assert bundle.tokenizer is tokenizer
    assert len(target_factory.calls) == 1
    assert draft_factory.calls == []
