"""Vanilla DFlash GPU fast-path control-flow tests."""

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from dflash_specblock.vanilla_engine import VanillaDFlashEngine


class _Target(nn.Module):
    def __init__(self, vocab_size: int = 32) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.calls: list[dict[str, object]] = []

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: object,
        use_cache: bool,
        output_hidden_states: bool,
        return_dict: bool,
        position_ids: torch.Tensor | None = None,
        cache_position: torch.Tensor | None = None,
        **kwargs: object,
    ) -> object:
        del use_cache, return_dict
        self.calls.append(
            {
                "input_ids": input_ids.clone(),
                "position_ids": None if position_ids is None else position_ids.clone(),
                "cache_position": (
                    None if cache_position is None else cache_position.clone()
                ),
                "kwargs": kwargs,
            }
        )

        hidden = input_ids.to(torch.float32).unsqueeze(-1)
        key_values = hidden.unsqueeze(1)
        past_key_values.update(key_values, key_values, layer_idx=0)
        predicted = (input_ids + 1) % self.vocab_size
        logits = torch.full(
            (*input_ids.shape, self.vocab_size),
            -100.0,
            device=input_ids.device,
        )
        logits.scatter_(-1, predicted.unsqueeze(-1), 100.0)
        hidden_states = (hidden, hidden) if output_hidden_states else None
        return SimpleNamespace(
            logits=logits,
            hidden_states=hidden_states,
            past_key_values=past_key_values,
        )


class _Adapter:
    block_size = 2
    target_layer_ids = [0]

    @staticmethod
    def extract_target_context(hidden_states: tuple[torch.Tensor, ...]) -> torch.Tensor:
        return hidden_states[1]

    @staticmethod
    def draft_first_raw(
        target_context: torch.Tensor,
        anchor_ids: torch.Tensor,
        draft_cache: object,
        cache_prefix_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del target_context, draft_cache, cache_prefix_length
        vocab_size = 32
        offsets = torch.arange(1, 3, device=anchor_ids.device)
        tokens = (anchor_ids[:, None] + offsets) % vocab_size
        logits = torch.full(
            (anchor_ids.shape[0], 2, vocab_size),
            -100.0,
            device=anchor_ids.device,
        )
        logits.scatter_(-1, tokens.unsqueeze(-1), 100.0)
        return logits, tokens.to(torch.float32).unsqueeze(-1)


def test_vanilla_verify_uses_native_causal_path_and_1d_cache_position() -> None:
    target = _Target()
    engine = VanillaDFlashEngine(
        target=target,
        adapter=_Adapter(),  # type: ignore[arg-type]
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    result = engine.generate(torch.tensor([[5]]), max_new_tokens=4)
    assert result.generated_ids.tolist() == [[6, 7, 8, 9]]

    verify_call = target.calls[1]
    assert "attention_mask" not in verify_call["kwargs"]
    assert verify_call["cache_position"].ndim == 1  # type: ignore[union-attr]
    assert verify_call["cache_position"].tolist() == [1, 2, 3]  # type: ignore[union-attr]
    assert verify_call["position_ids"].tolist() == [[1, 2, 3]]  # type: ignore[union-attr]
