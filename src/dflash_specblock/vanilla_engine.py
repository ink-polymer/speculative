"""正版 DFlash 线性 speculative decoding（无 SpecBlock 树扩展）。

与 DFlash-SpecBlock 的唯一区别：
- 草稿阶段只取每个 block 的 greedy top-1，形成长度 K 的线性链（无兄弟/分支/树）。
- 验证阶段用标准因果注意力（无 ancestor-only tree mask），接受最长匹配前缀 + bonus token。
- 不需要 rank head（不做分支决策）。

DFlash block diffusion drafter、target KV injection、draft cache 增量维护与官方完全一致，
对应 DFlash 论文第 4.1 节和附录 A.3。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

from .device import DeviceTimer
from .dflash_adapter import DFlashBlockAdapter


@dataclass(slots=True)
class VanillaIterationStats:
    draft_ms: float
    verify_ms: float
    block_size: int
    accepted_draft_tokens: int
    committed_tokens: int


@dataclass(slots=True)
class VanillaGenerationResult:
    output_ids: torch.Tensor
    generated_ids: torch.Tensor
    iterations: list[VanillaIterationStats] = field(default_factory=list)
    prefill_ms: float = 0.0

    @property
    def average_accepted_length(self) -> float:
        if not self.iterations:
            return 0.0
        return sum(item.committed_tokens for item in self.iterations) / len(self.iterations)

    @property
    def total_decode_ms(self) -> float:
        return sum(item.draft_ms + item.verify_ms for item in self.iterations)


class VanillaDFlashEngine:
    """正版 DFlash：线性 block draft + 线性因果 verify，batch=1。"""

    def __init__(
        self,
        target: nn.Module,
        adapter: DFlashBlockAdapter,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.target = target
        self.adapter = adapter
        self.device = device
        self.dtype = dtype
        self.block_size = adapter.block_size

    @torch.inference_mode()
    def _prefill(self, input_ids: torch.Tensor) -> tuple[int, object, torch.Tensor, float]:
        from transformers import DynamicCache

        cache = DynamicCache()
        with DeviceTimer(self.device) as timer:
            output = self.target(
                input_ids=input_ids,
                past_key_values=cache,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
        anchor = int(output.logits[0, -1].argmax(dim=-1).item())
        target_update = self.adapter.extract_target_context(output.hidden_states)
        return anchor, output.past_key_values, target_update, timer.elapsed_ms

    def _context_from_hidden(
        self,
        hidden_states: tuple[torch.Tensor, ...],
        rows: torch.Tensor,
    ) -> torch.Tensor:
        """提取指定行的多层目标 hidden 并拼接，与 verification._context_from_hidden 一致。"""
        selected = [
            hidden_states[layer_id + 1].index_select(1, rows)
            for layer_id in self.adapter.target_layer_ids
        ]
        return torch.cat(selected, dim=-1)

    def _build_causal_mask(
        self,
        past_length: int,
        current_length: int,
    ) -> torch.Tensor:
        """标准因果 4D mask：current 各位置可看全部 past + 自身及之前的 current。"""
        total = past_length + current_length
        minimum = torch.finfo(self.dtype).min
        mask = torch.full(
            (1, 1, current_length, total),
            minimum,
            dtype=self.dtype,
            device=self.device,
        )
        if past_length:
            mask[..., :past_length] = 0
        causal = torch.tril(
            torch.ones(current_length, current_length, dtype=torch.bool, device=self.device)
        )
        mask[..., past_length:].masked_fill_(causal, 0)
        return mask

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        stop_token_ids: set[int] | None = None,
    ) -> VanillaGenerationResult:
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("当前实验实现只支持 batch=1 的 input_ids")
        if max_new_tokens < 1:
            empty = torch.empty((1, 0), dtype=torch.long, device=input_ids.device)
            return VanillaGenerationResult(output_ids=input_ids, generated_ids=empty)

        stop_token_tokens = stop_token_ids or set()
        anchor, cache, target_update, prefill_ms = self._prefill(input_ids)

        from transformers import DynamicCache

        draft_cache = DynamicCache()
        generated: list[int] = [anchor]
        stats: list[VanillaIterationStats] = []
        K = self.block_size

        while len(generated) < max_new_tokens and anchor not in stop_token_tokens:
            # ── Draft: DFlash 官方 block forward，取 greedy top-1 线性链 ──────────
            with DeviceTimer(self.device) as draft_timer:
                logits, _hidden = self.adapter.draft_first_raw(
                    target_context=target_update,
                    anchor_ids=torch.tensor([anchor], dtype=torch.long, device=self.device),
                    draft_cache=draft_cache,
                    cache_prefix_length=int(cache.get_seq_length()),
                )
            draft_tokens: list[int] = logits[0].argmax(dim=-1).tolist()

            # ── Verify: 目标一次因果前向，接受最长匹配前缀 + bonus ────────────────
            with DeviceTimer(self.device) as verify_timer:
                past_length = int(cache.get_seq_length())
                verify_length = K + 1
                current_ids = torch.tensor(
                    [[anchor] + draft_tokens], dtype=torch.long, device=self.device
                )
                position_ids = torch.arange(
                    past_length,
                    past_length + verify_length,
                    dtype=torch.long,
                    device=self.device,
                ).unsqueeze(0)
                cache_position = position_ids.clone()
                attention_mask = self._build_causal_mask(past_length, verify_length)

                output = self.target(
                    input_ids=current_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    cache_position=cache_position,
                    past_key_values=cache,
                    use_cache=True,
                    output_hidden_states=True,
                    return_dict=True,
                )
                # target_logits[0, i, :] 是位置 i 之后的预测（即应该匹配 draft_tokens[i]）。
                target_argmax: list[int] = output.logits[0].argmax(dim=-1).tolist()

            accepted = 0
            for i in range(K):
                if target_argmax[i] == draft_tokens[i]:
                    accepted += 1
                else:
                    break
            bonus = target_argmax[accepted]

            committed = draft_tokens[:accepted] + [bonus]
            remaining = max_new_tokens - len(generated)
            committed = committed[:remaining]
            stop_reached = False
            for index, token in enumerate(committed):
                if token in stop_token_tokens:
                    committed = committed[: index + 1]
                    stop_reached = True
                    break
            generated.extend(committed)
            stats.append(
                VanillaIterationStats(
                    draft_ms=draft_timer.elapsed_ms,
                    verify_ms=verify_timer.elapsed_ms,
                    block_size=K,
                    accepted_draft_tokens=accepted,
                    committed_tokens=len(committed),
                )
            )

            # ── Cache 压缩：保留 [旧前缀 + anchor + accepted 个 draft token] ──────
            # current_ids row 0 = anchor, row i+1 = draft_tokens[i]。
            # 接受 accepted 个 draft token → 保留 rows 0..accepted（共 accepted+1 行）。
            # bonus 的 KV 尚未进入 cache（它是 target 预测，未经 forward）。
            keep_length = past_length + 1 + accepted
            extended_cache = output.past_key_values
            if hasattr(extended_cache, "crop"):
                extended_cache.crop(keep_length)
            else:
                raise TypeError("target cache 必须实现 crop(length)")
            cache = extended_cache

            # ── 提取下一轮 draft 所需的增量 target hidden ────────────────────────
            # rows [0..accepted] 对应 current_ids 中的 anchor + 已接受 draft tokens。
            kept_rows = torch.arange(
                0, accepted + 1, dtype=torch.long, device=self.device
            )
            target_update = self._context_from_hidden(output.hidden_states, kept_rows)

            anchor = bonus
            if stop_reached:
                break

        generated_tensor = torch.tensor(
            generated, dtype=torch.long, device=input_ids.device
        ).unsqueeze(0)
        output_ids = torch.cat([input_ids, generated_tensor], dim=1)
        return VanillaGenerationResult(
            output_ids=output_ids,
            generated_ids=generated_tensor,
            iterations=stats,
            prefill_ms=prefill_ms,
        )
