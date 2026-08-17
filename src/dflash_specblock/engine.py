"""DFlash-SpecBlock 端到端 greedy 推理循环。"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

from .device import DeviceTimer
from .dflash_adapter import DFlashBlockAdapter
from .tree import SpecBlockTreeBuilder
from .verification import TargetTreeVerifier


@dataclass(slots=True)
class IterationStats:
    draft_ms: float
    verify_ms: float
    tree_nodes: int
    accepted_draft_tokens: int
    committed_tokens: int
    # 分项计时（sub-components of draft_ms / verify_ms），用于定位实现开销：
    # tree_build_ms ⊂ draft_ms，cache_compact_ms ⊂ verify_ms。
    tree_build_ms: float = 0.0
    cache_compact_ms: float = 0.0


@dataclass(slots=True)
class GenerationResult:
    output_ids: torch.Tensor
    generated_ids: torch.Tensor
    iterations: list[IterationStats] = field(default_factory=list)
    prefill_ms: float = 0.0

    @property
    def average_accepted_length(self) -> float:
        if not self.iterations:
            return 0.0
        return sum(item.committed_tokens for item in self.iterations) / len(self.iterations)

    @property
    def total_decode_ms(self) -> float:
        return sum(item.draft_ms + item.verify_ms for item in self.iterations)


class DFlashSpecBlockEngine:
    def __init__(
        self,
        target: nn.Module,
        adapter: DFlashBlockAdapter,
        tree_builder: SpecBlockTreeBuilder,
        verifier: TargetTreeVerifier,
        device: torch.device,
    ) -> None:
        self.target = target
        self.adapter = adapter
        self.tree_builder = tree_builder
        self.verifier = verifier
        self.device = device

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
        # 首轮把完整 prompt hidden 作为 DFlash 增量；后续轮只传刚验证的新 token hidden。
        target_update = self.adapter.extract_target_context(output.hidden_states)
        return anchor, output.past_key_values, target_update, timer.elapsed_ms

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        stop_token_ids: set[int] | None = None,
    ) -> GenerationResult:
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("当前实验实现只支持 batch=1 的 input_ids")
        if max_new_tokens < 1:
            empty = torch.empty((1, 0), dtype=torch.long, device=input_ids.device)
            return GenerationResult(output_ids=input_ids, generated_ids=empty)

        stop_token_ids = stop_token_ids or set()
        from transformers import DynamicCache

        anchor, cache, target_update, prefill_ms = self._prefill(input_ids)
        # DFlash 官方使用独立于 target cache 的 DynamicCache；每轮 draft 后裁回已验证前缀。
        draft_cache = DynamicCache()
        generated: list[int] = [anchor]
        stats: list[IterationStats] = []

        while len(generated) < max_new_tokens and anchor not in stop_token_ids:
            with DeviceTimer(self.device) as draft_timer:
                first = self.adapter.propose_first(
                    target_context=target_update,
                    anchor_ids=torch.tensor([anchor], dtype=torch.long, device=self.device),
                    draft_cache=draft_cache,
                    cache_prefix_length=int(cache.get_seq_length()),
                )
            with DeviceTimer(self.device) as tree_timer:
                tree = self.tree_builder.build(first, self.adapter.propose_continuation)

            with DeviceTimer(self.device) as verify_timer:
                verified = self.verifier.verify(
                    anchor_token_id=anchor,
                    tree=tree,
                    cache=cache,
                    target_context=target_update,
                )

            appended = verified.path.token_ids + [verified.path.bonus_token_id]
            remaining = max_new_tokens - len(generated)
            committed = appended[:remaining]
            stop_reached = False
            for index, token in enumerate(committed):
                if token in stop_token_ids:
                    committed = committed[: index + 1]
                    stop_reached = True
                    break
            generated.extend(committed)
            stats.append(
                IterationStats(
                    draft_ms=draft_timer.elapsed_ms + tree_timer.elapsed_ms,
                    verify_ms=verify_timer.elapsed_ms,
                    tree_nodes=len(tree),
                    accepted_draft_tokens=len(verified.path.token_ids),
                    committed_tokens=len(committed),
                    tree_build_ms=tree_timer.elapsed_ms,
                    cache_compact_ms=verified.cache_compact_ms,
                )
            )
            cache = verified.cache
            target_update = verified.target_context
            anchor = verified.path.bonus_token_id

            if stop_reached:
                break

        generated_tensor = torch.tensor(
            generated,
            dtype=torch.long,
            device=self.device,
        ).unsqueeze(0)
        output_ids = torch.cat([input_ids, generated_tensor], dim=1)
        return GenerationResult(
            output_ids=output_ids,
            generated_ids=generated_tensor,
            iterations=stats,
            prefill_ms=prefill_ms,
        )
