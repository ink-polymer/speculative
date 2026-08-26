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
        self._anchor_buffer = torch.empty(1, dtype=torch.long, device=device)
        # DDTree 构建器声明 requires_rank=False：它只用 log-prob 做全局预算分配，
        # 因此可以跳过 draft 阶段的 top-20 摘要与 rank head 前向。
        self._compute_rank = bool(getattr(tree_builder, "requires_rank", True))
        self._builder_manages_budget = bool(getattr(tree_builder, "manages_budget", False))

    @property
    def _uses_static_target_cache(self) -> bool:
        return bool(getattr(self.verifier, "manages_static_cache", False))

    def _target_cache_length(self, cache: object) -> int:
        if self._uses_static_target_cache:
            return int(self.verifier.past_length)
        return int(cache.get_seq_length())

    @torch.inference_mode()
    def _prefill(self, input_ids: torch.Tensor) -> tuple[int, object, torch.Tensor, float]:
        if self._uses_static_target_cache:
            anchor, target_update, elapsed_ms = self.verifier.prefill(input_ids)
            return anchor, self.verifier.static_cache, target_update, elapsed_ms

        from transformers import DynamicCache

        cache = DynamicCache()
        with DeviceTimer(self.device) as timer:
            output = self.target(
                input_ids=input_ids,
                past_key_values=cache,
                use_cache=True,
                output_hidden_states=True,
                # Keep all hidden states needed by DFlash, but project only the
                # final prompt row through the 151K-token LM head.
                logits_to_keep=1,
                return_dict=True,
            )
        anchor = int(output.logits[0, -1].argmax(dim=-1).item())
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

        # 自适应 tree_budget：根据上一轮接受长度缩放预算，减少低接受时的 verify token 数。
        # 首轮乐观使用满预算；后续按 ratio = accepted / block_size 在 [block_size*2, tree_budget] 间插值。
        prev_accepted = self.tree_builder.block_size
        K = self.tree_builder.block_size
        max_budget = self.tree_builder.tree_budget
        min_budget = min(max_budget, K * 2)

        while len(generated) < max_new_tokens and anchor not in stop_token_ids:
            if self._builder_manages_budget:
                # LatencyAwareDDTree 使用当前 block 的概率质量和实测 GPU 延迟选预算。
                # 传入最大预算，让它一次枚举后选择嵌套前缀；不再使用上一轮接受长度。
                adaptive_budget = max_budget
            else:
                ratio = prev_accepted / K
                adaptive_budget = int(min_budget + ratio * (max_budget - min_budget))
                adaptive_budget = max(min_budget, min(max_budget, adaptive_budget))

            cache_prefix = self._target_cache_length(cache)

            with DeviceTimer(self.device) as draft_timer:
                self._anchor_buffer.fill_(anchor)
                first = self.adapter.propose_first(
                    target_context=target_update,
                    anchor_ids=self._anchor_buffer,
                    draft_cache=draft_cache,
                    cache_prefix_length=cache_prefix,
                    compute_rank=self._compute_rank,
                )
            # Tree topology is Python/host control flow. CUDA events would only time
            # its few kernels and miss the host work, so retain an honest wall clock.
            with DeviceTimer(self.device, use_cuda_events=False) as tree_timer:
                tree = self.tree_builder.build(
                    first, self.adapter.propose_continuation, budget=adaptive_budget
                )

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
            draft_ms = draft_timer.elapsed_ms + tree_timer.elapsed_ms
            observer = getattr(self.tree_builder, "observe", None)
            if observer is not None:
                observer(
                    tree_nodes=len(tree),
                    draft_ms=draft_ms,
                    verify_ms=verify_timer.elapsed_ms,
                    accepted_draft_tokens=len(verified.path.token_ids),
                )
            stats.append(
                IterationStats(
                    draft_ms=draft_ms,
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
            prev_accepted = len(verified.path.token_ids)

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
