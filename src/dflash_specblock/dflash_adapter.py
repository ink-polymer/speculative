"""把 DFlash 官方 Qwen3 checkpoint 暴露为 SpecBlock 所需的 block 接口。

第一块完全沿用 DFlash：目标多层 hidden 经 checkpoint 自带的融合层后，在每个 draft layer
中作为持久 Key/Value，并使用 DFlash 官方相同的 DynamicCache 增量维护方式。后续块是本组合
方法唯一新增的接口：遵循 SpecBlock ``use_draft_condition``，把起点位置缓存的 draft h(L)
作为逐层 KV 条件，绕过只适用于目标多层特征的 ``fc`` projection，但仍保留 ``hidden_norm``。
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from .tree import BlockProposal


class DFlashBlockAdapter:
    def __init__(
        self,
        target: nn.Module,
        draft: nn.Module,
        ranker: nn.Module,
        block_size: int,
        mask_token_id: int | None = None,
    ) -> None:
        self.target = target
        self.draft = draft
        self.ranker = ranker
        self.block_size = int(block_size)
        required = ("target_layer_ids", "layers", "norm", "rotary_emb", "fc", "hidden_norm")
        missing = [name for name in required if not hasattr(draft, name)]
        if missing:
            raise TypeError(f"DFlash checkpoint 缺少官方接口字段: {missing}")
        self.target_layer_ids: list[int] = [int(x) for x in draft.target_layer_ids]
        dflash_config = getattr(draft.config, "dflash_config", {}) or {}
        configured_mask = getattr(draft, "mask_token_id", None)
        if configured_mask is None:
            configured_mask = dflash_config.get("mask_token_id")
        self.mask_token_id = configured_mask if mask_token_id is None else mask_token_id
        if self.mask_token_id is None:
            raise ValueError("DFlash checkpoint 未提供 mask_token_id，请在适配器中显式传入")
        checkpoint_block = int(getattr(draft, "block_size", draft.config.block_size))
        # DFlash 的 block_size 包含一个 clean anchor；本工程 block_size 表示未来位置 K。
        if self.block_size + 1 > checkpoint_block:
            raise ValueError(
                f"请求 K={self.block_size}，但 checkpoint 最多支持 {checkpoint_block - 1} 个未来位置"
            )
        # 方向 A：记录 checkpoint 训练时的完整 block_size（含 anchor），用于构造训练分布
        # 一致的输入。checkpoint 在 block_size=16（1 anchor + 15 mask）上训练，is_causal=False
        # 使块内双向 attention——mask 数量改变每个位置能看到的邻居集合，只喂 K+1 个 token
        # 属于 OOD 输入（实测 hidden 相对差异 45.9%，top-1 一致率仅 75%）。
        self._checkpoint_block = checkpoint_block

    @property
    def hidden_size(self) -> int:
        return int(self.draft.config.hidden_size)

    def extract_target_context(self, hidden_states: Sequence[torch.Tensor]) -> torch.Tensor:
        # Transformers hidden_states[0] 是 embedding 输出，所以 layer id 需要 +1。
        selected = [hidden_states[layer_id + 1] for layer_id in self.target_layer_ids]
        return torch.cat(selected, dim=-1)

    def _noise_embedding(self, anchor_ids: torch.Tensor) -> torch.Tensor:
        if anchor_ids.ndim != 1:
            raise ValueError("anchor_ids 必须是 [B]")
        batch = anchor_ids.shape[0]
        # 方向 A：始终构造 checkpoint 训练分布对应的完整输入长度（1 anchor + (block_size-1) mask），
        # 即使只取用前 K 位输出用于建树，避免 OOD 输入导致 draft 质量下降。
        # 实测：喂 5 token vs 16 token，hidden 相对差异 45.9%，top-1 一致率仅 75%。
        full_noise_length = self._checkpoint_block - 1
        masks = torch.full(
            (batch, full_noise_length),
            int(self.mask_token_id),
            dtype=torch.long,
            device=anchor_ids.device,
        )
        noise_ids = torch.cat([anchor_ids[:, None], masks], dim=1)
        return self.target.get_input_embeddings()(noise_ids)

    @staticmethod
    def _position_ids(
        batch: int,
        start: int,
        added_length: int,
        device: torch.device,
    ) -> torch.Tensor:
        """生成 DFlash 新增 KV 的绝对 RoPE 位置，与官方 cache slice 语义一致。"""
        return (
            torch.arange(start, start + added_length, dtype=torch.long, device=device)
            .unsqueeze(0)
            .expand(batch, -1)
        )

    @staticmethod
    def _cache_length(cache: object | None) -> int:
        return 0 if cache is None else int(cache.get_seq_length())

    @torch.inference_mode()
    def draft_first_raw(
        self,
        target_context: torch.Tensor,
        anchor_ids: torch.Tensor,
        draft_cache: object | None = None,
        cache_prefix_length: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """执行 DFlash 官方第一块前向，返回 ``[B,K,V]`` logits 与 ``[B,K,H]`` hidden。

        ``target_context`` 是自上次调用后刚被 target 验证的增量 hidden。若提供 draft cache，
        调用后会像 DFlash 官方生成循环一样裁回 ``cache_prefix_length``，丢弃当前未验证的
        anchor/mask KV，只保留已经验证前缀的条件 KV。
        """
        noise_embedding = self._noise_embedding(anchor_ids)
        cache_length = self._cache_length(draft_cache)
        context_length = int(target_context.shape[1])
        if draft_cache is not None:
            if cache_prefix_length is None:
                raise ValueError("使用 draft_cache 时必须给出 cache_prefix_length")
            if cache_length + context_length != int(cache_prefix_length):
                raise ValueError(
                    "DFlash cache 与 target 增量不连续: "
                    f"cache={cache_length}, update={context_length}, prefix={cache_prefix_length}"
                )
        position_ids = self._position_ids(
            batch=anchor_ids.shape[0],
            start=cache_length,
            added_length=context_length + noise_embedding.shape[1],
            device=anchor_ids.device,
        )
        hidden_all = self.draft(
            target_hidden=target_context,
            noise_embedding=noise_embedding,
            position_ids=position_ids,
            past_key_values=draft_cache,
            use_cache=draft_cache is not None,
            is_causal=False,
        )
        if not isinstance(hidden_all, torch.Tensor):
            hidden_all = hidden_all.last_hidden_state
        # 方向 A：输出长度为 checkpoint_block（如 16），取前 K 位（跳过 position 0 的 anchor），
        # 而非末尾 K 位。当输入长度 == K+1 时两种切法等价；补齐到 checkpoint_block 后只有
        # [1:1+K] 是正确的——[-K:] 会取到末尾无关位置。
        hidden = hidden_all[:, 1 : 1 + self.block_size, :]
        logits = self.target.get_output_embeddings()(hidden)
        if draft_cache is not None:
            if not hasattr(draft_cache, "crop"):
                raise TypeError("DFlash 增量缓存必须实现 crop(length)")
            draft_cache.crop(int(cache_prefix_length))
        return logits, hidden

    @torch.inference_mode()
    def propose_first(
        self,
        target_context: torch.Tensor,
        anchor_ids: torch.Tensor,
        draft_cache: object | None = None,
        cache_prefix_length: int | None = None,
    ) -> BlockProposal:
        logits, hidden = self.draft_first_raw(
            target_context,
            anchor_ids,
            draft_cache=draft_cache,
            cache_prefix_length=cache_prefix_length,
        )
        # Top-k ordering only depends on the already-quantized logits.  Keep
        # the full vocabulary tensor in its model dtype and promote the 20
        # retained values, avoiding a B x K x V FP32 temporary on GPU.
        top20_values, top20_ids = torch.topk(logits, k=20, dim=-1)
        top20_values = top20_values.float()
        rank_logits = self.ranker(hidden, logits, top20_values=top20_values)
        return BlockProposal(
            logits=logits,
            hidden=hidden,
            rank_logits=rank_logits,
            top20_values=top20_values,
            top20_ids=top20_ids,
        )

    @torch.inference_mode()
    def draft_continuation_raw(
        self,
        draft_context: torch.Tensor,
        anchor_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """执行创新部分的后续 diffusion block，返回原始 logits 与 hidden。"""
        if draft_context.ndim != 2:
            raise ValueError("draft_context 必须是 [B,H]")
        hidden_states = self._noise_embedding(anchor_ids)
        # SpecBlock 后续块用 cached h(L) 替代 target 多层特征，因此绕过只适用于
        # ``len(target_layer_ids) * H -> H`` 的 ``fc``；但官方 use_draft_condition 分支仍然
        # 执行 condition_norm，本工程对应 DFlash 的 ``hidden_norm``。保留它，注入 KV 的尺度
        # 才与第一块 ``hidden_norm(fc(target_hidden))`` 一致。
        context = self.draft.hidden_norm(draft_context[:, None, :])
        position_ids = self._position_ids(
            batch=anchor_ids.shape[0],
            start=0,
            added_length=context.shape[1] + hidden_states.shape[1],
            device=anchor_ids.device,
        )
        position_embeddings = self.draft.rotary_emb(hidden_states, position_ids)

        # 直接进入 draft layers；每层都把 cached h(L) 作为 target_hidden 注入 Key/Value。
        for layer in self.draft.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                target_hidden=context,
                attention_mask=None,
                position_ids=position_ids,
                past_key_value=None,
                use_cache=False,
                position_embeddings=position_embeddings,
                is_causal=False,
            )
        hidden_all = self.draft.norm(hidden_states)
        # 方向 A：与 draft_first_raw 一致，取前 K 位输出（跳过 anchor）。
        hidden = hidden_all[:, 1 : 1 + self.block_size, :]
        logits = self.target.get_output_embeddings()(hidden)
        return logits, hidden

    @torch.inference_mode()
    def propose_continuation(
        self,
        draft_context: torch.Tensor,
        anchor_ids: torch.Tensor,
    ) -> BlockProposal:
        """批量扩展 pending 节点；draft_context 是各起点缓存的 h(L)。"""
        logits, hidden = self.draft_continuation_raw(draft_context, anchor_ids)
        top20_values, top20_ids = torch.topk(logits, k=20, dim=-1)
        top20_values = top20_values.float()
        rank_logits = self.ranker(hidden, logits, top20_values=top20_values)
        return BlockProposal(
            logits=logits,
            hidden=hidden,
            rank_logits=rank_logits,
            top20_values=top20_values,
            top20_ids=top20_ids,
        )
