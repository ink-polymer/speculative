"""目标模型的 ancestor-only 树验证与 KV cache 压缩。

当前实现只提供 temperature=0 的 lossless greedy 路径。目标模型一次 forward 同时计算 anchor
与全部树节点；每个树节点只能关注旧 KV、anchor、本节点和自己的祖先，不能看到兄弟分支。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn

from .tree import DraftTree


@dataclass(slots=True)
class GreedyPath:
    node_indices: list[int]
    token_ids: list[int]
    bonus_token_id: int


@dataclass(slots=True)
class VerificationResult:
    path: GreedyPath
    cache: object
    target_context: torch.Tensor


def select_greedy_path(current_logits: torch.Tensor, tree: DraftTree) -> GreedyPath:
    """严格按 SpecBlock/EAGLE 的“枚举叶路径，选择最长匹配前缀”执行 greedy 验证。

    ``current_logits[0]`` 属于 anchor，``current_logits[i+1]`` 属于 tree node ``i``。
    不能只在树上局部贪心：同 token 的重复节点可能拥有不同后继，局部选择累计概率最高的
    重复节点会错过另一条更长的 target 匹配路径。
    """
    if current_logits.ndim != 2 or current_logits.shape[0] != len(tree) + 1:
        raise ValueError("current_logits 必须是 [1 + tree_nodes, vocab]")
    if len(tree) == 0:
        bonus = int(current_logits[0].argmax(dim=-1).item())
        return GreedyPath(node_indices=[], token_ids=[], bonus_token_id=bonus)

    paths = tree.retrieve_indices(device=current_logits.device)
    best_nodes: list[int] = []
    best_length = -1
    for row in range(paths.shape[0]):
        node_path = [int(index) for index in paths[row].tolist() if index >= 0]
        accepted: list[int] = []
        source_row = 0
        for node_index in node_path:
            target_token = int(current_logits[source_row].argmax(dim=-1).item())
            if tree.nodes[node_index].token_id != target_token:
                break
            accepted.append(node_index)
            source_row = node_index + 1
        # 与官方 torch.argmax(candidates_accept_length) 一致：同长度时保留第一条叶路径。
        if len(accepted) > best_length:
            best_length = len(accepted)
            best_nodes = accepted

    bonus_row = 0 if not best_nodes else best_nodes[-1] + 1
    bonus = int(current_logits[bonus_row].argmax(dim=-1).item())
    return GreedyPath(
        node_indices=best_nodes,
        token_ids=[tree.nodes[index].token_id for index in best_nodes],
        bonus_token_id=bonus,
    )


def build_tree_attention_mask(
    tree: DraftTree,
    past_length: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """构造 [1,1,S,past+S] additive mask，S=anchor+tree nodes。"""
    node_count = len(tree)
    current_length = node_count + 1
    allowed = torch.zeros((current_length, current_length), dtype=torch.bool, device=device)
    allowed[0, 0] = True
    if node_count:
        allowed[1:, 0] = True
        allowed[1:, 1:] = tree.ancestor_mask(device=device)

    minimum = torch.finfo(dtype).min
    mask = torch.full(
        (1, 1, current_length, past_length + current_length),
        minimum,
        dtype=dtype,
        device=device,
    )
    if past_length:
        mask[..., :past_length] = 0
    current = mask[0, 0, :, past_length:]
    current.masked_fill_(allowed, 0)
    return mask


def _compact_cache(cache: object, keep: torch.Tensor) -> object:
    """按官方 SpecBlock 方式重排并裁剪 target KV cache。

    对真实 ``DynamicCache`` 先把选中 KV 拷贝到前缀，再调用 ``crop``，从而同步 cache layer
    内部长度状态；无 ``crop`` 的测试替身才直接替换张量。
    """
    if keep.ndim != 1 or keep.dtype != torch.long:
        raise ValueError("keep 必须是一维 torch.long 索引")
    target_length = int(keep.numel())

    if hasattr(cache, "layers"):
        for layer in cache.layers:
            if getattr(layer, "keys", None) is not None:
                selected_keys = layer.keys.index_select(-2, keep)
                if hasattr(cache, "crop"):
                    layer.keys[..., :target_length, :].copy_(selected_keys)
                else:
                    layer.keys = selected_keys
            if getattr(layer, "values", None) is not None:
                selected_values = layer.values.index_select(-2, keep)
                if hasattr(cache, "crop"):
                    layer.values[..., :target_length, :].copy_(selected_values)
                else:
                    layer.values = selected_values
        if hasattr(cache, "crop"):
            cache.crop(target_length)
        return cache
    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        selected_keys = [tensor.index_select(-2, keep) for tensor in cache.key_cache]
        selected_values = [tensor.index_select(-2, keep) for tensor in cache.value_cache]
        if hasattr(cache, "crop"):
            for original, selected in zip(cache.key_cache, selected_keys):
                original[..., :target_length, :].copy_(selected)
            for original, selected in zip(cache.value_cache, selected_values):
                original[..., :target_length, :].copy_(selected)
            cache.crop(target_length)
        else:
            cache.key_cache = selected_keys
            cache.value_cache = selected_values
        return cache
    if isinstance(cache, (tuple, list)):
        return tuple(
            (key.index_select(-2, keep), value.index_select(-2, keep))
            for key, value in cache
        )
    raise TypeError(f"无法压缩的 cache 类型: {type(cache)!r}")


class TargetTreeVerifier:
    def __init__(
        self,
        target: nn.Module,
        target_layer_ids: Sequence[int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.target = target
        self.target_layer_ids = [int(x) for x in target_layer_ids]
        self.device = device
        self.dtype = dtype

    def _context_from_hidden(
        self,
        hidden_states: Sequence[torch.Tensor],
        rows: torch.Tensor,
    ) -> torch.Tensor:
        selected = [
            hidden_states[layer_id + 1].index_select(1, rows)
            for layer_id in self.target_layer_ids
        ]
        return torch.cat(selected, dim=-1)

    @torch.inference_mode()
    def verify(
        self,
        anchor_token_id: int,
        tree: DraftTree,
        cache: object,
        target_context: torch.Tensor,
    ) -> VerificationResult:
        past_length = int(cache.get_seq_length())
        tree_tokens = tree.token_tensor(self.device)
        current_ids = torch.cat(
            [torch.tensor([anchor_token_id], dtype=torch.long, device=self.device), tree_tokens]
        ).unsqueeze(0)
        depths = torch.tensor(
            [node.depth for node in tree.nodes],
            dtype=torch.long,
            device=self.device,
        )
        position_ids = torch.cat(
            [
                torch.tensor([past_length], dtype=torch.long, device=self.device),
                past_length + depths,
            ]
        ).unsqueeze(0)
        cache_position = torch.arange(
            past_length, past_length + current_ids.shape[1], dtype=torch.long, device=self.device
        )
        attention_mask = build_tree_attention_mask(
            tree=tree,
            past_length=past_length,
            dtype=self.dtype,
            device=self.device,
        )

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
        path = select_greedy_path(output.logits[0], tree)

        # anchor row=0，tree node i 对应 current row=i+1。
        kept_current_rows = torch.tensor(
            [0] + [index + 1 for index in path.node_indices],
            dtype=torch.long,
            device=self.device,
        )
        # 下一轮 DFlash 只接收本轮新验证的增量 hidden；旧前缀已经保存在 draft cache 中。
        del target_context
        new_target_context = self._context_from_hidden(output.hidden_states, kept_current_rows)

        absolute_keep = torch.cat(
            [
                torch.arange(past_length, dtype=torch.long, device=self.device),
                past_length + kept_current_rows,
            ]
        )
        compacted_cache = _compact_cache(output.past_key_values, absolute_keep)
        return VerificationResult(
            path=path,
            cache=compacted_cache,
            target_context=new_target_context,
        )
