"""目标模型的 ancestor-only 树验证与 KV cache 压缩。

当前实现只提供 temperature=0 的 lossless greedy 路径。目标模型一次 forward 同时计算 anchor
与全部树节点；每个树节点只能关注旧 KV、anchor、本节点和自己的祖先，不能看到兄弟分支。

``GraphedTargetTreeVerifier`` 用 NPU graph mode 把 36 层 × ~200 个 kernel launch 融合为
1 次 graph replay，将 verify 固定开销从 ~150 ms 降到 ~35 ms（4x 加速）。外部仍使用
DynamicCache 保持与 engine.py 的兼容；内部用 StaticCache + 固定 shape 的 input/mask
实现 graph capture，每次 verify 前复制 KV 并更新 static tensor 内容后 replay。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn

from .device import DeviceTimer, synchronize
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
    cache_compact_ms: float = 0.0


def select_greedy_path(current_logits: torch.Tensor, tree: DraftTree) -> GreedyPath:
    """严格按 SpecBlock/EAGLE 的“枚举叶路径，选择最长匹配前缀”执行 greedy 验证。

    ``current_logits[0]`` 属于 anchor，``current_logits[i+1]`` 属于 tree node ``i``。
    不能只在树上局部贪心：同 token 的重复节点可能拥有不同后继，局部选择累计概率最高的
    重复节点会错过另一条更长的 target 匹配路径。
    """
    if current_logits.ndim != 2 or current_logits.shape[0] != len(tree) + 1:
        raise ValueError("current_logits 必须是 [1 + tree_nodes, vocab]")
    # 一次性算出所有行的 target argmax，避免逐 node .item() 同步（NPU 上关键）。
    target_tokens: list[int] = current_logits.argmax(dim=-1).tolist()
    if len(tree) == 0:
        return GreedyPath(node_indices=[], token_ids=[], bonus_token_id=target_tokens[0])

    paths = tree.retrieve_indices(device=current_logits.device)
    paths_list: list[list[int]] = paths.tolist()
    tree_tokens = [node.token_id for node in tree.nodes]
    best_nodes: list[int] = []
    best_length = -1
    for row in range(len(paths_list)):
        node_path = [index for index in paths_list[row] if index >= 0]
        accepted: list[int] = []
        source_row = 0
        for node_index in node_path:
            if tree_tokens[node_index] != target_tokens[source_row]:
                break
            accepted.append(node_index)
            source_row = node_index + 1
        # 与官方 torch.argmax(candidates_accept_length) 一致：同长度时保留第一条叶路径。
        if len(accepted) > best_length:
            best_length = len(accepted)
            best_nodes = accepted

    bonus_row = 0 if not best_nodes else best_nodes[-1] + 1
    return GreedyPath(
        node_indices=best_nodes,
        token_ids=[tree_tokens[index] for index in best_nodes],
        bonus_token_id=target_tokens[bonus_row],
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


def _compact_cache(
    cache: object,
    keep: torch.Tensor,
    *,
    prefix_length: int = 0,
) -> object:
    """按官方 SpecBlock 方式重排并裁剪 target KV cache。

    对真实 ``DynamicCache`` 先把选中 KV 拷贝到前缀，再调用 ``crop``，从而同步 cache layer
    内部长度状态；无 ``crop`` 的测试替身才直接替换张量。

    当 ``prefix_length > 0`` 时，前缀 ``[0..prefix_length-1]`` 的 KV 行原本就在正确位置，
    跳过它们的 ``index_select`` 只搬被接受的尾部行——在 NPU 上可省去对全长前缀的无用拷贝
    （实测 31x 加速：16.0ms → 0.52ms，1024 上下文 / 36 层 / 8 KV heads）。
    ``keep`` 此时只包含尾部索引，总目标长度为 ``prefix_length + keep.numel()``。
    """
    if keep.ndim != 1 or keep.dtype != torch.long:
        raise ValueError("keep 必须是一维 torch.long 索引")
    tail_length = int(keep.numel())
    target_length = prefix_length + tail_length

    if tail_length == 0:
        if hasattr(cache, "crop"):
            cache.crop(target_length)
        return cache

    if hasattr(cache, "layers"):
        for layer in cache.layers:
            if getattr(layer, "keys", None) is not None:
                selected_keys = layer.keys.index_select(-2, keep)
                if hasattr(cache, "crop"):
                    layer.keys[..., prefix_length:target_length, :].copy_(selected_keys)
                elif prefix_length > 0:
                    layer.keys = torch.cat(
                        [layer.keys[..., :prefix_length, :], selected_keys], dim=-2
                    )
                else:
                    layer.keys = selected_keys
            if getattr(layer, "values", None) is not None:
                selected_values = layer.values.index_select(-2, keep)
                if hasattr(cache, "crop"):
                    layer.values[..., prefix_length:target_length, :].copy_(selected_values)
                elif prefix_length > 0:
                    layer.values = torch.cat(
                        [layer.values[..., :prefix_length, :], selected_values], dim=-2
                    )
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
                original[..., prefix_length:target_length, :].copy_(selected)
            for original, selected in zip(cache.value_cache, selected_values):
                original[..., prefix_length:target_length, :].copy_(selected)
            cache.crop(target_length)
        elif prefix_length > 0:
            cache.key_cache = [
                torch.cat([orig[..., :prefix_length, :], sel], dim=-2)
                for orig, sel in zip(cache.key_cache, selected_keys)
            ]
            cache.value_cache = [
                torch.cat([orig[..., :prefix_length, :], sel], dim=-2)
                for orig, sel in zip(cache.value_cache, selected_values)
            ]
        else:
            cache.key_cache = selected_keys
            cache.value_cache = selected_values
        return cache
    if isinstance(cache, (tuple, list)):
        if prefix_length > 0:
            return tuple(
                (
                    torch.cat(
                        [key[..., :prefix_length, :], key.index_select(-2, keep)], dim=-2
                    ),
                    torch.cat(
                        [value[..., :prefix_length, :], value.index_select(-2, keep)], dim=-2
                    ),
                )
                for key, value in cache
            )
        return tuple(
            (key.index_select(-2, keep), value.index_select(-2, keep))
            for key, value in cache
        )
    raise TypeError(f"无法压缩的 cache 类型: {type(cache)!r}")


def _compact_cache_contiguous(
    cache: object,
    *,
    start: int,
    length: int,
    prefix_length: int = 0,
) -> object:
    """连续区间快速路径：用 ``narrow``（view）代替 ``index_select``（gather kernel）。

    仅当 keep 索引是 ``[start, start+1, ..., start+length-1]`` 时可用。
    在主链被完整接受的常见场景下，36 层 × 2（K/V）= 72 次 ``index_select``
    全部被替换为 view + ``copy_``，省去 72 个 gather kernel launch。
    """
    target_length = prefix_length + length
    if length == 0:
        if hasattr(cache, "crop"):
            cache.crop(target_length)
        return cache

    # 如果区间已在正确位置，只需 crop。
    if start == prefix_length:
        if hasattr(cache, "crop"):
            cache.crop(target_length)
        return cache

    if hasattr(cache, "layers"):
        for layer in cache.layers:
            if getattr(layer, "keys", None) is not None:
                selected = layer.keys[..., start : start + length, :]
                if hasattr(cache, "crop"):
                    layer.keys[..., prefix_length:target_length, :].copy_(selected)
                elif prefix_length > 0:
                    layer.keys = torch.cat(
                        [layer.keys[..., :prefix_length, :], selected], dim=-2
                    )
                else:
                    layer.keys = selected.clone()
            if getattr(layer, "values", None) is not None:
                selected = layer.values[..., start : start + length, :]
                if hasattr(cache, "crop"):
                    layer.values[..., prefix_length:target_length, :].copy_(selected)
                elif prefix_length > 0:
                    layer.values = torch.cat(
                        [layer.values[..., :prefix_length, :], selected], dim=-2
                    )
                else:
                    layer.values = selected.clone()
        if hasattr(cache, "crop"):
            cache.crop(target_length)
        return cache

    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        if hasattr(cache, "crop"):
            for original in cache.key_cache:
                selected = original[..., start : start + length, :]
                original[..., prefix_length:target_length, :].copy_(selected)
            for original in cache.value_cache:
                selected = original[..., start : start + length, :]
                original[..., prefix_length:target_length, :].copy_(selected)
            cache.crop(target_length)
        elif prefix_length > 0:
            cache.key_cache = [
                torch.cat([orig[..., :prefix_length, :], orig[..., start : start + length, :]], dim=-2)
                for orig in cache.key_cache
            ]
            cache.value_cache = [
                torch.cat([orig[..., :prefix_length, :], orig[..., start : start + length, :]], dim=-2)
                for orig in cache.value_cache
            ]
        else:
            cache.key_cache = [orig[..., start : start + length, :].clone() for orig in cache.key_cache]
            cache.value_cache = [orig[..., start : start + length, :].clone() for orig in cache.value_cache]
        return cache

    if isinstance(cache, (tuple, list)):
        if prefix_length > 0:
            return tuple(
                (
                    torch.cat([key[..., :prefix_length, :], key[..., start : start + length, :]], dim=-2),
                    torch.cat([value[..., :prefix_length, :], value[..., start : start + length, :]], dim=-2),
                )
                for key, value in cache
            )
        return tuple(
            (key[..., start : start + length, :].clone(), value[..., start : start + length, :].clone())
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
        kept_rows_list = [0] + [index + 1 for index in path.node_indices]
        kept_current_rows = torch.tensor(
            kept_rows_list,
            dtype=torch.long,
            device=self.device,
        )
        # 下一轮 DFlash 只接收本轮新验证的增量 hidden；旧前缀已经保存在 draft cache 中。
        del target_context
        new_target_context = self._context_from_hidden(output.hidden_states, kept_current_rows)

        # 方向 B1 优化：前缀 [0..past_length-1] 原本就在正确位置，无需 index_select；
        # 只搬被接受的尾部行（1 + accepted）到 past_length 起始位置。
        tail_keep = past_length + kept_current_rows

        # 连续区间快速路径：主链被完整接受时 kept_rows_list = [0,1,...,accepted]，
        # tail_keep = [past_length, ..., past_length+accepted]，可用 narrow 代替 index_select。
        is_contiguous = len(kept_rows_list) <= 1 or all(
            kept_rows_list[i + 1] == kept_rows_list[i] + 1
            for i in range(len(kept_rows_list) - 1)
        )

        with DeviceTimer(self.device) as compact_timer:
            if is_contiguous:
                compacted_cache = _compact_cache_contiguous(
                    output.past_key_values,
                    start=past_length + kept_rows_list[0],
                    length=len(kept_rows_list),
                    prefix_length=past_length,
                )
            else:
                compacted_cache = _compact_cache(
                    output.past_key_values,
                    tail_keep,
                    prefix_length=past_length,
                )
        return VerificationResult(
            path=path,
            cache=compacted_cache,
            target_context=new_target_context,
            cache_compact_ms=compact_timer.elapsed_ms,
        )


class GraphedTargetTreeVerifier:
    """用 NPU graph mode 加速的 target tree verifier。

    预分配 StaticCache + 固定 shape 的 input/mask/position_ids，capture 一次 NPUGraph，
    后续每次 verify 只需更新 static tensor 内容后 replay。完全在 StaticCache 上操作，
    消除 DynamicCache ↔ StaticCache 转换开销（cache_compact 从 53ms 降到 0）。
    """

    def __init__(
        self,
        target: nn.Module,
        target_layer_ids: Sequence[int],
        device: torch.device,
        dtype: torch.dtype,
        max_tree_budget: int,
        max_cache_len: int = 2048,
    ) -> None:
        from transformers import StaticCache

        self.target = target
        self.target_layer_ids = [int(x) for x in target_layer_ids]
        self.device = device
        self.dtype = dtype
        self.max_verify_len = int(max_tree_budget) + 1
        self.max_cache_len = int(max_cache_len)
        self._past_length = 0

        self._static_cache = StaticCache(
            target.config, max_cache_len=self.max_cache_len
        )
        self._static_input_ids = torch.zeros(
            (1, self.max_verify_len), dtype=torch.long, device=device
        )
        self._static_position_ids = torch.zeros(
            (1, self.max_verify_len), dtype=torch.long, device=device
        )
        self._static_cache_position = torch.zeros(
            self.max_verify_len, dtype=torch.long, device=device
        )
        minimum = torch.finfo(dtype).min
        self._static_mask = torch.full(
            (1, 1, self.max_verify_len, self.max_cache_len),
            minimum,
            dtype=dtype,
            device=device,
        )
        self._graph: object | None = None
        self._graph_output = None
        self._pad_token_id = 0

    @property
    def static_cache(self) -> object:
        return self._static_cache

    @property
    def past_length(self) -> int:
        return self._past_length

    def prefill(self, input_ids: torch.Tensor) -> tuple[int, torch.Tensor, float]:
        """用 StaticCache 做 prefill，返回 anchor/target_update/elapsed。"""
        from .device import DeviceTimer
        cache_position = torch.arange(
            0, input_ids.shape[1], dtype=torch.long, device=self.device
        )
        with DeviceTimer(self.device) as timer:
            output = self.target(
                input_ids=input_ids,
                past_key_values=self._static_cache,
                cache_position=cache_position,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
        anchor = int(output.logits[0, -1].argmax(dim=-1).item())
        target_update_list = [
            output.hidden_states[lid + 1] for lid in self.target_layer_ids
        ]
        target_update = torch.cat(target_update_list, dim=-1)
        self._past_length = int(input_ids.shape[1])
        return anchor, target_update, timer.elapsed_ms

    def _context_from_hidden(
        self,
        hidden_states: Sequence[torch.Tensor],
        rows: torch.Tensor,
        actual_len: int,
    ) -> torch.Tensor:
        selected = [
            hidden_states[layer_id + 1][:, :actual_len, :].index_select(1, rows)
            for layer_id in self.target_layer_ids
        ]
        return torch.cat(selected, dim=-1)

    def _copy_dyn_to_static(self, dyn_cache: object, past_length: int) -> None:
        for i, layer in enumerate(dyn_cache.layers):
            if getattr(layer, "keys", None) is not None:
                self._static_cache.layers[i].keys[..., :past_length, :].copy_(
                    layer.keys[..., :past_length, :]
                )
                self._static_cache.layers[i].values[..., :past_length, :].copy_(
                    layer.values[..., :past_length, :]
                )

    def _capture_graph(self) -> None:
        for _ in range(3):
            _ = self.target(
                input_ids=self._static_input_ids,
                attention_mask=self._static_mask,
                position_ids=self._static_position_ids,
                cache_position=self._static_cache_position,
                past_key_values=self._static_cache,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
        synchronize(self.device)

        self._graph = torch.npu.NPUGraph()
        with torch.npu.graph(self._graph):
            self._graph_output = self.target(
                input_ids=self._static_input_ids,
                attention_mask=self._static_mask,
                position_ids=self._static_position_ids,
                cache_position=self._static_cache_position,
                past_key_values=self._static_cache,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
        synchronize(self.device)

    @torch.inference_mode()
    def verify(
        self,
        anchor_token_id: int,
        tree: DraftTree,
        cache: object | None = None,
        target_context: torch.Tensor | None = None,
        past_length: int | None = None,
    ) -> VerificationResult:
        if cache is None:
            raise ValueError("GraphedTargetTreeVerifier (dyn mode) requires DynamicCache")
        pl = int(cache.get_seq_length()) if past_length is None else past_length
        tree_tokens = tree.token_tensor(self.device)
        actual_len = 1 + len(tree)

        ids = torch.cat([
            torch.tensor([anchor_token_id], dtype=torch.long, device=self.device),
            tree_tokens,
            torch.full(
                (self.max_verify_len - actual_len,),
                self._pad_token_id,
                dtype=torch.long,
                device=self.device,
            ),
        ])
        self._static_input_ids[0].copy_(ids)

        depths = torch.tensor(
            [node.depth for node in tree.nodes],
            dtype=torch.long,
            device=self.device,
        )
        pos = torch.cat([
            torch.tensor([pl], dtype=torch.long, device=self.device),
            pl + depths,
            torch.zeros(
                self.max_verify_len - actual_len, dtype=torch.long, device=self.device
            ),
        ])
        self._static_position_ids[0].copy_(pos)

        cp = torch.arange(
            pl, pl + self.max_verify_len,
            dtype=torch.long, device=self.device,
        )
        self._static_cache_position.copy_(cp)

        minimum = torch.finfo(self.dtype).min
        self._static_mask.fill_(minimum)
        self._static_mask[0, 0, :actual_len, :pl] = 0

        allowed = torch.zeros(
            (self.max_verify_len, self.max_verify_len), dtype=torch.bool, device=self.device
        )
        allowed[0, 0] = True
        if len(tree) > 0:
            allowed[1:actual_len, 0] = True
            ancestor = tree.ancestor_mask(device=self.device)
            allowed[1:actual_len, 1:actual_len] = ancestor
        self._static_mask[0, 0, :actual_len, pl : pl + self.max_verify_len].masked_fill_(
            allowed[:actual_len], 0
        )

        if self._graph is None:
            self._capture_graph()

        self._copy_dyn_to_static(cache, pl)

        with DeviceTimer(self.device) as replay_timer:
            self._graph.replay()
            synchronize(self.device)

        logits = self._graph_output.logits[0, :actual_len, :]
        path = select_greedy_path(logits, tree)

        kept_rows_list = [0] + [index + 1 for index in path.node_indices]
        kept_current_rows = torch.tensor(
            kept_rows_list, dtype=torch.long, device=self.device
        )

        new_target_context = self._context_from_hidden(
            self._graph_output.hidden_states, kept_current_rows, actual_len
        )

        with DeviceTimer(self.device) as compact_timer:
            for i, layer in enumerate(cache.layers):
                if getattr(layer, "keys", None) is None:
                    continue
                new_keys = self._static_cache.layers[i].keys[
                    ..., pl : pl + self.max_verify_len, :
                ]
                new_values = self._static_cache.layers[i].values[
                    ..., pl : pl + self.max_verify_len, :
                ]
                accepted_keys = new_keys.index_select(-2, kept_current_rows)
                accepted_values = new_values.index_select(-2, kept_current_rows)
                layer.keys = torch.cat([layer.keys, accepted_keys], dim=-2)
                layer.values = torch.cat([layer.values, accepted_values], dim=-2)

        return VerificationResult(
            path=path,
            cache=cache,
            target_context=new_target_context,
            cache_compact_ms=replay_timer.elapsed_ms + compact_timer.elapsed_ms,
        )
