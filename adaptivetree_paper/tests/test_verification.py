"""greedy 最长路径与 ancestor-only additive mask 测试。"""

import torch

from dflash_specblock.tree import DraftTree
from dflash_specblock.verification import (
    _compact_cache,
    build_tree_attention_mask,
    select_greedy_path,
)


def _tree() -> DraftTree:
    tree = DraftTree()
    node_a = tree.add_node(4, -1, -0.1, 0, 0, 0)
    tree.add_node(5, -1, -0.2, 0, 0, 1)
    tree.add_node(6, node_a, -0.3, 0, 1, 0)
    return tree


def test_selects_longest_matching_path_and_bonus() -> None:
    tree = _tree()
    logits = torch.zeros(4, 10)
    logits[0, 4] = 10  # anchor -> node 0
    logits[1, 6] = 10  # node 0 -> node 2
    logits[3, 9] = 10  # node 2 -> bonus 9
    path = select_greedy_path(logits, tree)
    assert path.node_indices == [0, 2]
    assert path.token_ids == [4, 6]
    assert path.bonus_token_id == 9


def test_longest_path_beats_local_probability_when_duplicate_tokens_exist() -> None:
    """两个同 token 根节点中，高 draft 概率节点不能覆盖另一条更长的 target 路径。"""
    tree = DraftTree()
    high_probability = tree.add_node(4, -1, -0.01, 0, 0, 0)
    low_probability = tree.add_node(4, -1, -2.0, 0, 0, 0)
    tree.add_node(7, high_probability, -0.02, 0, 1, 0)
    long_child = tree.add_node(6, low_probability, -2.1, 0, 1, 0)
    tree.add_node(8, long_child, -2.2, 0, 2, 0)

    logits = torch.zeros(len(tree) + 1, 12)
    logits[0, 4] = 10
    logits[high_probability + 1, 9] = 10  # 高概率重复节点立刻失败
    logits[low_probability + 1, 6] = 10
    logits[long_child + 1, 8] = 10
    logits[5, 11] = 10

    path = select_greedy_path(logits, tree)
    assert path.node_indices == [low_probability, long_child, 4]
    assert path.token_ids == [4, 6, 8]
    assert path.bonus_token_id == 11


def test_tree_attention_mask_allows_past_and_ancestors_only() -> None:
    tree = _tree()
    mask = build_tree_attention_mask(
        tree,
        past_length=2,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    assert mask.shape == (1, 1, 4, 6)
    assert torch.equal(mask[..., :2], torch.zeros_like(mask[..., :2]))
    # node 2 的 query row=3：允许 anchor col=2、自身 col=5、祖先 node0 col=3；屏蔽 sibling col=4。
    assert mask[0, 0, 3, 2].item() == 0
    assert mask[0, 0, 3, 3].item() == 0
    assert mask[0, 0, 3, 5].item() == 0
    assert mask[0, 0, 3, 4].item() < -1e20


def test_transformers_new_cache_layout_can_be_compacted() -> None:
    class Layer:
        keys = torch.arange(10).reshape(1, 1, 5, 2)
        values = keys + 100

    class Cache:
        layers = [Layer()]

    cache = _compact_cache(Cache(), torch.tensor([0, 2, 4]))
    assert cache.layers[0].keys.shape[-2] == 3
    assert cache.layers[0].keys[0, 0, :, 0].tolist() == [0, 4, 8]


def test_dynamic_cache_style_compaction_updates_length_with_crop() -> None:
    class Layer:
        def __init__(self) -> None:
            self.keys = torch.arange(12).reshape(1, 1, 6, 2)
            self.values = self.keys + 100

    class Cache:
        def __init__(self) -> None:
            self.layers = [Layer()]

        def crop(self, length: int) -> None:
            for layer in self.layers:
                layer.keys = layer.keys[..., :length, :]
                layer.values = layer.values[..., :length, :]

    cache = _compact_cache(Cache(), torch.tensor([0, 1, 4, 5], dtype=torch.long))
    assert cache.layers[0].keys.shape[-2] == 4
    assert cache.layers[0].keys[0, 0, :, 0].tolist() == [0, 2, 8, 10]


def test_installed_transformers_dynamic_cache_can_be_compacted() -> None:
    from transformers import DynamicCache

    cache = DynamicCache()
    keys = torch.arange(12, dtype=torch.float32).reshape(1, 1, 6, 2)
    values = keys + 100
    cache.update(keys, values, layer_idx=0)
    compacted = _compact_cache(cache, torch.tensor([0, 2, 5], dtype=torch.long))
    assert compacted.get_seq_length() == 3
    if hasattr(compacted, "layers"):
        actual = compacted.layers[0].keys
    else:
        actual = compacted.key_cache[0]
    assert actual[0, 0, :, 0].tolist() == [0, 4, 10]


# ---------------------------------------------------------------------------
# 方向 B1：_compact_cache prefix_length 优化路径测试
# ---------------------------------------------------------------------------


def _crop_cache_factory(keys: torch.Tensor) -> type:
    """构建带 crop 的 layers 风格 cache 测试替身。"""

    class Layer:
        def __init__(self) -> None:
            self.keys = keys.clone()
            self.values = self.keys + 100

    class Cache:
        def __init__(self) -> None:
            self.layers = [Layer()]

        def crop(self, length: int) -> None:
            for layer in self.layers:
                layer.keys = layer.keys[..., :length, :]
                layer.values = layer.values[..., :length, :]

    return Cache


def test_compact_cache_with_prefix_length_skips_identity_prefix() -> None:
    """prefix_length > 0 时只搬尾部行，前缀原位不动。"""
    keys = torch.arange(16).reshape(1, 1, 8, 2)  # col 0 = [0,2,4,6,8,10,12,14]
    Cache = _crop_cache_factory(keys)
    # past_length=5, accepted tail = positions [5,7] -> values [10,14]
    cache = _compact_cache(
        Cache(),
        torch.tensor([5, 7], dtype=torch.long),
        prefix_length=5,
    )
    assert cache.layers[0].keys.shape[-2] == 7
    assert cache.layers[0].keys[0, 0, :, 0].tolist() == [0, 2, 4, 6, 8, 10, 14]


def test_compact_cache_prefix_length_matches_full_keep() -> None:
    """prefix_length 优化路径与全量 keep 结果完全一致。"""
    keys = torch.arange(16).reshape(1, 1, 8, 2)
    FullCache = _crop_cache_factory(keys)
    OptCache = _crop_cache_factory(keys)

    full_result = _compact_cache(
        FullCache(),
        torch.tensor([0, 1, 2, 3, 4, 5, 7], dtype=torch.long),
    )
    opt_result = _compact_cache(
        OptCache(),
        torch.tensor([5, 7], dtype=torch.long),
        prefix_length=5,
    )
    assert torch.equal(full_result.layers[0].keys, opt_result.layers[0].keys)
    assert torch.equal(full_result.layers[0].values, opt_result.layers[0].values)


def test_compact_cache_prefix_length_zero_matches_original_behavior() -> None:
    """prefix_length=0 (默认) 时行为与原始完全一致。"""
    keys = torch.arange(10).reshape(1, 1, 5, 2)
    Cache = _crop_cache_factory(keys)
    cache = _compact_cache(Cache(), torch.tensor([0, 2, 4]))
    assert cache.layers[0].keys.shape[-2] == 3
    assert cache.layers[0].keys[0, 0, :, 0].tolist() == [0, 4, 8]


def test_compact_cache_prefix_length_with_dynamic_cache() -> None:
    """DynamicCache + prefix_length 优化路径。"""
    from transformers import DynamicCache

    cache = DynamicCache()
    keys = torch.arange(16, dtype=torch.float32).reshape(1, 1, 8, 2)
    values = keys + 100
    cache.update(keys, values, layer_idx=0)
    compacted = _compact_cache(
        cache,
        torch.tensor([5, 7], dtype=torch.long),
        prefix_length=5,
    )
    assert compacted.get_seq_length() == 7
    if hasattr(compacted, "layers"):
        actual = compacted.layers[0].keys
    else:
        actual = compacted.key_cache[0]
    assert actual[0, 0, :, 0].tolist() == [0, 2, 4, 6, 8, 10, 14]


def test_compact_cache_prefix_length_without_crop_concatenates() -> None:
    """无 crop 的 cache 替身在 prefix_length > 0 时通过 cat 重建。"""

    class Layer:
        keys = torch.arange(16).reshape(1, 1, 8, 2)
        values = keys + 100

    class Cache:
        layers = [Layer()]

    cache = _compact_cache(
        Cache(),
        torch.tensor([5, 7], dtype=torch.long),
        prefix_length=5,
    )
    assert cache.layers[0].keys.shape[-2] == 7
    assert cache.layers[0].keys[0, 0, :, 0].tolist() == [0, 2, 4, 6, 8, 10, 14]


def test_compact_cache_empty_tail_with_prefix_only_crops() -> None:
    """tail 为空时只做 crop，不触发 index_select。"""
    keys = torch.arange(16).reshape(1, 1, 8, 2)
    Cache = _crop_cache_factory(keys)
    cache = _compact_cache(
        Cache(),
        torch.tensor([], dtype=torch.long),
        prefix_length=5,
    )
    assert cache.layers[0].keys.shape[-2] == 5
    assert cache.layers[0].keys[0, 0, :, 0].tolist() == [0, 2, 4, 6, 8]
