import heapq
import time
from collections import Counter, defaultdict
from functools import lru_cache
from types import SimpleNamespace

from loguru import logger
import numpy as np
import torch
from transformers import AutoModelForCausalLM, DynamicCache

from model import DFlashDraftModel, sample, extract_context_feature
from dflash import dflash_generate, cuda_time, empty_stage_times


DDTREE_STAGE_ORDER = ("draft", "tree_build", "tree_compile", "verify", "commit")
DDTREE_TREE_BUILD_STAGE_ORDER = ("tree_build_copy", "tree_build_heap", "tree_build_visibility")


_CPP_COMPACT_ENABLED = False


class NgramContinuationStore:
    """Frequency-ranked continuation paths built only from training token ids."""

    def __init__(
        self,
        token_sequences: list[list[int]],
        *,
        horizon: int,
        max_ngram: int = 4,
    ) -> None:
        if horizon <= 0 or max_ngram <= 0:
            raise ValueError("invalid ngram continuation store configuration")
        self.horizon = int(horizon)
        self.max_ngram = int(max_ngram)
        counts: dict[tuple[int, ...], Counter[tuple[int, ...]]] = defaultdict(Counter)
        for sequence in token_sequences:
            values = [int(token) for token in sequence]
            for continuation_start in range(1, len(values)):
                continuation = tuple(
                    values[continuation_start : continuation_start + self.horizon]
                )
                if not continuation:
                    continue
                for ngram in range(1, min(self.max_ngram, continuation_start) + 1):
                    key = tuple(values[continuation_start - ngram : continuation_start])
                    counts[key][continuation] += 1
        self._continuations = {
            key: tuple(path for path, _count in counter.most_common(16))
            for key, counter in counts.items()
        }

    def find_paths(
        self, history_token_ids: list[int], path_count: int
    ) -> list[list[int]]:
        paths: list[list[int]] = []
        seen: set[tuple[int, ...]] = set()
        for ngram in range(min(self.max_ngram, len(history_token_ids)), 0, -1):
            key = tuple(history_token_ids[-ngram:])
            for continuation in self._continuations.get(key, ()):
                if continuation in seen:
                    continue
                seen.add(continuation)
                paths.append(list(continuation))
                if len(paths) >= path_count:
                    return paths
        return paths


@lru_cache(maxsize=1)
def load_cpp_compact_module():
    try:
        from torch.utils.cpp_extension import load_inline
    except Exception as exc:
        logger.warning(f"torch.utils.cpp_extension is unavailable; falling back to Python cache compaction. {exc}")
        return None

    cpp_source = r"""
torch::Tensor compact_tail_inplace(torch::Tensor cache_tensor, int64_t past_length, torch::Tensor keep_current_indices) {
    TORCH_CHECK(cache_tensor.dim() >= 2, "cache_tensor must have rank >= 2");
    TORCH_CHECK(keep_current_indices.dim() == 1, "keep_current_indices must be a 1D tensor");
    TORCH_CHECK(keep_current_indices.scalar_type() == torch::kLong, "keep_current_indices must have dtype torch.long");
    TORCH_CHECK(cache_tensor.device() == keep_current_indices.device(), "cache_tensor and keep_current_indices must be on the same device");

    const int64_t seq_dim = cache_tensor.dim() - 2;
    TORCH_CHECK(past_length >= 0, "past_length must be non-negative");
    TORCH_CHECK(past_length <= cache_tensor.size(seq_dim), "past_length exceeds cache sequence length");

    const int64_t current_length = cache_tensor.size(seq_dim) - past_length;
    if (current_length <= 0) {
        return cache_tensor;
    }

    const int64_t keep_count = keep_current_indices.numel();
    TORCH_CHECK(keep_count >= 0, "keep_count must be non-negative");
    TORCH_CHECK(keep_count <= current_length, "keep_count exceeds appended window length");

    if (keep_count == 0 || keep_count == current_length) {
        return cache_tensor;
    }

    auto tail = cache_tensor.narrow(seq_dim, past_length, current_length);
    auto kept_tail = tail.index_select(seq_dim, keep_current_indices);
    cache_tensor.narrow(seq_dim, past_length, keep_count).copy_(kept_tail);
    return cache_tensor;
}
"""
    try:
        module = load_inline(
            name="ddtree_compact_tail_ext_v1",
            cpp_sources=[cpp_source],
            functions=["compact_tail_inplace"],
            extra_cflags=["-O3"],
            verbose=False,
        )
        logger.info("Loaded inline C++ tail cache compaction extension for DDTree.")
        return module
    except Exception as exc:
        logger.warning(
            f"Failed to build inline C++ tail cache compaction extension; falling back to Python implementation. {exc}"
        )
        return None


def maybe_enable_cpp_compact(enabled: bool) -> None:
    global _CPP_COMPACT_ENABLED
    _CPP_COMPACT_ENABLED = enabled
    if enabled:
        load_cpp_compact_module()


def build_ddtree_tree(
    draft_logits: torch.Tensor,
    budget: int,
    tree_builder=None,
    tree_temperature: float = 1.0,
    tree_depth_bonus: float = 0.0,
    rank_bucket_logits: torch.Tensor | None = None,
    rank_choice_logits: torch.Tensor | None = None,
    rank_score_blend: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, list[int], list[dict[int, int]], torch.Tensor, dict[str, float]]:
    build_subtimes = empty_stage_times(DDTREE_TREE_BUILD_STAGE_ORDER)

    if budget <= 0 or draft_logits.shape[0] == 0:
        visibility = torch.zeros((1, 1), dtype=torch.bool)
        visibility[0, 0] = True
        return (
            torch.empty(0, dtype=torch.long),
            torch.empty(0, dtype=torch.long),
            [-1],
            [dict()],
            visibility,
            build_subtimes,
        )

    topk = min(budget, draft_logits.shape[-1])
    depth_limit = int(draft_logits.shape[0])

    copy_start = cuda_time()
    if tree_temperature <= 0.0:
        raise ValueError("tree_temperature must be positive")
    logits = draft_logits.float() / float(tree_temperature)
    top_logits, top_token_ids = torch.topk(logits, k=topk, dim=-1)
    log_z = torch.logsumexp(logits, dim=-1, keepdim=True)
    draft_top_log_probs = top_logits - log_z
    if rank_bucket_logits is not None and rank_choice_logits is not None:
        raise ValueError("only one rank calibration head may be used")
    if rank_bucket_logits is None and rank_choice_logits is None:
        top_log_probs = draft_top_log_probs
    else:
        if not 0.0 <= rank_score_blend <= 1.0:
            raise ValueError("rank_score_blend must be in [0, 1]")
        rank_probabilities = torch.empty_like(top_logits, dtype=torch.float32)
        if rank_choice_logits is not None:
            if rank_choice_logits.shape != (depth_limit, 11):
                raise ValueError("rank_choice_logits must have shape [horizon, 11]")
            choice_probabilities = torch.softmax(
                rank_choice_logits.float(), dim=-1
            )
            direct_count = min(topk, 10)
            rank_probabilities[:, :direct_count] = choice_probabilities[
                :, :direct_count
            ]
            if topk > 10:
                rank_probabilities[:, 10:] = (
                    choice_probabilities[:, 10:11] / float(topk - 10)
                )
        else:
            if rank_bucket_logits.shape != (depth_limit, 4):
                raise ValueError("rank_bucket_logits must have shape [horizon, 4]")
            bucket_probabilities = torch.softmax(
                rank_bucket_logits.float(), dim=-1
            )
            rank_probabilities[:, :1] = bucket_probabilities[:, :1]
            rank_probabilities[:, 1:4] = bucket_probabilities[:, 1:2] / 3.0
            rank_probabilities[:, 4:10] = bucket_probabilities[:, 2:3] / 6.0
            if topk > 10:
                rank_probabilities[:, 10:] = (
                    bucket_probabilities[:, 3:4] / float(topk - 10)
                )
        rank_log_probs = rank_probabilities.clamp_min(1e-12).log()
        blended_scores = (
            (1.0 - float(rank_score_blend)) * draft_top_log_probs
            + float(rank_score_blend) * rank_log_probs
        )
        top_log_probs = torch.log_softmax(blended_scores, dim=-1)
    top_log_probs_cpu = top_log_probs.to(device="cpu", dtype=torch.float32)
    top_token_ids_cpu = top_token_ids.to(device="cpu", dtype=torch.long)
    build_subtimes["tree_build_copy"] = cuda_time() - copy_start

    top_log_probs_np = top_log_probs_cpu.numpy()
    top_token_ids_np = top_token_ids_cpu.numpy()

    heap_start = time.perf_counter()
    first_logw = float(top_log_probs_np[0, 0]) + float(tree_depth_bonus)
    heap: list[tuple[float, tuple[int, ...], int, int, int, float]] = [(-first_logw, (0,), 0, 1, 0, first_logw)]

    node_token_ids_np = np.empty(budget, dtype=np.int64)
    node_depths_np = np.empty(budget, dtype=np.int64)
    node_scores_np = np.empty(budget, dtype=np.float64)
    parents_np = np.empty(budget + 1, dtype=np.int32)
    parents_np[0] = -1
    node_count = 0

    while heap and node_count < budget:
        _, ranks, parent_index, depth, rank, logw = heapq.heappop(heap)

        token_id = int(top_token_ids_np[depth - 1, rank])
        current_index = node_count + 1
        node_token_ids_np[node_count] = token_id
        node_depths_np[node_count] = depth
        node_scores_np[node_count] = logw
        parents_np[current_index] = parent_index
        node_count += 1

        if rank + 1 < topk:
            sibling_ranks = ranks[:-1] + (rank + 1,)
            sibling_logw = logw - float(top_log_probs_np[depth - 1, rank]) + float(top_log_probs_np[depth - 1, rank + 1])
            heapq.heappush(heap, (-sibling_logw, sibling_ranks, parent_index, depth, rank + 1, sibling_logw))

        if depth < depth_limit:
            child_ranks = ranks + (0,)
            child_logw = (
                logw
                + float(top_log_probs_np[depth, 0])
                + float(tree_depth_bonus)
            )
            heapq.heappush(heap, (-child_logw, child_ranks, current_index, depth + 1, 0, child_logw))

    if tree_builder is not None:
        node_count = int(tree_builder._select_node_count(node_scores_np[:node_count]))
        if node_count < 0 or node_count > budget:
            raise ValueError(f"RL tree policy selected invalid node count {node_count}")

    # Reconstruct only the selected prefix.  Building maps after policy
    # truncation prevents links to discarded nodes and keeps verification
    # identical to a fixed-budget official DDTree of the selected size.
    child_maps: list[dict[int, int]] = [dict() for _ in range(node_count + 1)]
    for current_index in range(1, node_count + 1):
        parent_index = int(parents_np[current_index])
        token_id = int(node_token_ids_np[current_index - 1])
        child_maps[parent_index][token_id] = current_index

    build_subtimes["tree_build_heap"] = time.perf_counter() - heap_start

    visibility_start = time.perf_counter()
    current_length = 1 + node_count
    visibility_np = np.zeros((current_length, current_length), dtype=np.bool_)
    visibility_np[0, 0] = True
    for index in range(1, current_length):
        parent_index = int(parents_np[index])
        visibility_np[index, :index] = visibility_np[parent_index, :index]
        visibility_np[index, index] = True
    build_subtimes["tree_build_visibility"] = time.perf_counter() - visibility_start

    node_token_ids = torch.from_numpy(node_token_ids_np[:node_count])
    node_depths = torch.from_numpy(node_depths_np[:node_count])
    visibility = torch.from_numpy(visibility_np)
    parents = parents_np[:current_length].tolist()

    return node_token_ids, node_depths, parents, child_maps, visibility, build_subtimes


def build_ddtree_tree_with_policy(draft_logits: torch.Tensor, tree_builder):
    """Build the official tensor representation from an RL-selected DDTree prefix.

    The policy builder emits the same parent-before-child best-first prefix as
    ``build_ddtree_tree``.  This adapter changes only the selected node count;
    target-side tree compilation and verification continue through the original
    DDTree functions below.
    """
    tree = tree_builder.build_from_logits(
        draft_logits,
        budget=int(tree_builder.tree_budget),
    )


def find_prompt_lookup_paths(
    history_token_ids: list[int],
    horizon: int,
    path_count: int,
    max_ngram: int = 4,
) -> list[list[int]]:
    """Find recent prompt/history continuations matching the current suffix."""
    if horizon <= 0 or path_count <= 0 or max_ngram <= 0:
        return []
    current = len(history_token_ids)
    paths: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    for ngram in range(min(max_ngram, current), 0, -1):
        suffix = history_token_ids[current - ngram : current]
        latest_start = current - ngram - 1
        for match_start in range(latest_start, -1, -1):
            if history_token_ids[match_start : match_start + ngram] != suffix:
                continue
            continuation_start = match_start + ngram
            continuation = history_token_ids[
                continuation_start : min(continuation_start + horizon, current)
            ]
            key = tuple(continuation)
            if not key or key in seen:
                continue
            seen.add(key)
            paths.append(continuation)
            if len(paths) >= path_count:
                return paths
    return paths


def add_paths_to_tree(
    tree_result,
    paths: list[list[int]],
):
    """Union arbitrary lookup paths into an existing DDTree."""
    (
        node_token_ids,
        node_depths,
        parents,
        child_maps,
        _visibility,
        build_subtimes,
    ) = tree_result
    node_tokens = [int(value) for value in node_token_ids.tolist()]
    depths = [int(value) for value in node_depths.tolist()]
    parents = list(parents)
    child_maps = [dict(children) for children in child_maps]
    for path in paths:
        parent = 0
        for depth, token in enumerate(path, start=1):
            child = child_maps[parent].get(int(token))
            if child is None:
                child = len(parents)
                child_maps[parent][int(token)] = child
                parents.append(parent)
                child_maps.append({})
                node_tokens.append(int(token))
                depths.append(depth)
            parent = child

    count = len(parents)
    visibility_np = np.zeros((count, count), dtype=np.bool_)
    visibility_np[0, 0] = True
    for index in range(1, count):
        visibility_np[index, :index] = visibility_np[parents[index], :index]
        visibility_np[index, index] = True
    return (
        torch.tensor(node_tokens, dtype=torch.long),
        torch.tensor(depths, dtype=torch.long),
        parents,
        child_maps,
        torch.from_numpy(visibility_np),
        build_subtimes,
    )


def reorder_tree_depth_first(tree_result):
    """Reorder an unchanged tree so its greedy-first paths become contiguous."""
    (
        node_token_ids,
        node_depths,
        parents,
        child_maps,
        _visibility,
        build_subtimes,
    ) = tree_result
    order: list[int] = []

    def visit(parent: int) -> None:
        for child in child_maps[parent].values():
            order.append(child)
            visit(child)

    visit(0)
    if len(order) != len(parents) - 1:
        raise ValueError("tree contains unreachable nodes")
    old_to_new = {0: 0}
    old_to_new.update({old: new for new, old in enumerate(order, start=1)})
    tokens = [int(node_token_ids[old - 1]) for old in order]
    depths = [int(node_depths[old - 1]) for old in order]
    new_parents = [-1] + [old_to_new[int(parents[old])] for old in order]
    new_child_maps: list[dict[int, int]] = [dict() for _ in new_parents]
    for new_index, (token, parent) in enumerate(
        zip(tokens, new_parents[1:]), start=1
    ):
        new_child_maps[parent][token] = new_index
    count = len(new_parents)
    visibility_np = np.zeros((count, count), dtype=np.bool_)
    visibility_np[0, 0] = True
    for index in range(1, count):
        parent = new_parents[index]
        visibility_np[index, :index] = visibility_np[parent, :index]
        visibility_np[index, index] = True
    return (
        torch.tensor(tokens, dtype=torch.long),
        torch.tensor(depths, dtype=torch.long),
        new_parents,
        new_child_maps,
        torch.from_numpy(visibility_np),
        build_subtimes,
    )
    node_token_ids = torch.tensor(
        [int(node.token_id) for node in tree.nodes], dtype=torch.long
    )
    node_depths = torch.tensor(
        [int(node.slot_index) + 1 for node in tree.nodes], dtype=torch.long
    )
    parents = [-1]
    child_maps: list[dict[int, int]] = [dict() for _ in range(len(tree) + 1)]
    for index, node in enumerate(tree.nodes, start=1):
        parent = int(node.parent) + 1
        parents.append(parent)
        child_maps[parent][int(node.token_id)] = index
    visibility = torch.zeros((len(tree) + 1, len(tree) + 1), dtype=torch.bool)
    visibility[0, 0] = True
    for index in range(1, len(tree) + 1):
        parent = parents[index]
        visibility[index, :index] = visibility[parent, :index]
        visibility[index, index] = True
    return (
        node_token_ids,
        node_depths,
        parents,
        child_maps,
        visibility,
        empty_stage_times(DDTREE_TREE_BUILD_STAGE_ORDER),
    )


def compile_ddtree_tree(
    root_token_id: torch.Tensor,
    start: int,
    node_token_ids: torch.Tensor,
    node_depths: torch.Tensor,
    visibility_cpu: torch.Tensor,
    past_length: int,
    dtype: torch.dtype,
    device: torch.device,
    verify_input_ids_buffer: torch.Tensor,
    verify_position_ids_buffer: torch.Tensor,
    attention_mask_buffer: torch.Tensor,
    tree_visibility_buffer: torch.Tensor,
    previous_tree_start: int,
    previous_tree_length: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    current_length = 1 + int(node_token_ids.numel())

    if previous_tree_length > 0:
        attention_mask_buffer[0, 0, :previous_tree_length, previous_tree_start : previous_tree_start + previous_tree_length] = 0

    verify_input_ids = verify_input_ids_buffer[:, :current_length]
    verify_input_ids[0, 0] = root_token_id
    if current_length > 1:
        verify_input_ids[0, 1:current_length].copy_(node_token_ids, non_blocking=False)

    verify_position_ids = verify_position_ids_buffer[:, :current_length]
    verify_position_ids[0, 0] = start
    if current_length > 1:
        verify_position_ids[0, 1:current_length].copy_(node_depths, non_blocking=False)
        verify_position_ids[0, 1:current_length].add_(start)

    visibility = tree_visibility_buffer[:current_length, :current_length]
    visibility.copy_(visibility_cpu, non_blocking=False)

    tree_block = attention_mask_buffer[0, 0, :current_length, past_length : past_length + current_length]
    tree_block.fill_(torch.finfo(dtype).min)
    tree_block.masked_fill_(visibility, 0)

    attention_mask = attention_mask_buffer[:, :, :current_length, : past_length + current_length]
    return verify_input_ids, verify_position_ids, attention_mask, past_length, current_length


def follow_verified_tree(child_maps: list[dict[int, int]], posterior: torch.Tensor) -> tuple[list[int], int]:
    posterior_tokens = posterior[0].tolist()
    accepted_indices = [0]
    current_index = 0
    next_token = int(posterior_tokens[current_index])

    while next_token in child_maps[current_index]:
        current_index = child_maps[current_index][next_token]
        accepted_indices.append(current_index)
        next_token = int(posterior_tokens[current_index])

    return accepted_indices, next_token


def follow_verified_tree_sparse_lm_head(
    child_maps: list[dict[int, int]],
    last_hidden_state: torch.Tensor,
    lm_head: torch.nn.Module,
    temperature: float,
) -> tuple[list[int], int]:
    """Project likely root-to-leaf segments until the exact walk terminates.

    A dense causal-LM forward projects every verified tree node to the full
    vocabulary even though DDTree commits exactly one root-to-leaf walk.  The
    target transformer and ancestor mask are unchanged here.  Each phase
    projects the first-child path below the current node as one efficient GEMM.
    If target verification takes a different child, another segment starts at
    that child.  Thus every returned token is still computed by the target
    LM head at its exact DDTree row, while unvisited branches are not projected.
    """
    accepted_indices = [0]
    current_index = 0
    while True:
        segment = [current_index]
        while child_maps[segment[-1]]:
            predicted_child = next(iter(child_maps[segment[-1]].values()))
            segment.append(int(predicted_child))

        segment_tensor = torch.tensor(
            segment, dtype=torch.long, device=last_hidden_state.device
        )
        segment_hidden = last_hidden_state.index_select(1, segment_tensor)
        segment_tokens = sample(lm_head(segment_hidden), temperature)[0].tolist()

        for offset, node_index in enumerate(segment):
            next_token = int(segment_tokens[offset])
            child_index = child_maps[node_index].get(next_token)
            if child_index is None:
                return accepted_indices, next_token
            child_index = int(child_index)
            accepted_indices.append(child_index)
            if offset + 1 < len(segment) and child_index == segment[offset + 1]:
                continue
            current_index = child_index
            break


def _compact_appended_window(cache_tensor: torch.Tensor, past_length: int, keep_current_indices: torch.Tensor) -> None:
    current_length = cache_tensor.shape[-2] - past_length
    if current_length <= 0:
        return

    keep_count = keep_current_indices.numel()
    if keep_count == 0 or keep_count == current_length:
        return

    if _CPP_COMPACT_ENABLED:
        module = load_cpp_compact_module()
        if module is not None:
            module.compact_tail_inplace(cache_tensor, past_length, keep_current_indices)
            return

    kept_tail = cache_tensor.narrow(-2, past_length, current_length).index_select(-2, keep_current_indices)
    cache_tensor.narrow(-2, past_length, keep_count).copy_(kept_tail)


def compact_dynamic_cache(past_key_values: DynamicCache, past_length: int, keep_current_indices: list[int]) -> None:
    if len(keep_current_indices) == 0:
        past_key_values.crop(past_length)
        return

    # A DFS-ordered tree makes the common greedy path a leading contiguous
    # slice.  Cropping is then sufficient and avoids two index_select/copy
    # kernels per transformer layer.
    if keep_current_indices == list(range(len(keep_current_indices))):
        past_key_values.crop(past_length + len(keep_current_indices))
        return

    keep_tensor_by_device: dict[torch.device, torch.Tensor] = {}

    def get_keep_tensor(device: torch.device) -> torch.Tensor:
        if device not in keep_tensor_by_device:
            keep_tensor_by_device[device] = torch.tensor(keep_current_indices, dtype=torch.long, device=device)
        return keep_tensor_by_device[device]

    if hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
        for layer_idx in range(len(past_key_values.key_cache)):
            key_cache = past_key_values.key_cache[layer_idx]
            value_cache = past_key_values.value_cache[layer_idx]
            keep_tensor = get_keep_tensor(key_cache.device)
            _compact_appended_window(key_cache, past_length, keep_tensor)
            _compact_appended_window(value_cache, past_length, keep_tensor)
        past_key_values.crop(past_length + len(keep_current_indices))
        return

    if hasattr(past_key_values, "layers"):
        for layer in past_key_values.layers:
            if not hasattr(layer, "keys") or layer.keys is None or layer.keys.numel() == 0:
                continue
            keep_tensor = get_keep_tensor(layer.keys.device)
            _compact_appended_window(layer.keys, past_length, keep_tensor)
            _compact_appended_window(layer.values, past_length, keep_tensor)
        past_key_values.crop(past_length + len(keep_current_indices))
        return

    raise RuntimeError("Unsupported DynamicCache layout for DDTree cache compaction.")


@torch.inference_mode()
def ddtree_generate(
    model: DFlashDraftModel,
    target: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    mask_token_id: int,
    max_new_tokens: int,
    block_size: int,
    stop_token_ids: list[int],
    temperature: float = 0.0,
    tree_budget: int | None = None,
    save_tree_traces: bool = False,
    tree_builder=None,
    tree_temperature: float = 1.0,
    tree_depth_bonus: float = 0.0,
    lookup_path_count: int = 0,
    lookup_max_ngram: int = 4,
    corpus_lookup_store: NgramContinuationStore | None = None,
    corpus_lookup_path_count: int = 0,
    depth_first_order: bool = False,
    rank_head=None,
    rank_choice_head=None,
    rank_score_blend: float = 1.0,
    draft_refinement_steps: int = 1,
    fast_timing: bool = False,
    sparse_lm_head: bool = False,
) -> SimpleNamespace:
    if block_size <= 1:
        return dflash_generate(
            model=model,
            target=target,
            input_ids=input_ids,
            mask_token_id=mask_token_id,
            max_new_tokens=max_new_tokens,
            block_size=block_size,
            stop_token_ids=stop_token_ids,
            temperature=temperature,
        )
    if draft_refinement_steps < 1:
        raise ValueError("draft_refinement_steps must be at least 1")

    num_input_tokens = input_ids.shape[1]
    max_length = num_input_tokens + max_new_tokens
    draft_horizon = block_size - 1
    tree_budget = draft_horizon if tree_budget is None else max(tree_budget, 0)
    if tree_builder is not None and int(tree_builder.tree_budget) != tree_budget:
        raise ValueError("tree_builder.tree_budget must match tree_budget")
    if (
        lookup_path_count < 0
        or corpus_lookup_path_count < 0
        or lookup_max_ngram < 1
    ):
        raise ValueError("invalid prompt-lookup configuration")
    if corpus_lookup_path_count and corpus_lookup_store is None:
        raise ValueError("corpus lookup paths require a continuation store")
    if tree_builder is not None and (lookup_path_count or corpus_lookup_path_count):
        raise ValueError("prompt lookup cannot be combined with tree_builder")
    max_tree_nodes = (
        1
        + tree_budget
        + (lookup_path_count + corpus_lookup_path_count) * draft_horizon
    )

    output_ids = torch.full(
        (1, max_length + max_tree_nodes),
        mask_token_id,
        dtype=torch.long,
        device=model.device,
    )
    position_ids = torch.arange(output_ids.shape[1], device=model.device).unsqueeze(0)
    stop_token_ids_tensor = None if stop_token_ids is None else torch.tensor(stop_token_ids, device=model.device)

    verify_input_ids_buffer = torch.empty((1, max_tree_nodes), dtype=torch.long, device=model.device)
    verify_position_ids_buffer = torch.empty((1, max_tree_nodes), dtype=torch.long, device=model.device)
    attention_mask_buffer = torch.zeros(
        (1, 1, max_tree_nodes, max_length + max_tree_nodes),
        dtype=target.dtype,
        device=model.device,
    )
    tree_visibility_buffer = torch.empty((max_tree_nodes, max_tree_nodes), dtype=torch.bool, device=model.device)

    past_key_values_target = DynamicCache()
    past_key_values_draft = DynamicCache()
    stage_times = empty_stage_times(DDTREE_STAGE_ORDER + DDTREE_TREE_BUILD_STAGE_ORDER)
    stage_clock = time.perf_counter if fast_timing else cuda_time

    prefill_start = cuda_time()
    output = target(
        input_ids,
        position_ids=position_ids[:, :num_input_tokens],
        past_key_values=past_key_values_target,
        use_cache=True,
        logits_to_keep=1,
        output_hidden_states=True,
    )

    output_ids[:, :num_input_tokens] = input_ids
    output_ids[:, num_input_tokens : num_input_tokens + 1] = sample(output.logits, temperature)
    target_hidden = extract_context_feature(output.hidden_states, model.target_layer_ids)

    time_to_first_token = cuda_time() - prefill_start

    decode_start = cuda_time()
    round_clock_start = cuda_time()
    start = input_ids.shape[1]
    acceptance_lengths = []
    round_timestamps = []
    round_trees = [] if save_tree_traces else None
    selected_tree_budgets = []
    draft_prefill = True
    previous_tree_start = 0
    previous_tree_length = 0
    if tree_builder is not None and hasattr(tree_builder, "begin_episode"):
        tree_builder.begin_episode()
    history_token_ids = output_ids[0, : num_input_tokens + 1].detach().cpu().tolist()

    while start < max_length:
        block_output_ids = output_ids[:, start : start + block_size].clone()
        root_token = block_output_ids[:, :1]

        draft_stage_start = stage_clock()
        refinement_input = block_output_ids
        draft_hidden = draft_logits = None
        for refinement_step in range(draft_refinement_steps):
            noise_embedding = target.model.embed_tokens(refinement_input)
            # The first pass appends the newly accepted target_hidden rows to
            # the persistent draft cache before cropping away its noise tail.
            # Refinement therefore supplies an empty context slice: re-adding
            # target_hidden would duplicate those rows and misalign RoPE.
            refinement_context = (
                target_hidden
                if refinement_step == 0
                else target_hidden[:, :0, :]
            )
            draft_hidden = model(
                target_hidden=refinement_context,
                noise_embedding=noise_embedding,
                position_ids=position_ids[
                    :, past_key_values_draft.get_seq_length() : start + block_size
                ],
                past_key_values=past_key_values_draft,
                use_cache=True,
                is_causal=False,
            )[:, -draft_horizon:, :]
            draft_logits = target.lm_head(draft_hidden)
            past_key_values_draft.crop(start)
            if refinement_step + 1 < draft_refinement_steps:
                refinement_input = block_output_ids.clone()
                refinement_input[:, 1:] = draft_logits.argmax(dim=-1)
        rank_bucket_logits = (
            None
            if rank_head is None
            else rank_head(draft_hidden, draft_logits)[0]
        )
        rank_choice_logits = (
            None
            if rank_choice_head is None
            else rank_choice_head(draft_hidden, draft_logits)[0]
        )
        draft_stage_elapsed = stage_clock() - draft_stage_start
        if draft_prefill:
            draft_prefill = False
            decode_start = cuda_time()
        else:
            stage_times["draft"] += draft_stage_elapsed

        tree_build_start = stage_clock()
        if tree_builder is None:
            tree_result = build_ddtree_tree(
                draft_logits[0],
                tree_budget,
                tree_temperature=tree_temperature,
                tree_depth_bonus=tree_depth_bonus,
                rank_bucket_logits=rank_bucket_logits,
                rank_choice_logits=rank_choice_logits,
                rank_score_blend=rank_score_blend,
            )
        else:
            if (
                tree_temperature != 1.0
                or tree_depth_bonus != 0.0
                or rank_head is not None
                or rank_choice_head is not None
            ):
                raise ValueError(
                    "tree calibration cannot be combined with tree_builder"
                )
            tree_builder.set_runtime_context(prefix_length=start)
            tree_result = build_ddtree_tree(
                draft_logits[0], tree_budget, tree_builder=tree_builder
            )
        if lookup_path_count:
            lookup_paths = find_prompt_lookup_paths(
                history_token_ids,
                draft_horizon,
                lookup_path_count,
                lookup_max_ngram,
            )
            tree_result = add_paths_to_tree(tree_result, lookup_paths)
        if corpus_lookup_path_count:
            corpus_paths = corpus_lookup_store.find_paths(
                history_token_ids, corpus_lookup_path_count
            )
            tree_result = add_paths_to_tree(tree_result, corpus_paths)
        if depth_first_order:
            tree_result = reorder_tree_depth_first(tree_result)
        (
            node_token_ids,
            node_depths,
            parents,
            child_maps,
            visibility_cpu,
            tree_build_subtimes,
        ) = tree_result
        tree_build_elapsed = stage_clock() - tree_build_start
        stage_times["tree_build"] += tree_build_elapsed
        selected_tree_budgets.append(int(node_token_ids.numel()))
        for stage_name, stage_elapsed in tree_build_subtimes.items():
            stage_times[stage_name] += stage_elapsed

        tree_compile_start = stage_clock()
        verify_input_ids, verify_position_ids, verify_attention_mask, previous_tree_start, previous_tree_length = compile_ddtree_tree(
            root_token_id=root_token[0, 0],
            start=start,
            node_token_ids=node_token_ids,
            node_depths=node_depths,
            visibility_cpu=visibility_cpu,
            past_length=start,
            dtype=target.dtype,
            device=model.device,
            verify_input_ids_buffer=verify_input_ids_buffer,
            verify_position_ids_buffer=verify_position_ids_buffer,
            attention_mask_buffer=attention_mask_buffer,
            tree_visibility_buffer=tree_visibility_buffer,
            previous_tree_start=previous_tree_start,
            previous_tree_length=previous_tree_length,
        )
        tree_compile_elapsed = stage_clock() - tree_compile_start
        stage_times["tree_compile"] += tree_compile_elapsed

        verify_stage_start = stage_clock()
        if sparse_lm_head:
            output = target.model(
                input_ids=verify_input_ids,
                position_ids=verify_position_ids,
                attention_mask=verify_attention_mask,
                past_key_values=past_key_values_target,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
        else:
            output = target(
                verify_input_ids,
                position_ids=verify_position_ids,
                attention_mask=verify_attention_mask,
                past_key_values=past_key_values_target,
                use_cache=True,
                output_hidden_states=True,
            )
        verify_stage_elapsed = stage_clock() - verify_stage_start
        stage_times["verify"] += verify_stage_elapsed

        commit_stage_start = stage_clock()
        if sparse_lm_head:
            accepted_indices, next_token = follow_verified_tree_sparse_lm_head(
                child_maps,
                output.last_hidden_state,
                target.lm_head,
                temperature,
            )
        else:
            posterior = sample(output.logits, temperature)
            accepted_indices, next_token = follow_verified_tree(child_maps, posterior)
        contiguous_accept = accepted_indices == list(range(len(accepted_indices)))
        if contiguous_accept:
            accepted_tokens = verify_input_ids[:, : len(accepted_indices)]
            accepted_index_tensor = None
        else:
            accepted_index_tensor = torch.tensor(
                accepted_indices,
                dtype=torch.long,
                device=verify_input_ids.device,
            )
            accepted_tokens = verify_input_ids.index_select(1, accepted_index_tensor)

        output_ids[:, start : start + len(accepted_indices)] = accepted_tokens
        output_ids[:, start + len(accepted_indices)] = next_token

        # The root is already the last token in history.  DDTree node ids live
        # on CPU, so updating lookup history adds no new device synchronization.
        history_token_ids.extend(
            int(node_token_ids[index - 1]) for index in accepted_indices[1:]
        )
        history_token_ids.append(int(next_token))

        compact_dynamic_cache(past_key_values_target, start, accepted_indices)
        verified_hidden = extract_context_feature(
            output.hidden_states, model.target_layer_ids
        )
        if contiguous_accept:
            target_hidden = verified_hidden[:, : len(accepted_indices)]
        else:
            target_hidden = verified_hidden.index_select(1, accepted_index_tensor)

        acceptance_lengths.append(len(accepted_indices))
        start += len(accepted_indices)
        commit_stage_elapsed = stage_clock() - commit_stage_start
        stage_times["commit"] += commit_stage_elapsed
        if tree_builder is not None:
            tree_builder.observe(
                tree_nodes=int(node_token_ids.numel()),
                draft_ms=draft_stage_elapsed * 1000.0,
                tree_build_ms=(tree_build_elapsed + tree_compile_elapsed) * 1000.0,
                verify_ms=verify_stage_elapsed * 1000.0,
                commit_ms=commit_stage_elapsed * 1000.0,
                accepted_draft_tokens=max(0, len(accepted_indices) - 1),
                committed_tokens=len(accepted_indices),
            )
        round_timestamps.append(stage_clock() - round_clock_start)
        if save_tree_traces:
            round_trees.append({
                "accepted_indices": [int(index) for index in accepted_indices],
                "tree": {
                    "node_token_ids": [int(token_id) for token_id in node_token_ids.tolist()],
                    "node_depths": [int(depth) for depth in node_depths.tolist()],
                    "parents": [int(parent) for parent in parents],
                },
            })

        if stop_token_ids_tensor is not None:
            new_tokens = output_ids[:, start - len(accepted_indices) : start + 1]
            if torch.isin(new_tokens[0], stop_token_ids_tensor).any():
                break

    if tree_builder is not None and hasattr(tree_builder, "end_episode"):
        tree_builder.end_episode()

    output_ids = output_ids[:, :max_length]
    output_ids = output_ids[:, output_ids[0] != mask_token_id]
    if stop_token_ids_tensor is not None:
        stop_token_indices = torch.isin(output_ids[0][num_input_tokens:], stop_token_ids_tensor).nonzero(as_tuple=True)[0]
        if stop_token_indices.numel() > 0:
            output_ids = output_ids[:, : num_input_tokens + stop_token_indices[0] + 1]

    num_output_tokens = output_ids.shape[1] - num_input_tokens
    total_decode_time = cuda_time() - decode_start
    time_per_output_token = total_decode_time / max(num_output_tokens, 1)

    return SimpleNamespace(
        output_ids=output_ids.cpu(),
        num_input_tokens=num_input_tokens,
        num_output_tokens=num_output_tokens,
        time_to_first_token=time_to_first_token,
        time_per_output_token=time_per_output_token,
        acceptance_lengths=acceptance_lengths,
        decode_rounds=len(acceptance_lengths),
        stage_times=stage_times,
        round_timestamps=round_timestamps,
        round_trees=round_trees,
        selected_tree_budgets=selected_tree_budgets,
        tree_policy=(
            tree_builder.policy_diagnostics() if tree_builder is not None else None
        ),
    )
