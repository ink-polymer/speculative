"""Corrected AdaptiveTree inserted into the pinned official DDTree loop.

The generation function below is derived from c96427a.../ddtree.py.
Only tree construction, controller feedback and diagnostics are added.
Target verification, sampling, cache compaction, EOS and TPOT boundaries remain upstream.
See third_party/ddtree_pinned/LICENSE (MIT, Copyright 2026 Liran Ringel).
"""
from __future__ import annotations

from types import SimpleNamespace

import torch

from .official_spec import upstream
from .triton_cache import BatchedTritonCacheCompactor


def _compact_dynamic_cache_with_tensor(
    ddtree_module, past_key_values, past_length, keep_current_indices,
    batched_compactor=None,
):
    """Adaptive-only cache compaction reusing an existing device index tensor."""
    keep_count = int(keep_current_indices.numel())
    if keep_count == 0:
        past_key_values.crop(past_length)
        return

    keep_tensor_by_device = {keep_current_indices.device: keep_current_indices}

    def get_keep_tensor(device):
        if device not in keep_tensor_by_device:
            keep_tensor_by_device[device] = keep_current_indices.to(device)
        return keep_tensor_by_device[device]

    if hasattr(past_key_values, "key_cache") and hasattr(
            past_key_values, "value_cache"):
        cache_tensors = [
            tensor
            for pair in zip(past_key_values.key_cache,
                            past_key_values.value_cache)
            for tensor in pair
        ]
        if (batched_compactor is not None
                and batched_compactor.compact(
                    cache_tensors, past_length, keep_current_indices)):
            past_key_values.crop(past_length + keep_count)
            return
        for key_cache, value_cache in zip(
                past_key_values.key_cache, past_key_values.value_cache):
            keep_tensor = get_keep_tensor(key_cache.device)
            ddtree_module._compact_appended_window(
                key_cache, past_length, keep_tensor)
            ddtree_module._compact_appended_window(
                value_cache, past_length, keep_tensor)
        past_key_values.crop(past_length + keep_count)
        return

    if hasattr(past_key_values, "layers"):
        populated_layers = [
            layer for layer in past_key_values.layers
            if (hasattr(layer, "keys") and layer.keys is not None
                and layer.keys.numel() > 0)
        ]
        cache_tensors = [
            tensor for layer in populated_layers
            for tensor in (layer.keys, layer.values)
        ]
        if (batched_compactor is not None
                and batched_compactor.compact(
                    cache_tensors, past_length, keep_current_indices)):
            past_key_values.crop(past_length + keep_count)
            return
        for layer in populated_layers:
            keep_tensor = get_keep_tensor(layer.keys.device)
            ddtree_module._compact_appended_window(
                layer.keys, past_length, keep_tensor)
            ddtree_module._compact_appended_window(
                layer.values, past_length, keep_tensor)
        past_key_values.crop(past_length + keep_count)
        return

    raise RuntimeError("Unsupported DynamicCache layout for AdaptiveTree cache compaction.")


def build_with_controller(logits, builder):
    if getattr(builder, "variant", None) in {
            "guarded_raw_prefix", "contextual_prefix_v8"}:
        return builder.build_official_tree_from_logits(logits)
    tree = builder.build_from_logits(logits)
    nodes = tree.nodes
    token_ids = torch.tensor([n.token_id for n in nodes], dtype=torch.long)
    depths = torch.tensor([n.depth for n in nodes], dtype=torch.long)
    parents = [-1] + [n.parent + 1 for n in nodes]
    children = [{} for _ in parents]
    for index, node in enumerate(nodes, 1):
        children[parents[index]][node.token_id] = index
    visibility = torch.zeros((len(nodes)+1, len(nodes)+1), dtype=torch.bool)
    visibility[:, 0] = True
    if nodes:
        visibility[1:, 1:] = tree.ancestor_mask(torch.device("cpu"))
    # Total tree_build is measured by the unchanged enclosing official timer.
    # Substage attribution is unavailable for this builder, not zero-cost work.
    return token_ids, depths, parents, children, visibility, {}


@torch.inference_mode()
def adaptive_generate(
    model: torch.nn.Module,
    target: torch.nn.Module,
    input_ids: torch.Tensor,
    mask_token_id: int,
    max_new_tokens: int,
    block_size: int,
    stop_token_ids: list[int],
    temperature: float = 0.0,
    tree_budget: int | None = None,
    save_tree_traces: bool = False,
    builder: object | None = None,
) -> SimpleNamespace:
    from transformers import DynamicCache

    u = upstream()
    cuda_time, empty_stage_times = u.dflash.cuda_time, u.dflash.empty_stage_times
    sample, extract_context_feature = u.model.sample, u.model.extract_context_feature
    compile_ddtree_tree = u.ddtree.compile_ddtree_tree
    follow_verified_tree = u.ddtree.follow_verified_tree
    compact_dynamic_cache = u.ddtree.compact_dynamic_cache
    dflash_generate = u.dflash.dflash_generate
    DDTREE_STAGE_ORDER = u.ddtree.DDTREE_STAGE_ORDER
    DDTREE_TREE_BUILD_STAGE_ORDER = u.ddtree.DDTREE_TREE_BUILD_STAGE_ORDER
    if builder is None or temperature != 0 or block_size - 1 != builder.block_size:
        raise ValueError("Adaptive requires the original T=0 controller and matching draft horizon")
    tree_budget = builder.tree_budget
    builder.trace = []
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

    num_input_tokens = input_ids.shape[1]
    max_length = num_input_tokens + max_new_tokens
    draft_horizon = block_size - 1
    tree_budget = draft_horizon if tree_budget is None else max(tree_budget, 0)
    max_tree_nodes = 1 + tree_budget

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
    accepted_index_buffer = torch.empty(
        max_tree_nodes, dtype=torch.long, device=model.device)
    if model.device.type == "cuda":
        target_layers = int(target.config.num_hidden_layers)
        key_value_heads = int(target.config.num_key_value_heads)
        head_dimension = int(getattr(
            target.config, "head_dim",
            target.config.hidden_size // target.config.num_attention_heads))
        cache_compactor = BatchedTritonCacheCompactor(
            2 * target_layers, model.device,
            maximum_copy_elements=(
                2 * target_layers * key_value_heads * block_size
                * head_dimension
            ),
            element_size=torch.empty((), dtype=target.dtype).element_size(),
        )
    else:
        cache_compactor = BatchedTritonCacheCompactor(0, model.device)

    past_key_values_target = DynamicCache()
    past_key_values_draft = DynamicCache()
    stage_times = empty_stage_times(DDTREE_STAGE_ORDER + DDTREE_TREE_BUILD_STAGE_ORDER)

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
    draft_prefill = True
    previous_tree_start = 0
    previous_tree_length = 0

    while start < max_length:
        stage_before = stage_times.copy()
        block_output_ids = output_ids[:, start : start + block_size].clone()
        root_token = block_output_ids[:, :1]

        draft_stage_start = cuda_time()
        noise_embedding = target.model.embed_tokens(block_output_ids)
        draft_logits = target.lm_head(model(
            target_hidden=target_hidden,
            noise_embedding=noise_embedding,
            position_ids=position_ids[:, past_key_values_draft.get_seq_length() : start + block_size],
            past_key_values=past_key_values_draft,
            use_cache=True,
            is_causal=False,
        )[:, -draft_horizon:, :])
        past_key_values_draft.crop(start)
        draft_stage_elapsed = cuda_time() - draft_stage_start
        if draft_prefill:
            draft_prefill = False
            decode_start = cuda_time()
        else:
            stage_times["draft"] += draft_stage_elapsed

        tree_build_start = cuda_time()
        node_token_ids, node_depths, parents, child_maps, visibility_cpu, tree_build_subtimes = build_with_controller(draft_logits[0], builder)
        stage_times["tree_build"] += cuda_time() - tree_build_start
        for stage_name, stage_elapsed in tree_build_subtimes.items():
            stage_times[stage_name] += stage_elapsed

        tree_compile_start = cuda_time()
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
        stage_times["tree_compile"] += cuda_time() - tree_compile_start

        verify_stage_start = cuda_time()
        output = target(
            verify_input_ids,
            position_ids=verify_position_ids,
            attention_mask=verify_attention_mask,
            past_key_values=past_key_values_target,
            use_cache=True,
            output_hidden_states=True,
        )
        stage_times["verify"] += cuda_time() - verify_stage_start

        commit_stage_start = cuda_time()
        posterior = sample(output.logits, temperature)
        if child_maps is None:
            accepted_indices_host, next_token = builder.follow_compiled_tree(
                posterior)
            accepted_count = int(accepted_indices_host.numel())
            accepted_index_tensor = accepted_index_buffer[:accepted_count]
            accepted_index_tensor.copy_(accepted_indices_host, non_blocking=True)
            accepted_indices = accepted_indices_host.tolist()
        else:
            accepted_indices, next_token = follow_verified_tree(child_maps, posterior)
            accepted_count = len(accepted_indices)
            accepted_index_tensor = torch.tensor(
                accepted_indices, dtype=torch.long,
                device=verify_input_ids.device)
        accepted_tokens = verify_input_ids.index_select(1, accepted_index_tensor)

        output_ids[:, start : start + accepted_count] = accepted_tokens
        output_ids[:, start + accepted_count] = next_token

        if child_maps is None:
            _compact_dynamic_cache_with_tensor(
                u.ddtree, past_key_values_target, start,
                accepted_index_tensor, cache_compactor)
        else:
            compact_dynamic_cache(past_key_values_target, start, accepted_indices)
        target_hidden = extract_context_feature(
            output.hidden_states, model.target_layer_ids
        ).index_select(1, accepted_index_tensor)

        acceptance_lengths.append(accepted_count)
        start += accepted_count
        stage_times["commit"] += cuda_time() - commit_stage_start
        if getattr(builder, "timing_partition", "legacy") == "legacy":
            # Preserve the frozen protocol's exact expression grouping and trace
            # schema.  Even a mathematically equivalent regrouping can move an
            # EWMA tie by an ulp and is therefore kept out of the official path.
            builder.observe(
                tree_nodes=int(node_token_ids.numel()),
                draft_ms=1000 * (draft_stage_elapsed + stage_times["tree_build"]
                                   - stage_before["tree_build"]),
                verify_ms=1000 * sum(stage_times[k] - stage_before[k]
                                     for k in ("tree_compile", "verify", "commit")),
                accepted_draft_tokens=accepted_count - 1,
            )
        else:
            # Corrected formal path: tree construction varies with the selected
            # budget and belongs in its per-budget latency estimate.
            builder.observe_stages(
                tree_nodes=int(node_token_ids.numel()),
                draft_ms=1000 * draft_stage_elapsed,
                tree_build_ms=1000 * (stage_times["tree_build"]
                                      - stage_before["tree_build"]),
                tree_compile_ms=1000 * (stage_times["tree_compile"] - stage_before["tree_compile"]),
                target_verify_ms=1000 * (stage_times["verify"] - stage_before["verify"]),
                commit_ms=1000 * (stage_times["commit"] - stage_before["commit"]),
                accepted_draft_tokens=accepted_count - 1,
                accepted_node_indices=accepted_indices,
            )
        round_timestamps.append(cuda_time() - round_clock_start)
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
            new_tokens = output_ids[:, start - accepted_count : start + 1]
            if torch.isin(new_tokens[0], stop_token_ids_tensor).any():
                break

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
        adaptive_decisions=list(builder.trace),
        cache_compaction={
            "backend": ("triton_batched" if cache_compactor.batched_calls
                        else "official_per_tensor_fallback"),
            "batched_calls": cache_compactor.batched_calls,
            "decode_rounds": len(acceptance_lengths),
        },
    )
