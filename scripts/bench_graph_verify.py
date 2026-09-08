"""验证 CUDA Graph 支持输入内容变化，并与 eager 做同范围稳态计时。"""

from __future__ import annotations

import inspect
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache

from dflash_specblock.config import ExperimentConfig
from dflash_specblock.device import dtype_from_name, resolve_device, synchronize


def make_static_cache(model, max_cache_len: int, device: torch.device, dtype: torch.dtype):
    parameters = inspect.signature(StaticCache).parameters
    kwargs: dict[str, object] = {"max_cache_len": max_cache_len}
    if "max_batch_size" in parameters:
        kwargs["max_batch_size"] = 1
    if "device" in parameters:
        kwargs["device"] = device
    if "dtype" in parameters:
        kwargs["dtype"] = dtype
    return StaticCache(model.config, **kwargs)


@torch.inference_mode()
def prefill_cache(model, input_ids: torch.Tensor, cache, device: torch.device) -> None:
    cache_position = torch.arange(input_ids.shape[1], dtype=torch.long, device=device)
    _ = model(
        input_ids=input_ids,
        past_key_values=cache,
        cache_position=cache_position,
        use_cache=True,
        return_dict=True,
    )


def validate_cache_position(cache_position: torch.Tensor, max_cache_len: int) -> None:
    if cache_position.ndim != 1 or cache_position.numel() == 0:
        raise ValueError("cache_position must be a non-empty 1D tensor")
    first = int(cache_position[0].item())
    last = int(cache_position[-1].item())
    if first < 0 or last >= max_cache_len:
        raise ValueError(
            f"cache_position range [{first}, {last}] exceeds max_cache_len={max_cache_len}"
        )


@torch.inference_mode()
def main() -> None:
    project = Path(__file__).resolve().parent.parent
    config = ExperimentConfig.from_json(
        str(project / "configs/qwen3_4b_cuda_tree15_graph.json")
    )
    device = resolve_device(config.device)
    dtype = dtype_from_name(config.dtype)
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        dtype = torch.float16

    print(f"Loading Qwen3-4B on {device}, dtype={dtype}")
    tokenizer = AutoTokenizer.from_pretrained(config.target_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        config.target_path,
        trust_remote_code=True,
        dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation=config.attn_implementation,
    ).eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    prompt = "Explain quantum computing. " * 5
    input_ids_full = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    past_length = int(input_ids_full.shape[1])
    cur_len = 61
    max_cache_len = int(config.cuda_graph_max_cache_len)
    if past_length + cur_len > max_cache_len:
        raise ValueError(
            f"prefill({past_length}) + verify({cur_len}) exceeds "
            f"cuda_graph_max_cache_len={max_cache_len}"
        )

    cache_position = torch.arange(
        past_length, past_length + cur_len, dtype=torch.long, device=device
    )
    validate_cache_position(cache_position, max_cache_len)
    position_ids = cache_position.unsqueeze(0)

    minimum = torch.finfo(dtype).min
    mask = torch.full(
        (1, 1, cur_len, max_cache_len), minimum, dtype=dtype, device=device
    )
    mask[..., :past_length] = 0
    causal = torch.tril(torch.ones(cur_len, cur_len, dtype=torch.bool, device=device))
    mask[..., past_length : past_length + cur_len].masked_fill_(causal, 0)

    graph_input_ids = torch.zeros((1, cur_len), dtype=torch.long, device=device)
    eager_input_ids = torch.zeros_like(graph_input_ids)
    graph_position_ids = position_ids.clone()
    eager_position_ids = position_ids.clone()
    graph_cache_position = cache_position.clone()
    eager_cache_position = cache_position.clone()
    graph_mask = mask.clone()
    eager_mask = mask.clone()

    graph_cache = make_static_cache(model, max_cache_len, device, dtype)
    eager_cache = make_static_cache(model, max_cache_len, device, dtype)
    prefill_cache(model, input_ids_full, graph_cache, device)
    prefill_cache(model, input_ids_full, eager_cache, device)

    current_stream = torch.cuda.current_stream(device)
    warmup_stream = torch.cuda.Stream(device=device)
    warmup_stream.wait_stream(current_stream)
    with torch.cuda.stream(warmup_stream):
        for _ in range(3):
            _ = model(
                input_ids=graph_input_ids,
                attention_mask=graph_mask,
                position_ids=graph_position_ids,
                past_key_values=graph_cache,
                cache_position=graph_cache_position,
                use_cache=True,
                return_dict=True,
            )
    current_stream.wait_stream(warmup_stream)
    synchronize(device)

    print("Capturing CUDA Graph...")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = model(
            input_ids=graph_input_ids,
            attention_mask=graph_mask,
            position_ids=graph_position_ids,
            past_key_values=graph_cache,
            cache_position=graph_cache_position,
            use_cache=True,
            return_dict=True,
        )
    synchronize(device)

    print("Checking replay with five different token tensors...")
    for index in range(5):
        new_ids = torch.randint(
            0,
            int(model.config.vocab_size),
            (1, cur_len),
            dtype=torch.long,
            device=device,
        )
        graph_input_ids.copy_(new_ids)
        eager_input_ids.copy_(new_ids)

        graph.replay()
        eager_output = model(
            input_ids=eager_input_ids,
            attention_mask=eager_mask,
            position_ids=eager_position_ids,
            past_key_values=eager_cache,
            cache_position=eager_cache_position,
            use_cache=True,
            return_dict=True,
        )
        synchronize(device)

        graph_logits = graph_output.logits
        eager_logits = eager_output.logits
        max_abs = float((graph_logits - eager_logits).abs().max().item())
        argmax_equal = bool(
            torch.equal(graph_logits.argmax(dim=-1), eager_logits.argmax(dim=-1))
        )
        tolerance = 2e-2 if dtype == torch.bfloat16 else 5e-3
        torch.testing.assert_close(
            graph_logits,
            eager_logits,
            rtol=tolerance,
            atol=tolerance,
            msg=lambda message: f"CUDA Graph/eager mismatch at iteration {index}: {message}",
        )
        if not argmax_equal:
            raise AssertionError(f"CUDA Graph/eager argmax mismatch at iteration {index}")
        print(f"  iter {index}: max_abs_diff={max_abs:.6g}, argmax_equal={argmax_equal}")

    warmup = 3
    iters = 20
    for _ in range(warmup):
        graph.replay()
    synchronize(device)
    start = time.perf_counter()
    for _ in range(iters):
        graph.replay()
    synchronize(device)
    graph_ms = (time.perf_counter() - start) / iters * 1000

    for _ in range(warmup):
        _ = model(
            input_ids=eager_input_ids,
            attention_mask=eager_mask,
            position_ids=eager_position_ids,
            past_key_values=eager_cache,
            cache_position=eager_cache_position,
            use_cache=True,
            return_dict=True,
        )
    synchronize(device)
    start = time.perf_counter()
    for _ in range(iters):
        _ = model(
            input_ids=eager_input_ids,
            attention_mask=eager_mask,
            position_ids=eager_position_ids,
            past_key_values=eager_cache,
            cache_position=eager_cache_position,
            use_cache=True,
            return_dict=True,
        )
    synchronize(device)
    eager_ms = (time.perf_counter() - start) / iters * 1000

    print(f"\nRESULTS (cur_len={cur_len}, forward-only steady state)")
    print(f"  Eager forward:   {eager_ms:.1f} ms/iter")
    print(f"  Graph replay:    {graph_ms:.1f} ms/iter")
    print(f"  Speedup:         {eager_ms / graph_ms:.2f}x")


if __name__ == "__main__":
    main()
