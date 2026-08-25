"""比较固定 shape verify forward 的 eager 与 CUDA Graph 稳态开销。"""

from __future__ import annotations

import inspect
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache

from dflash_specblock.config import ExperimentConfig
from dflash_specblock.device import resolve_device, synchronize


def make_static_cache(model, max_cache_len: int, device: torch.device, dtype: torch.dtype):
    """兼容新旧 Transformers StaticCache 构造签名。"""
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


@torch.inference_mode()
def bench_eager(
    model,
    input_ids,
    mask,
    position_ids,
    cache,
    cache_position,
    device,
    warmup=3,
    iters=20,
):
    for _ in range(warmup):
        _ = model(
            input_ids=input_ids,
            attention_mask=mask,
            position_ids=position_ids,
            past_key_values=cache,
            cache_position=cache_position,
            use_cache=True,
            return_dict=True,
        )
    synchronize(device)
    start = time.perf_counter()
    for _ in range(iters):
        _ = model(
            input_ids=input_ids,
            attention_mask=mask,
            position_ids=position_ids,
            past_key_values=cache,
            cache_position=cache_position,
            use_cache=True,
            return_dict=True,
        )
    synchronize(device)
    return (time.perf_counter() - start) / iters * 1000


@torch.inference_mode()
def bench_graph(
    model,
    input_ids,
    mask,
    position_ids,
    cache,
    cache_position,
    device,
    warmup=3,
    iters=20,
):
    static_input_ids = input_ids.clone()
    static_mask = mask.clone()
    static_position_ids = position_ids.clone()
    static_cache_position = cache_position.clone()

    current_stream = torch.cuda.current_stream(device)
    warmup_stream = torch.cuda.Stream(device=device)
    warmup_stream.wait_stream(current_stream)
    with torch.cuda.stream(warmup_stream):
        for _ in range(warmup):
            _ = model(
                input_ids=static_input_ids,
                attention_mask=static_mask,
                position_ids=static_position_ids,
                past_key_values=cache,
                cache_position=static_cache_position,
                use_cache=True,
                return_dict=True,
            )
    current_stream.wait_stream(warmup_stream)
    synchronize(device)

    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            graph_output = model(
                input_ids=static_input_ids,
                attention_mask=static_mask,
                position_ids=static_position_ids,
                past_key_values=cache,
                cache_position=static_cache_position,
                use_cache=True,
                return_dict=True,
            )
        synchronize(device)
        for _ in range(warmup):
            graph.replay()
        synchronize(device)
        start = time.perf_counter()
        for _ in range(iters):
            graph.replay()
        synchronize(device)
        elapsed_ms = (time.perf_counter() - start) / iters * 1000
        # 保持 capture 输出存活，避免其 storage 在 replay 完成前被回收。
        _ = graph_output.logits
        return elapsed_ms
    except Exception as exc:
        print(f"  CUDA Graph capture failed: {type(exc).__name__}: {exc}")
        return None


@torch.inference_mode()
def main() -> None:
    project = Path(__file__).resolve().parent.parent
    config = ExperimentConfig.from_json(str(project / "configs/qwen3_4b_cuda_tree15.json"))
    device = resolve_device(config.device)
    dtype = (
        torch.bfloat16
        if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float16
    )

    print(f"Loading Qwen3-4B (SDPA) on {device}, dtype={dtype}")
    tokenizer = AutoTokenizer.from_pretrained(config.target_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        config.target_path,
        trust_remote_code=True,
        dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    ).eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    prompt = "Explain quantum computing in simple terms."
    input_ids_full = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    past_length = int(input_ids_full.shape[1])
    cur_len = 31
    max_cache_len = past_length + cur_len

    input_ids = torch.randint(
        0,
        int(model.config.vocab_size),
        (1, cur_len),
        dtype=torch.long,
        device=device,
    )
    cache_position = torch.arange(
        past_length, past_length + cur_len, dtype=torch.long, device=device
    )
    if int(cache_position[-1]) >= max_cache_len:
        raise ValueError(
            f"cache_position={int(cache_position[-1])} exceeds max_cache_len={max_cache_len}"
        )
    position_ids = cache_position.unsqueeze(0)

    minimum = torch.finfo(dtype).min
    mask = torch.full(
        (1, 1, cur_len, max_cache_len), minimum, dtype=dtype, device=device
    )
    mask[..., :past_length] = 0
    causal = torch.tril(torch.ones(cur_len, cur_len, dtype=torch.bool, device=device))
    mask[..., past_length:].masked_fill_(causal, 0)

    eager_cache = make_static_cache(model, max_cache_len, device, dtype)
    graph_cache = make_static_cache(model, max_cache_len, device, dtype)
    prefill_cache(model, input_ids_full, eager_cache, device)
    prefill_cache(model, input_ids_full, graph_cache, device)
    synchronize(device)

    print(f"\nTesting cur_len={cur_len}, past_length={past_length}")
    eager_ms = bench_eager(
        model, input_ids, mask, position_ids, eager_cache, cache_position, device
    )
    print(f"  Eager forward:    {eager_ms:.1f} ms/iter")

    graph_ms = bench_graph(
        model, input_ids, mask, position_ids, graph_cache, cache_position, device
    )
    if graph_ms is not None:
        print(f"  Graph replay:     {graph_ms:.1f} ms/iter")
        print(f"  Speedup:          {eager_ms / graph_ms:.2f}x")


if __name__ == "__main__":
    main()
