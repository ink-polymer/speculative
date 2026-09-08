"""Real tiny Qwen attention/KV/decoding checks, not large-checkpoint GPU validation."""
from __future__ import annotations

import time

import pytest
import torch
from transformers import DynamicCache, Qwen3Config, Qwen3ForCausalLM

from dflash_specblock.paper.adaptive_official import adaptive_generate
from dflash_specblock.paper.common import ROOT, VARIANTS, load_json
from dflash_specblock.paper.controller import PaperAdaptiveBuilder
from dflash_specblock.paper.official_spec import upstream


@pytest.fixture(scope="module", autouse=True)
def small_model_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


@torch.inference_mode()
def serial_reference(target, prompt, maximum, stops):
    cache = DynamicCache()
    output = target(prompt, past_key_values=cache, use_cache=True, logits_to_keep=1)
    values = []
    for j in range(maximum):
        token = int(output.logits[0, -1].argmax())
        values.append(token)
        if token in stops or j+1 == maximum:
            break
        output = target(torch.tensor([[token]]), past_key_values=cache, use_cache=True)
    return values


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("ending", ["one", "limit", "first_eos", "later_eos"])
def test_real_qwen_outputs_match_independent_serial(seed, dtype, ending, monkeypatch):
    u = upstream()
    monkeypatch.setattr(u.dflash, "cuda_time", time.perf_counter)
    monkeypatch.setattr(u.ddtree, "cuda_time", time.perf_counter)
    monkeypatch.setattr(u.ddtree, "_CPP_COMPACT_ENABLED", False)
    torch.manual_seed(seed)
    base = dict(vocab_size=160, hidden_size=32, intermediate_size=64,
                num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                max_position_embeddings=512, tie_word_embeddings=False,
                eos_token_id=158, pad_token_id=0, bos_token_id=1)
    tc = Qwen3Config(**base, num_hidden_layers=4)
    dc = Qwen3Config(**base, num_hidden_layers=2, num_target_layers=4, block_size=16,
                    dflash_config={"mask_token_id":159, "target_layer_ids":[0,2]})
    tc._attn_implementation = dc._attn_implementation = "sdpa"
    target = Qwen3ForCausalLM(tc).eval().to(dtype=dtype)
    draft = u.model.DFlashDraftModel(dc).eval().to(dtype=dtype)
    prompt = torch.tensor([[1,4,8,3]])
    reference_long = serial_reference(target, prompt, 33, [])
    maximum = 1 if ending == "one" else 33
    stops = ([reference_long[0]] if ending == "first_eos"
             else [reference_long[5]] if ending == "later_eos" else [])
    ref = serial_reference(target, prompt, maximum, stops)
    assert 159 not in ref, "Random test fixture generated the reserved mask token"
    common = dict(model=draft, target=target, input_ids=prompt, mask_token_id=159,
                  max_new_tokens=maximum, stop_token_ids=stops, temperature=0.)
    results = {"official_ar":u.dflash.dflash_generate(**common, block_size=1),
               "dflash":u.dflash.dflash_generate(**common, block_size=16)}
    for budget in (16,64,128):
        results[f"ddtree_{budget}"] = u.ddtree.ddtree_generate(**common, block_size=16, tree_budget=budget)
    cfg = load_json(ROOT/"configs/paper_t0_full.json")
    for variant in VARIANTS:
        builder = PaperAdaptiveBuilder(cfg["adaptive"], variant)
        results[variant] = adaptive_generate(**common, block_size=16, builder=builder)
    for method, result in results.items():
        assert result.output_ids[0, result.num_input_tokens:].tolist() == ref, method
        assert result.num_output_tokens == len(ref), method
