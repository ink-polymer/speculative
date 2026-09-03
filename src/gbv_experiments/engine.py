from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
import importlib.util
import sys
import time

import torch

from .common import ROOT
from .config import Variant
from .sampling import block_verify, probabilities, sample, select_and_reweight, token_verify
from .tree import compact_cache, probability_tree, sampled_tree


class StageMeter:
    def __init__(self, device, enabled):
        self.device, self.enabled = device, enabled
        self.host = defaultdict(float)
        self.events = []

    @contextmanager
    def measure(self, name):
        if not self.enabled:
            yield
            return
        start = time.perf_counter()
        pair = None
        if self.device.type == "cuda":
            pair = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            pair[0].record()
        yield
        if pair:
            pair[1].record()
            self.events.append((name, pair))
        self.host[name] += 1000 * (time.perf_counter() - start)

    def result(self):
        gpu = defaultdict(float)
        for name, pair in self.events:
            gpu[name] += pair[0].elapsed_time(pair[1])
        return {"host_ms": dict(self.host), "cuda_event_ms": dict(gpu)}


def draft_model_class():
    # Load the vendored package under a private name to avoid 'model' collisions.
    path = ROOT / "third_party/ddtree_official/model/__init__.py"
    spec = importlib.util.spec_from_file_location("_gbv_dflash_model", path,
                                                 submodule_search_locations=[str(path.parent)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.DFlashDraftModel


def load_models(cfg: dict, device: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Formal model benchmarks require an available CUDA GPU")
    torch.cuda.set_device(device)
    dtype = getattr(torch, cfg.get("dtype", "bfloat16"))
    target = AutoModelForCausalLM.from_pretrained(
        cfg["target"], revision=cfg.get("target_revision"), torch_dtype=dtype,
        attn_implementation=cfg.get("target_attention", "sdpa"),
    ).to(device).eval()
    draft = draft_model_class().from_pretrained(
        cfg["draft"], revision=cfg.get("draft_revision"), torch_dtype=dtype,
        attn_implementation=cfg.get("draft_attention", "sdpa"),
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(cfg["target"], revision=cfg.get("target_revision"))
    if target.config.vocab_size != draft.config.vocab_size:
        raise ValueError("Target and draft vocabularies differ")
    if target.config.hidden_size != draft.config.hidden_size:
        raise ValueError("Target/draft hidden sizes differ")
    if draft.mask_token_id is None or not (0 <= draft.mask_token_id < target.config.vocab_size):
        raise ValueError("Invalid draft mask token")
    if any(i < 0 or i >= target.config.num_hidden_layers for i in draft.target_layer_ids):
        raise ValueError("Invalid target feature layer IDs")
    target.requires_grad_(False)
    draft.requires_grad_(False)
    torch.backends.cuda.matmul.allow_tf32 = bool(cfg.get("allow_tf32", False))
    torch.backends.cudnn.allow_tf32 = bool(cfg.get("allow_tf32", False))
    return Engine(target, draft), tokenizer


class Engine:
    def __init__(self, target, draft, cache_factory=None):
        self.target, self.draft = target, draft
        self.device = next(target.parameters()).device
        if cache_factory is None:
            from transformers import DynamicCache
            cache_factory = DynamicCache
        self.cache_factory = cache_factory

    def sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def features(self, hidden_states, rows=None):
        selected = [hidden_states[i + 1] for i in self.draft.target_layer_ids]
        if rows is not None:
            selected = [x.index_select(1, rows) for x in selected]
        return torch.cat(selected, dim=-1)

    def target_forward(self, ids, cache, *, hidden, positions=None, mask=None, last_only=False):
        kwargs = dict(input_ids=ids, past_key_values=cache, use_cache=True,
                      output_hidden_states=hidden, return_dict=True)
        if positions is not None:
            kwargs["position_ids"] = positions
        if mask is not None:
            kwargs["attention_mask"] = mask
        if last_only:
            kwargs["logits_to_keep"] = 1
        return self.target(**kwargs)

    @torch.inference_mode()
    def generate(self, input_ids, variant: Variant, max_new_tokens: int, stop_ids,
                 seed=0, profile=False):
        variant.validate()
        if input_ids.shape[0] != 1 or max_new_tokens < 1:
            raise ValueError("Expected one prompt and max_new_tokens >= 1")
        if input_ids.shape[1] + max_new_tokens + self.draft.block_size > self.target.config.max_position_embeddings:
            raise ValueError("Prompt plus generation/verification exceeds model context length; no silent truncation")
        if variant.length >= self.draft.block_size and variant.method != "target":
            raise ValueError("Candidate length exceeds the checkpoint's future slots")
        generator = torch.Generator(device=self.device).manual_seed(seed)
        dtype = getattr(torch, variant.probability_dtype)
        stops = set(stop_ids)
        target_cache = self.cache_factory()
        draft_cache = self.cache_factory()
        meter = StageMeter(self.device, profile)
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        self.sync()
        started = time.perf_counter()
        with meter.measure("prefill"):
            initial = self.target_forward(input_ids, target_cache,
                                          hidden=variant.method != "target", last_only=True)
            anchor = int(sample(probabilities(initial.logits[0, -1], variant.temperature, dtype), generator))
            update = self.features(initial.hidden_states) if variant.method != "target" else None
            full_features = update if not variant.reuse_draft_cache else None
            del initial
        self.sync()
        prefill_end = time.perf_counter()
        generated = [anchor]
        rounds = []
        target_calls, draft_calls = 1, 0
        target_tokens = input_ids.shape[1]
        while len(generated) < max_new_tokens and generated[-1] not in stops:
            if variant.method == "target":
                with meter.measure("target_decode"):
                    output = self.target_forward(torch.tensor([[generated[-1]]], device=self.device),
                                                 target_cache, hidden=False, last_only=True)
                    token = int(sample(probabilities(output.logits[0, -1], variant.temperature, dtype), generator))
                generated.append(token)
                target_calls += 1
                target_tokens += 1
                continue
            prefix_len = int(target_cache.get_seq_length())
            block_width = int(self.draft.block_size)
            with meter.measure("draft"):
                if not variant.reuse_draft_cache:
                    draft_cache = self.cache_factory()
                    context = full_features
                else:
                    context = update
                if int(draft_cache.get_seq_length()) + context.shape[1] != prefix_len:
                    raise RuntimeError("Draft cache and feature-update lengths disagree")
                if variant.condition_features == "zero":
                    context = torch.zeros_like(context)
                noise_ids = torch.full((1, block_width), int(self.draft.mask_token_id),
                                       dtype=torch.long, device=self.device)
                noise_ids[0, 0] = generated[-1]
                noise = self.target.get_input_embeddings()(noise_ids)
                positions = torch.arange(draft_cache.get_seq_length(), prefix_len + block_width,
                                         device=self.device)[None]
                draft_mask = None
                if variant.draft_attention == "causal":
                    draft_mask = torch.zeros((block_width, prefix_len + block_width),
                                             dtype=noise.dtype, device=self.device)
                    forbidden = torch.ones((block_width, block_width), device=self.device,
                                           dtype=torch.bool).triu(1)
                    draft_mask[:, prefix_len:].masked_fill_(forbidden, float("-inf"))
                    draft_mask = draft_mask[None, None]
                hidden = self.draft(target_hidden=context, noise_embedding=noise,
                                    position_ids=positions, attention_mask=draft_mask,
                                    past_key_values=draft_cache, use_cache=True, is_causal=False)
                if not isinstance(hidden, torch.Tensor):
                    hidden = hidden.last_hidden_state
                logits = self.target.get_output_embeddings()(hidden[:, 1:variant.length + 1])[0]
                draft_cache.crop(prefix_len)
                draft_temp = variant.draft_temperature or variant.temperature or 1.0
                q = probabilities(logits, draft_temp, dtype)
                draft_calls += 1
            with meter.measure("tree_build"):
                if variant.method == "ddtree":
                    tree = probability_tree(q, variant.tree_budget)
                    paths = None
                else:
                    paths = sample(q[None].expand(variant.paths, -1, -1), generator)
                    tree = sampled_tree(paths, variant.share_prefixes)
                ids = torch.tensor([[generated[-1]] + tree.tokens], device=self.device)
                positions = (torch.tensor(tree.depths, device=self.device) + prefix_len)[None]
                mask = tree.mask(prefix_len, next(self.target.parameters()).dtype, self.device)
            with meter.measure("verify"):
                output = self.target_forward(ids, target_cache, hidden=True,
                                             positions=positions, mask=mask)
                all_p = probabilities(output.logits[0], variant.temperature, dtype)
                target_calls += 1
                target_tokens += ids.shape[1]
            with meter.measure("select_and_correct"):
                if variant.method == "ddtree":
                    children = {(tree.parents[i], tree.tokens[i - 1]): i for i in range(1, len(tree.parents))}
                    nodes, tokens, node = [], [], 0
                    while True:
                        bonus = int(sample(all_p[node], generator))
                        child = children.get((node, bonus))
                        if child is None:
                            break
                        nodes.append(child)
                        tokens.append(bonus)
                        node = child
                    accepted = len(nodes)
                else:
                    p_by_path = torch.stack([all_p[[0] + nodes] for nodes in tree.path_nodes])
                    if variant.method == "gbv":
                        chosen, r = select_and_reweight(paths, p_by_path, q)
                    else:
                        chosen, r = 0, q
                    verifier = token_verify if variant.method == "token" else block_verify
                    accepted, bonus = verifier(paths[chosen], p_by_path[chosen], r, generator)
                    nodes = tree.path_nodes[chosen][:accepted]
                    tokens = paths[chosen, :accepted].tolist()
            with meter.measure("commit"):
                appended = tokens + [bonus]
                committed = appended[:max_new_tokens - len(generated)]
                for i, token in enumerate(committed):
                    if token in stops:
                        committed = committed[:i + 1]
                        break
                generated.extend(committed)
                keep = [0] + nodes
                index = torch.tensor(keep, device=self.device)
                update = self.features(output.hidden_states, index)
                if full_features is not None:
                    full_features = torch.cat((full_features, update), dim=1)
                compact_cache(target_cache, prefix_len, keep, self.device)
                rounds.append({"accepted_draft_tokens": accepted, "committed_tokens": len(committed),
                               "committed_draft_tokens": min(accepted, len(committed)),
                               "proposed_tokens": variant.paths * variant.length if paths is not None else len(tree.tokens),
                               "tree_nodes": len(tree.tokens), "verify_tokens": len(tree.parents)})
                del output, all_p, hidden, logits
                if paths is not None:
                    del p_by_path, r
                del paths, q, tree, ids, mask, noise, noise_ids
        self.sync()
        ended = time.perf_counter()
        prefill_ms = (prefill_end - started) * 1000
        decode_ms = (ended - prefill_end) * 1000
        decode_tokens = len(generated) - 1
        return {
            "generated_token_ids": generated, "generated_tokens": len(generated),
            "decode_tokens": decode_tokens, "prefill_ms": prefill_ms, "decode_ms": decode_ms,
            "e2e_ms": (ended - started) * 1000,
            "decode_tokens_per_second": 1000 * decode_tokens / decode_ms if decode_tokens else None,
            "e2e_tokens_per_second": 1000 * len(generated) / ((ended - started) * 1000),
            "finish_reason": "eos" if generated[-1] in stops else "length",
            "target_forward_calls": target_calls, "draft_forward_calls": draft_calls,
            "target_tokens_processed": target_tokens, "rounds": rounds,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(self.device) if self.device.type == "cuda" else None,
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(self.device) if self.device.type == "cuda" else None,
            "stages": meter.result(),
        }
