"""Controlled eager loop. No changes to the existing inference engine or verifier."""
from __future__ import annotations

import copy
import random
import re
import statistics
import time
from dataclasses import dataclass

import torch

from ..benchmark import baseline_greedy
from ..config import ExperimentConfig
from ..ddtree_builder import DDTreeBuilder
from ..device import configure_cuda_runtime, dtype_from_name, resolve_device, synchronize
from ..dflash_adapter import DFlashBlockAdapter
from ..models import ModelBundle, _freeze_and_place, _pretrained_kwargs, _validate_model_pair, render_prompt
from ..rank_head import HeuristicRanker
from ..vanilla_engine import VanillaDFlashEngine
from ..verification import TargetTreeVerifier
from .common import ROOT
from .structure import K, build_layered, features, static_action


def commit(tokens, remaining, stops):
    result = list(tokens[:remaining])
    for i, token in enumerate(result):
        if token in stops:
            return result[:i + 1]
    return result


@dataclass
class RoundState:
    logits: torch.Tensor
    target_context: torch.Tensor
    cache: object
    anchor: int
    remaining: int
    history: list
    features: torch.Tensor | None = None
    draft_ms: float = 0.


class PaperRuntime:
    def __init__(self, cfg, *, bundle=None, device=None):
        self.cfg = cfg
        self.model_cfg = ExperimentConfig.from_json(ROOT / cfg["model_config"])
        mc = self.model_cfg
        if mc.block_size != K or mc.max_blocks != 1 or mc.tree_budget != cfg["baseline_budget"]:
            raise ValueError("Model/protocol block size or baseline budget mismatch")
        if mc.device != "cuda:0" or mc.dtype != "bfloat16" or mc.allow_tf32:
            raise ValueError("Formal protocol uses cuda:0, BF16, TF32 off; select GPU via CUDA_VISIBLE_DEVICES")
        if any(not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision)
               for revision in (mc.target_revision, mc.draft_revision)):
            raise ValueError("Both model revisions must be immutable commit SHAs")
        if mc.use_cuda_graphs or mc.torch_compile_mode or mc.attn_implementation != "eager":
            raise ValueError("The controlled protocol requires eager target, no compile/graphs")
        if mc.draft_attn_implementation != "eager" or mc.rank_checkpoint or mc.ddtree_reserve_greedy_chain:
            raise ValueError("Keep draft backend/rank/greedy-reserve identical across methods")
        self.device = device or resolve_device(mc.device)
        configure_cuda_runtime(self.device, mc.allow_tf32)
        if bundle is None:
            # Intentionally ignore local_dir shortcuts: a fine-tuned local draft must
            # not silently override the immutable original model revisions.
            from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(mc.target_model_id, revision=mc.target_revision,
                                                     trust_remote_code=True)
            target = AutoModelForCausalLM.from_pretrained(mc.target_model_id,
                **_pretrained_kwargs(mc, mc.target_revision, self.device, "eager"))
            draft = AutoModel.from_pretrained(mc.draft_model_id,
                **_pretrained_kwargs(mc, mc.draft_revision, self.device, "eager"))
            bundle = ModelBundle(_freeze_and_place(target, self.device),
                                 _freeze_and_place(draft, self.device), tokenizer)
            _validate_model_pair(bundle.target, bundle.draft)
        self.target, self.tokenizer = bundle.target, bundle.tokenizer
        self.adapter = DFlashBlockAdapter(bundle.target, bundle.draft, HeuristicRanker(), K)
        self.verifier = TargetTreeVerifier(bundle.target, self.adapter.target_layer_ids,
                                          self.device, dtype_from_name(mc.dtype))
        self.ddtree = DDTreeBuilder(K, cfg["baseline_budget"])
        self.vanilla = VanillaDFlashEngine(bundle.target, self.adapter, self.device, dtype_from_name(mc.dtype))
        eos = getattr(bundle.target.generation_config, "eos_token_id", None)
        if eos is None:
            eos = self.tokenizer.eos_token_id
        self.stops = set(eos if isinstance(eos, (list, tuple)) else ([] if eos is None else [eos]))

    def encode(self, prompt):
        return self._check_encoded(render_prompt(self.tokenizer, prompt, self.model_cfg.enable_thinking))

    def encode_messages(self, messages):
        if (not messages or len(messages) % 2 != 1
                or any(m["role"] != ("user" if i % 2 == 0 else "assistant")
                       or not isinstance(m["content"], str) for i, m in enumerate(messages))):
            raise ValueError("Expected alternating user/assistant messages ending with user")
        kwargs = dict(return_tensors="pt", add_generation_prompt=True, tokenize=True)
        try:
            ids = self.tokenizer.apply_chat_template(messages, enable_thinking=self.model_cfg.enable_thinking, **kwargs)
        except TypeError:
            ids = self.tokenizer.apply_chat_template(messages, **kwargs)
        return self._check_encoded(ids)

    def _check_encoded(self, ids):
        ids = ids.to(self.device)
        limit = getattr(self.target.config, "max_position_embeddings", None)
        if limit and ids.shape[1] + self.cfg["max_new_tokens"] + K > limit:
            raise ValueError("Prompt + output exceeds model context; no silent truncation/drop is allowed")
        return ids

    def verify_clone(self, state, tree):
        cache = copy.deepcopy(state.cache)
        synchronize(self.device)
        started = time.perf_counter()
        result = self.verifier.verify(state.anchor, tree, cache, state.target_context)
        synchronize(self.device)
        elapsed = (time.perf_counter() - started) * 1000
        tokens = commit(result.path.token_ids + [result.path.bonus_token_id], state.remaining, self.stops)
        accepted = len(result.path.token_ids)
        del result, cache
        return tokens, accepted, elapsed

    def counterfactuals(self, state, policy, rng):
        actions = [("ddtree_" + str(b), None, b, None) for b in self.cfg["oracle_budgets"]]
        for j in range(self.cfg["cf_actions"]):
            action, indices = policy.random_action(rng)
            actions.append((f"layered_{j}", action, None, indices))
        # Randomize order at each state; candidate cache copies are outside timers.
        rng.shuffle(actions)
        results = []
        for name, action, budget, indices in actions:
            latencies, reference = [], None
            for _ in range(self.cfg["cf_repeats"]):
                synchronize(self.device)
                started = time.perf_counter()
                tree = (self.ddtree.build_from_logits(state.logits, budget) if action is None
                        else build_layered(state.logits, action))
                synchronize(self.device)
                build_ms = (time.perf_counter() - started) * 1000
                tokens, accepted, verify_ms = self.verify_clone(state, tree)
                if reference is not None and reference != tokens:
                    raise RuntimeError("Repeated counterfactual verification changed output")
                reference = tokens
                latencies.append(state.draft_ms + build_ms + verify_ms)
            results.append({"name": name, "indices": indices,
                            "action": action.json() if action else None,
                            "committed": len(tokens), "accepted": accepted, "nodes": len(tree),
                            "latency_ms": statistics.median(latencies), "tokens": tokens})
        # Compare common prefixes, not lengths: trees may commit different amounts.
        longest = max(results, key=lambda r: r["committed"])["tokens"]
        if any(r["tokens"] != longest[:r["committed"]] for r in results):
            raise RuntimeError("Counterfactual trees disagree with the same greedy continuation")
        return {"features": state.features.tolist(), "candidates": results}

    @torch.inference_mode()
    def generate(self, ids, method, policy=None, *, sample=False, generator=None, observer=None):
        maximum = self.cfg["max_new_tokens"]
        if maximum < 0:
            raise ValueError("max_new_tokens must be nonnegative")
        if maximum == 0:
            return {"tokens": [], "wall_ms": 0., "rounds": []}
        synchronize(self.device)
        start = time.perf_counter()
        if method == "ar":
            tokens, _ = baseline_greedy(self.target, ids, maximum, self.stops, self.device)
            rounds = []
        elif method == "dflash":
            result = self.vanilla.generate(ids, maximum, self.stops)
            tokens = result.generated_ids[0].cpu()
            rounds = [{"accepted": r.accepted_draft_tokens, "committed": r.committed_tokens,
                       "nodes": K, "latency_ms": r.draft_ms + r.verify_ms} for r in result.iterations]
        else:
            from transformers import DynamicCache
            cache, draft_cache = DynamicCache(), DynamicCache()
            output = self.target(input_ids=ids, past_key_values=cache, use_cache=True,
                                 output_hidden_states=True, logits_to_keep=1, return_dict=True)
            cache = output.past_key_values
            anchor = int(output.logits[0, -1].argmax())
            context = self.adapter.extract_target_context(output.hidden_states)
            generated, rounds = [anchor], []
            del output
            while len(generated) < maximum and anchor not in self.stops:
                synchronize(self.device)
                tick = time.perf_counter()
                prefix = int(cache.get_seq_length())
                first = self.adapter.propose_first(target_context=context,
                    anchor_ids=torch.tensor([anchor], device=self.device), draft_cache=draft_cache,
                    cache_prefix_length=prefix, compute_rank=False)
                logits = first.logits[0]
                synchronize(self.device)
                draft_ms = (time.perf_counter() - tick) * 1000
                feature_start = time.perf_counter()
                x = features(logits, context, rounds, prefix + 1, maximum - len(generated)) if policy or observer else None
                action, indices = None, None
                if policy:
                    action, indices = policy.choose(x, sample=sample, generator=generator)
                policy_ms = (time.perf_counter() - feature_start) * 1000
                observer_ms = 0.
                if observer:
                    observer_start = time.perf_counter()
                    observer(RoundState(logits, context, cache, anchor, maximum - len(generated), rounds, x, draft_ms))
                    synchronize(self.device)
                    observer_ms = (time.perf_counter() - observer_start) * 1000
                build_start = time.perf_counter()
                if action:
                    tree = build_layered(logits, action)
                elif method == "static_layered":
                    tree = build_layered(logits, static_action())
                elif method in {"ddtree", "acceptance_budget_control"}:
                    budget = self.cfg["baseline_budget"]
                    if method == "acceptance_budget_control" and rounds:
                        budget = max(30, min(60, int(30 + rounds[-1]["accepted"] / K * 30)))
                    tree = self.ddtree.build_from_logits(logits, budget)
                else:
                    raise ValueError(f"Unknown method or missing policy: {method}")
                synchronize(self.device)
                build_ms = (time.perf_counter() - build_start) * 1000
                verified = self.verifier.verify(anchor, tree, cache, context)
                appended = commit(verified.path.token_ids + [verified.path.bonus_token_id],
                                  maximum - len(generated), self.stops)
                generated.extend(appended)
                cache, context = verified.cache, verified.target_context
                anchor = verified.path.bonus_token_id
                # Disagreement along *verified* positions, not a target future signal.
                accepted_tokens = verified.path.token_ids
                draft_top = logits.argmax(-1).cpu().tolist()
                compared = accepted_tokens + [anchor]
                n = min(len(compared), K)
                disagreement = sum(draft_top[i] != compared[i] for i in range(n)) / max(n, 1)
                synchronize(self.device)
                elapsed = (time.perf_counter() - tick) * 1000 - observer_ms
                rounds.append({"accepted": len(accepted_tokens), "committed": len(appended),
                    "nodes": len(tree), "latency_ms": elapsed, "draft_ms": draft_ms,
                    "policy_ms": policy_ms, "build_ms": build_ms,
                    "greedy_disagreement": disagreement, "action": action.json() if action else None,
                    "indices": indices, "features": x.tolist() if x is not None else None})
                if appended[-1] in self.stops:
                    break
            tokens = torch.tensor(generated)
        synchronize(self.device)
        wall_ms = (time.perf_counter() - start) * 1000
        return {"tokens": tokens.tolist(), "wall_ms": wall_ms, "rounds": rounds}

    def warmup(self, policies):
        ids = self.encode("Explain why one plus one equals two.")
        old = self.cfg["max_new_tokens"]
        self.cfg = {**self.cfg, "max_new_tokens": min(old, 32)}
        try:
            for _ in range(self.cfg["warmup_runs"]):
                for name in ("ar", "dflash", "ddtree", "acceptance_budget_control", "static_layered"):
                    self.generate(ids, name)
                for policy in policies:
                    self.generate(ids, "layered", policy)
        finally:
            self.cfg = {**self.cfg, "max_new_tokens": old}
