"""Historical non-default controlled runtime; current official runner is official_worker.py."""
from __future__ import annotations

import re
import time

import torch

from ..benchmark import baseline_greedy
from ..config import ExperimentConfig
from ..engine import DFlashSpecBlockEngine
from ..device import configure_cuda_runtime, dtype_from_name, resolve_device, synchronize
from ..dflash_adapter import DFlashBlockAdapter
from ..models import ModelBundle, _freeze_and_place, _pretrained_kwargs, _validate_model_pair, render_prompt
from ..rank_head import HeuristicRanker
from ..vanilla_engine import VanillaDFlashEngine
from ..verification import TargetTreeVerifier
from .common import ROOT, K, BASELINES
from .controller import PaperAdaptiveBuilder, make_builder


def commit(tokens, remaining, stops):
    result = list(tokens[:remaining])
    for i, token in enumerate(result):
        if token in stops:
            return result[:i + 1]
    return result


class PaperRuntime:
    def __init__(self, cfg, *, bundle=None, device=None):
        self.cfg = cfg
        self.model_cfg = ExperimentConfig.from_json(ROOT / cfg["model_config"])
        mc = self.model_cfg
        if mc.block_size != K or mc.max_blocks != 1 or mc.tree_budget != cfg["baseline_budget"]:
            raise ValueError("Model/protocol block size or baseline budget mismatch")
        if mc.device != "cuda:0" or mc.dtype != "bfloat16" or not mc.allow_tf32:
            raise ValueError("Formal protocol uses cuda:0, BF16; select GPU via CUDA_VISIBLE_DEVICES")
        if any(not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision)
               for revision in (mc.target_revision, mc.draft_revision)):
            raise ValueError("Both model revisions must be immutable commit SHAs")
        if mc.use_cuda_graphs or mc.torch_compile_mode or mc.attn_implementation != "sdpa":
            raise ValueError("The controlled protocol requires SDPA target, no compile/graphs")
        if mc.draft_attn_implementation != "sdpa" or mc.rank_checkpoint or mc.ddtree_reserve_greedy_chain:
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
                **_pretrained_kwargs(mc, mc.target_revision, self.device, mc.attn_implementation))
            draft = AutoModel.from_pretrained(mc.draft_model_id,
                **_pretrained_kwargs(mc, mc.draft_revision, self.device, mc.draft_attn_implementation))
            bundle = ModelBundle(_freeze_and_place(target, self.device),
                                 _freeze_and_place(draft, self.device), tokenizer)
            _validate_model_pair(bundle.target, bundle.draft)
        self.target, self.tokenizer = bundle.target, bundle.tokenizer
        self.adapter = DFlashBlockAdapter(bundle.target, bundle.draft, HeuristicRanker(), K)
        self.verifier = TargetTreeVerifier(bundle.target, self.adapter.target_layer_ids,
                                          self.device, dtype_from_name(mc.dtype))
        self.engines = {}
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

    def engine_for(self, method, replica):
        key = f"{method}:{replica}"
        if key not in self.engines:
            builder = make_builder(self.cfg, method)
            self.engines[key] = DFlashSpecBlockEngine(self.target, self.adapter, builder,
                                                    self.verifier, self.device)
        return self.engines[key]

    def reset_controllers(self):
        self.engines = {}

    def controller_states(self):
        return {key: engine.tree_builder.state_dict() for key, engine in sorted(self.engines.items())
                if isinstance(engine.tree_builder, PaperAdaptiveBuilder)}

    def restore_controllers(self, states):
        for key, state in states.items():
            method, replica = key.rsplit(":", 1)
            builder = self.engine_for(method, int(replica)).tree_builder
            if not isinstance(builder, PaperAdaptiveBuilder):
                raise ValueError("Invalid serialized controller method")
            builder.load_state_dict(state)
        if set(self.controller_states()) != set(states):
            raise ValueError("Unexpected extra controller states")

    @torch.inference_mode()
    def generate(self, ids, method, replica=0):
        maximum = self.cfg["max_new_tokens"]
        if maximum < 0:
            raise ValueError("max_new_tokens must be nonnegative")
        if maximum == 0:
            return {"tokens": [], "wall_ms": 0., "rounds": []}
        engine = None if method in {"ar", "dflash"} else self.engine_for(method, replica)
        if engine and isinstance(engine.tree_builder, PaperAdaptiveBuilder):
            engine.tree_builder.trace = []
        synchronize(self.device)
        start = time.perf_counter()
        if method == "ar":
            tokens, _ = baseline_greedy(self.target, ids, maximum, self.stops, self.device)
            rounds, decisions = [], []
        else:
            result = (self.vanilla if method == "dflash" else engine).generate(ids, maximum, self.stops)
            tokens = result.generated_ids[0].detach().cpu()
            rounds = [{"accepted": r.accepted_draft_tokens, "committed": r.committed_tokens,
                       "nodes": K if method == "dflash" else r.tree_nodes,
                       "latency_ms": r.draft_ms + r.verify_ms,
                       "draft_ms": r.draft_ms, "verify_ms": r.verify_ms,
                       "build_ms": getattr(r, "tree_build_ms", 0.)} for r in result.iterations]
            decisions = (list(engine.tree_builder.trace)
                         if engine and isinstance(engine.tree_builder, PaperAdaptiveBuilder) else [])
        synchronize(self.device)
        wall_ms = (time.perf_counter() - start) * 1000
        return {"tokens": tokens.tolist(), "wall_ms": wall_ms, "rounds": rounds,
                "decisions": decisions}

    def warmup(self):
        # Synthetic, non-benchmark prompt. Kernel warmup is not controller calibration:
        # discard all its controller observations before the measured stream.
        ids = self.encode("Explain why one plus one equals two.")
        old = self.cfg
        self.cfg = {**old, "max_new_tokens": min(old["max_new_tokens"], 32)}
        try:
            for _ in range(self.cfg["warmup_runs"]):
                for method in (*BASELINES, *self.cfg["variants"]):
                    self.generate(ids, method)
        finally:
            self.cfg = old
            self.reset_controllers()
