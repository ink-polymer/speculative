from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
import importlib.util
import sys
import time

import torch

from .common import ROOT
from .config import Variant
from . import root_marginalized_bv as root_marginal
from . import protected_tree_bv
from . import atom_tree_bv
from . import diffusion_tree_bv
from .config import (SHARED_SUFFIX_METHODS, ATOM_TREE_METHODS, DIFFUSION_SCAFFOLD_METHODS,
                     DIFFUSION_LAW_METHODS as DIFFUSION_TREE_METHODS)
from .sampling import (block_verify_batched, block_verify_sparse,
                       block_verify_sparse_lazy, greedy_max_distribution,
                       matching_verify, probabilities, reweight_selected_path,
                       sample, select_and_reweight, select_greedy_path,
                       tree_block_verify_recycle,
                       tree_block_verify_terminal_mass,
                       tree_verify_ancestral_lazy_projection,
                       tree_verify_ancestral_batched,
                       token_verify)
from .fused_tree_sampling import (tree_verify_ancestral_fused,
                                  tree_verify_ancestral_fused_parallel,
                                  tree_verify_ancestral_fused_scan)
from .tree import (adaptive_path_proposal, adaptive_prefix_proposal,
                   budgeted_prefix_proposal, compact_cache, probability_tree,
                   sampled_tree)


SPARSE_FULL_TREE_METHODS = {
    "tree_gbv_full", "tree_gbv_prefix", "tree_gbv_full_sparse_ref",
    "tree_gbv_full_sparse_lazy",
}
FINITE_TREE_METHODS = SPARSE_FULL_TREE_METHODS | {"tree_gbv_full_dense_ref"}
RECYCLE_TREE_METHODS = {
    "tree_gbv_recycle", "tree_gbv_prefix_recycle",
    "tree_gbv_budgeted_prefix_recycle",
    "tree_gbv_budgeted_prefix_recycle_sparse_lazy_prefetch",
    "tree_gbv_budgeted_prefix_recycle_sparse_lazy",
    "tree_gbv_budgeted_prefix_recycle_host",
    "tree_gbv_slot_mixer_recycle",
    "tree_gbv_ratio_transport_recycle",
    "tree_gbv_prefix_recycle_packed",
}
PACKED_TREE_METHODS = {"tree_gbv_prefix_recycle_packed", "tree_gbv_packed"}
HOST_RECYCLE_METHODS = {"tree_gbv_budgeted_prefix_recycle_host"}
FINITE_TREE_METHODS.update(RECYCLE_TREE_METHODS)
TERMINAL_TREE_METHODS = {
    "ddtree_terminal_block", "ddtree_terminal_serial", "ddtree_terminal_dense",
}
FUSED_TREE_METHODS = {
    "ddtree_fused", "ddtree_fused_parallel", "ddtree_fused_scan",
}
LAZY_HEAD_TREE_METHODS = {"ddtree_lazy_projection"}


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
        self.proposal_adapter = None

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

    def target_hidden_forward(self, ids, cache, *, positions=None, mask=None):
        """Run the Target transformer without eagerly applying its LM head."""
        kwargs = dict(
            input_ids=ids, past_key_values=cache, use_cache=True,
            output_hidden_states=True, return_dict=True,
        )
        if positions is not None:
            kwargs["position_ids"] = positions
        if mask is not None:
            kwargs["attention_mask"] = mask
        return self.target.model(**kwargs)

    @staticmethod
    def greedy_trace_point(logits, generated_index):
        """Compact evidence for a greedy decision without serializing the vocabulary logits."""
        values = logits.float()
        token_id = int(values.argmax())
        runner_values = values.clone()
        runner_values[token_id] = float("-inf")
        runner_up_token_id = int(runner_values.argmax())
        return {
            "generated_index": generated_index,
            # torch.argmax is also the T=0 decoder's deterministic tie-break.
            # torch.topk may return another tied index first, so it must not be
            # used as the audited token identity.
            "token_id": token_id,
            "runner_up_token_id": runner_up_token_id,
            "top1_margin": float(values[token_id] - values[runner_up_token_id]),
        }

    @torch.inference_mode()
    def generate(self, input_ids, variant: Variant, max_new_tokens: int, stop_ids,
                 seed=0, profile=False, audit_greedy=False, tree_observer=None,
                 shared_suffix_observer=None, atom_observer=None, diffusion_observer=None,
                 ar_observer=None, scaffold_observer=None):
        variant.validate()
        if input_ids.shape[0] != 1 or max_new_tokens < 1:
            raise ValueError("Expected one prompt and max_new_tokens >= 1")
        if input_ids.shape[1] + max_new_tokens + self.draft.block_size > self.target.config.max_position_embeddings:
            raise ValueError("Prompt plus generation/verification exceeds model context length; no silent truncation")
        if variant.length >= self.draft.block_size and variant.method != "target":
            raise ValueError("Candidate length exceeds the checkpoint's future slots")
        if audit_greedy and variant.temperature != 0:
            raise ValueError("Greedy logit auditing requires temperature=0")
        generator = torch.Generator(device=self.device).manual_seed(seed)
        host_generator = torch.Generator(device="cpu").manual_seed(
            seed ^ 0x5A17_2026
        )
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
            if ar_observer is not None:
                # Diagnostic only, before sampling so invalid rows fail closed.
                ar_observer(0, initial.logits[0, -1:])
            anchor = int(sample(probabilities(initial.logits[0, -1], variant.temperature, dtype), generator))
            target_greedy_trace = []
            if audit_greedy and variant.method == "target":
                target_greedy_trace.append(self.greedy_trace_point(initial.logits[0, -1], 0))
            update = self.features(initial.hidden_states) if variant.method != "target" else None
            full_features = update if not variant.reuse_draft_cache else None
            del initial
        self.sync()
        prefill_end = time.perf_counter()
        generated = [anchor]
        rounds = []
        greedy_audit = []
        target_calls, draft_calls = 1, 0
        target_tokens = input_ids.shape[1]
        while len(generated) < max_new_tokens and generated[-1] not in stops:
            if variant.method == "target":
                with meter.measure("target_decode"):
                    output = self.target_forward(torch.tensor([[generated[-1]]], device=self.device),
                                                 target_cache, hidden=False, last_only=True)
                    if ar_observer is not None:
                        ar_observer(len(generated), output.logits[0, -1:])
                    token = int(sample(probabilities(output.logits[0, -1], variant.temperature, dtype), generator))
                    if audit_greedy:
                        target_greedy_trace.append(
                            self.greedy_trace_point(output.logits[0, -1], len(generated))
                        )
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
                proposal_hidden = hidden[:, 1:variant.length + 1]
                if variant.method == "tree_gbv_slot_mixer_recycle":
                    if self.proposal_adapter is None:
                        raise RuntimeError(
                            "tree_gbv_slot_mixer_recycle requires a loaded proposal adapter"
                        )
                    proposal_hidden = self.proposal_adapter(proposal_hidden)
                logits = self.target.get_output_embeddings()(proposal_hidden)[0]
                if variant.method == "tree_gbv_ratio_transport_recycle":
                    if self.proposal_adapter is None:
                        raise RuntimeError(
                            "tree_gbv_ratio_transport_recycle requires a loaded ratio head"
                        )
                    logits = self.proposal_adapter.correct_logits(
                        proposal_hidden, logits[None],
                        self.target.get_output_embeddings().weight,
                    )[0]
                draft_cache.crop(prefix_len)
                draft_temp = variant.draft_temperature or variant.temperature or 1.0
                q = (None if variant.method == "dflash" or variant.method in DIFFUSION_TREE_METHODS
                     else probabilities(logits, draft_temp, dtype))
                if variant.method in DIFFUSION_TREE_METHODS:
                    diffusion_law = diffusion_tree_bv.DiffusionBlockLaw.from_logits(
                        logits, noise_ids[0], draft_temp, int(self.draft.mask_token_id),
                        support_size=variant.diffusion_support_size)
                draft_calls += 1
            with meter.measure("tree_build"):
                if (variant.method == "ddtree" or variant.method in
                        TERMINAL_TREE_METHODS | FUSED_TREE_METHODS | LAZY_HEAD_TREE_METHODS):
                    tree = probability_tree(q, variant.tree_budget)
                    paths = None
                    tree_proposal = None
                elif variant.method == "dflash":
                    paths = logits.argmax(-1)[None]
                    tree = sampled_tree(paths)
                    tree_proposal = None
                elif variant.method in SHARED_SUFFIX_METHODS:
                    paths = root_marginal.propose(q, variant.paths, generator)
                    tree = sampled_tree(paths, variant.share_prefixes)
                    tree_proposal = None
                elif variant.method in DIFFUSION_TREE_METHODS:
                    coupling = "aligned" if variant.method == "diffusion_tree_bv_aligned" else "depth_permuted"
                    tree_proposal = diffusion_tree_bv.propose(diffusion_law, variant.paths, generator,
                                                             coupling=coupling)
                    paths = tree_proposal.paths()
                    if variant.method in DIFFUSION_SCAFFOLD_METHODS:
                        tree = diffusion_tree_bv.scaffold_tree(
                            tree_proposal, logits.argmax(-1), variant.tree_budget,
                            fill=variant.method != "diffusion_scaffold_no_fill")
                    else:
                        tree = sampled_tree(paths, variant.share_prefixes and variant.method != "diffusion_tree_bv_unmerged")
                elif variant.method in ATOM_TREE_METHODS:
                    coupling = (
                        "aligned" if variant.method == "atom_tree_bv_aligned"
                        else "fixed" if variant.method == "atom_tree_bv_fixed"
                        else "depth_permuted"
                    )
                    tree_proposal = atom_tree_bv.propose(
                        q, variant.paths, generator,
                        coupling=coupling,
                    )
                    paths = tree_proposal.paths()
                    tree = sampled_tree(paths, variant.share_prefixes)
                elif variant.method in {"tree_gbv", "tree_gbv_packed"}:
                    tree_proposal = adaptive_path_proposal(q, variant.tree_budget)
                    paths, proposal_token_probabilities = tree_proposal.sample(variant.paths, generator)
                    tree = sampled_tree(paths, variant.share_prefixes)
                elif variant.method in FINITE_TREE_METHODS:
                    if variant.method in {
                        "tree_gbv_budgeted_prefix_recycle",
                        "tree_gbv_budgeted_prefix_recycle_sparse_lazy",
                        "tree_gbv_budgeted_prefix_recycle_sparse_lazy_prefetch",
                        "tree_gbv_budgeted_prefix_recycle_host",
                        "tree_gbv_slot_mixer_recycle",
                        "tree_gbv_ratio_transport_recycle",
                    }:
                        constructor = budgeted_prefix_proposal
                    elif variant.method in {
                        "tree_gbv_prefix", "tree_gbv_prefix_recycle",
                        "tree_gbv_prefix_recycle_packed",
                    }:
                        constructor = adaptive_prefix_proposal
                    else:
                        constructor = adaptive_path_proposal
                    tree_proposal = constructor(q, variant.tree_budget)
                    paths = tree_proposal.paths
                    proposal_token_probabilities = tree_proposal.token_probabilities
                    tree = sampled_tree(paths, variant.share_prefixes)
                else:
                    paths = sample(q[None].expand(variant.paths, -1, -1), generator)
                    tree = sampled_tree(paths, variant.share_prefixes)
                    tree_proposal = None
                ids = torch.tensor([[generated[-1]] + tree.tokens], device=self.device)
                positions = (torch.tensor(tree.depths, device=self.device) + prefix_len)[None]
                mask = tree.mask(prefix_len, next(self.target.parameters()).dtype, self.device)
            with meter.measure("verify"):
                packed_cache = None
                if variant.method in PACKED_TREE_METHODS:
                    # Verify complete support paths as a regular causal batch.
                    # This duplicates shared tree nodes, but avoids the arbitrary
                    # 4-D tree mask and lets SDPA select its fast causal kernel.
                    packed_cache = self.cache_factory()
                    for layer_index, cached_layer in enumerate(target_cache.layers):
                        packed_cache.update(
                            cached_layer.keys.expand(paths.shape[0], -1, -1, -1),
                            cached_layer.values.expand(paths.shape[0], -1, -1, -1),
                            layer_index,
                        )
                    packed_ids = torch.cat((
                        torch.full((paths.shape[0], 1), generated[-1],
                                   dtype=torch.long, device=self.device),
                        paths,
                    ), dim=1)
                    packed_positions = torch.arange(
                        prefix_len, prefix_len + packed_ids.shape[1],
                        device=self.device,
                    )[None].expand(paths.shape[0], -1)
                    output = self.target_forward(
                        packed_ids, packed_cache, hidden=True,
                        positions=packed_positions,
                    )
                    packed_p = probabilities(output.logits, variant.temperature, dtype)
                    node_sources = [None] * len(tree.parents)
                    node_sources[0] = (0, 0)
                    for leaf, leaf_nodes in enumerate(tree.path_nodes):
                        for depth, node in enumerate(leaf_nodes, start=1):
                            if node_sources[node] is None:
                                node_sources[node] = (leaf, depth)
                    all_p = torch.stack([
                        packed_p[leaf, depth]
                        for leaf, depth in node_sources
                    ])
                    target_tokens += packed_ids.numel()
                else:
                    if variant.method in LAZY_HEAD_TREE_METHODS:
                        output = self.target_hidden_forward(
                            ids, target_cache, positions=positions, mask=mask
                        )
                        all_p = None
                    else:
                        output = self.target_forward(ids, target_cache, hidden=True,
                                                     positions=positions, mask=mask)
                        # Diffusion BV only needs the labelled proposal rows up
                        # front.  Its scaffold continuation normalizes the
                        # reached subtree lazily, so materializing FP64
                        # full-vocabulary probabilities for every tree row here
                        # wastes bandwidth without changing the sampling law.
                        all_p = (None if variant.method in (ATOM_TREE_METHODS - {"atom_tree_ancestral"})
                                 | (DIFFUSION_TREE_METHODS - {"diffusion_tree_ancestral"})
                                 else probabilities(output.logits[0], variant.temperature, dtype))
                    target_tokens += ids.shape[1]
                target_calls += 1
            if tree_observer is not None:
                # Only diagnostic runs attach an observer. Captures and their
                # synchronization must never contaminate primary throughput.
                tree_observer(tree.parents, tree.tokens,
                              all_p if all_p is not None else probabilities(
                                  self.target.get_output_embeddings()(
                                      output.last_hidden_state[0]
                                  ) if variant.method in LAZY_HEAD_TREE_METHODS
                                  else output.logits[0],
                                  variant.temperature, dtype,
                              ))
            if shared_suffix_observer is not None and variant.method in SHARED_SUFFIX_METHODS:
                # Diagnostic-only capture includes the ACTUAL proposal Q;
                # marginal verifier replay cannot be reconstructed from p alone.
                shared_suffix_observer(tree.parents, tree.tokens, paths, all_p, q)
            if atom_observer is not None and variant.method in ATOM_TREE_METHODS:
                # Diagnostic-only: preserve the latent law, not only its marginals.
                atom_observer(tree.parents, tree.tokens, tree_proposal, output.logits[0], variant.temperature)
            if diffusion_observer is not None and variant.method in DIFFUSION_TREE_METHODS:
                diffusion_observer(tree, tree_proposal, output.logits[0], variant.temperature)
            with meter.measure("select_and_correct"):
                if variant.method in RECYCLE_TREE_METHODS:
                    node_paths = torch.tensor(
                        tree.path_nodes, dtype=torch.long, device=self.device
                    )
                    recycle_children = {
                        (tree.parents[i], tree.tokens[i - 1]): i
                        for i in range(1, len(tree.parents))
                    }
                    nodes, tokens, bonus, recycle_stats = tree_block_verify_recycle(
                        paths, node_paths, recycle_children, all_p,
                        tree_proposal.leaf_probabilities,
                        proposal_token_probabilities, variant.paths, generator,
                        # All recycle verifier variants share the same exact
                        # block-verification law; this switch chooses whether
                        # to use dense per-segment BV rows or deferred-tails
                        # lazy sampling.
                        segment_verifier=(
                            "sparse_lazy"
                            if variant.method in {
                                "tree_gbv_budgeted_prefix_recycle_sparse_lazy",
                                "tree_gbv_budgeted_prefix_recycle_sparse_lazy_prefetch",
                            }
                            else "sparse" if variant.method in {
                                "tree_gbv_prefix_recycle",
                                "tree_gbv_budgeted_prefix_recycle",
                                "tree_gbv_budgeted_prefix_recycle_host",
                                "tree_gbv_slot_mixer_recycle",
                                "tree_gbv_ratio_transport_recycle",
                                "tree_gbv_prefix_recycle_packed",
                            } else "sparse_lazy"
                        ),
                        control_device=(
                            "cpu" if variant.method in HOST_RECYCLE_METHODS else "device"
                        ),
                        host_generator=host_generator,
                        precompute_subtrees=(
                            variant.method == "tree_gbv_budgeted_prefix_recycle_sparse_lazy_prefetch"
                        ),
                    )
                    accepted = len(nodes)
                elif variant.method in TERMINAL_TREE_METHODS:
                    nodes, tokens, bonus = tree_block_verify_terminal_mass(
                        # The shared FP64 softmax constructs these probability
                        # rows; topology checks remain enabled in the kernel.
                        tree.parents, tree.tokens, all_p, generator,
                        validate=False,
                        prefix_mode=("serial" if variant.method == "ddtree_terminal_serial"
                                     else "batched"),
                        exit_mode=("dense" if variant.method == "ddtree_terminal_dense"
                                   else "internal"),
                    )
                    accepted = len(nodes)
                elif variant.method in FUSED_TREE_METHODS:
                    verifier = {
                        "ddtree_fused": tree_verify_ancestral_fused,
                        "ddtree_fused_parallel": tree_verify_ancestral_fused_parallel,
                        "ddtree_fused_scan": tree_verify_ancestral_fused_scan,
                    }[variant.method]
                    nodes, tokens, bonus = verifier(
                        tree.parents, tree.tokens, all_p, generator,
                        validate=False,
                    )
                    accepted = len(nodes)
                elif variant.method in LAZY_HEAD_TREE_METHODS:
                    nodes, tokens, bonus, lazy_projection_stats = (
                        tree_verify_ancestral_lazy_projection(
                            tree.parents, tree.tokens,
                            output.last_hidden_state[0],
                            self.target.get_output_embeddings(),
                            variant.temperature, dtype, generator,
                            validate=False,
                        )
                    )
                    accepted = len(nodes)
                elif variant.method in {"ddtree", "root_shared_ddtree", "atom_tree_ancestral", "diffusion_tree_ancestral"}:
                    nodes, tokens, bonus = tree_verify_ancestral_batched(
                        tree.parents, tree.tokens, all_p, generator,
                        validate=False,
                    )
                    accepted = len(nodes)
                elif variant.method in DIFFUSION_SCAFFOLD_METHODS:
                    nodes, tokens, bonus = diffusion_tree_bv.verify_scaffold_logits(
                        output.logits[0], tree, tree_proposal, variant.temperature, generator,
                        recycle=variant.method != "diffusion_scaffold_no_recycle",
                        continuation="ancestral" if variant.method == "diffusion_scaffold_ancestral" else "terminal",
                        validate=False, node_probabilities=all_p)
                    accepted = len(nodes)
                elif variant.method in DIFFUSION_TREE_METHODS:
                    nodes, tokens, bonus = diffusion_tree_bv.verify_logits(
                        output.logits[0], tree, tree_proposal, variant.temperature, generator,
                        pool=variant.method != "diffusion_tree_bv_no_pool", validate=False)
                    accepted = len(nodes)
                elif variant.method in ATOM_TREE_METHODS:
                    # Distinct first tokens keep complete branches contiguous.
                    # Do not materialize all branch/depth probability or residual rows.
                    chosen, accepted_suffix, bonus = atom_tree_bv.verify_logits(
                        output.logits[0, 0],
                        output.logits[0, 1:].view(variant.paths, variant.length, -1),
                        tree_proposal, variant.temperature, generator,
                        pool=variant.method != "atom_tree_bv_no_pool", validate=False,
                    )
                    nodes = [] if chosen == -1 else tree.path_nodes[chosen][:accepted_suffix + 1]
                    tokens = [tree.tokens[node - 1] for node in nodes]
                    accepted = len(nodes)
                elif variant.method in SHARED_SUFFIX_METHODS:
                    # Distinct roots imply no cross-branch prefix sharing.
                    # sampled_tree stores complete branches consecutively, so
                    # use a view instead of copying a K x L x V probability tensor.
                    branch_p = all_p[1:].view(variant.paths, variant.length, -1)
                    verifier = (
                        protected_tree_bv.verify if variant.method == "root_protected_bv"
                        else root_marginal.verify if variant.method == "root_marginal_bv_ref"
                        else root_marginal.verify_early_root if variant.method == "root_early_bv"
                        else root_marginal.verify_tensorized
                    )
                    options = {"rule": "token"} if variant.method == "root_marginal_token" else {}
                    chosen, accepted_suffix, bonus = verifier(
                        paths[:, 0], all_p[0], paths[0, 1:], branch_p, q[1:],
                        generator, validate=False, **options,
                    )
                    nodes = [] if chosen == -1 else tree.path_nodes[chosen][:accepted_suffix + 1]
                    tokens = [tree.tokens[node - 1] for node in nodes]
                    accepted = len(nodes)
                    del branch_p
                else:
                    if variant.method in SPARSE_FULL_TREE_METHODS:
                        # Selecting a finite-tree leaf needs only the Target
                        # probability of each proposed token.  Avoid building a
                        # leaves x length x vocabulary tensor before one leaf is
                        # selected; materialize dense Target rows only for it.
                        target_row_nodes = torch.tensor(
                            [[0] + nodes[:-1] for nodes in tree.path_nodes],
                            dtype=torch.long, device=self.device,
                        )
                        target_token_probabilities = all_p[
                            target_row_nodes, paths
                        ]
                        selected_leaf_probabilities = greedy_max_distribution(
                            paths, target_token_probabilities, proposal_token_probabilities,
                            tree_proposal.leaf_probabilities, variant.paths,
                        )
                        chosen = int(sample(selected_leaf_probabilities, generator).item())
                        chosen_nodes = tree.path_nodes[chosen]
                        chosen_p = all_p[[0] + chosen_nodes]
                        proposal_tokens, proposal_probabilities = tree_proposal.conditional_sparse(
                            paths[chosen], selected_leaf_probabilities
                        )
                        sparse_verifier = (
                            block_verify_sparse_lazy
                            if variant.method == "tree_gbv_full_sparse_lazy"
                            else block_verify_sparse
                        )
                        accepted, bonus = sparse_verifier(
                            paths[chosen], chosen_p, proposal_tokens,
                            proposal_probabilities, generator,
                        )
                    else:
                        p_by_path = torch.stack([all_p[[0] + nodes]
                                                 for nodes in tree.path_nodes])
                    if variant.method == "dflash":
                        chosen, r = 0, None
                        accepted, bonus = matching_verify(paths[0], p_by_path[0], generator)
                    elif variant.method == "gbv":
                        chosen, r = select_and_reweight(paths, p_by_path, q)
                    elif variant.method in {"tree_gbv", "tree_gbv_packed"}:
                        target_token_probabilities = p_by_path[:, :-1].gather(
                            2, paths[:, :, None]
                        )[:, :, 0]
                        chosen = select_greedy_path(
                            paths, target_token_probabilities, proposal_token_probabilities
                        )
                        proposal_rows = tree_proposal.conditional_rows(paths[chosen])
                        r = reweight_selected_path(
                            paths[chosen], p_by_path[chosen], proposal_rows, variant.paths
                        )
                    elif variant.method == "tree_gbv_full_dense_ref":
                        target_token_probabilities = p_by_path[:, :-1].gather(
                            2, paths[:, :, None]
                        )[:, :, 0]
                        selected_leaf_probabilities = greedy_max_distribution(
                            paths, target_token_probabilities, proposal_token_probabilities,
                            tree_proposal.leaf_probabilities, variant.paths,
                        )
                        chosen = int(sample(selected_leaf_probabilities, generator).item())
                        r = tree_proposal.conditional_rows(
                            paths[chosen], selected_leaf_probabilities
                        )
                    elif variant.method not in SPARSE_FULL_TREE_METHODS:
                        chosen, r = 0, q
                    if variant.method not in {"dflash", *SPARSE_FULL_TREE_METHODS}:
                        verifier = token_verify if variant.method == "token" else block_verify_batched
                        accepted, bonus = verifier(paths[chosen], p_by_path[chosen], r, generator)
                    nodes = tree.path_nodes[chosen][:accepted]
                    tokens = paths[chosen, :accepted].tolist()
            if scaffold_observer is not None and variant.method in DIFFUSION_SCAFFOLD_METHODS:
                # Diagnostic-only witness of the actual draft and selected exit.
                # Tree logits alone cannot certify fixed greedy coverage or
                # maximal continuation, the premises of the DFlash tail bound.
                scaffold_observer(tree, tree_proposal, logits, nodes, tokens, bonus)
            if audit_greedy and variant.method != "target":
                row_index = torch.tensor([0] + nodes, device=self.device)
                tree_logits = (
                    self.target.get_output_embeddings()(
                        output.last_hidden_state[0].index_select(0, row_index)
                    )
                    if variant.method in LAZY_HEAD_TREE_METHODS
                    else output.logits[0].index_select(0, row_index)
                )
                # Match the production AR baseline exactly: prompt prefill followed
                # by one cached token at a time. A full-sequence forward may use a
                # different SDPA kernel and is not the baseline being audited.
                sequential_cache = self.cache_factory()
                self.target_forward(input_ids, sequential_cache, hidden=False, last_only=True)
                sequential_rows = []
                for audit_token in generated + tokens:
                    audit_output = self.target_forward(
                        torch.tensor([[audit_token]], device=self.device), sequential_cache,
                        hidden=False, last_only=True
                    )
                    sequential_rows.append(audit_output.logits[0, -1])
                sequential_logits = torch.stack(sequential_rows[-(len(tokens) + 1):])
                tree_argmax = tree_logits.argmax(-1)
                sequential_argmax = sequential_logits.argmax(-1)
                tree_top2 = tree_logits.float().topk(2, dim=-1)
                sequential_top2 = sequential_logits.float().topk(2, dim=-1)
                greedy_audit.append({
                    "generated_start_index": len(generated),
                    "tree_token_ids": tree_argmax.tolist(),
                    "sequential_token_ids": sequential_argmax.tolist(),
                    "argmax_equal": tree_argmax.eq(sequential_argmax).tolist(),
                    "max_absolute_logit_error": (tree_logits.float() - sequential_logits.float()).abs().amax(-1).tolist(),
                    "tree_top1_margin": (tree_top2.values[:, 0] - tree_top2.values[:, 1]).tolist(),
                    "sequential_top1_margin": (sequential_top2.values[:, 0] - sequential_top2.values[:, 1]).tolist(),
                })
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
                if variant.method in PACKED_TREE_METHODS:
                    cache_leaf = next(
                        leaf for leaf, leaf_nodes in enumerate(tree.path_nodes)
                        if leaf_nodes[:len(nodes)] == nodes
                    ) if nodes else 0
                    selected_hidden = tuple(
                        layer[cache_leaf:cache_leaf + 1]
                        for layer in output.hidden_states
                    )
                    packed_rows = torch.arange(
                        len(keep), dtype=torch.long, device=self.device
                    )
                    update = self.features(selected_hidden, packed_rows)
                    for layer_index, packed_layer in enumerate(packed_cache.layers):
                        new_keys = packed_layer.keys[
                            cache_leaf:cache_leaf + 1, :, prefix_len:prefix_len + len(keep), :
                        ]
                        new_values = packed_layer.values[
                            cache_leaf:cache_leaf + 1, :, prefix_len:prefix_len + len(keep), :
                        ]
                        target_cache.update(new_keys, new_values, layer_index)
                else:
                    update = self.features(output.hidden_states, index)
                if full_features is not None:
                    full_features = torch.cat((full_features, update), dim=1)
                if variant.method not in PACKED_TREE_METHODS:
                    compact_cache(target_cache, prefix_len, keep, self.device)
                round_stats = {
                    "accepted_draft_tokens": accepted,
                    "committed_tokens": len(committed),
                    "committed_draft_tokens": min(accepted, len(committed)),
                    "proposed_tokens": (
                        variant.paths * variant.length
                        if paths is not None else len(tree.tokens)
                    ),
                    "tree_nodes": len(tree.tokens),
                    "verify_tokens": len(tree.parents),
                }
                if variant.method in RECYCLE_TREE_METHODS:
                    round_stats.update({
                        "tree_bv_segments": recycle_stats["segments"],
                        "recycled_corrections": recycle_stats["recycled_corrections"],
                    })
                if variant.method in ATOM_TREE_METHODS:
                    round_stats.update({"proposal_kind": "correlated_latent_atoms",
                                        "proposal_support_size": tree_proposal.tokens.shape[-1],
                                        "proposal_atom_columns": tree_proposal.source.shape[-1],
                                        "coupling": coupling,
                                        "joint_block_verification": variant.method != "atom_tree_ancestral"})
                if variant.method in DIFFUSION_TREE_METHODS:
                    round_stats.update({"proposal_kind": "one_step_masked_block_diffusion",
                                        "denoising_steps": 1, "stochastic_positions": variant.length,
                                        "proposal_support_size": tree_proposal.tokens.shape[-1],
                                        "proposal_atom_columns": tree_proposal.source.shape[-1],
                                        "labelled_paths": variant.paths, "coupling": coupling,
                                        "joint_block_verification": variant.method != "diffusion_tree_ancestral"})
                    if variant.method in DIFFUSION_SCAFFOLD_METHODS:
                        round_stats.update({"fixed_greedy_scaffold": True,
                                            "scaffold_fill": variant.method != "diffusion_scaffold_no_fill",
                                            "correction_recycling": variant.method != "diffusion_scaffold_no_recycle",
                                            "continuation_backend": "ancestral" if variant.method == "diffusion_scaffold_ancestral" else "terminal"})
                if variant.method in LAZY_HEAD_TREE_METHODS:
                    round_stats.update({
                        "vocabulary_projection_rows": lazy_projection_stats["projected_rows"],
                        "full_vocabulary_projection_rows": lazy_projection_stats["total_tree_rows"],
                        "internal_projection_rows": lazy_projection_stats["internal_projected_rows"],
                        "leaf_projection_rows": lazy_projection_stats["leaf_projected_rows"],
                    })
                rounds.append(round_stats)
                del output, all_p, hidden, logits
                if variant.method in PACKED_TREE_METHODS:
                    del packed_cache, packed_ids, packed_positions, packed_p, selected_hidden
                if variant.method in RECYCLE_TREE_METHODS:
                    del node_paths, recycle_children
                elif variant.method in SPARSE_FULL_TREE_METHODS:
                    del (chosen_p, proposal_tokens, proposal_probabilities,
                         selected_leaf_probabilities, target_token_probabilities)
                elif paths is not None and variant.method not in SHARED_SUFFIX_METHODS | ATOM_TREE_METHODS | DIFFUSION_TREE_METHODS:
                    del p_by_path, r
                if variant.method in DIFFUSION_TREE_METHODS:
                    del diffusion_law
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
            "greedy_audit": greedy_audit,
            "target_greedy_trace": target_greedy_trace,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(self.device) if self.device.type == "cuda" else None,
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(self.device) if self.device.type == "cuda" else None,
            "stages": meter.result(),
        }
