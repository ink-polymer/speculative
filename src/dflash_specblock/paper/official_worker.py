"""One official (dataset, model, target backend) run, with additive Adaptive methods."""
from __future__ import annotations

import hashlib
import os
import random
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from .common import atomic_json, digest, file_hash, load_json
from .controller import PaperAdaptiveBuilder
from .official_data import check_manifest
from .official_spec import BUDGETS, LIMITS, MODELS, upstream
from .official_audit import gpu_identity, validate_hardware


def method_names(backend, variants):
    names = ["baseline", "dflash"]
    if backend == "sdpa":
        names += [f"ddtree_tb{budget}" for budget in BUDGETS] + list(variants)
    elif backend != "flash_attention_2":
        raise ValueError("Unexpected official target backend")
    return names


def response_tokens(result):
    return result.output_ids[0, result.num_input_tokens:].tolist()


def audit_response(response, *, index, turn, input_ids, diagnostic_path):
    reference = response_tokens(response["baseline"])
    mismatches = {name: response_tokens(value) for name, value in response.items()
                  if response_tokens(value) != reference}
    if mismatches:
        atomic_json(diagnostic_path, {"index": index, "turn": turn,
            "baseline_tokens": reference, "mismatching_tokens": mismatches,
            "message": "Official greedy token mismatch; no successful-run marker is written."})
        raise RuntimeError(f"Greedy mismatch; diagnostic saved to {diagnostic_path}")
    return {"index": index, "turn": turn,
            "input_sha256": digest(input_ids.detach().cpu().tolist()), "exact_match": True}


def worker(args, config):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from .adaptive_official import adaptive_generate
    u = upstream()
    u.dist.init()
    local_rank, rank, world = u.dist.local_rank(), u.dist.rank(), u.dist.size()
    if not torch.cuda.is_available():
        raise RuntimeError("Official benchmark requires NVIDIA CUDA")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # These are the official defaults: no forced TF32 setting, no CUDA Graph.
    import flash_attn  # required for the draft, also in the SDPA target run
    environment = load_json(args.run_dir / "environment.json")
    hardware = {**gpu_identity(device, rank), "flash_attn":getattr(flash_attn, "__version__", "unknown")}
    validate_hardware([hardware], environment, world)
    u.ddtree.maybe_enable_cpp_compact(True)
    # Unlike upstream's silent fallback, refuse to label a different compaction
    # implementation an exact official reproduction.
    if u.ddtree.load_cpp_compact_module() is None:
        raise RuntimeError("Official C++ cache compaction failed; fix the compiler environment")
    check_manifest(args.data_dir)
    lock = load_json(args.data_dir / "source_revisions.json")
    target_name, draft_name = MODELS[args.model_index]
    target = AutoModelForCausalLM.from_pretrained(target_name,
        revision=lock["models"][target_name], attn_implementation=args.backend,
        dtype=torch.bfloat16).to(device).eval()
    draft = u.model.DFlashDraftModel.from_pretrained(draft_name,
        revision=lock["models"][draft_name], attn_implementation="flash_attention_2",
        dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(target_name, revision=lock["models"][target_name])
    block_size = draft.block_size
    if block_size != 16:
        raise ValueError(f"Original AdaptiveTree requires K=15; loaded official draft has block_size={block_size}")
    maximum = min(config["max_new_tokens"], 32) if args.smoke_count else config["max_new_tokens"]
    rows = load_json(args.data_dir / f"{args.dataset}.json")
    if args.smoke_count:
        rows = rows[:args.smoke_count]
    methods = method_names(args.backend, config["variants"])
    controllers = {name: PaperAdaptiveBuilder(config["adaptive"], name)
                   for name in config["variants"]} if args.backend == "sdpa" else {}

    def generate(ids, method, max_tokens):
        kwargs = dict(model=draft, target=target, input_ids=ids,
            mask_token_id=draft.mask_token_id, max_new_tokens=max_tokens,
            block_size=1 if method == "baseline" else block_size,
            stop_token_ids=[tokenizer.eos_token_id], temperature=0.)
        if method in {"baseline", "dflash"}:
            return u.dflash.dflash_generate(**kwargs)
        if method.startswith("ddtree_tb"):
            return u.ddtree.ddtree_generate(**kwargs, tree_budget=int(method.removeprefix("ddtree_tb")))
        return adaptive_generate(**kwargs, builder=controllers[method])

    def encode(messages):
        text = tokenizer.apply_chat_template(messages, tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        return tokenizer.encode(text, return_tensors="pt").to(target.device)

    # Same warmup string, order and length as official benchmark.py.
    warmup_ids = encode([{"role":"user", "content":"Warmup"}])
    for method in methods:
        generate(warmup_ids, method, min(maximum, 16))
    controllers = {name: PaperAdaptiveBuilder(config["adaptive"], name) for name in controllers}
    responses = []
    for idx in range(rank, len(rows), world):
        messages = []
        for turn, user_content in enumerate(rows[idx]["turns"]):
            messages.append({"role":"user", "content":user_content})
            ids = encode(messages)
            response = {}
            for method in methods:
                response[method] = generate(ids, method, maximum)
            audit = audit_response(response, index=idx, turn=turn, input_ids=ids,
                diagnostic_path=args.output.with_name(args.output.stem + f".rank{rank}.mismatch.json"))
            # Adding ablations must NOT change the official multi-turn conditioning:
            # SDPA uses the last original DDTree budget (1024); FA2 uses DFlash.
            history_method = "ddtree_tb1024" if args.backend == "sdpa" else "dflash"
            text = tokenizer.decode(response_tokens(response[history_method]), skip_special_tokens=True)
            messages.append({"role":"assistant", "content":text})
            response["_audit"] = audit
            responses.append(response)
        print(f"{args.dataset} model={args.model_index} {args.backend} rank={rank} case={idx+1}/{len(rows)}", flush=True)
    hardware = {**gpu_identity(device, rank), "flash_attn":getattr(flash_attn, "__version__", "unknown")}
    validate_hardware([hardware], environment, world)
    hardware = u.dist.all_gather(hardware)
    if world > 1:
        gathered = u.dist.gather(responses, dst=0)
        if not u.dist.is_main():
            return
        responses = [item for group in gathered for item in group]
    run_data = {"responses":responses, "block_size":block_size,
                "draft_attn_implementation":"flash_attention_2",
                "target_attn_implementation":args.backend,
                "args":{"dataset":args.dataset, "model_name_or_path":target_name,
                        "draft_name_or_path":draft_name, "temperature":0.,
                        "max_samples":LIMITS[args.dataset], "max_new_tokens":maximum,
                        "tree_budget":",".join(map(str, BUDGETS)), "flash_attn":args.backend!="sdpa"},
                "protocol_identity":args.identity, "source_lock":lock,
                "world_size":world, "hardware":hardware,
                "smoke":bool(args.smoke_count), "methods":methods,
                "adaptive_timing":"proposal+build; compile+verify+KV/commit; all controller overhead included in official decode timer",
                "substage_note":"Adaptive fine-grained tree_build_* attribution unavailable; use aggregate tree_build"}
    expected_turns = sum(len(r["turns"]) for r in rows)
    if len(responses) != expected_turns:
        raise RuntimeError("Missing official response turns")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite an unvalidated run artifact; use a new run directory")
    temp = args.output.with_suffix(".pt.tmp")
    torch.save(run_data, temp)
    temp.replace(args.output)
    atomic_json(args.output.with_suffix(".complete.json"), {
        "identity":args.identity, "sha256":file_hash(args.output), "turns":expected_turns,
        "cases":len(rows), "methods":methods, "smoke":bool(args.smoke_count)})
