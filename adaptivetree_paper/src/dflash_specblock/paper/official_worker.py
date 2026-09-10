"""One official (dataset, model, target backend) run, with additive Adaptive methods."""
from __future__ import annotations

import random

import numpy as np
import torch

from .common import (METHOD_SCHEMA_VERSION, PRIMARY_ADAPTIVE_METHOD, atomic_json,
                     digest, file_hash, load_json)
from .controller import (DIAGNOSTIC_VARIANTS, controller_config,
                         deprecated_experiment_flags,
                         expected_official_controller_configs,
                         make_paper_builder)
from .official_data import check_manifest
from .official_spec import BUDGETS, LIMITS, MODELS, upstream
from .official_audit import gpu_identity, validate_hardware
from .wandb_monitor import finish_wandb, initialize_wandb, log_response, wandb_contract


def method_names(backend, variants, diagnostic_variants=()):
    if set(diagnostic_variants) - set(DIAGNOSTIC_VARIANTS):
        raise ValueError("Unknown diagnostic AdaptiveTree variant")
    names = ["baseline", "dflash"]
    if backend == "sdpa":
        names += ([f"ddtree_tb{budget}" for budget in BUDGETS]
                  + list(variants) + list(diagnostic_variants))
    elif backend != "flash_attention_2":
        raise ValueError("Unexpected official target backend")
    return names


def scheduled_methods(methods, response_ordinal, policy):
    """Return a deterministic order with balanced timed positions.

    A cyclic rotation gives every method each execution position equally often
    over every complete cycle.  The legacy order remains available for exact
    upstream protocol reproduction.
    """
    if policy == "official-fixed":
        return list(methods)
    if policy != "balanced-rotation":
        raise ValueError("Unknown method order policy")
    offset = response_ordinal % len(methods)
    return list(methods[offset:] + methods[:offset])


def response_tokens(result):
    return result.output_ids[0, result.num_input_tokens:].tolist()


def _first_difference(reference, candidate):
    common = min(len(reference), len(candidate))
    for position in range(common):
        if reference[position] != candidate[position]:
            return position
    return common if len(reference) != len(candidate) else None


def audit_response(response, *, index, turn, input_ids, diagnostic_path, policy="strict"):
    if policy not in {"strict", "record-bf16-mismatches"}:
        raise ValueError("Unknown greedy audit policy")
    reference = response_tokens(response["baseline"])
    mismatches = {name: response_tokens(value) for name, value in response.items()
                  if response_tokens(value) != reference}
    first_differences = {name:_first_difference(reference, tokens)
                         for name,tokens in mismatches.items()}
    if mismatches:
        atomic_json(diagnostic_path, {"index": index, "turn": turn,
            "baseline_tokens": reference, "mismatching_tokens": mismatches,
            "first_mismatch_indices":first_differences,
            "greedy_audit_policy":policy,
            "message":("Official greedy token mismatch; no successful-run marker is written."
                       if policy == "strict" else
                       "Greedy output divergence observed during BF16 execution; "
                       "the divergence cause is not inferred, and timings are not eligible "
                       "for a strict lossless claim.")})
        if policy == "strict":
            raise RuntimeError(f"Greedy mismatch; diagnostic saved to {diagnostic_path}")
    return {"index": index, "turn": turn,
            "input_sha256": digest(input_ids.detach().cpu().tolist()),
            "exact_match":not mismatches, "greedy_audit_policy":policy,
            "mismatching_methods":sorted(mismatches),
            "first_mismatch_indices":first_differences}


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
        raise ValueError(f"AdaptiveTree T=0 requires K=15; loaded official draft has block_size={block_size}")
    maximum = min(config["max_new_tokens"], 32) if args.smoke_count else config["max_new_tokens"]
    rows = load_json(args.data_dir / f"{args.dataset}.json")
    if args.smoke_count:
        rows = rows[:args.smoke_count]
    legacy_cli_aliases = deprecated_experiment_flags(
        cost_attribution=getattr(args, "experimental_cost_attribution", False),
        extended_budgets=getattr(args, "experimental_extended_budgets", False),
    )
    controller_names = tuple(config["variants"])
    methods = method_names(args.backend, config["variants"])
    controllers = {name: make_paper_builder(config["adaptive"], name)
                   for name in controller_names} if args.backend == "sdpa" else {}
    controller_configs = {
        name:controller_config(builder) for name,builder in controllers.items()
    }
    expected_controller_configs = (
        expected_official_controller_configs(controller_names)
        if args.backend == "sdpa" else {}
    )
    if controller_configs != expected_controller_configs:
        raise RuntimeError("Constructed AdaptiveTree controllers differ from the registry")
    audit_policy = getattr(args, "greedy_audit_policy", "strict")
    order_policy = getattr(args, "method_order_policy", "official-fixed")
    wandb_run = initialize_wandb(
        args, model_name=target_name, draft_name=draft_name, backend=args.backend,
        rank=rank, world_size=world, controller_configs=controller_configs)

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
    controllers = {name: make_paper_builder(config["adaptive"], name)
                   for name in controllers}
    responses = []
    response_offsets = []
    response_count = 0
    for row in rows:
        response_offsets.append(response_count)
        response_count += len(row["turns"])
    for idx in range(rank, len(rows), world):
        messages = []
        conditioning_history = []
        for turn, user_content in enumerate(rows[idx]["turns"]):
            messages.append({"role":"user", "content":user_content})
            ids = encode(messages)
            response = {}
            execution_order = scheduled_methods(
                methods, response_offsets[idx] + turn, order_policy
            )
            for method in execution_order:
                response[method] = generate(ids, method, maximum)
            audit = audit_response(response, index=idx, turn=turn, input_ids=ids,
                diagnostic_path=args.output.with_name(args.output.stem + f".rank{rank}.mismatch.json"),
                policy=audit_policy)
            # Bind both the immutable source prompt and the exact generated-token
            # history that conditioned this turn.  In mismatch-recording mode the
            # two target backends may legitimately have different MT-Bench
            # continuation contexts, but neither may silently change the user
            # turns or use an unexplained history.
            audit["source_turns_sha256"] = digest(rows[idx]["turns"])
            audit["user_turn_sha256"] = digest(user_content)
            audit["conditioning_history_sha256"] = digest(conditioning_history)
            audit["method_order"] = execution_order
            log_response(wandb_run, response, step=len(responses), index=idx, turn=turn)
            # Adding ablations must NOT change the official multi-turn conditioning:
            # SDPA uses the registered fixed DDTree reference; FA2 uses DFlash.
            history_method = f"ddtree_tb{BUDGETS[-1]}" if args.backend == "sdpa" else "dflash"
            history_tokens = response_tokens(response[history_method])
            text = tokenizer.decode(history_tokens, skip_special_tokens=True)
            messages.append({"role":"assistant", "content":text})
            conditioning_history.append(history_tokens)
            response["_audit"] = audit
            responses.append(response)
        print(f"{args.dataset} model={args.model_index} {args.backend} rank={rank} case={idx+1}/{len(rows)}", flush=True)
    hardware = {**gpu_identity(device, rank), "flash_attn":getattr(flash_attn, "__version__", "unknown")}
    validate_hardware([hardware], environment, world)
    hardware = u.dist.all_gather(hardware)
    local_responses = responses
    if world > 1:
        gathered = u.dist.gather(responses, dst=0)
        if not u.dist.is_main():
            finish_wandb(wandb_run, local_responses, methods)
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
                "method_schema_version":METHOD_SCHEMA_VERSION,
                "primary_adaptive_method":PRIMARY_ADAPTIVE_METHOD,
                "greedy_audit_policy":audit_policy,
                "method_order_policy":order_policy,
                "adaptive_timing":"timing attribution is method-specific and frozen in controller_configs; all controller overhead is included in the official decode timer",
                "controller_configs":controller_configs,
                "substage_note":"Adaptive fine-grained tree_build_* attribution unavailable; use aggregate tree_build"}
    if wandb_contract(args):
        run_data["wandb"] = wandb_contract(args)
    if legacy_cli_aliases:
        run_data["deprecated_cli_aliases"] = list(legacy_cli_aliases)
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
        "cases":len(rows), "methods":methods, "smoke":bool(args.smoke_count),
        "greedy_audit_policy":audit_policy})
    finish_wandb(wandb_run, responses, methods)
