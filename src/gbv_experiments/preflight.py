"""Production-checkpoint checks executed on the benchmark GPU before long runs."""
from __future__ import annotations

from dataclasses import replace
import importlib.metadata
from pathlib import Path
import platform
import subprocess

from .common import digest, source_hashes, write_json
from .config import Variant, build_variants


def check_environment(cfg, code_backend="docker", device="cuda:0"):
    import torch
    import math_verify  # noqa: F401
    from transformers import Qwen3ForCausalLM  # noqa: F401
    if not torch.cuda.is_available() or torch.device(device).type != "cuda":
        raise RuntimeError("CUDA is unavailable")
    torch.cuda.set_device(device)
    if cfg["model"].get("dtype", "bfloat16") == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("The GPU does not support BF16")
    image_id = None
    if code_backend == "docker":
        image_id = subprocess.check_output(["docker", "image", "inspect", "gbv-code-eval:py311",
                                           "--format", "{{.Id}}"], text=True).strip()
        subprocess.run(["docker", "run", "--rm", "--network=none", "gbv-code-eval:py311",
                        "python", "-c", "import numpy, sympy; print('code evaluator ready')"], check=True)
    versions = {name: importlib.metadata.version(name) for name in
                ("torch", "transformers", "huggingface-hub", "datasets", "numpy", "math-verify")}
    evaluator_checks = []
    if "livecodebench" in cfg["datasets"]:
        from .lcb import evaluate_lcb
        # The chosen production backend must execute both LCB calling conventions.
        for functional in (False, True):
            evaluation = {"fn_name": "add" if functional else None, "tests": [
                {"input": "1\n2" if functional else "1 2\n", "output": "3"},
                {"input": "7\n9" if functional else "7 9\n", "output": "16"}]}
            for correct in (False, True):
                expression = "a+b" if correct else "3"
                code = ("class Solution:\n    def add(self,a,b): return " + expression if functional else
                        "import sys\na,b=map(int,sys.stdin.buffer.read().split())\nprint(" + expression + ")")
                result = evaluate_lcb(code, evaluation, backend=code_backend, timeout=2)
                if result["passed"] is not correct:
                    raise RuntimeError(f"LiveCodeBench evaluator preflight failed: {result}")
                evaluator_checks.append({"functional": functional, "expected_pass": correct, "result": result})
    return {"versions": versions, "python": platform.python_version(), "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device), "total_memory_bytes": torch.cuda.get_device_properties(device).total_memory,
            "code_backend": code_backend, "code_image_id": image_id, "lcb_evaluator_checks": evaluator_checks}


def check_model(cfg, output: Path, device="cuda:0", code_backend="docker", only_variants=None):
    import torch
    from .engine import load_models
    from .runner import model_identity
    from .tree import sampled_tree, compact_cache

    environment = check_environment(cfg, code_backend, device)
    model = model_identity(cfg["model"])
    engine, tokenizer = load_models(model, device)
    prompts = ["Compute 19 + 23. Give a brief explanation.",
               "Write a Python function that reverses a list.",
               "Explain why the sum of two even integers is even."]
    from .config import select_variants
    variants = [Variant(**e["variant"]) for e in select_variants(build_variants(cfg), only_variants)]
    # Keep every structural variation; set T=0 solely for the equality gate.
    unique = {}
    for v in variants:
        v = replace(v, temperature=0., draft_temperature=1.)
        unique[tuple((k, value) for k, value in v.to_dict().items() if k != "name")] = v
    checks = []
    with torch.inference_mode():
        for prompt in prompts:
            ids = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                    tokenize=True, add_generation_prompt=True, enable_thinking=bool(model.get("enable_thinking", False)), return_tensors="pt").to(device)
            reference = engine.generate(ids, Variant(name="ar", method="target", paths=1, temperature=0), 64, [], seed=105)
            for v in unique.values():
                result = engine.generate(ids, v, 64, [], seed=105)
                checks.append({"prompt_sha256": digest(prompt), "variant": v.to_dict(),
                               "greedy_equal": result["generated_token_ids"] == reference["generated_token_ids"]})
            # Verify actual checkpoint logits and KV compaction on a branched tree.
            future = reference["generated_token_ids"][:4]
            paths = torch.tensor([future[1:4], [future[1], future[3], future[2]]], device=device)
            tree = sampled_tree(paths)
            prefix_length = ids.shape[1]
            cache = engine.cache_factory()
            engine.target_forward(ids, cache, hidden=False)
            tree_out = engine.target_forward(torch.tensor([[future[0]] + tree.tokens], device=device), cache,
                hidden=False, positions=(torch.tensor(tree.depths, device=device) + prefix_length)[None],
                mask=tree.mask(prefix_length, next(engine.target.parameters()).dtype, device))
            for path, nodes in zip(paths, tree.path_nodes):
                sequence = torch.cat([ids, torch.tensor([[future[0]]], device=device), path[None]], 1)
                sequential = engine.target(sequence).logits[:, prefix_length:]
                actual = tree_out.logits[:, [0] + nodes]
                equal = bool((actual.argmax(-1) == sequential.argmax(-1)).all())
                checks.append({"check": "tree_argmax", "passed": equal,
                               "max_absolute_logit_error": float((actual - sequential).abs().max())})
            keep = [0] + tree.path_nodes[1][:2]
            compact_cache(cache, prefix_length, keep, device)
            token = future[-1]
            actual = engine.target_forward(torch.tensor([[token]], device=device), cache, hidden=False).logits[:, -1]
            sequence = torch.cat([ids, torch.tensor([[future[0], int(paths[1, 0]), int(paths[1, 1]), token]], device=device)], 1)
            sequential = engine.target(sequence).logits[:, -1]
            checks.append({"check": "compacted_cache_argmax", "passed": bool((actual.argmax(-1) == sequential.argmax(-1)).all()),
                           "max_absolute_logit_error": float((actual - sequential).abs().max())})
    passed = all(c.get("greedy_equal", c.get("passed", False)) for c in checks)
    result = {"passed": passed, "scope": "checkpoint smoke correctness, not full benchmark results",
              "environment": environment, "model": model, "source_hashes": source_hashes(), "checks": checks}
    write_json(output, result)
    if not passed:
        raise RuntimeError(f"GPU correctness gate failed; inspect {output}. Do not launch formal timing runs.")
    return result
