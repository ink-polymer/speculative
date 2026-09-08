"""Same-flow fusion ablation on saved states and held-out development prompts.

DEFERRED by user direction on 2026-09-07: do not submit as the research path.
This is an implementation probe, not a new architecture or formal benchmark.
It has not run on CUDA. The eager implementation stays the production default.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import random
import time
from unittest.mock import patch

from . import protected_tree_bv as protected
from .common import ROOT, file_hash, write_json
from .root_marginalized_bv import _normalize
from .runner import output_lock, stop_token_ids
from .terminal_formal import (DIAGNOSTIC_PROMPTS, allocation_gate, load_frozen_engine,
                              runtime, telemetry)
from .terminal_protocol import load_study, study_sources, variants


def run(capture_run, output):
    import torch
    from .conversation import encode_messages
    device = "cuda:0"
    study = load_study(ROOT / "configs/protected_tree_formal.json")
    study["spec"]["gpu_family"] = "GH200"
    with output_lock(output):
        if (output / "manifest.json").exists():
            raise ValueError("Choose a fresh probe output")
        env = runtime(study, device)
        allocation_gate(device)
        write_json(output / "manifest.json", {"runtime": env, "sources": study_sources(),
            "formal_complete": False, "kind": "same_flow_implementation_ablation",
            "compile": {"fullgraph": True, "dynamic": False, "mode": "default"},
            "capture_run": str(capture_run), "capture_index_sha256": file_hash(capture_run / "capture_index.json")})
        eager = protected._prepare
        compiled = torch.compile(eager, fullgraph=True, dynamic=False)
        backends = {"protected_eager": eager, "protected_compiled": compiled}
        index = json.loads((capture_run / "capture_index.json").read_text())
        checks, replay = [], []
        for entry in index:
            if entry["kind"] != "shared_suffix":
                continue
            path = capture_run / entry["file"]
            if file_hash(path) != entry["sha256"]:
                raise ValueError("Captured state hash mismatch")
            state = torch.load(path, map_location=device, weights_only=True)
            p, paths, q = state["all_p"], state["paths"], state["q"]
            args = (paths[:, 0], p[0], paths[0, 1:], p[1:].view(*paths.shape, -1), q[1:])
            flow_args = (_normalize(p[0, paths[:, 0]]), *args[2:])
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            expected, actual = eager(*flow_args), compiled(*flow_args)
            torch.cuda.synchronize(device)
            first_call_ms = (time.perf_counter() - start) * 1000
            errors = []
            for x, y in zip(expected, actual):
                torch.testing.assert_close(x, y, rtol=1e-10, atol=1e-12)
                errors.append(float((x - y).abs().max()))
            for seed in range(50):
                results = []
                for implementation in backends.values():
                    with patch.object(protected, "_prepare", implementation):
                        results.append(protected.verify(*args, torch.Generator(device=device).manual_seed(seed)))
                if results[0] != results[1]:
                    raise AssertionError("Eager/compiled sample disagreement; retain evidence before deployment")
            checks.append({"capture": entry, "max_absolute_error": max(errors),
                           "same_seed_samples": 50, "first_call_including_compile_ms": first_call_ms})
            print("compiled parity passed", checks[-1], flush=True)
            for repeat in range(3):
                order = list(backends)
                random.Random(repeat).shuffle(order)
                for name in order:
                    g = torch.Generator(device=device).manual_seed(17)
                    with patch.object(protected, "_prepare", backends[name]):
                        for _ in range(5):
                            protected.verify(*args, g, validate=False)
                        allocation_gate(device)
                        torch.cuda.synchronize(device)
                        start = time.perf_counter()
                        for _ in range(50):
                            protected.verify(*args, g, validate=False)
                        torch.cuda.synchronize(device)
                        replay.append({"capture": entry["file"], "variant": name, "repeat": repeat,
                                       "ms_per_call": (time.perf_counter() - start) * 20})
            write_json(output / "replay.json", {"checks": checks, "rows": replay, "primary_timing": False})
            del state, p, paths, q, args, flow_args, expected, actual, x, y
        if len(checks) != 3:
            raise ValueError("Expected all three captured development states")
        engine, tokenizer = load_frozen_engine(study, device)
        stop_ids = stop_token_ids(engine, tokenizer)
        selected = {v.name: v for v in variants(study) if v.name in {"ddtree", "dflash_match", "protected_full"}}
        candidate = selected.pop("protected_full")
        selected.update({name: replace(candidate, name=name) for name in backends})
        rows = []
        for prompt_index, prompt in enumerate(DIAGNOSTIC_PROMPTS):
            ids = encode_messages(tokenizer, [{"role": "user", "content": prompt}], study["config"]["model"], device)
            for name, variant in selected.items():
                with patch.object(protected, "_prepare", backends.get(name, eager)):
                    engine.generate(ids, variant, 32, [], seed=0)
            for repeat in range(3):
                order = list(selected)
                random.Random(prompt_index * 10 + repeat).shuffle(order)
                allocation_gate(device)
                before = telemetry(device)
                for name in order:
                    with patch.object(protected, "_prepare", backends.get(name, eager)):
                        result = engine.generate(ids, selected[name], 96, stop_ids, seed=105 + prompt_index)
                    rows.append({"prompt": prompt_index, "repeat": repeat, "variant": name, "before": before, **result})
                    write_json(output / "timings.json", {"primary_timing": False, "rows": rows})
                    print(f"fusion pilot prompt={prompt_index} repeat={repeat} method={name}", flush=True)
                allocation_gate(device)
        metrics = {}
        for name in selected:
            rs = [r for r in rows if r["variant"] == name]
            metrics[name] = sum(r["decode_ms"] for r in rs) / sum(r["decode_tokens"] for r in rs)
        eager_tokens = {(r["prompt"], r["repeat"]): r["generated_token_ids"] for r in rows if r["variant"] == "protected_eager"}
        same_tokens = all(r["generated_token_ids"] == eager_tokens[r["prompt"], r["repeat"]] for r in rows if r["variant"] == "protected_compiled")
        write_json(output / "report.json", {"probe_complete": True, "formal_complete": False,
            "runtime": env, "metrics_decode_ms_per_token": metrics, "same_candidate_tokens": same_tokens,
            "limitations": ["Three diagnostic prompts, not evaluation datasets",
                            "Implementation fusion only; no architecture novelty claim",
                            "Not a strict-lossless or complete formal experiment"]})
        print("fusion probe complete", metrics, "same_tokens", same_tokens, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.capture_run.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
