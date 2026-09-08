"""Isolated CUDA diagnostic; never labels development prompts as formal results."""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import random
import time

from .common import ROOT, digest, file_hash, write_json
from .runner import output_lock, stop_token_ids
from .terminal_formal import (DIAGNOSTIC_PROMPTS, allocation_gate, capture_states,
                              load_frozen_engine, replay_functions, runtime, telemetry)
from .terminal_protocol import load_study, paired_order, primary_candidate, study_sources, variants


def numerical_probe(device, *, include_atoms=False):
    """Check CUDA flow tensors against CPU FP64, not stochastic seed equality."""
    import torch
    from . import protected_tree_bv as protected
    errors = []
    g = torch.Generator().manual_seed(20260907)
    for m in (0, 1, 14, 128):
        p = torch.randn(3, m + 1, 17, generator=g, dtype=torch.float64).softmax(-1)
        q = torch.randn(m, 17, generator=g, dtype=torch.float64).softmax(-1)
        path = torch.zeros(m, dtype=torch.long)
        alpha = torch.tensor([.2, .3, .5], dtype=torch.float64)
        for zeros in (False, True):
            if zeros and m:
                p[0, m // 2, 0] = 0
                p /= p.sum(-1, keepdim=True)
            floor, score = protected._scores(alpha, path, p, q)
            residual, capacity = protected._residuals(alpha, floor, score, p, q)
            ga, gp, gq = alpha.to(device), p.to(device), q.to(device)
            gf, gs = protected._scores(ga, path.to(device), gp, gq)
            gr, gc = protected._residuals(ga, gf, gs, gp, gq)
            for reference, actual in zip((floor, score, residual, capacity), (gf, gs, gr, gc)):
                torch.testing.assert_close(actual.cpu(), reference, rtol=1e-10, atol=1e-12)
                errors.append(float((actual.cpu() - reference).abs().max()))
            if not bool(torch.isfinite(gr).all() & (gr >= 0).all()
                        & (gs >= gf - 1e-12).all() & (gs.sum(0) <= 1 + 1e-12).all()):
                raise AssertionError("CUDA protected flow invariant failed")
    if include_atoms:
        from . import atom_tree_bv as atom
        for m in (0, 1, 14, 128):
            q = torch.randn(m + 1, 17, generator=g, dtype=torch.float64).softmax(-1)
            proposal = atom.propose(q, 3, g)
            gpu_proposal = atom.AtomProposal(**{key: value.to(device) for key, value in vars(proposal).items()})
            p = torch.randn(3, m, 17, generator=g, dtype=torch.float64).softmax(-1)
            support = p.gather(-1, proposal.tokens)
            alpha = torch.tensor([.2, .3, .5], dtype=torch.float64)
            for pool in (False, True):
                reference = atom.plan(alpha, support, proposal, pool=pool)
                actual = atom.plan(alpha.to(device), support.to(device), gpu_proposal, pool=pool)
                for key, value in vars(reference).items():
                    gpu_value = getattr(actual, key).cpu()
                    torch.testing.assert_close(gpu_value, value, rtol=1e-10, atol=1e-12)
                    if value.numel():
                        errors.append(float((gpu_value - value).abs().max()))
                if not bool((actual.scores >= actual.floors - 1e-12).all()
                            & (actual.scores.sum(0) <= 1 + 1e-12).all()
                            & (actual.flow.sum(0) <= gpu_proposal.source + 1e-12).all()):
                    raise AssertionError("CUDA atom-flow invariant failed")
    return {"passed": True, "max_absolute_error": max(errors), "includes_atom_flow": include_atoms,
            "scope": "CPU/CUDA FP64 recurrence and residual parity on synthetic states; not a proof of unbiased CUDA RNG"}


def run(output, device, gpu_family, tokens, study_path=None):
    import gc
    import torch
    from .conversation import encode_messages
    from .preflight import check_model

    study = load_study(study_path or ROOT / "configs/protected_tree_formal.json")
    candidate = primary_candidate(study)
    # This is a separate pilot, never a relabeling of an H200 registration.
    study["spec"]["gpu_family"] = gpu_family
    with output_lock(output):
        if (output / "pilot_manifest.json").exists():
            raise ValueError("Pilot output already exists; choose a new directory")
        env = runtime(study, device)
        allocation_gate(device)
        manifest = {"kind": "development_pilot", "formal_complete": False,
                    "gpu_executed": True, "runtime": env, "study": study,
                    "sources": study_sources(), "prompt_sha256": list(map(digest, DIAGNOSTIC_PROMPTS)),
                    "tokens": tokens, "seeds": [105, 106, 107], "timing_repeats": 3,
                    "note": "Three hand-written diagnostic prompts; no evaluation dataset inspected or tuned on"}
        write_json(output / "pilot_manifest.json", manifest)
        if candidate == "diffusion_full":
            from .diffusion_preflight import numerical_probe as diffusion_numerical_probe
            numerics = diffusion_numerical_probe(device)
        else:
            numerics = numerical_probe(device, include_atoms=candidate == "atom_full")
        write_json(output / "numerical_probe.json", numerics)
        print("CUDA numerical probe passed; loading frozen checkpoints", flush=True)
        engine, tokenizer = load_frozen_engine(study, device)
        stops = stop_token_ids(engine, tokenizer)
        by_name = {v.name: v for v in variants(study)}
        rows, profiles, replay, captures, greedy = [], [], [], [], []
        for index, prompt in enumerate(DIAGNOSTIC_PROMPTS):
            ids = encode_messages(tokenizer, [{"role": "user", "content": prompt}], study["config"]["model"], device)
            states = capture_states(study, engine, ids, 105 + index, limit=1, tokens=32)
            for state_index, state in enumerate(states):
                relative = f"captures/p{index}-s{state_index}.pt"
                (output / "captures").mkdir(exist_ok=True)
                torch.save(state, output / relative)
                captures.append({"file": relative, "sha256": file_hash(output / relative), "kind": state["kind"]})
                gs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in state.items()}
                functions = {k: v for k, v in replay_functions(gs).items() if k in by_name}
                for function in functions.values():
                    g = torch.Generator(device=device).manual_seed(1)
                    for _ in range(5):
                        function(g)
                for repeat in range(3):
                    order = list(functions)
                    random.Random(index * 100 + repeat).shuffle(order)
                    for name in order:
                        g = torch.Generator(device=device).manual_seed(105 + repeat)
                        torch.cuda.synchronize(device)
                        started = time.perf_counter()
                        for _ in range(20):
                            functions[name](g)
                        torch.cuda.synchronize(device)
                        replay.append({"capture": relative, "kind": state["kind"], "variant": name,
                                       "repeat": repeat, "iterations": 20,
                                       "ms_per_call": (time.perf_counter() - started) * 50})
                del functions, function, gs
            for v in by_name.values():
                engine.generate(ids, v, 32, [], seed=0)
            # Greedy is an integration probe, not a substitute for the T=1 law tests.
            if candidate != "diffusion_full":
                reference = engine.generate(ids, replace(by_name["target_t1"], temperature=0.), 64, stops, seed=105)
                for name in ("ddtree", "dflash_match", candidate):
                    result = engine.generate(ids, replace(by_name[name], temperature=0.), 64, stops, seed=105)
                    greedy.append({"prompt": index, "variant": name,
                                   "equal": result["generated_token_ids"] == reference["generated_token_ids"],
                                   "reference": reference["generated_token_ids"], "actual": result["generated_token_ids"]})
            for repeat in range(3):
                allocation_gate(device)
                before = telemetry(device)
                for name in paired_order(105 + index, repeat, study):
                    result = engine.generate(ids, by_name[name], tokens, stops, seed=105 + index)
                    if any(r["tree_nodes"] > 45 for r in result["rounds"]):
                        raise RuntimeError("Registered tree budget exceeded")
                    rows.append({"prompt": index, "variant": name, "repeat": repeat, "before": before, **result})
                    write_json(output / "timings.json", {"primary_timing": False, "rows": rows})
                    print(f"pilot prompt={index + 1}/3 repeat={repeat} method={name} tokens={result['generated_tokens']}", flush=True)
                allocation_gate(device)
            if index == 0:
                for v in by_name.values():
                    profiles.append({"variant": v.name, **engine.generate(ids, v, tokens, stops, seed=105, profile=True)})
            write_json(output / "capture_index.json", captures)
            write_json(output / "replay.json", {"primary_timing": False, "rows": replay})
            write_json(output / "profiles.json", {"primary_timing": False, "rows": profiles})
            if candidate != "diffusion_full":
                write_json(output / "greedy.json", {"passed": all(r["equal"] for r in greedy), "rows": greedy})
            del states
        del engine, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        # Preserve detailed mismatch evidence; failure does not turn diagnostic
        # timings into publishable results or silently relax the formal gate.
        try:
            if candidate == "diffusion_full":
                from .diffusion_preflight import check_diffusion_model
                checked = check_diffusion_model(study["config"], output / "checkpoint_checks.json", device)
            else:
                checked = check_model(study["config"], output / "checkpoint_checks.json", device, "process")
        except RuntimeError as exc:
            checked = {"passed": False, "error": str(exc)}
        allocation_gate(device)
        first = {(r["prompt"], r["variant"]): r["generated_token_ids"] for r in rows if r["repeat"] == 0}
        summary = {}
        for name in by_name:
            selected = [r for r in rows if r["variant"] == name]
            summary[name] = {"decode_ms_per_token": sum(r["decode_ms"] for r in selected) / sum(r["decode_tokens"] for r in selected),
                             "e2e_ms_per_token": sum(r["e2e_ms"] for r in selected) / sum(r["generated_tokens"] for r in selected),
                             "records": len(selected)}
        result = {"pilot_complete": True, "formal_complete": False, "gpu": env["gpu"],
                  "checkpoint_gate": checked.get("passed", False),
                  "greedy_exact_gate": checked.get("greedy_exact_passed"),
                  "temperature_scope": "positive_only" if candidate == "diffusion_full" else "T1_with_greedy_diagnostic",
                  "within_method_repeat_equal": all(r["generated_token_ids"] == first[r["prompt"], r["variant"]] for r in rows),
                  "metrics": summary, "limitations": ["Development prompts only; no benchmark or novelty claim",
                      "Unbiasedness theorem assumes exact arithmetic and valid target conditionals",
                      "No confidence intervals or formal evaluation quality scores in this pilot"]}
        write_json(output / "pilot_report.json", result)
        print(result, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-family", choices=["H200", "GH200"], required=True)
    parser.add_argument("--tokens", type=int, default=96)
    parser.add_argument("--study", type=Path, default=ROOT / "configs/protected_tree_formal.json")
    args = parser.parse_args()
    if not 32 <= args.tokens <= 256:
        parser.error("Pilot token budget must be 32..256")
    run(args.output.resolve(), args.device, args.gpu_family, args.tokens, args.study)


if __name__ == "__main__":
    main()
