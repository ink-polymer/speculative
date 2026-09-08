"""Synthetic verifier-only screen; NOT model inference or a speedup claim.

PYTHONPATH=src python scripts/benchmark_diffusion_scaffold.py --device cpu
CUDA is optional. The same script records synchronization and backend identity.
"""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import platform
import random
import statistics
import time

import torch

from gbv_experiments import diffusion_tree_bv as diffusion, sampling
from gbv_experiments.common import ROOT, file_hash, write_json
from gbv_experiments.tree import probability_tree, sampled_tree


@torch.inference_mode()
def run(device, vocab, iterations):
    torch.set_num_threads(1)
    if vocab < 9 or iterations < 1:
        raise ValueError("Need vocabulary >=9 and positive iterations")
    device = torch.device(device)
    sync = (lambda: torch.cuda.synchronize(device)) if device.type == "cuda" else lambda: None
    result = {"kind": "synthetic_verifier_microbenchmark", "formal_complete": False,
              "model_forward_executed": False, "device": str(device), "torch": torch.__version__,
              "python": platform.python_version(), "measured_at_utc": datetime.now(timezone.utc).isoformat(),
              "source_sha256": {str(path.relative_to(ROOT)): file_hash(path) for path in
                                [ROOT / "scripts/benchmark_diffusion_scaffold.py",
                                 *sorted((ROOT / "src/gbv_experiments").glob("*.py"))]},
              "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
              "vocab": vocab, "length": 15, "budget": 45, "iterations_per_seed": iterations,
              "excludes": ["model forwards", "draft logits normalization", "proposal sampling",
                           "tree construction", "KV cache operations"],
              "acceptance_scope": "conditional on fixed latent snapshots, not an acceptance/quality estimate", "rows": []}
    for case, draft_head, target_head in (
        ("matched_diffuse", [1 / 8] * 8, [1 / 8] * 8),
        ("sharp_target_mismatch", [.51, .49], [.99, .01]),
        ("matched_sharp", [.995, .005], [.995, .005]),
    ):
        def row(head):
            values = torch.full((vocab,), 1e-6 / (vocab - len(head)), dtype=torch.float64, device=device)
            values[:len(head)] = values.new_tensor(head) * (1 - 1e-6)
            return values

        q, p = row(draft_head), row(target_head)
        logits = q.log()[None].expand(15, -1).contiguous()
        law = diffusion.DiffusionBlockLaw.from_logits(logits, torch.tensor([0] + [vocab - 1] * 15, device=device),
                                                     1., vocab - 1)
        greedy = logits.argmax(-1)
        for seed in (17, 29, 43):
            old = diffusion.propose(law, 3, torch.Generator(device=device).manual_seed(seed))
            new = diffusion.propose(law, 1, torch.Generator(device=device).manual_seed(seed))
            trees = {"old_k3": sampled_tree(old.paths()),
                     "scaffold_k1": diffusion.scaffold_tree(new, greedy, 45),
                     "ddtree": probability_tree(q[None].expand(15, -1), 45),
                     "dflash": sampled_tree(greedy[None])}
            trees["scaffold_ancestral"] = trees["scaffold_k1"]
            target_logits = {name: p.log()[None].expand(len(tree.parents), -1).contiguous()
                             for name, tree in trees.items()}
            generators = {name: torch.Generator(device=device).manual_seed(seed) for name in trees}

            def call(name):
                tree, raw, generator = trees[name], target_logits[name], generators[name]
                if name == "old_k3":
                    return diffusion.verify_logits(raw, tree, old, 1., generator, validate=False)
                if name in {"scaffold_k1", "scaffold_ancestral"}:
                    return diffusion.verify_scaffold_logits(
                        raw, tree, new, 1., generator, validate=False,
                        continuation="ancestral" if name == "scaffold_ancestral" else "terminal")
                probabilities = sampling.probabilities(raw, 1.)
                if name == "ddtree":
                    return sampling.tree_verify_ancestral_batched(tree.parents, tree.tokens, probabilities,
                                                                  generator, validate=False)
                accepted, bonus = sampling.matching_verify(greedy, probabilities, generator)
                return tree.path_nodes[0][:accepted], tree.tokens[:accepted], bonus

            for name in trees:
                for _ in range(3):
                    call(name)
            timings, counts = {name: [] for name in trees}, {name: [] for name in trees}
            for iteration in range(iterations):
                order = list(trees)
                random.Random(seed * 1000 + iteration).shuffle(order)
                for name in order:
                    sync()
                    start = time.perf_counter()
                    _, accepted, _ = call(name)
                    sync()
                    timings[name].append((time.perf_counter() - start) * 1000)
                    counts[name].append(len(accepted))
            for name, tree in trees.items():
                result["rows"].append({"case": case, "method": name, "seed": seed,
                                       "nodes": len(tree.tokens), "median_verifier_ms": statistics.median(timings[name]),
                                       "mean_accepted_toy_samples": statistics.mean(counts[name])})
    result["summary"] = []
    for case in sorted({r["case"] for r in result["rows"]}):
        for method in ("old_k3", "scaffold_k1", "scaffold_ancestral", "ddtree", "dflash"):
            rows = [r for r in result["rows"] if r["case"] == case and r["method"] == method]
            result["summary"].append({"case": case, "method": method,
                                      "median_of_seed_medians_ms": statistics.median(r["median_verifier_ms"] for r in rows)})
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--vocab", type=int, default=151936)
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--output", type=Path, default=Path(".artifacts/diffusion-scaffold-microbench.json"))
    args = parser.parse_args()
    measured = run(args.device, args.vocab, args.iterations)
    write_json(args.output, measured)
    for row in measured["summary"]:
        print(row, flush=True)
