"""Audit fused DDTree scan against explicit FP64 inverse CDF at full vocab."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import torch

from gbv_experiments.common import file_hash, write_json
from gbv_experiments.fused_tree_sampling import tree_verify_ancestral_fused_scan


@torch.inference_mode()
def run(output: Path, device: str, seeds: int, vocabulary: int) -> dict:
    if seeds < 1 or vocabulary < 1045:
        raise ValueError("Need positive seeds and vocabulary >= 1045")
    parents = [-1]
    tokens = []
    for depth in range(15):
        parents.append(depth)
        tokens.append(100 + depth)
    for extra in range(30):
        parents.append(0)
        tokens.append(1000 + extra)

    p = torch.full(
        (46, vocabulary), 0.2 / (vocabulary - 1),
        dtype=torch.float64, device=device,
    )
    for node in range(15):
        p[node, 100 + node] = 0.8
    p[15:].fill_(1.0 / vocabulary)
    cdf = p.cpu().cumsum(-1)
    children = {
        (parents[node], tokens[node - 1]): node
        for node in range(1, len(parents))
    }
    accepted_lengths = []
    for seed in range(seeds):
        expected_generator = torch.Generator(device=device).manual_seed(seed)
        actual_generator = torch.Generator(device=device).manual_seed(seed)
        uniforms = torch.rand(
            16, dtype=torch.float64, device=device,
            generator=expected_generator,
        ).cpu().tolist()
        expected_nodes, node, bonus = [], 0, None
        for uniform in uniforms:
            threshold = uniform * float(cdf[node, -1])
            bonus = int(torch.searchsorted(
                cdf[node], torch.tensor(threshold, dtype=torch.float64),
            ))
            child = children.get((node, bonus))
            if child is None:
                break
            expected_nodes.append(child)
            node = child
        expected = (
            expected_nodes,
            [tokens[index - 1] for index in expected_nodes],
            bonus,
        )
        actual = tree_verify_ancestral_fused_scan(
            parents, tokens, p, actual_generator, validate=False,
        )
        if actual != expected:
            raise AssertionError({
                "seed": seed, "actual": actual, "expected": expected,
            })
        accepted_lengths.append(len(actual[0]))

    report = {
        "kind": "full_vocabulary_fused_scan_inverse_cdf_audit",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": True,
        "seeds": seeds,
        "vocabulary": vocabulary,
        "tree_probability_shape": list(p.shape),
        "max_depth": 15,
        "probability_dtype": str(p.dtype),
        "gpu": torch.cuda.get_device_name(torch.device(device)),
        "accepted_length_min": min(accepted_lengths),
        "accepted_length_max": max(accepted_lengths),
        "accepted_length_mean": sum(accepted_lengths) / len(accepted_lengths),
        "source_sha256": file_hash(
            Path(__file__).resolve().parents[1]
            / "src/gbv_experiments/fused_tree_sampling.py"
        ),
        "script_sha256": file_hash(Path(__file__).resolve()),
    }
    write_json(output, report)
    print(report, flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seeds", type=int, default=1024)
    parser.add_argument("--vocabulary", type=int, default=151936)
    args = parser.parse_args()
    run(args.output.resolve(), args.device, args.seeds, args.vocabulary)


if __name__ == "__main__":
    main()
