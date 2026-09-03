#!/usr/bin/env python3
"""Build a portable code-only bundle, excluding models, data, and old results."""
from pathlib import Path
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    files = list((ROOT / "src/gbv_experiments").glob("*.py"))
    files += list((ROOT / "tests/gbv_paper").glob("*.py"))
    files += list((ROOT / "third_party/ddtree_official/model").glob("*.py"))
    files += list((ROOT / "third_party/livecodebench_official").glob("*.py"))
    files += [ROOT / p for p in (
        "configs/gbv_paper_ddtree_counts.json", "configs/gbv_paper_qwen3_8b.json", "configs/gbv_paper_suite.json",
        "configs/gbv_paper_full.json", "requirements-gbv-paper.txt",
        "scripts/gbv_paper.py", "scripts/run_gbv_paper.sh", "scripts/run_gbv_paper_full.sh", "scripts/package_gbv_paper.py",
        "docs/GBV_PAPER_EXPERIMENTS.md", "experiments/gbv_paper/Dockerfile", "experiments/gbv_paper/qwen3_8b_metadata.json",
        "third_party/ddtree_official/LICENSE", "third_party/livecodebench_official/LICENSE",
        "experiments/gbv_paper/THIRD_PARTY_NOTICES.md", "LICENSE",
    )]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
    output = ROOT / "outputs/gbv_paper_experiment_code.zip"
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files):
            archive.write(path, str(path.relative_to(ROOT)))
        archive.writestr("SOURCE_SHA256.json", json.dumps(hashes, indent=2) + "\n")
        archive.writestr("README.md", "# Three-path GBV: Qwen3-4B and Qwen3-8B\n\nRun bash scripts/run_gbv_paper.sh gbv-first to evaluate three-path GBV first on both model pairs; main and complete resume the same model-specific result directories. Default: 786 questions/conversations from seven datasets, selected once with seed 0; 866 answer turns per method and generation seed.\n\nThe reduced study has seven configurations on 4B and five on 8B, all at T=1; only 4B adds K=2 and K=4. Small GPU greedy-correctness checks remain required. See docs/GBV_PAPER_EXPERIMENTS.md for the experimental design and commands. Count alignment and matching algorithm rules do not reproduce the complete optimized official DDTree/DFlash runtime.\n\nThis is source code; it contains no formal GPU benchmark results.\n")
    print(output)


if __name__ == "__main__":
    main()
