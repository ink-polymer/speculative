#!/usr/bin/env python3
"""Regenerate or verify the AdaptiveTree distribution SHA256 manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PACKAGE_ROOT.parent
MANIFEST = PACKAGE_ROOT / "SOURCE_SHA256.json"


def distribution_files() -> list[Path]:
    output = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard",
         "--", PACKAGE_ROOT.name],
        cwd=REPOSITORY_ROOT,
        text=True,
    )
    paths = [REPOSITORY_ROOT / value for value in output.splitlines() if value]
    return sorted(path for path in paths if path.is_file() and path != MANIFEST)


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def expected_manifest() -> dict[str, str]:
    return {
        str(path.relative_to(PACKAGE_ROOT)): sha256(path)
        for path in distribution_files()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    expected = expected_manifest()
    if args.check:
        actual = json.loads(MANIFEST.read_text(encoding="utf-8"))
        if actual != expected:
            raise SystemExit("SOURCE_SHA256.json is stale; regenerate it")
        return
    temporary = MANIFEST.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(expected, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(MANIFEST)


if __name__ == "__main__":
    main()
