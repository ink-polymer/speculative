"""Dedicated Qwen3-8B entry; the shared official runner implements every method."""
from __future__ import annotations

import sys

from .common import ROOT
from .official import main as official_main


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        args = ["plan"]
    if any(arg.split("=", 1)[0] == "--model-index" for arg in args):
        raise ValueError("Qwen3-8B entry fixes --model-index=1; use the general entry for other models")
    official_main([args[0], "--model-index", "1", "--run-dir",
                   str(ROOT / "outputs/adaptive_ddtree_official_t0_qwen3_8b"), *args[1:]])


if __name__ == "__main__":
    main()
