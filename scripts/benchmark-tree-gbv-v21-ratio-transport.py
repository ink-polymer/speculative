"""Strict architecture screen of candidate ratio transport plus tree BRBV."""
from __future__ import annotations

import os
from pathlib import Path
import runpy


root = Path(os.environ["FUYILE_ROOT"])
os.environ["GBV_STRICT_SCHEMA"] = "21"
os.environ["GBV_STRICT_TAG"] = "tree-gbv-v21-ratio-transport"
os.environ["GBV_CANDIDATE_NAME"] = "ratio_transport_tree_brbv"
os.environ["GBV_CANDIDATE_METHOD"] = "tree_gbv_ratio_transport_recycle"
os.environ["GBV_PROPOSAL_ADAPTER"] = str(
    root / "checkpoints/tree_gbv_ratio_transport_v21.pt"
)
os.environ["GBV_PROPOSAL_ADAPTER_KIND"] = "ratio_transport"
runpy.run_path(
    str(Path(__file__).with_name("benchmark-tree-gbv-v18-strict-budget.py")),
    run_name="__main__",
)
