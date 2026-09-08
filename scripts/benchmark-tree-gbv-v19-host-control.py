"""Run the strict harness with hybrid host control for finite-tree BRBV."""
from __future__ import annotations

import os
from pathlib import Path
import runpy


os.environ["GBV_STRICT_SCHEMA"] = "19"
os.environ["GBV_STRICT_TAG"] = "tree-gbv-v19-host-control"
os.environ["GBV_CANDIDATE_NAME"] = "hard_budget_brbv_host"
os.environ["GBV_CANDIDATE_METHOD"] = "tree_gbv_budgeted_prefix_recycle_host"
runpy.run_path(
    str(Path(__file__).with_name("benchmark-tree-gbv-v18-strict-budget.py")),
    run_name="__main__",
)
