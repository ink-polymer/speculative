"""Strict architecture-only screen of the parallel causal slot mixer."""
from __future__ import annotations

import os
from pathlib import Path
import runpy


root = Path(os.environ["FUYILE_ROOT"])
os.environ["GBV_STRICT_SCHEMA"] = "20"
os.environ["GBV_STRICT_TAG"] = "tree-gbv-v20-slot-mixer"
os.environ["GBV_CANDIDATE_NAME"] = "slot_mixer_brbv"
os.environ["GBV_CANDIDATE_METHOD"] = "tree_gbv_slot_mixer_recycle"
os.environ["GBV_PROPOSAL_ADAPTER"] = str(
    root / "checkpoints/tree_gbv_slot_mixer_v20.pt"
)
runpy.run_path(
    str(Path(__file__).with_name("benchmark-tree-gbv-v18-strict-budget.py")),
    run_name="__main__",
)
