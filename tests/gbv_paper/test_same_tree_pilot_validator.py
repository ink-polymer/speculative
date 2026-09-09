from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil

import pytest

from gbv_experiments.common import ROOT


VALIDATOR_PATH = ROOT / "scripts/validate_same_tree_pilot.py"
SPEC = importlib.util.spec_from_file_location("same_tree_pilot_validator", VALIDATOR_PATH)
assert SPEC is not None and SPEC.loader is not None
VALIDATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDATOR)
validate = VALIDATOR.validate


RESULT = (
    ROOT / "results/pilots/20260909/h20_same_tree_fast_verifier/"
    "h20-same-tree-block-official-pilot-20260910-r2"
)
ENGINE = (
    ROOT / "results/pilots/20260909/h20_same_tree_fast_verifier/"
    "reproduction/r2_source/engine.py"
)


def copy_result(tmp_path: Path) -> Path:
    result = tmp_path / "result"
    result.mkdir()
    for name in ("manifest.json", "report.json", "rows.json"):
        shutil.copyfile(RESULT / name, result / name)
    return result


def test_archived_r2_is_source_bound_and_independently_recalculated():
    evidence = validate(RESULT, ROOT, ENGINE)
    assert evidence["integrity"] == "PASS"
    assert evidence["records"] == 504
    assert evidence["strict_speed_gate"]["passed"] is True


@pytest.mark.parametrize("tamper", ["source", "mapping", "sampling_seed", "order"])
def test_validator_rejects_experiment_identity_tampering(tmp_path, tamper):
    result = copy_result(tmp_path)
    if tamper == "mapping":
        path = result / "manifest.json"
        document = json.loads(path.read_text())
        document["variants"][0]["method"] = "ddtree"
    else:
        path = result / "rows.json"
        document = json.loads(path.read_text())
        if tamper == "source":
            document["rows"][0]["source_id"] = "forged-source"
        elif tamper == "sampling_seed":
            document["rows"][0]["sampling_seed"] += 1
        else:
            document["rows"][0]["execution_position"] = 2
    path.write_text(json.dumps(document))
    with pytest.raises(AssertionError):
        validate(result, ROOT, ENGINE)
