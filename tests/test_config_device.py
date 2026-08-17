"""配置验证与 CPU 回退测试。"""

from pathlib import Path

import torch

from dflash_specblock.config import ExperimentConfig
from dflash_specblock.device import resolve_device


def test_cpu_is_explicitly_supported_for_unit_tests() -> None:
    assert resolve_device("cpu") == torch.device("cpu")


def test_config_rejects_invalid_branch_table() -> None:
    config = ExperimentConfig(branch_factors=(1, 2, 3))
    try:
        config.validate()
    except ValueError as error:
        assert "branch_factors" in str(error)
    else:
        raise AssertionError("invalid branch table was accepted")


def test_config_rejects_non_npu_prefix_devices() -> None:
    config = ExperimentConfig(device="npu-not-a-device")
    try:
        config.validate()
    except ValueError as error:
        assert "device" in str(error)
    else:
        raise AssertionError("invalid NPU device string was accepted")

    try:
        resolve_device("npu-not-a-device")
    except ValueError as error:
        assert "npu" in str(error).lower()
    else:
        raise AssertionError("device resolver accepted an invalid NPU device string")


def test_formal_a2_config_is_fail_closed_and_uses_paper_defaults() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config = ExperimentConfig.from_json(project_root / "configs" / "qwen3_4b_a2.json")
    assert config.device == "npu:0"
    assert config.rank_mode == "learned"
    assert config.rank_checkpoint == "checkpoints/rank_head_tree15.pt"
    assert config.block_size == 15
    assert config.max_blocks == 1
    assert config.beam_width == 4
    assert config.branch_factors == (2, 4, 10, 0)
    assert config.target_model_id == "Qwen/Qwen3-4B"
    assert config.draft_model_id == "z-lab/Qwen3-4B-DFlash-b16"
    assert config.target_revision == "1cfa9a7208912126459214e8b04321603b3df60c"
    assert config.draft_revision == "b74e3a329c4d963783143b1e970d95b002be72bd"
