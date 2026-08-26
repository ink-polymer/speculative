"""CUDA 配置验证与 CPU 单元测试回退。"""

from pathlib import Path

import torch

from dflash_specblock.config import ExperimentConfig
from dflash_specblock.device import configure_cuda_runtime, resolve_device


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


def test_config_rejects_invalid_cuda_device() -> None:
    config = ExperimentConfig(device="cuda-not-a-device")
    try:
        config.validate()
    except ValueError as error:
        assert "device" in str(error)
    else:
        raise AssertionError("invalid CUDA device string was accepted")

    try:
        resolve_device("cuda-not-a-device")
    except ValueError as error:
        assert "cuda" in str(error).lower()
    else:
        raise AssertionError("device resolver accepted an invalid CUDA device string")


def test_cuda_device_resolution_is_explicit(monkeypatch) -> None:
    selected: list[torch.device] = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "set_device", selected.append)

    assert resolve_device("cuda") == torch.device("cuda:0")
    assert resolve_device("cuda:1") == torch.device("cuda:1")
    assert selected == [torch.device("cuda:0"), torch.device("cuda:1")]


def test_cuda_runtime_configures_tf32(monkeypatch) -> None:
    precisions: list[str] = []
    monkeypatch.setattr(torch, "set_float32_matmul_precision", precisions.append)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)

    configure_cuda_runtime(torch.device("cuda:0"), allow_tf32=True)

    assert precisions == ["high"]
    assert torch.backends.cuda.matmul.allow_tf32 is True
    assert torch.backends.cudnn.allow_tf32 is True


def test_target_flash_attention_and_nested_cuda_graphs_are_rejected() -> None:
    invalid_attention = ExperimentConfig(attn_implementation="flash_attention_2")
    try:
        invalid_attention.validate()
    except ValueError as error:
        assert "tree mask" in str(error)
    else:
        raise AssertionError("target FlashAttention was accepted with a raw 4D tree mask")

    nested_graphs = ExperimentConfig(
        use_cuda_graphs=True,
        torch_compile_mode="reduce-overhead",
    )
    try:
        nested_graphs.validate()
    except ValueError as error:
        assert "CUDA Graph" in str(error)
    else:
        raise AssertionError("nested CUDA Graph execution was accepted")


def test_formal_cuda_config_is_fail_closed_and_uses_paper_defaults() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config = ExperimentConfig.from_json(project_root / "configs" / "qwen3_4b_cuda.json")
    assert config.device == "cuda:0"
    assert config.attn_implementation == "sdpa"
    assert config.draft_attn_implementation is None
    assert config.allow_tf32 is True
    assert config.torch_compile_mode is None
    assert config.use_cuda_graphs is False
    assert config.cuda_graph_max_cache_len == 4096
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


def test_latency_aware_ddtree_config() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config = ExperimentConfig.from_json(
        project_root / "configs" / "qwen3_4b_cuda_ddtree_adaptive.json"
    )
    assert config.tree_mode == "ddtree_adaptive"
    assert config.tree_budget == 128
    assert config.ddtree_budget_candidates == (30, 45, 60, 80, 100, 128)
    assert config.ddtree_initial_budget == 60
    assert config.max_blocks == 1
    assert config.rank_mode == "heuristic"
    assert config.rank_checkpoint is None
    assert config.use_cuda_graphs is False


def test_latency_aware_ddtree_rejects_fixed_shape_cuda_graph() -> None:
    config = ExperimentConfig(
        tree_mode="ddtree_adaptive",
        block_size=15,
        max_blocks=1,
        tree_budget=60,
        ddtree_budget_candidates=(30, 60),
        ddtree_initial_budget=60,
        rank_mode="heuristic",
        rank_checkpoint=None,
        use_cuda_graphs=True,
    )
    try:
        config.validate()
    except ValueError as error:
        assert "CUDA Graph" in str(error)
    else:
        raise AssertionError("adaptive DDTree 错误接受了固定最大形状 CUDA Graph")
