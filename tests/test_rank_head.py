"""rank 特征、bucket 与 valid-prefix 的 CPU 数学测试。"""

from pathlib import Path

import torch

from dflash_specblock.rank_head import (
    DFlashRankHead,
    distribution_summary,
    load_rank_head,
    target_rank_buckets,
    valid_prefix_mask,
)


def test_distribution_summary_has_paper_dimension() -> None:
    logits = torch.randn(2, 4, 32)
    summary = distribution_summary(logits)
    assert summary.shape == (2, 4, 15)
    assert torch.isfinite(summary).all()


def test_target_rank_bucket_boundaries() -> None:
    logits = torch.arange(20, dtype=torch.float32).repeat(4, 1)
    targets = torch.tensor([19, 17, 12, 0])  # ranks 1, 3, 8, 20
    assert target_rank_buckets(logits, targets).tolist() == [0, 1, 2, 3]


def test_valid_prefix_stops_after_first_error() -> None:
    logits = torch.zeros(1, 4, 8)
    logits[0, 0, 1] = 10
    logits[0, 1, 7] = 10  # 第二位错误
    logits[0, 2, 3] = 10
    logits[0, 3, 4] = 10
    targets = torch.tensor([[1, 2, 3, 4]])
    assert valid_prefix_mask(logits, targets).tolist() == [[True, True, False, False]]


def test_rank_head_output_shape() -> None:
    head = DFlashRankHead(hidden_size=16, projection_size=8)
    output = head(torch.randn(2, 4, 16), torch.randn(2, 4, 32))
    assert output.shape == (2, 4, 4)
    assert head.classifier[0].in_features == 31
    assert head.classifier[0].bias is None
    assert head.classifier[2].bias is None


def test_rank_head_can_train_from_inference_mode_dflash_outputs() -> None:
    head = DFlashRankHead(hidden_size=16, projection_size=8)
    with torch.inference_mode():
        hidden = torch.randn(2, 4, 16)
        logits = torch.randn(2, 4, 32)
    loss = head(hidden, logits).sum()
    loss.backward()
    assert head.classifier[0].weight.grad is not None


def test_masked_rank_loss_works_with_inference_mode_labels() -> None:
    """回归：inference_mode 产生的 mask/label 必须先 clone 才能索引 autograd 张量。

    训练脚本在 inference_mode 里算draft 分布、bucket 与 valid-prefix mask，随后在
    autograd 下计算 rank loss。若直接用这些 inference tensor 做布尔索引，PyTorch 会抛
    "Inference tensors cannot be saved for backward"，导致训练一步都跑不起来。
    """
    import torch.nn.functional as functional

    head = DFlashRankHead(hidden_size=16, projection_size=8)
    with torch.inference_mode():
        hidden = torch.randn(1, 4, 16)
        logits = torch.randn(1, 4, 32)
        targets = torch.tensor([[3, 5, 7, 9]])
        buckets = target_rank_buckets(logits, targets)
        valid = valid_prefix_mask(logits, targets)

    buckets = buckets.clone()
    valid = valid.clone()
    class_logits = head(hidden, logits)
    loss = functional.cross_entropy(class_logits[valid], buckets[valid])
    loss.backward()
    assert head.classifier[0].weight.grad is not None


def test_rank_checkpoint_architecture_is_strictly_validated() -> None:
    head = DFlashRankHead(hidden_size=16, projection_size=8)
    checkpoint = Path(__file__).with_name(".rank_head_test.pt")
    try:
        torch.save(
            {
                "state_dict": head.state_dict(),
                "metadata": {
                    "hidden_size": 16,
                    "projection_size": 8,
                    "architecture": "specblock_h15_mlp_v1",
                    "updates": 1,
                },
            },
            checkpoint,
        )
        loaded = load_rank_head(
            checkpoint,
            hidden_size=16,
            device=torch.device("cpu"),
        )
        assert loaded.classifier[0].in_features == 31
    finally:
        checkpoint.unlink(missing_ok=True)


def test_rank_checkpoint_rejects_different_block_size() -> None:
    head = DFlashRankHead(hidden_size=16, projection_size=8)
    checkpoint = Path(__file__).with_name(".rank_head_mismatch_test.pt")
    try:
        torch.save(
            {
                "state_dict": head.state_dict(),
                "metadata": {
                    "hidden_size": 16,
                    "projection_size": 8,
                    "architecture": "specblock_h15_mlp_v1",
                    "updates": 1,
                    "block_size": 8,
                },
            },
            checkpoint,
        )
        try:
            load_rank_head(
                checkpoint,
                hidden_size=16,
                device=torch.device("cpu"),
                expected_metadata={"block_size": 4},
            )
        except ValueError as error:
            assert "不一致" in str(error)
        else:
            raise AssertionError("不同 K 的 rank checkpoint 被静默加载")
    finally:
        checkpoint.unlink(missing_ok=True)
