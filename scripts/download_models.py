"""下载目标模型与 DFlash draft 权重，不在仓库内保存任何访问令牌。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download


def _validate_downloaded_pair(
    target_dir: Path,
    draft_dir: Path,
    required_future_tokens: int,
) -> None:
    """下载后立即验证 target/DFlash 结构，避免到 A2 加载数 GB 权重后才发现配错。"""
    with (target_dir / "config.json").open("r", encoding="utf-8") as stream:
        target = json.load(stream)
    with (draft_dir / "config.json").open("r", encoding="utf-8") as stream:
        draft = json.load(stream)

    errors: list[str] = []
    if target.get("model_type") != "qwen3" or draft.get("model_type") != "qwen3":
        errors.append("target/draft model_type 必须都是 qwen3")
    if target.get("hidden_size") != draft.get("hidden_size"):
        errors.append("target/draft hidden_size 不一致")
    if target.get("num_hidden_layers") != draft.get("num_target_layers"):
        errors.append("draft.num_target_layers 与 target.num_hidden_layers 不一致")
    if target.get("vocab_size") != draft.get("vocab_size"):
        errors.append("target/draft vocab_size 不一致")
    if "DFlashDraftModel" not in draft.get("architectures", []):
        errors.append("draft architectures 不包含 DFlashDraftModel")
    if (draft.get("dflash_config") or {}).get("mask_token_id") is None:
        errors.append("draft.dflash_config 缺少 mask_token_id")
    required_block = required_future_tokens + 1
    if int(draft.get("block_size", 0)) < required_block:
        errors.append(
            "DFlash checkpoint block_size 小于实验所需的 "
            f"anchor+K={required_block}"
        )
    target_layer_ids = (draft.get("dflash_config") or {}).get("target_layer_ids", [])
    target_layer_count = int(target.get("num_hidden_layers", 0))
    if not target_layer_ids or any(
        not 0 <= int(layer_id) < target_layer_count for layer_id in target_layer_ids
    ):
        errors.append("draft target_layer_ids 为空或越过 target 层数")
    if errors:
        raise RuntimeError("下载模型结构校验失败: " + "; ".join(errors))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download Qwen3 and DFlash checkpoints")
    parser.add_argument("--config", default="configs/qwen3_4b_a2.json")
    parser.add_argument("--token", default=None, help="也可使用环境变量 HF_TOKEN，切勿提交 token")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config_path = Path(args.config).resolve()
    project_root = config_path.parent.parent
    with config_path.open("r", encoding="utf-8") as stream:
        config = json.load(stream)

    pairs = [
        (
            config["target_model_id"],
            config["target_revision"],
            project_root / config["target_local_dir"],
        ),
        (
            config["draft_model_id"],
            config["draft_revision"],
            project_root / config["draft_local_dir"],
        ),
    ]
    for repo_id, revision, destination in pairs:
        destination.parent.mkdir(parents=True, exist_ok=True)
        print(f"正在下载 {repo_id}@{revision} -> {destination}")
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_dir=destination,
            token=args.token,
        )
    _validate_downloaded_pair(
        pairs[0][2],
        pairs[1][2],
        required_future_tokens=int(config["block_size"]),
    )
    print("模型下载完成。")


if __name__ == "__main__":
    main()
