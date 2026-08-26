"""单提示词实验入口。"""

from __future__ import annotations

import argparse
import json
import warnings
from collections import Counter

import torch

from .config import ExperimentConfig
from .ddtree_builder import DDTreeBuilder, LatencyAwareDDTreeBuilder
from .device import configure_cuda_runtime, dtype_from_name, resolve_device
from .dflash_adapter import DFlashBlockAdapter
from .engine import DFlashSpecBlockEngine
from .models import load_models, render_prompt
from .rank_head import HeuristicRanker, load_rank_head
from .tree import SpecBlockTreeBuilder
from .verification import GraphedTargetTreeVerifier, TargetTreeVerifier


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DFlash + SpecBlock NVIDIA CUDA experiment")
    parser.add_argument("--config", default="configs/qwen3_4b_cuda.json")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--device", default=None, help="auto/cpu/cuda:0")
    return parser


def create_engine(config: ExperimentConfig, device: torch.device):
    rank_checkpoint = config.rank_checkpoint_path
    if config.rank_mode == "learned" and (
        rank_checkpoint is None or not rank_checkpoint.is_file()
    ):
        raise FileNotFoundError(
            "正式 learned 模式缺少 rank checkpoint。请先运行 "
            "dflash-specblock-train-rank，或仅为连通性检查显式使用 smoke 配置。"
        )
    bundle = load_models(config, device)
    if config.rank_mode == "learned":
        ranker = load_rank_head(
            rank_checkpoint,
            hidden_size=int(bundle.draft.config.hidden_size),
            device=device,
            expected_metadata={
                "block_size": config.block_size,
                "max_blocks": config.max_blocks,
                "target_model_id": config.target_model_id,
                "target_revision": config.target_revision,
                "draft_model_id": config.draft_model_id,
                "draft_revision": config.draft_revision,
            },
        )
    elif config.tree_mode in {"ddtree", "ddtree_adaptive"}:
        # DDTree 的宽度分配完全由 draft log-prob 决定，rank head 的输出不会被读取。
        # 这里仍构造一个占位 ranker 以保持 adapter 的字段契约，但 engine 会通过
        # ``requires_rank=False`` 跳过它的前向。
        ranker = HeuristicRanker().to(device).eval()
    else:
        warnings.warn(
            "当前使用 heuristic ranker，只能做工程连通性测试；正式实验请训练 rank head。",
            stacklevel=2,
        )
        ranker = HeuristicRanker().to(device).eval()

    adapter = DFlashBlockAdapter(
        target=bundle.target,
        draft=bundle.draft,
        ranker=ranker,
        block_size=config.block_size,
    )
    if config.tree_mode == "ddtree_adaptive":
        tree_builder = LatencyAwareDDTreeBuilder(
            block_size=config.block_size,
            tree_budget=config.tree_budget,
            budget_candidates=config.ddtree_budget_candidates,
            initial_budget=config.ddtree_initial_budget,
            warmup_rounds_per_budget=config.ddtree_warmup_rounds_per_budget,
            ewma_alpha=config.ddtree_policy_ewma_alpha,
            exploration_interval=config.ddtree_exploration_interval,
        )
    elif config.tree_mode == "ddtree":
        tree_builder = DDTreeBuilder(
            block_size=config.block_size,
            tree_budget=config.tree_budget,
            reserve_greedy_chain=config.ddtree_reserve_greedy_chain,
        )
    else:
        tree_builder = SpecBlockTreeBuilder(
            block_size=config.block_size,
            max_blocks=config.max_blocks,
            tree_budget=config.tree_budget,
            beam_width=config.beam_width,
            branch_factors=config.branch_factors,
        )
    if config.use_cuda_graphs:
        if device.type != "cuda":
            raise ValueError("use_cuda_graphs=true 只能用于 NVIDIA CUDA 设备")
        verifier = GraphedTargetTreeVerifier(
            target=bundle.target,
            target_layer_ids=adapter.target_layer_ids,
            device=device,
            dtype=dtype_from_name(config.dtype),
            max_tree_budget=config.tree_budget,
            max_cache_len=config.cuda_graph_max_cache_len,
        )
    else:
        verifier = TargetTreeVerifier(
            target=bundle.target,
            target_layer_ids=adapter.target_layer_ids,
            device=device,
            dtype=dtype_from_name(config.dtype),
        )
    engine = DFlashSpecBlockEngine(
        target=bundle.target,
        adapter=adapter,
        tree_builder=tree_builder,
        verifier=verifier,
        device=device,
    )
    return engine, bundle.tokenizer


def main() -> None:
    args = build_parser().parse_args()
    config = ExperimentConfig.from_json(args.config)
    if args.device:
        config.device = args.device
    device = resolve_device(config.device)
    configure_cuda_runtime(device, allow_tf32=config.allow_tf32)
    if device.type != "cuda":
        warnings.warn("未使用 NVIDIA GPU；真实模型实验应在 cuda:0 上运行。", stacklevel=2)

    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    engine, tokenizer = create_engine(config, device)
    input_ids = render_prompt(tokenizer, args.prompt, config.enable_thinking).to(device)
    max_new_tokens = (
        config.max_new_tokens if args.max_new_tokens is None else args.max_new_tokens
    )
    if max_new_tokens < 1:
        raise ValueError("max-new-tokens 必须为正整数")
    stop_ids = {int(tokenizer.eos_token_id)} if tokenizer.eos_token_id is not None else set()
    result = engine.generate(input_ids, max_new_tokens=max_new_tokens, stop_token_ids=stop_ids)

    print(tokenizer.decode(result.output_ids[0], skip_special_tokens=True))
    budget_histogram = Counter(item.tree_nodes for item in result.iterations)
    summary = {
        "device": str(device),
        "prefill_ms": result.prefill_ms,
        "decode_ms": result.total_decode_ms,
        "generated_tokens": int(result.generated_ids.shape[1]),
        "verify_iterations": len(result.iterations),
        "average_committed_per_verify": result.average_accepted_length,
        "tree_budget_histogram": {
            str(budget): budget_histogram[budget] for budget in sorted(budget_histogram)
        },
        "iterations": [
            {
                "draft_ms": item.draft_ms,
                "verify_ms": item.verify_ms,
                "tree_nodes": item.tree_nodes,
                "accepted_draft_tokens": item.accepted_draft_tokens,
                "committed_tokens": item.committed_tokens,
            }
            for item in result.iterations
        ],
    }
    print("\n[实验统计]")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
