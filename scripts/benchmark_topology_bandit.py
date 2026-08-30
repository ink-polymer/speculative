#!/usr/bin/env python3
"""Train/evaluate a contextual bandit over topology and capacity actions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_ROOT = PROJECT_ROOT / "third_party" / "ddtree_official"
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(OFFICIAL_ROOT))
sys.path.insert(0, str(SRC_ROOT))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--prompts", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--checkpoint", type=Path, required=True)
    result.add_argument("--model", default="models/Qwen3-4B")
    result.add_argument("--draft", default="models/Qwen3-4B-DFlash-b16")
    result.add_argument(
        "--fixed-draft",
        default=None,
        help="Optional original draft used by baseline/fixed-DDTree comparators only.",
    )
    result.add_argument("--rank-checkpoint", type=Path, default=None)
    result.add_argument("--rank-choice-checkpoint", type=Path, default=None)
    result.add_argument("--corpus-training-text", type=Path, default=None)
    result.add_argument("--temperature", type=float, required=True)
    result.add_argument("--actions", required=True)
    result.add_argument("--initial-action", default="ddtree:60")
    result.add_argument("--fixed-tree-budget", type=int, default=60)
    result.add_argument("--max-samples", type=int, default=None)
    result.add_argument("--max-new-tokens", type=int, default=128)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--bandit-ridge", type=float, default=2.0)
    result.add_argument("--bandit-exploration-scale", type=float, default=0.25)
    result.add_argument("--bandit-warmup-episodes", type=int, default=12)
    result.add_argument("--bandit-reward-noise-scale", type=float, default=0.12)
    result.add_argument("--enable-cpp-compact", action="store_true")
    result.add_argument(
        "--cuda-graph-policy",
        action="store_true",
        help="Use a reusable fixed-shape CUDA Graph verifier for policy actions.",
    )
    result.add_argument("--cuda-graph-max-cache-len", type=int, default=4096)
    result.add_argument("--skip-comparators", action="store_true")
    result.add_argument("--resume", action="store_true")
    mode = result.add_mutually_exclusive_group(required=True)
    mode.add_argument("--bandit-train", action="store_true")
    mode.add_argument("--bandit-eval", action="store_true")
    return result


def load_rows(path: Path, limit: int | None) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return rows if limit is None else rows[:limit]


def metrics(result: Any) -> dict[str, Any]:
    decode_ms = float(result.time_per_output_token) * int(result.num_output_tokens) * 1000.0
    return {
        "generated_token_ids": result.output_ids[
            0, result.num_input_tokens :
        ].tolist(),
        "generated_tokens": int(result.num_output_tokens),
        "prefill_ms": float(result.time_to_first_token) * 1000.0,
        "decode_ms": decode_ms,
        "tokens_per_second": int(result.num_output_tokens) * 1000.0 / max(decode_ms, 1e-9),
        "mean_committed": sum(result.acceptance_lengths) / max(len(result.acceptance_lengths), 1),
        "decode_rounds": int(result.decode_rounds),
        "stage_times_s": {key: float(value) for key, value in result.stage_times.items()},
        "mean_tree_nodes": (
            sum(getattr(result, "tree_node_counts", []))
            / max(len(getattr(result, "tree_node_counts", [])), 1)
        ),
    }


def parse_action(action: str) -> tuple[str, tuple[str, ...]]:
    pieces = action.split(":")
    kind, values = pieces[0], tuple(pieces[1:])
    expected_values = {
        "ddtree": 1, "dpv": 1, "gbv": 1, "dfs": 1, "sparse": 1,
        "calibrated": 3, "lookup": 4, "corpus": 3,
        "rank": 2, "rankchoice": 2, "slookup": 4,
        "scorpus": 3, "srankchoice": 2,
    }
    if kind not in expected_values or len(values) != expected_values[kind]:
        raise ValueError(f"invalid topology action: {action}")
    if int(values[0]) <= 0:
        raise ValueError(f"action capacity must be positive: {action}")
    return kind, values


def main() -> None:
    cfg = parser().parse_args()
    import torch
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from ddtree import ddtree_generate, maybe_enable_cpp_compact
    from dflash import dflash_generate
    from model import DFlashDraftModel
    from dflash_specblock.topology_bandit import (
        TopologyRatioBandit,
        prompt_context_features,
    )

    action_names = tuple(
        dict.fromkeys(action.strip() for action in cfg.actions.split(",") if action.strip())
    )
    parsed_actions = {action: parse_action(action) for action in action_names}
    if any(kind == "dpv" for kind, _values in parsed_actions.values()):
        if cfg.temperature != 0.0:
            raise ValueError("dpv actions require temperature=0")
        from dpv import dpv_generate
    if any(kind == "gbv" for kind, _values in parsed_actions.values()):
        if cfg.temperature <= 0.0:
            raise ValueError("gbv actions require temperature>0")
        from gbv import gbv_generate

    maybe_enable_cpp_compact(cfg.enable_cpp_compact)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    target = AutoModelForCausalLM.from_pretrained(
        cfg.model, attn_implementation="sdpa", dtype=torch.bfloat16
    ).cuda().eval()
    draft = DFlashDraftModel.from_pretrained(
        cfg.draft, attn_implementation="flash_attention_2", dtype=torch.bfloat16
    ).cuda().eval()
    fixed_draft = draft
    if cfg.fixed_draft is not None:
        fixed_draft = DFlashDraftModel.from_pretrained(
            cfg.fixed_draft,
            attn_implementation="flash_attention_2",
            dtype=torch.bfloat16,
        ).cuda().eval()
        if int(fixed_draft.block_size) != int(draft.block_size):
            raise ValueError("policy and fixed-comparator drafts must use the same block size")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model)
    rank_head = None
    if any(kind == "rank" for kind, _values in parsed_actions.values()):
        if cfg.rank_checkpoint is None:
            raise ValueError("rank actions require --rank-checkpoint")
        from dflash_specblock.rank_head import load_rank_head
        rank_head = load_rank_head(
            cfg.rank_checkpoint,
            hidden_size=int(target.config.hidden_size),
            device=torch.device("cuda"),
            expected_metadata={"block_size": int(draft.block_size) - 1, "max_blocks": 1},
        )
    rank_choice_head = None
    if any(kind in {"rankchoice", "srankchoice"} for kind, _values in parsed_actions.values()):
        if cfg.rank_choice_checkpoint is None:
            raise ValueError("rankchoice actions require --rank-choice-checkpoint")
        from dflash_specblock.rank_head import load_rank_choice_head
        rank_choice_head = load_rank_choice_head(
            cfg.rank_choice_checkpoint,
            hidden_size=int(target.config.hidden_size),
            device=torch.device("cuda"),
        )
    block_size = int(draft.block_size)
    graph_verifier = None
    if cfg.cuda_graph_policy:
        if cfg.temperature != 0.0:
            raise ValueError("CUDA Graph policy verification requires temperature=0")
        if any(kind in {"dpv", "gbv"} for kind, _values in parsed_actions.values()):
            raise ValueError("CUDA Graph policy supports DDTree-family actions only")
        from dflash_specblock.verification import GraphedTargetTreeVerifier

        graph_budget = 0
        for kind, values in parsed_actions.values():
            capacity = int(values[0])
            extra_paths = 0
            if kind in {"lookup", "slookup"}:
                extra_paths = int(values[1])
            elif kind in {"corpus", "scorpus"}:
                extra_paths = int(values[1])
            graph_budget = max(
                graph_budget,
                capacity + extra_paths * (block_size - 1),
            )
        graph_verifier = GraphedTargetTreeVerifier(
            target=target,
            target_layer_ids=draft.target_layer_ids,
            device=next(target.parameters()).device,
            dtype=target.dtype,
            max_tree_budget=graph_budget,
            max_cache_len=cfg.cuda_graph_max_cache_len,
        )
    corpus_store = None
    if any(kind in {"corpus", "scorpus"} for kind, _values in parsed_actions.values()):
        if cfg.corpus_training_text is None:
            raise ValueError("corpus actions require --corpus-training-text")
        from ddtree import NgramContinuationStore
        texts = [
            json.loads(line)["text"]
            for line in cfg.corpus_training_text.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        corpus_store = NgramContinuationStore(
            [tokenizer.encode(text, add_special_tokens=True) for text in texts],
            horizon=block_size - 1,
            max_ngram=4,
        )
    stop_ids = target.generation_config.eos_token_id or tokenizer.eos_token_id
    stop_ids = [int(stop_ids)] if isinstance(stop_ids, int) else [int(x) for x in stop_ids]
    metadata = {
        "temperature": str(cfg.temperature),
        "target_model": str(cfg.model),
        "draft_model": str(cfg.draft),
        "fixed_comparator_draft_model": str(cfg.fixed_draft or cfg.draft),
        "cuda_graph_policy": str(bool(cfg.cuda_graph_policy)).lower(),
        "block_size": str(block_size),
        "decision_level": "one_joint_topology_capacity_action_per_prompt",
        "dataset_label_feature": "false",
    }
    policy = TopologyRatioBandit(
        action_names,
        cfg.initial_action,
        ridge=cfg.bandit_ridge,
        exploration_scale=cfg.bandit_exploration_scale,
        warmup_episodes_per_action=cfg.bandit_warmup_episodes,
        reward_noise_scale=cfg.bandit_reward_noise_scale,
        learning_enabled=cfg.bandit_train,
        random_seed=cfg.seed,
        policy_metadata=metadata,
    )
    checkpoint = cfg.checkpoint.expanduser().resolve()
    if checkpoint.is_file():
        policy.load_policy(checkpoint)
    elif cfg.bandit_eval:
        raise FileNotFoundError(f"frozen evaluation requires {checkpoint}")

    def encode(prompt: str) -> torch.Tensor:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        return tokenizer.encode(
            text, return_tensors="pt", add_special_tokens=False
        ).cuda()

    def run_target(ids: torch.Tensor, length: int):
        return dflash_generate(
            model=fixed_draft,
            target=target,
            input_ids=ids,
            mask_token_id=fixed_draft.mask_token_id,
            max_new_tokens=length,
            block_size=1,
            stop_token_ids=stop_ids,
            temperature=cfg.temperature,
            verification_mode="target_match",
        )

    def run_action(
        ids: torch.Tensor,
        action: str,
        length: int,
        *,
        draft_model=None,
        graphed: bool = False,
    ):
        kind, values = parse_action(action)
        capacity = int(values[0])
        draft_model = draft if draft_model is None else draft_model
        common = dict(
            model=draft_model,
            target=target,
            input_ids=ids,
            mask_token_id=draft_model.mask_token_id,
            max_new_tokens=length,
            block_size=block_size,
            stop_token_ids=stop_ids,
            temperature=cfg.temperature,
        )
        verifier_arg = {
            "target_verifier": graph_verifier if graphed else None,
        }
        if kind == "ddtree":
            return ddtree_generate(**common, **verifier_arg, tree_budget=capacity)
        if kind == "dfs":
            return ddtree_generate(
                **common, **verifier_arg, tree_budget=capacity,
                depth_first_order=True, fast_timing=True,
            )
        if kind == "sparse":
            return ddtree_generate(
                **common, **verifier_arg, tree_budget=capacity, sparse_lm_head=True,
                depth_first_order=True, fast_timing=True,
            )
        if kind == "calibrated":
            return ddtree_generate(
                **common, **verifier_arg, tree_budget=capacity,
                tree_temperature=float(values[1]),
                tree_depth_bonus=float(values[2]),
            )
        if kind == "lookup":
            return ddtree_generate(
                **common, **verifier_arg, tree_budget=capacity,
                lookup_path_count=int(values[1]),
                lookup_max_ngram=int(values[2]),
                tree_temperature=float(values[3]),
                depth_first_order=True, fast_timing=True,
            )
        if kind == "slookup":
            return ddtree_generate(
                **common, **verifier_arg, tree_budget=capacity,
                lookup_path_count=int(values[1]),
                lookup_max_ngram=int(values[2]),
                tree_temperature=float(values[3]),
                depth_first_order=True, fast_timing=True,
                sparse_lm_head=True,
            )
        if kind == "corpus":
            return ddtree_generate(
                **common, **verifier_arg, tree_budget=capacity,
                corpus_lookup_store=corpus_store,
                corpus_lookup_path_count=int(values[1]),
                tree_temperature=float(values[2]),
                depth_first_order=True, fast_timing=True,
            )
        if kind == "scorpus":
            return ddtree_generate(
                **common, **verifier_arg, tree_budget=capacity,
                corpus_lookup_store=corpus_store,
                corpus_lookup_path_count=int(values[1]),
                tree_temperature=float(values[2]),
                depth_first_order=True, fast_timing=True,
                sparse_lm_head=True,
            )
        if kind == "rank":
            return ddtree_generate(
                **common, **verifier_arg, tree_budget=capacity, rank_head=rank_head,
                rank_score_blend=float(values[1]),
                depth_first_order=True, fast_timing=True,
            )
        if kind == "rankchoice":
            return ddtree_generate(
                **common, **verifier_arg, tree_budget=capacity,
                rank_choice_head=rank_choice_head,
                rank_score_blend=float(values[1]),
                depth_first_order=True, fast_timing=True,
            )
        if kind == "srankchoice":
            return ddtree_generate(
                **common, **verifier_arg, tree_budget=capacity,
                rank_choice_head=rank_choice_head,
                rank_score_blend=float(values[1]),
                depth_first_order=True, fast_timing=True,
                sparse_lm_head=True,
            )
        if kind == "dpv":
            return dpv_generate(**common, path_count=capacity)
        return gbv_generate(**common, path_count=capacity)

    # Compile/cache every candidate shape outside measured samples.
    warmup = encode("Warmup")
    run_target(warmup, 16)
    run_action(
        warmup,
        f"ddtree:{cfg.fixed_tree_budget}",
        16,
        draft_model=fixed_draft,
    )
    for action in action_names:
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)
        run_action(warmup, action, 16, graphed=cfg.cuda_graph_policy)

    rows = load_rows(cfg.prompts.expanduser().resolve(), cfg.max_samples)
    output = cfg.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = 0
    mode = "w"
    if cfg.resume and output.is_file():
        completed = len(load_rows(output, None))
        mode = "a"
    with output.open(mode, encoding="utf-8") as stream:
        for index, row in enumerate(
            tqdm(rows[completed:], initial=completed, total=len(rows), desc="topology-bandit"),
            start=completed,
        ):
            prompt = row["prompt"]
            ids = encode(prompt)
            context = prompt_context_features(prompt, int(ids.shape[1]))
            action = policy.select(context)
            seed = cfg.seed + index
            baseline_metrics = fixed_metrics = None
            if not cfg.skip_comparators:
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                baseline_metrics = metrics(run_target(ids, cfg.max_new_tokens))
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                fixed_metrics = metrics(
                    run_action(
                        ids,
                        f"ddtree:{cfg.fixed_tree_budget}",
                        cfg.max_new_tokens,
                        draft_model=fixed_draft,
                    )
                )
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            selected_metrics = metrics(
                run_action(
                    ids,
                    action,
                    cfg.max_new_tokens,
                    graphed=cfg.cuda_graph_policy,
                )
            )
            policy.observe(
                committed_tokens=selected_metrics["generated_tokens"],
                decode_ms=selected_metrics["decode_ms"],
            )
            if cfg.bandit_train:
                policy.save_policy(checkpoint)
            record = {
                "index": index,
                "dataset": row.get("dataset"),
                "source_id": row.get("source_id"),
                "prompt": prompt,
                "prompt_tokens": int(ids.shape[1]),
                "temperature": cfg.temperature,
                "action": action,
                "policy_runtime": {
                    "cuda_graph": bool(cfg.cuda_graph_policy),
                    "cuda_graph_max_cache_len": (
                        cfg.cuda_graph_max_cache_len
                        if cfg.cuda_graph_policy
                        else None
                    ),
                    "draft_model": str(cfg.draft),
                    "fixed_comparator_draft_model": str(
                        cfg.fixed_draft or cfg.draft
                    ),
                },
                "decision": {
                    "predicted_tokens_per_ms": policy.last_decision.predicted_tokens_per_ms,
                    "uncertainty": policy.last_decision.uncertainty,
                    "forced_exploration": policy.last_decision.forced_exploration,
                },
                "baseline": baseline_metrics,
                "fixed_ddtree": fixed_metrics,
                "policy": selected_metrics,
                "policy_diagnostics": policy.diagnostics(),
            }
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
    if cfg.bandit_train:
        policy.save_policy(checkpoint)


if __name__ == "__main__":
    main()
