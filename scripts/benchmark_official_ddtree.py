#!/usr/bin/env python3
"""Run the vendored official DDTree implementation on this paper's JSONL suite."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_ROOT = PROJECT_ROOT / "third_party" / "ddtree_official"
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(OFFICIAL_ROOT))
sys.path.insert(0, str(SRC_ROOT))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Official liranringel/DDTree on the exact DFlash-SpecBlock prompt JSONL"
    )
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--draft", default="z-lab/Qwen3-4B-DFlash-b16")
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--draft-revision", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--draft-attn-implementation",
        choices=("flash_attention_2", "sdpa", "eager"),
        default="flash_attention_2",
    )
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--tree-budget", type=int, default=256)
    parser.add_argument("--fixed-tree-budget", type=int, default=60)
    parser.add_argument(
        "--budget-candidates",
        default="30,40,50,60,70,80,90,100,112,128,144,160,192,224,256",
    )
    parser.add_argument("--initial-budget", type=int, default=60)
    parser.add_argument("--ppo-hidden-size", type=int, default=64)
    parser.add_argument("--ppo-learning-rate", type=float, default=3e-4)
    parser.add_argument("--ppo-gamma", type=float, default=0.99)
    parser.add_argument("--ppo-gae-lambda", type=float, default=0.95)
    parser.add_argument("--ppo-clip-range", type=float, default=0.2)
    parser.add_argument("--ppo-value-coefficient", type=float, default=0.5)
    parser.add_argument("--ppo-entropy-coefficient", type=float, default=0.01)
    parser.add_argument("--ppo-rollout-steps", type=int, default=256)
    parser.add_argument("--ppo-update-epochs", type=int, default=4)
    parser.add_argument("--ppo-minibatch-size", type=int, default=64)
    parser.add_argument("--ppo-max-grad-norm", type=float, default=0.5)
    parser.add_argument("--ppo-tree-build-cost-weight", type=float, default=2.0)
    parser.add_argument("--ppo-context-length-scale", type=int, default=4096)
    parser.add_argument("--ppo-checkpoint", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--ppo-train", action="store_true")
    mode.add_argument("--ppo-eval", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--enable-cpp-compact", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def _load_rows(path: Path, limit: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.expanduser().resolve().open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            prompt = row.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"{path}:{line_number} is missing a non-empty prompt")
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    if not rows:
        raise ValueError(f"No prompts found in {path}")
    return rows


def _metrics(result: Any) -> dict[str, Any]:
    generated = result.output_ids[0, result.num_input_tokens :].detach().cpu().tolist()
    decode_s = float(result.time_per_output_token) * int(result.num_output_tokens)
    budgets = Counter(int(x) for x in getattr(result, "selected_tree_budgets", []))
    return {
        "generated_token_ids": generated,
        "generated_tokens": int(result.num_output_tokens),
        "prefill_ms": float(result.time_to_first_token) * 1000.0,
        "decode_ms": decode_s * 1000.0,
        "tokens_per_second": int(result.num_output_tokens) / max(decode_s, 1e-12),
        "acceptance_lengths": [int(value) for value in result.acceptance_lengths],
        "decode_rounds": int(result.decode_rounds),
        "stage_times_s": {key: float(value) for key, value in result.stage_times.items()},
        "tree_budget_histogram": {str(k): budgets[k] for k in sorted(budgets)},
        "tree_policy": getattr(result, "tree_policy", None),
    }


def main() -> None:
    args = _parser().parse_args()
    if args.temperature <= 0:
        raise ValueError(
            "sampling comparison requires --temperature > 0 (recommended: 1.0)"
        )
    if not OFFICIAL_ROOT.is_dir():
        raise FileNotFoundError(f"Official DDTree checkout not found: {OFFICIAL_ROOT}")

    import torch
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from ddtree import ddtree_generate, maybe_enable_cpp_compact
    from dflash import dflash_generate
    from dflash_specblock.ppo_builder import PPODDTreeBuilder
    from model import DFlashDraftModel

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Official DDTree benchmark requires an available CUDA GPU")
    if args.draft_attn_implementation == "flash_attention_2":
        try:
            import flash_attn  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "--draft-attn-implementation=flash_attention_2 requires flash-attn; "
                "use sdpa on hosts without a matching prebuilt wheel or nvcc"
            ) from exc

    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    maybe_enable_cpp_compact(args.enable_cpp_compact)

    target_kwargs: dict[str, Any] = {
        "attn_implementation": "sdpa",
        "dtype": torch.bfloat16,
    }
    draft_kwargs: dict[str, Any] = {
        "attn_implementation": args.draft_attn_implementation,
        "dtype": torch.bfloat16,
    }
    tokenizer_kwargs: dict[str, Any] = {}
    if args.model_revision:
        target_kwargs["revision"] = args.model_revision
        tokenizer_kwargs["revision"] = args.model_revision
    if args.draft_revision:
        draft_kwargs["revision"] = args.draft_revision

    target = AutoModelForCausalLM.from_pretrained(args.model, **target_kwargs).to(device).eval()
    draft = DFlashDraftModel.from_pretrained(args.draft, **draft_kwargs).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, **tokenizer_kwargs)
    block_size = int(draft.block_size if args.block_size is None else args.block_size)
    budget_candidates = tuple(
        int(value.strip())
        for value in args.budget_candidates.split(",")
        if value.strip()
    )
    policy_metadata = {
        "target_model_id": str(args.model),
        "target_revision": str(args.model_revision),
        "draft_model_id": str(args.draft),
        "draft_revision": str(args.draft_revision),
        "dtype": "bfloat16",
        "draft_attn_implementation": args.draft_attn_implementation,
        "temperature": str(args.temperature),
        "verification": "official_target_sampling_tree_walk",
        "policy_algorithm": "ppo_discrete",
    }
    ppo_builder = PPODDTreeBuilder(
        block_size=block_size - 1,
        tree_budget=args.tree_budget,
        budget_candidates=budget_candidates,
        initial_budget=args.initial_budget,
        hidden_size=args.ppo_hidden_size,
        learning_rate=args.ppo_learning_rate,
        gamma=args.ppo_gamma,
        gae_lambda=args.ppo_gae_lambda,
        clip_range=args.ppo_clip_range,
        value_coefficient=args.ppo_value_coefficient,
        entropy_coefficient=args.ppo_entropy_coefficient,
        rollout_steps=args.ppo_rollout_steps,
        update_epochs=args.ppo_update_epochs,
        minibatch_size=args.ppo_minibatch_size,
        max_grad_norm=args.ppo_max_grad_norm,
        tree_build_cost_weight=args.ppo_tree_build_cost_weight,
        context_length_scale=args.ppo_context_length_scale,
        learning_enabled=args.ppo_train,
        policy_metadata=policy_metadata,
    )
    checkpoint = args.ppo_checkpoint.expanduser().resolve()
    checkpoint_loaded = checkpoint.is_file()
    if checkpoint_loaded:
        ppo_builder.load_policy(checkpoint)
    elif args.ppo_eval:
        raise FileNotFoundError(f"Frozen PPO evaluation requires {checkpoint}")
    stop_ids = target.generation_config.eos_token_id or tokenizer.eos_token_id
    stop_ids = [int(stop_ids)] if isinstance(stop_ids, int) else [int(x) for x in stop_ids]
    rows = _load_rows(args.prompts, args.max_samples)

    def encode(prompt: str):
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        return tokenizer.encode(text, return_tensors="pt", add_special_tokens=False).to(device)

    def run_dflash(input_ids, size: int, verification_mode: str = "target_match"):
        return dflash_generate(
            model=draft,
            target=target,
            input_ids=input_ids,
            mask_token_id=draft.mask_token_id,
            max_new_tokens=args.max_new_tokens,
            block_size=size,
            stop_token_ids=stop_ids,
            temperature=args.temperature,
            verification_mode=verification_mode,
        )

    def run_ddtree(input_ids, *, use_policy: bool = True):
        selected_budget = args.tree_budget if use_policy else args.fixed_tree_budget
        return ddtree_generate(
            model=draft,
            target=target,
            input_ids=input_ids,
            mask_token_id=draft.mask_token_id,
            max_new_tokens=args.max_new_tokens,
            block_size=block_size,
            tree_budget=selected_budget,
            stop_token_ids=stop_ids,
            temperature=args.temperature,
            tree_builder=ppo_builder if use_policy else None,
        )

    warmup_ids = encode("Warmup")
    original_max_new_tokens = args.max_new_tokens
    args.max_new_tokens = min(16, original_max_new_tokens)
    run_dflash(warmup_ids, 1)
    run_dflash(warmup_ids, block_size, "token")
    # Warm the original target tree-walk without updating or consulting PPO.
    run_ddtree(warmup_ids, use_policy=False)
    args.max_new_tokens = original_max_new_tokens

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = 0
    mode = "w"
    if args.resume and output.exists():
        existing = _load_rows(output, None)
        completed = len(existing)
        if args.ppo_train and completed and not checkpoint_loaded:
            raise FileNotFoundError(
                "Cannot resume PPO training output without its policy checkpoint"
            )
        if completed > len(rows):
            raise ValueError(f"resume output has {completed} rows, but input has only {len(rows)}")
        for index, record in enumerate(existing):
            if int(record.get("index", -1)) != index or record.get("prompt") != rows[index]["prompt"]:
                raise ValueError(f"resume output/input mismatch at row {index}")
            if float(record.get("temperature", -1.0)) != args.temperature:
                raise ValueError(f"resume temperature mismatch at row {index}")
            if record.get("ppo_mode") != ("train" if args.ppo_train else "eval"):
                raise ValueError(f"resume PPO mode mismatch at row {index}")
        mode = "a"
        print(f"Resuming {output} from row {completed}/{len(rows)}")
    with output.open(mode, encoding="utf-8") as stream:
        pending = rows[completed:]
        iterator = tqdm(
            enumerate(pending, start=completed),
            total=len(pending),
            initial=0,
            desc="official-ddtree",
            unit="prompt",
        )
        for index, row in iterator:
            input_ids = encode(row["prompt"])
            # Common per-prompt seed keeps the comparison reproducible.  Each method
            # is reset independently because it consumes a different number of random
            # variates after the first rejection.
            prompt_seed = args.seed + index
            comparison: dict[str, Any] = {}
            if args.ppo_eval:
                torch.manual_seed(prompt_seed)
                torch.cuda.manual_seed_all(prompt_seed)
                comparison["baseline"] = _metrics(run_dflash(input_ids, 1))
                torch.manual_seed(prompt_seed)
                torch.cuda.manual_seed_all(prompt_seed)
                comparison["dflash"] = _metrics(run_dflash(input_ids, block_size, "token"))
                torch.manual_seed(prompt_seed)
                torch.cuda.manual_seed_all(prompt_seed)
                comparison["ddtree"] = _metrics(run_ddtree(input_ids, use_policy=False))
            torch.manual_seed(prompt_seed)
            torch.cuda.manual_seed_all(prompt_seed)
            ddtree_ppo = _metrics(run_ddtree(input_ids))
            if args.ppo_train:
                # Checkpointing is outside measured decode time and makes long jobs resumable.
                ppo_builder.save_policy(checkpoint)
            record = {
                "index": index,
                "dataset": row.get("dataset"),
                "source_id": row.get("source_id"),
                "prompt": row["prompt"],
                "implementation": "temperature_sampling_comparison",
                "verification_modes": {
                    "dflash": "token_rejection",
                    "ddtree": "official_target_sampling_tree_walk",
                    "ddtree_ppo": "official_target_sampling_tree_walk",
                },
                "dtype": "bfloat16",
                "draft_attn_implementation": args.draft_attn_implementation,
                "block_size_including_anchor": block_size,
                "tree_budget": args.tree_budget,
                "fixed_tree_budget": args.fixed_tree_budget,
                "temperature": args.temperature,
                "seed": prompt_seed,
                "ppo_mode": "train" if args.ppo_train else "eval",
                "ppo_checkpoint": str(checkpoint),
                "ppo_budget_candidates": list(budget_candidates),
                "ppo_tree_build_cost_weight": args.ppo_tree_build_cost_weight,
                "ddtree_ppo": ddtree_ppo,
                **comparison,
            }
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()

    if args.ppo_train:
        ppo_builder.finalize_training()
        ppo_builder.save_policy(checkpoint)


if __name__ == "__main__":
    main()
