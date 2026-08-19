"""Profile per-iteration stage timings: tree15 vs vanilla DFlash on float32."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dflash_specblock.benchmark_vanilla import create_vanilla_engine
from dflash_specblock.cli import create_engine
from dflash_specblock.config import ExperimentConfig
from dflash_specblock.device import resolve_device
from dflash_specblock.models import render_prompt


def load_prompts(path: Path, n: int = 5) -> list[str]:
    prompts: list[str] = []
    with path.open() as f:
        for line in f:
            item = json.loads(line.strip())
            prompts.append(item["prompt"])
            if len(prompts) >= n:
                break
    return prompts


def profile_tree15(config, device, prompts, max_new_tokens, stop_ids):
    engine, tokenizer = create_engine(config, device)
    print("\n" + "=" * 70)
    print("Tree15 DFlash-SpecBlock (float32) — per-iteration breakdown")
    print("=" * 70)
    all_stats = []
    for idx, prompt in enumerate(prompts):
        input_ids = render_prompt(tokenizer, prompt, config.enable_thinking).to(device)
        result = engine.generate(input_ids, max_new_tokens, stop_ids)
        iters = result.iterations
        print(f"\n[Prompt {idx}] prefill={result.prefill_ms:.1f}ms  iters={len(iters)}  tokens={result.generated_ids.shape[1]}")
        total_draft = total_tree_build = total_verify = total_compact = 0.0
        for i, s in enumerate(iters):
            draft_only = s.draft_ms - s.tree_build_ms
            verify_only = s.verify_ms - s.cache_compact_ms
            total_draft += s.draft_ms
            total_tree_build += s.tree_build_ms
            total_verify += s.verify_ms
            total_compact += s.cache_compact_ms
            print(f"  iter {i:2d}: nodes={s.tree_nodes:2d}  "
                  f"draft={s.draft_ms:6.1f}ms (fwd={draft_only:5.1f}+tree={s.tree_build_ms:5.1f})  "
                  f"verify={s.verify_ms:6.1f}ms (fwd={verify_only:5.1f}+compact={s.cache_compact_ms:4.1f})  "
                  f"accepted={s.accepted_draft_tokens}  committed={s.committed_tokens}")
        total = result.prefill_ms + sum(s.draft_ms + s.verify_ms for s in iters)
        print(f"  TOTAL: prefill={result.prefill_ms:.1f}  draft={total_draft:.1f}  "
              f"tree_build={total_tree_build:.1f}  verify={total_verify:.1f}  "
              f"compact={total_compact:.1f}  wall={total:.1f}ms")
        all_stats.append({
            "prompt": idx,
            "prefill_ms": result.prefill_ms,
            "total_draft_ms": total_draft,
            "total_tree_build_ms": total_tree_build,
            "total_verify_ms": total_verify,
            "total_compact_ms": total_compact,
            "wall_ms": total,
            "iterations": len(iters),
            "tokens": result.generated_ids.shape[1],
        })
    return all_stats


def profile_vanilla(config, device, prompts, max_new_tokens, stop_ids):
    engine, tokenizer = create_vanilla_engine(config, device)
    print("\n" + "=" * 70)
    print("Vanilla DFlash (float32) — per-iteration breakdown")
    print("=" * 70)
    all_stats = []
    for idx, prompt in enumerate(prompts):
        input_ids = render_prompt(tokenizer, prompt, config.enable_thinking).to(device)
        result = engine.generate(input_ids, max_new_tokens, stop_ids)
        iters = result.iterations
        print(f"\n[Prompt {idx}] prefill={result.prefill_ms:.1f}ms  iters={len(iters)}  tokens={result.generated_ids.shape[1]}")
        total_draft = total_verify = 0.0
        for i, s in enumerate(iters):
            total_draft += s.draft_ms
            total_verify += s.verify_ms
            print(f"  iter {i:2d}: block={s.block_size:2d}  "
                  f"draft={s.draft_ms:6.1f}ms  "
                  f"verify={s.verify_ms:6.1f}ms  "
                  f"accepted={s.accepted_draft_tokens}  committed={s.committed_tokens}")
        total = result.prefill_ms + sum(s.draft_ms + s.verify_ms for s in iters)
        print(f"  TOTAL: prefill={result.prefill_ms:.1f}  draft={total_draft:.1f}  "
              f"verify={total_verify:.1f}  wall={total:.1f}ms")
        all_stats.append({
            "prompt": idx,
            "prefill_ms": result.prefill_ms,
            "total_draft_ms": total_draft,
            "total_verify_ms": total_verify,
            "wall_ms": total,
            "iterations": len(iters),
            "tokens": result.generated_ids.shape[1],
        })
    return all_stats


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tree-config", default="configs/qwen3_4b_a2_tree15_float32.json")
    parser.add_argument("--vanilla-config", default="configs/qwen3_4b_a2_float32.json")
    parser.add_argument("--n-prompts", type=int, default=5)
    parser.add_argument("--skip-vanilla", action="store_true")
    args = parser.parse_args()

    project = Path(__file__).resolve().parent.parent
    config_tree = ExperimentConfig.from_json(str(project / args.tree_config))
    config_vanilla = ExperimentConfig.from_json(str(project / args.vanilla_config))

    device = resolve_device(config_tree.device)
    prompts = load_prompts(project / "datasets/processed/specblock_official/prompts_benchmark_tree15.jsonl", n=args.n_prompts)
    max_new_tokens = config_tree.max_new_tokens
    stop_ids = set()

    tree_stats = profile_tree15(config_tree, device, prompts, max_new_tokens, stop_ids)
    vanilla_stats = None
    if not args.skip_vanilla:
        del device
        torch.npu.empty_cache() if hasattr(torch.npu, "empty_cache") else None
        device2 = resolve_device(config_vanilla.device)
        vanilla_stats = profile_vanilla(config_vanilla, device2, prompts, max_new_tokens, stop_ids)

    print("\n" + "=" * 70)
    print(f"AGGREGATE ({len(tree_stats)} prompts)")
    print("=" * 70)
    t_d = sum(s["total_draft_ms"] for s in tree_stats)
    t_tb = sum(s.get("total_tree_build_ms", 0) for s in tree_stats)
    t_v = sum(s["total_verify_ms"] for s in tree_stats)
    t_c = sum(s.get("total_compact_ms", 0) for s in tree_stats)
    t_w = sum(s["wall_ms"] for s in tree_stats)
    t_iters = sum(s["iterations"] for s in tree_stats)
    t_tokens = sum(s["tokens"] for s in tree_stats)

    if vanilla_stats:
        v_d = sum(s["total_draft_ms"] for s in vanilla_stats)
        v_v = sum(s["total_verify_ms"] for s in vanilla_stats)
        v_w = sum(s["wall_ms"] for s in vanilla_stats)
        v_iters = sum(s["iterations"] for s in vanilla_stats)
        v_tokens = sum(s["tokens"] for s in vanilla_stats)
    else:
        v_d = v_v = v_w = v_iters = v_tokens = 0

    print(f"{'metric':<22} {'tree (ms)':>14} {'vanilla (ms)':>14} {'delta':>14} {'tree/van':>10}")
    print("-" * 74)
    print(f"{'draft_total':<22} {t_d:>14.1f} {v_d:>14.1f} {t_d-v_d:>14.1f} {t_d/v_d if v_d else 0:>9.2f}x")
    if t_tb:
        print(f"{'  tree_build':<22} {t_tb:>14.1f} {'—':>14} {'—':>14}")
    print(f"{'verify_total':<22} {t_v:>14.1f} {v_v:>14.1f} {t_v-v_v:>14.1f} {t_v/v_v if v_v else 0:>9.2f}x")
    if t_c:
        print(f"{'  cache_compact':<22} {t_c:>14.1f} {'—':>14} {'—':>14}")
    print(f"{'wall_total':<22} {t_w:>14.1f} {v_w:>14.1f} {t_w-v_w:>14.1f} {t_w/v_w if v_w else 0:>9.2f}x")
    print(f"{'total_iterations':<22} {t_iters:>14d} {v_iters:>14d} {t_iters-v_iters:>14d}")
    print(f"{'total_tokens':<22} {t_tokens:>14d} {v_tokens:>14d}")
    if t_iters and t_tokens:
        print(f"{'avg committed/iter':<22} {t_tokens/t_iters:>14.2f}")
        print(f"{'avg verify_ms/iter':<22} {t_v/t_iters:>14.1f}")
        print(f"{'avg wall_ms/iter':<22} {t_w/t_iters:>14.1f}")

    print(f"\n  tree_build 占 wall: {t_tb/t_w*100:.1f}%")
    print(f"  verify_fwd 占 wall: {(t_v-t_c)/t_w*100:.1f}%")
    print(f"  cache_compact 占 wall: {t_c/t_w*100:.1f}%")
    print(f"  draft_fwd 占 wall: {(t_d-t_tb)/t_w*100:.1f}%")


if __name__ == "__main__":
    main()
