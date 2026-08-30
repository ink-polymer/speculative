# DDTree tree-block-verification ablation: 2K rerun

Date: 2026-08-30

## Scope

This is a verification ablation, not a claim that the GBV verifier is a
distribution-preserving drop-in replacement for DDTree's deterministic
best-first node tree. The control keeps the official 60-node DDTree target
sampling tree walk. The ablation samples three i.i.d. DFlash paths, merges them
into a prefix tree, and applies the vendored Thomas--Pal greedy multi-path block
verification implementation.

All four methods use the same target and draft model, prompt order, per-prompt
seed, temperature, BF16 dtype, and maximum generation length. At temperature
one, sampled outputs and EOS positions are not expected to match pairwise.

## Configuration

- GPU: NVIDIA GeForce RTX 4090
- Target: Qwen3-4B
- Draft: Qwen3-4B-DFlash-b16
- Temperature: 1.0
- Maximum new tokens: 128
- DDTree budget: 60 nodes
- GBV paths: 3
- Block size: 16 including the anchor
- Prompts: 2,000 (GSM8K 1,169; HumanEval 149; MATH-500 450; MBPP 232)
- Seed: 42

## Overall results

| Method | Generated tokens | ms/token | tokens/s | Speedup vs target | Mean committed/verify | Decode rounds |
|---|---:|---:|---:|---:|---:|---:|
| Target | 255,699 | 27.6809 | 36.1259 | 1.0000x | 1.0000 | 255,683 |
| DFlash | 255,749 | 8.3167 | 120.2393 | 3.3283x | 5.0034 | 52,764 |
| DDTree | 255,783 | 5.9896 | 166.9552 | 4.6215x | 7.2079 | 36,846 |
| Tree block verification | 255,792 | 4.0613 | 246.2270 | 6.8158x | 13.0998 | 20,221 |

Relative to the DDTree control, tree block verification increased throughput by
47.48%, reduced mean time per token by 32.19%, increased mean committed tokens
per verification by 81.74%, and reduced decode rounds by 45.12%. It was faster
on 1,690 of 2,000 paired prompts. The mean paired DDTree/GBV speed ratio was
1.5660x and the median was 1.5346x.

## Per-dataset results

| Dataset | DDTree tokens/s | DDTree speedup | Tree-BV tokens/s | Tree-BV speedup |
|---|---:|---:|---:|---:|
| GSM8K | 160.0774 | 4.4298x | 249.3682 | 6.9008x |
| HumanEval | 166.1835 | 4.6033x | 250.8473 | 6.9485x |
| MATH-500 | 185.6607 | 5.1536x | 251.2424 | 6.9741x |
| MBPP | 171.0699 | 4.7144x | 221.0251 | 6.0911x |

## Artifacts

- `outputs/ddtree_tree_block_verification_2k_rerun_20260830/formal_2k_x128.jsonl`
  - 2,000 per-prompt records
  - SHA-256: `eb4e4a3a6d3d54e049068ee7fe0a168a3124eedb98ed9836bd789b0ae0d903fe`
- `outputs/ddtree_tree_block_verification_2k_rerun_20260830/formal_2k_x128_summary.json`
  - Overall and per-dataset aggregate metrics
  - SHA-256: `b4efea0844f5f2ef662cbe207e01ba89255c040ebc9b80819baa1381d7696fdc`

## Reproduction

```bash
TEMPERATURE=1.0 \
MAX_NEW_TOKENS=128 \
TREE_BUDGET=60 \
GBV_PATHS=3 \
MAX_SAMPLES=2000 \
OUTPUT=outputs/ddtree_tree_block_verification_2k_rerun_20260830/formal_2k_x128.jsonl \
SUMMARY=outputs/ddtree_tree_block_verification_2k_rerun_20260830/formal_2k_x128_summary.json \
bash scripts/run_temperature_block_verification_2k.sh
```
