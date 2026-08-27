# 16-vCPU temperature=1 GBV 2K rerun

Date: 2026-08-27

This branch records the complete 2,000-prompt rerun of the temperature=1,
three-path GBV implementation on the AutoDL RTX 4090 host.

## Run configuration

- Target: `Qwen/Qwen3-4B`
- Draft: `z-lab/Qwen3-4B-DFlash-b16`
- Target revision: `1cfa9a7208912126459214e8b04321603b3df60c`
- Draft revision: `b74e3a329c4d963783143b1e970d95b002be72bd`
- GPU: NVIDIA GeForce RTX 4090
- Allocated CPU: 16 vCPU (host CPU: Intel Xeon Platinum 8358P)
- Driver: 595.71.05
- PyTorch / CUDA: 2.8.0+cu128 / 12.8
- Transformers: 4.57.1
- Dtype: BF16
- Temperature: 1.0
- GBV paths: 3
- Block size: 15 draft tokens per path
- Maximum new tokens: 128
- Prompts: 2,000 (GSM8K 1,169; HumanEval 149; MATH500 450; MBPP 232)

## Overall result

| Metric | Result |
|---|---:|
| Prompts | 2,000 |
| Generated tokens | 255,756 |
| Mean time per token | 3.7620 ms |
| GBV throughput | 265.82 token/s |
| Mean speedup vs target | 7.3576x |
| Aggregate speedup vs target | 7.3604x |
| Mean committed tokens per verification | 12.4232 |
| Mean verification iterations per prompt | 10.213 |
| Mean tree nodes | 39.6044 |

## Per-dataset speedup

| Dataset | Prompts | Speedup |
|---|---:|---:|
| GSM8K | 1,169 | 7.4548x |
| HumanEval | 149 | 7.3342x |
| MATH500 | 450 | 7.6011x |
| MBPP | 232 | 6.5325x |

The fixed prompt file does not contain answer references or executable test
metadata. These results measure generation performance and verification
efficiency, not task accuracy or distributional equivalence.

## Artifacts and hashes

- Dataset SHA256: `c65f52a856f9ee60d7264596ce91b84e1ffdeb27e5d800a5f4453f1243a5ea27`
- Raw result: `outputs/rerun_t1_gbv_2k_20260827.jsonl`
- Raw result SHA256: `45a082d9cc1fbf2b3124911d7c6040966cb95b452e51746657ebefa26d9cc066`
- Summary: `outputs/rerun_t1_gbv_2k_20260827_summary.json`
- Summary SHA256: `24a61280d8182091fd34a349c11160bc0954dea6917f5d2a4e79363284e0306f`

The separately started four-method comparison was cancelled by request after
19 prompts and is intentionally not included in this branch.
