# NVIDIA GPU 迁移后的 Benchmark 验证协议

## 状态声明

仓库中的既有 `outputs/`、`logs/`、`backups/` 和 checkpoint 是迁移前实验留下的历史产物。
它们用于可追溯性，不得冒充 NVIDIA GPU 性能、吞吐、显存或稳定性结论；原测试日志
明确保留为 `logs/legacy_accelerator_test_speedup.log`。四个 rank-head checkpoint 仅将设备绑定的 tensor
storage tag 重存为可移植格式，权重数值和实验元数据未改，因此仍不代表 GPU 重训练结果。
迁移完成后必须在目标 GPU 上重新训练或验证 checkpoint，并重新生成全部基准结果。

在新的 GPU 结果落盘并通过以下协议前，本文件不报告任何 GPU 加速数字。

## 1. 固定实验输入

| 项目 | 要求 |
|---|---|
| 目标模型 | Qwen/Qwen3-4B，固定 revision |
| 草稿模型 | z-lab/Qwen3-4B-DFlash-b16，固定 revision |
| 配置 | `configs/qwen3_4b_cuda.json` 与对应消融配置 |
| prompt | 固定 `prompts_benchmark_tree15.jsonl` 及其内容哈希 |
| 解码 | greedy、相同 `max_new_tokens`、相同停止 token |
| 精度 | BF16 为吞吐基准；TF32/FP32 单独标注 |
| 设备 | 记录 GPU 型号、数量、显存、compute capability 与功耗状态 |
| 软件 | 记录驱动、CUDA runtime、cuDNN、PyTorch、Transformers 和 Python |

## 2. 必须对比的运行模式

1. 目标模型逐 token greedy baseline；
2. Vanilla DFlash；
3. DFlash-SpecBlock eager + SDPA；
4. DFlash-SpecBlock + CUDA Graph；
5. 可选的 `torch.compile` 模式；
6. 注意力后端消融：SDPA 自动调度，以及经验证可用的 Flash 路径。

每个优化必须独立消融，避免把接受长度变化、图捕获、编译和注意力内核收益混在一起。

## 3. 正确性门槛

- 每个样本与同配置目标模型 greedy 输出逐 token 完全一致；
- CUDA Graph 首次 capture 与后续 replay 输出一致；
- eager、SDPA、图模式和编译模式在声明的容差/精度协议下等价；
- rank checkpoint 的模型 revision、block size、`max_blocks` 和结构元数据匹配；
- 超过 graph cache 容量时明确报错或安全回退，不能静默产生错误结果。

任何 exact-match 失败都优先按正确性故障处理，不计入“有效加速”。

## 4. 性能采集

正式计时前至少执行 20 次 warmup；每种模式至少运行 3 个独立重复。计时边界使用 CUDA
event 或显式同步。至少记录：

- TTFT、prefill、draft、verify、cache compact 和端到端 decode；
- tokens/s、平均每次 verify 提交 token 数和 verify 次数；
- P50/P95/P99 延迟与跨重复置信区间；
- 峰值 allocated/reserved 显存；
- graph capture 一次性成本和稳态 replay 成本；
- OOM、fallback、重编译、graph recapture 和异常次数。

## 5. 精度口径

`allow_tf32=true` 是工业吞吐配置，会改变严格 IEEE FP32 数值口径。报告必须把以下结果分开：

- BF16/Tensor Core 吞吐基准；
- FP32 + TF32 加速基准；
- `allow_tf32=false` 的严格 FP32 对照。

不得把 TF32 结果标记为严格 FP32，也不得跨 dtype 直接归因于算法加速。

## 6. 推荐命令

```bash
export CUDA_VISIBLE_DEVICES=0
python scripts/check_cuda_env.py

bash scripts/run_vanilla_dflash_benchmark.sh
bash scripts/run_tree15_pipeline.sh
python scripts/profile_stages.py
```

新结果应写入新的、带 GPU 与配置标识的输出目录，不覆盖历史产物。报告只引用本次运行实际
生成的文件和环境清单。

## 7. 发布门槛

- correctness 全部通过；
- 在目标上下文长度和并发下无 OOM；
- 长稳测试无 graph replay 错误、显存持续增长或吞吐退化；
- 加速结论基于端到端延迟，而非单个 kernel microbenchmark；
- 原始 JSONL、环境信息、配置、命令和代码版本均可追溯。
