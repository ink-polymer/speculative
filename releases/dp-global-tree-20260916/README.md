# 全局异构树预算调度：冻结 DP 快速版

用户选择保留的 2026-09-16 审计 DP 版本，**不是 RL，也不是 canonical-reference 修复版**。核心源码 SHA256：`3f132a41b659d1306cbfc3fbb348c0f89284d097d69851f0f73d8a6e4518587f`。

## 阅读入口

- [中英文论文摘要](paper/ABSTRACT.md)
- [12 项数学性质、严格证明与反例](paper/DP_THEORY.md)
- [贡献、相关工作及证据边界](paper/CLAIMS_AND_RELATED_WORK.md)
- [全部 60 个配置结果](results/pilots/20260916/audited_global_tree_matrix_h20/ALL_TEMPERATURE_CONCURRENCY_RESULTS.md)
- [GPU 正确性报告，包括未通过项](results/pilots/20260916/shared-budget-correctness-20260916/CORRECTNESS_REPORT.md)
- [源码来源](SOURCE_PROVENANCE.json)、[重新计算的统计](EVIDENCE_SUMMARY.json)

## 框架

多个请求共享一次 Target 前向的行预算。冻结 DFlash Draft 生成位置概率，DDTree 构建候选树，调度器在 B11/B23/B45 档位中选择。B 是非根树节点数，验证行数为 B+1。先因果入场请求，再用 DP 保留每个总行数下的最大加权 Draft 代理收益，最后最大化收益/校准 Target 行代价。请求各自拥有 RNG、逻辑位置、Target KV 和特征历史。

当前还有门控：活跃请求超过8只允许B11；tier-fit 要求每个可选档位能统一覆盖全部入场请求。因此“全局最优”仅指已入场集合和门控后候选集合，不是所有树或未来全部轮次最优。

## 实测与限制

H20 **共同框架诊断复测**：Qwen3-4B/8B，温度0/1，预算96/193/384，并发1/4/8/16/32。12个运行目录、192条配对记录，每条含3次旋转顺序重复。GSM8K、seed17/29、64 token上限；C1每种子四个独立请求，其他并发每种子一个固定批次。

T=1的30配置：DP/DD与DP/DF的逐配置吞吐倍率几何均值为 **1.436907× / 1.225823×**，胜出 **26/30 / 30/30**。这是已有测量重新汇总，不是新增GPU实验、统计显著性或原生 serving 压测。表中综合吞吐另按总输出/总时间计算，不是这里的几何均值。

BF16严格AR数值等价未通过，没有完成任务准确率或完整序列分布认证。不能宣称现实GPU严格无损、质量已认证不变或已达到顶会标准。

## 安装与 CPU 验证

进入本目录。GPU重跑使用Python3.10/3.11，先安装与原实验匹配的CUDA PyTorch，再安装项目：

```bash
python -m pip install -e . pytest
python scripts/verify_release.py
python scripts/check_dp_theorems.py
PYTHONPATH=src:scripts python -m pytest -q tests/gbv_paper
```

定理检查为有理数枚举，不是Lean/Coq形式化证明；CPU测试不证明GPU速度。`verify_release.py` 检查原始文件哈希、六项被测源码、全部结果和汇总。

## H20 复现

模型ID和revision在configs与manifest；权重不上传，需要自行获取。以下脚本重跑该模型的全部30配置：

```bash
PYTHONPATH=src:scripts python scripts/rerun_audited_global_tree_matrix_h20.py \
  --config configs/adaptive_block_qwen3_4b.json \
  --output outputs/reproduction/qwen3_4b \
  --data-dir data/prepared --max-new-tokens 64 --repeats 3 --seeds 17,29
```

替换为8B配置再运行另一模型，不覆盖results。原脚本会覆盖通用配置的温度、预算、概率精度、Draft KV等；配置中2048 token/FP64不是本矩阵实际口径。校准使用合成64 token前缀，并不覆盖所有上下文。

## 发布范围

包括冻结DP执行路径、递归静态依赖、供应商基线及原许可证、全部原始性能记录、独立正确性记录和相关测试。支持模块中保留的旧方法定义不是本版被测方法。六项原始记录哈希和服务器快照已校验；其他支持文件是本次本地依赖收集，不伪称逐文件均有历史冻结认证。

不包含RL代码/结果、未部署修复、权重、缓存、密码、SSH连接脚本和无关架构历史实验。原项目许可见LICENSE；第三方和数据见[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
